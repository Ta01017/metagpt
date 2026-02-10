# models/transformer_sdpa.py
import math
import copy
from dataclasses import dataclass
from typing import Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F


def clones(module, N):
    return nn.ModuleList([copy.deepcopy(module) for _ in range(N)])


def _ensure_bool(mask: Optional[torch.Tensor]) -> Optional[torch.Tensor]:
    if mask is None:
        return None
    return mask.to(dtype=torch.bool)


def make_causal_keep_mask(L: int, device=None) -> torch.Tensor:
    """
    keep mask: True means allowed attention
    shape: (1, L, L) broadcastable to (B, 1, L, L)
    """
    upper = torch.triu(torch.ones(L, L, device=device, dtype=torch.bool), diagonal=1)
    keep = ~upper
    return keep.unsqueeze(0)


def make_pad_keep_mask(pad_mask: torch.Tensor) -> torch.Tensor:
    """
    pad_mask: (B, L) where True means 'is valid token' (keep)
    returns keep mask shape (B, 1, 1, L)
    """
    pad_mask = pad_mask.to(torch.bool)
    return pad_mask[:, None, None, :]


def combine_keep_masks(*masks: Optional[torch.Tensor]) -> Optional[torch.Tensor]:
    """
    All masks are keep-masks with True=allowed.
    Supports shapes:
      (1, L, L)
      (B, 1, 1, L)
      (B, 1, L, L)
      (B, H, L, L)
    Returns AND combination.
    """
    out = None
    for m in masks:
        if m is None:
            continue
        m = _ensure_bool(m)
        out = m if out is None else (out & m)
    return out


def keep_to_sdpa_maskout(keep_mask: Optional[torch.Tensor]) -> Optional[torch.Tensor]:
    """
    Convert keep-mask (True=allowed) -> mask-out (True=masked)
    for torch.scaled_dot_product_attention(bool mask).
    """
    if keep_mask is None:
        return None
    keep_mask = keep_mask.to(torch.bool)
    return ~keep_mask


def _to_sdpa_mask_shape(maskout: Optional[torch.Tensor], B: int, H: int, q_len: int, k_len: int) -> Optional[torch.Tensor]:
    """
    SDPA accepts maskout shapes like:
      (B, H, q, k) or broadcastable variants.
    We accept common keep-mask variants already broadcastable.
    """
    if maskout is None:
        return None
    m = maskout
    # allow (1,q,k)
    if m.dim() == 3 and m.size(0) == 1 and m.size(1) == q_len and m.size(2) == k_len:
        return m[:, None, :, :]  # (1,1,q,k) broadcast to (B,H,q,k)
    # (B,1,1,k)
    if m.dim() == 4 and m.size(0) == B and m.size(-1) == k_len:
        return m
    # (B,1,q,k) or (B,H,q,k)
    if m.dim() == 4 and m.size(0) == B and m.size(-2) == q_len and m.size(-1) == k_len:
        return m
    raise ValueError(f"Unsupported SDPA mask shape: {tuple(m.shape)}")


class MultiHeadedAttentionSDPA(nn.Module):
    """
    SDPA MHA with keep-mask API (True=allowed).
    Internally converts to SDPA mask-out (True=masked).
    """
    def __init__(self, h: int, d_model: int, dropout: float = 0.1, is_causal: bool = False):
        super().__init__()
        assert d_model % h == 0
        self.h = h
        self.d_k = d_model // h
        self.linears = clones(nn.Linear(d_model, d_model), 4)
        self.dropout_p = float(dropout)
        self.is_causal = bool(is_causal)

    def forward(self, query, key, value, keep_mask: Optional[torch.Tensor] = None):
        # query/key/value: (B, T, D)
        B, q_len, _ = query.shape
        _, k_len, _ = key.shape

        q, k, v = [
            l(x).view(B, -1, self.h, self.d_k).transpose(1, 2)
            for l, x in zip(self.linears, (query, key, value))
        ]  # (B,H,T,d_k)

        maskout = keep_to_sdpa_maskout(keep_mask)
        attn_mask = _to_sdpa_mask_shape(maskout, B=B, H=self.h, q_len=q_len, k_len=k_len)

        # keep_mask: True means "allowed"
        # SDPA bool attn_mask: True means "DISALLOWED"
        if attn_mask is not None:
            attn_mask = ~attn_mask

        out = F.scaled_dot_product_attention(
            q, k, v,
            attn_mask=attn_mask,
            dropout_p=self.dropout_p if self.training else 0.0,
            is_causal=self.is_causal
        )
        out = out.transpose(1, 2).contiguous().view(B, q_len, self.h * self.d_k)
        return self.linears[-1](out)


class SublayerConnection(nn.Module):
    """Pre-norm: x + dropout(sublayer(LN(x)))"""
    def __init__(self, size: int, dropout: float):
        super().__init__()
        self.norm = nn.LayerNorm(size)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, sublayer):
        return x + self.dropout(sublayer(self.norm(x)))


class PositionwiseFeedForward(nn.Module):
    def __init__(self, d_model: int, d_ff: int, dropout: float = 0.1):
        super().__init__()
        self.w1 = nn.Linear(d_model, d_ff)
        self.w2 = nn.Linear(d_ff, d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        return self.w2(self.dropout(F.gelu(self.w1(x))))


class DecoderLayer(nn.Module):
    def __init__(self, size: int, self_attn: nn.Module, feed_forward: nn.Module, dropout: float):
        super().__init__()
        self.size = size
        self.self_attn = self_attn
        self.feed_forward = feed_forward
        self.sublayer = clones(SublayerConnection(size, dropout), 2)

    def forward(self, x, keep_mask: Optional[torch.Tensor] = None):
        x = self.sublayer[0](x, lambda t: self.self_attn(t, t, t, keep_mask))
        x = self.sublayer[1](x, self.feed_forward)
        return x


class Decoder(nn.Module):
    def __init__(self, layer: DecoderLayer, N: int):
        super().__init__()
        self.layers = clones(layer, N)
        self.norm = nn.LayerNorm(layer.size)

    def forward(self, x, keep_mask: Optional[torch.Tensor] = None):
        for layer in self.layers:
            x = layer(x, keep_mask)
        return self.norm(x)


class Embeddings(nn.Module):
    def __init__(self, d_model: int, vocab: int):
        super().__init__()
        self.lut = nn.Embedding(vocab, d_model)
        self.d_model = d_model

    def forward(self, x):
        return self.lut(x) * math.sqrt(self.d_model)


class PositionalEncoding(nn.Module):
    def __init__(self, d_model: int, dropout: float, max_len: int = 8192):
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)

        pe = torch.zeros(max_len, d_model, dtype=torch.float32)
        position = torch.arange(0, max_len, dtype=torch.float32).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2, dtype=torch.float32) * (-(math.log(10000.0) / d_model)))
        pe_pos = position * div_term
        pe[:, 0::2] = torch.sin(pe_pos)
        pe[:, 1::2] = torch.cos(pe_pos)
        self.register_buffer("pe", pe.unsqueeze(0), persistent=False)

    def forward(self, x):
        x = x + self.pe[:, : x.size(1)]
        return self.dropout(x)


@dataclass
class TransformerConfig:
    vocab_size: int
    d_model: int = 512
    n_layers: int = 8
    n_heads: int = 8
    d_ff: int = 2048
    dropout: float = 0.1
    max_len: int = 1024
