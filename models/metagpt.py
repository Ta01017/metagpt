#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional

from .transformer_sdpa import (
    TransformerConfig,
    MultiHeadedAttentionSDPA,
    PositionwiseFeedForward,
    DecoderLayer,
    Decoder,
    Embeddings,
    PositionalEncoding,
)

# ============================================================
#  Spectrum → Prefix Encoder (Internal fallback)
# ============================================================

class SpectrumPrefixEncoder(nn.Module):
    """
    spectra: (B, spec_dim) → (B, prefix_len, d_model)
    简单 MLP：用于 Stage-1 fallback。
    Stage-2/3 会替换为外部 SpectrumEncoder。
    """
    def __init__(self, spec_dim: int, d_model: int, prefix_len: int):
        super().__init__()
        self.spec_dim = spec_dim
        self.prefix_len = prefix_len

        self.mlp = nn.Sequential(
            nn.Linear(spec_dim, d_model),
            nn.GELU(),
            nn.Linear(d_model, d_model * prefix_len),
        )

    def forward(self, spectra):
        B = spectra.size(0)
        x = self.mlp(spectra)              # (B, K*d)
        return x.view(B, self.prefix_len, -1)


# ============================================================
#  Prefix-aware Attention Mask
# ============================================================

def build_full_attn_mask(prefix_len: int, T: int, device):
    """
    prefix tokens：非因果，全连接
    target tokens：因果
    输出：bool mask，True=允许，False=禁止
    """
    L = prefix_len + T
    mask = torch.zeros((L, L), dtype=torch.bool, device=device)

    # prefix → 全可见（非因果）
    if prefix_len > 0:
        mask[:prefix_len, :L] = True             # prefix queries
        mask[prefix_len:, :prefix_len] = True     # targets can see prefix

    # token → causal
    causal = torch.tril(torch.ones((T, T), dtype=torch.bool, device=device))
    mask[prefix_len:, prefix_len:] = causal

    return mask.unsqueeze(0)  # (1,L,L)


# ============================================================
#  MetaGPT Model
# ============================================================

class MetaGPT(nn.Module):
    """
    Decoder-only Transformer supporting:
      - Stage-1 LM pretrain (no spectra)
      - Stage-2 SFT (spectra → prefix)
      - Stage-3 RL fine-tuning
    """
    def __init__(
        self,
        cfg: TransformerConfig,
        spec_dim: int,
        prefix_len: int,
        pad_id: int,
    ):
        super().__init__()

        self.cfg = cfg
        self.spec_dim = spec_dim
        self.prefix_len = prefix_len
        self.pad_id = int(pad_id)

        # token embedding
        self.tok_embed = Embeddings(cfg.d_model, cfg.vocab_size)

        # positional encoding
        self.pos_enc = PositionalEncoding(
            cfg.d_model,
            cfg.dropout,
            max_len=cfg.max_len + prefix_len + 8
        )

        # internal prefix encoder（Stage-2/3 将被外部 encoder 替换）
        self.prefix = SpectrumPrefixEncoder(spec_dim, cfg.d_model, prefix_len)

        # external encoder (可在 Stage2/3 注入)
        self.encoder = None

        # decoder layers
        self_attn = MultiHeadedAttentionSDPA(
            cfg.n_heads, cfg.d_model,
            dropout=cfg.dropout,
            is_causal=False
        )
        ff = PositionwiseFeedForward(cfg.d_model, cfg.d_ff, dropout=cfg.dropout)
        layer = DecoderLayer(cfg.d_model, self_attn, ff, cfg.dropout)
        self.decoder = Decoder(layer, cfg.n_layers)

        # LM head
        self.lm_head = nn.Linear(cfg.d_model, cfg.vocab_size, bias=False)

        self.reset_parameters()

    def reset_parameters(self):
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)

    # ========================================================
    # Forward
    # ========================================================
    def forward(
        self,
        input_ids: torch.Tensor,                # (B,T)
        spectra: Optional[torch.Tensor] = None, # (B, spec_dim) or None
        labels: Optional[torch.Tensor] = None,  # (B,T)
        attention_keep: Optional[torch.Tensor] = None,
    ):
        B, T = input_ids.size()
        device = input_ids.device

        # token embed
        tok = self.tok_embed(input_ids)  # (B,T,D)

        # ---------------------------------------------------
        # prefix build：关键逻辑（对齐 MetaGPT）
        # ---------------------------------------------------
        if spectra is None:
            # Stage-1
            x = tok
            prefix_len = 0

        else:
            # Stage-2/3
            if self.encoder is not None:
                pref = self.encoder(spectra)        # 外部 encoder（SFT/RL）
            else:
                pref = self.prefix(spectra)         # fallback

            prefix_len = pref.size(1)
            x = torch.cat([pref, tok], dim=1)       # (B,K+T,D)

        # pos encoding
        x = self.pos_enc(x)
        L = x.size(1)

        # ---------------------------------------------------
        # attention mask
        # ---------------------------------------------------
        attn_mask = build_full_attn_mask(prefix_len, T, device)

        # padding mask
        token_keep = (input_ids != self.pad_id) if attention_keep is None else attention_keep.to(torch.bool)

        if prefix_len > 0:
            prefix_keep = torch.ones((B, prefix_len), device=device, dtype=torch.bool)
            full_keep = torch.cat([prefix_keep, token_keep], dim=1)
        else:
            full_keep = token_keep

        pad_mask = full_keep[:, None, None, :]  # (B,1,1,L)
        final_mask = attn_mask & pad_mask       # broadcast

        # ---------------------------------------------------
        # decoder forward
        # ---------------------------------------------------
        out = self.decoder(x, keep_mask=final_mask)

        # LM on token region only
        tok_out = out[:, prefix_len:, :]
        logits = self.lm_head(tok_out)   # (B,T,V)

        loss = None
        if labels is not None:
            loss = F.cross_entropy(
                logits.reshape(-1, self.cfg.vocab_size),
                labels.reshape(-1),
                ignore_index=-100
            )

        aux = dict(prefix_len=prefix_len, seq_len=T, full_len=L)
        return logits, loss, aux

    # ========================================================
    # Autoregressive Generation
    # ========================================================
    @torch.no_grad()
    def generate(
        self,
        spectra: Optional[torch.Tensor],
        bos_id: int,
        eos_id: int,
        max_new_tokens: int,
        temperature: float = 1.0,
        top_k: int = 0,
        top_p: float = 1.0,
        greedy: bool = False,
    ):
        device = spectra.device if spectra is not None else "cpu"
        B = spectra.size(0) if spectra is not None else 1

        out = torch.full((B, 1), bos_id, device=device, dtype=torch.long)

        for _ in range(max_new_tokens):
            logits, _, _ = self.forward(out, spectra=spectra)
            logits = logits[:, -1, :]  # (B,V)

            if greedy:
                next_id = torch.argmax(logits, dim=-1)

            else:
                p = logits / max(temperature, 1e-8)

                # top-k
                if top_k > 0:
                    vals, _ = torch.topk(p, k=min(top_k, p.size(-1)))
                    kth = vals[:, -1].unsqueeze(-1)
                    p = torch.where(p < kth, torch.full_like(p, -1e9), p)

                # top-p
                if top_p < 1.0:
                    sorted_p, sorted_idx = torch.sort(p, descending=True)
                    probs = torch.softmax(sorted_p, dim=-1)
                    cumsum = probs.cumsum(dim=-1)

                    cutoff = cumsum > top_p
                    cutoff[:, 0] = False
                    sorted_p = torch.where(cutoff, torch.full_like(sorted_p, -1e9), sorted_p)

                    p = torch.zeros_like(p).scatter(-1, sorted_idx, sorted_p)

                probs = torch.softmax(p, dim=-1)
                next_id = torch.multinomial(probs, 1).squeeze(-1)

            out = torch.cat([out, next_id.unsqueeze(1)], dim=1)
            if (next_id == eos_id).all():
                break

        return out
