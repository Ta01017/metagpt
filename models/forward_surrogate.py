#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Forward surrogate model: structure token sequence -> spectrum vector.
Keep self-contained to avoid touching existing code.
"""

from __future__ import annotations

import math
import torch
import torch.nn as nn


class PositionalEncoding(nn.Module):
    def __init__(self, d_model: int, dropout: float = 0.1, max_len: int = 5000):
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float32).unsqueeze(1)
        div_term = torch.exp(
            torch.arange(0, d_model, 2, dtype=torch.float32) * (-(math.log(10000.0) / d_model))
        )
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)  # (1, T, D)
        self.register_buffer("pe", pe, persistent=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, T, D)
        x = x + self.pe[:, : x.size(1)]
        return self.dropout(x)


class ForwardSurrogate(nn.Module):
    def __init__(
        self,
        vocab_size: int,
        spec_dim: int,
        d_model: int = 256,
        n_heads: int = 8,
        n_layers: int = 2,
        d_ff: int = 512,
        dropout: float = 0.1,
        max_len: int = 512,
    ):
        super().__init__()
        self.embed = nn.Embedding(vocab_size, d_model)
        self.pos = PositionalEncoding(d_model, dropout=dropout, max_len=max_len)
        layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=n_heads,
            dim_feedforward=d_ff,
            dropout=dropout,
            batch_first=True,
        )
        self.encoder = nn.TransformerEncoder(layer, num_layers=n_layers)
        self.head = nn.Sequential(
            nn.Linear(d_model, d_model),
            nn.ReLU(),
            nn.Linear(d_model, spec_dim),
        )

    def forward(self, tokens: torch.Tensor, pad_id: int) -> torch.Tensor:
        """
        tokens: (B, T) long
        return: (B, spec_dim)
        """
        x = self.embed(tokens)
        x = self.pos(x)
        pad_mask = tokens.eq(pad_id)  # (B, T) True for PAD
        x = self.encoder(x, src_key_padding_mask=pad_mask)
        valid = (~pad_mask).unsqueeze(-1).float()
        lengths = valid.sum(dim=1).clamp(min=1.0)
        pooled = (x * valid).sum(dim=1) / lengths
        return self.head(pooled)


def build_surrogate_from_ckpt(ckpt: dict, device: torch.device):
    cfg = ckpt["model_cfg"]
    meta = ckpt["meta"]
    model = ForwardSurrogate(
        vocab_size=meta["vocab_size"],
        spec_dim=meta["spec_dim"],
        d_model=cfg["d_model"],
        n_heads=cfg["n_heads"],
        n_layers=cfg["n_layers"],
        d_ff=cfg["d_ff"],
        dropout=cfg["dropout"],
        max_len=cfg.get("max_len", 512),
    ).to(device)
    model.load_state_dict(ckpt["model"], strict=True)
    return model, meta
