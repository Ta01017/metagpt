#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
gen_stage2_synth_cache.py

符合 MetaGPT 思路的 Stage-2 数据生成版本：

structure  →  synthetic spectrum  →  SpectrumEncoder → prefix

输出：
  cache_stage2/stage2_latent_auto_cache.pt
    - prefix: (N, prefix_len, d_model)
    - struct: token-id list
    - vocab_size / d_model / prefix_len
"""

import os
import sys
import torch
import torch.nn as nn
import numpy as np
from pathlib import Path
from typing import List, Dict, Any

# ================================================================
# Path Setup
# ================================================================
_ROOT = Path(__file__).resolve().parents[1]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from structure_lang.tokenizer import StructureTokenizer
from structure_lang.parser import StructureParser
from structure_lang.validator import StructureValidator

from models.transformer_sdpa import TransformerConfig
from models.metagpt import MetaGPT


CACHE_DIR = "./cache_stage2"
os.makedirs(CACHE_DIR, exist_ok=True)
OUT_FILE = f"{CACHE_DIR}/stage2_latent_auto_cache.pt"


def _cfg_get(cfg, key):
    if isinstance(cfg, dict):
        return cfg[key]
    return getattr(cfg, key)


def _load_model_cfg_from_ckpt(ckpt) -> TransformerConfig:
    if "model_cfg" in ckpt:
        return TransformerConfig(**ckpt["model_cfg"])
    if "config" in ckpt:
        cfg = ckpt["config"]
        return TransformerConfig(
            vocab_size=_cfg_get(cfg, "vocab_size"),
            d_model=_cfg_get(cfg, "d_model"),
            d_ff=_cfg_get(cfg, "d_ff"),
            n_heads=_cfg_get(cfg, "n_heads"),
            n_layers=_cfg_get(cfg, "n_layers"),
            max_len=_cfg_get(cfg, "max_len") + 8,
            dropout=_cfg_get(cfg, "dropout"),
        )
    raise KeyError("Checkpoint missing 'model_cfg' or 'config'.")


# ================================================================
# 1. 轻量伪光谱生成器（无需 Meep/Tidy3D）
# ================================================================
def fake_spectrum_from_structure(toks: List[str], spec_dim=256) -> np.ndarray:
    """
    输入结构 tokens，输出一个光谱向量 (spec_dim,)
    可控；与特征大小相关；能用于训练 SFT。
    """
    spec = np.zeros(spec_dim, dtype="float32")

    # 提取几何特征
    P = 500
    H = 300
    W = 200
    L = 200
    R = 100

    for t in toks:
        if t.startswith("PX_"):
            P = int(t.split("_")[1])
        if t.startswith("L1_H_"):
            H = int(t.split("_")[2])
        if t.startswith("L1_R_"):
            R = int(t.split("_")[2])
        if t.startswith("L1_W_"):
            W = int(t.split("_")[2])
        if t.startswith("L1_L_"):
            L = int(t.split("_")[2])

    # 简单映射关系（可换成更复杂）
    xs = np.linspace(0, 1, spec_dim)

    base = np.sin(xs * np.pi * H / 400)
    shape_mod = np.cos(xs * np.pi * (W + L + R) / 800)

    spec = 0.4 * base + 0.6 * shape_mod
    spec = (spec - spec.min()) / (spec.max() - spec.min() + 1e-6)

    return spec.astype("float32")


# ================================================================
# 2. SpectrumEncoder（与 Stage-2/3 共享）
# ================================================================
class SpectrumEncoder(nn.Module):
    """MLP: (B, spec_dim) → (B, prefix_len, d_model)"""

    def __init__(self, spec_dim, d_model, prefix_len):
        super().__init__()
        self.spec_dim = spec_dim
        self.prefix_len = prefix_len

        self.net = nn.Sequential(
            nn.Linear(spec_dim, d_model),
            nn.GELU(),
            nn.Linear(d_model, d_model * prefix_len),
        )

    def forward(self, x):
        B = x.size(0)
        h = self.net(x)
        return h.view(B, self.prefix_len, -1)


# ================================================================
# 3. Stage-2 Cache Build
# ================================================================
def build_stage2_cache(
    stage1_ckpt: str,
    N: int = 20000,
    prefix_len: int = 16,
    spec_dim: int = 256,
):
    print("[Stage2-Cache] Loading Stage1:", stage1_ckpt)

    # ----------------------------
    # Load Stage-1 model
    # ----------------------------
    ckpt = torch.load(stage1_ckpt, map_location="cpu")
    model_cfg: TransformerConfig = _load_model_cfg_from_ckpt(ckpt)

    # Align cfg with checkpoint weights to avoid size mismatch
    state = ckpt.get("model", {})
    emb_w = state.get("tok_embed.lut.weight")
    if emb_w is not None:
        if model_cfg.vocab_size != emb_w.shape[0]:
            print(f"[Stage2-Cache] Override vocab_size {model_cfg.vocab_size} -> {emb_w.shape[0]}")
            model_cfg.vocab_size = int(emb_w.shape[0])
        if model_cfg.d_model != emb_w.shape[1]:
            print(f"[Stage2-Cache] Override d_model {model_cfg.d_model} -> {emb_w.shape[1]}")
            model_cfg.d_model = int(emb_w.shape[1])

    vocab_size = model_cfg.vocab_size
    d_model = model_cfg.d_model

    # 构建 Stage-1 模型（无 prefix）
    model = MetaGPT(
        cfg=model_cfg,
        spec_dim=1,
        prefix_len=0,
        pad_id=0,
    )
    model.load_state_dict(state, strict=False)
    model.eval()

    # -----------------------------
    # Tokenizer + StructureGenerator
    # -----------------------------
    tk = StructureTokenizer()
    parser = StructureParser()
    val = StructureValidator(min_feature_nm=20, margin_nm=50)

    from data.gen_stage1_synth import Stage1SynthGenerator
    gen = Stage1SynthGenerator(
        tokenizer=tk,
        parser=parser,
        validator=val,
        seed=0,
    )

    # -----------------------------
    # SpectrumEncoder
    # -----------------------------
    enc = SpectrumEncoder(spec_dim, d_model, prefix_len)
    enc.eval()

    all_prefix = []
    all_struct = []

    print(f"[Stage2-Cache] Generating {N} samples ...")
    for i in range(N):
        ids = gen.sample_one_ids()  # structure ids list
        toks = tk.decode(ids)

        # -----------------------------
        # 结构 → 伪光谱
        # -----------------------------
        spec = fake_spectrum_from_structure(toks, spec_dim)

        # -----------------------------
        # 伪光谱 → prefix (K, d_model)
        # -----------------------------
        spec_t = torch.tensor(spec, dtype=torch.float32).unsqueeze(0)
        pref = enc(spec_t).squeeze(0).detach().numpy()

        all_prefix.append(pref.astype("float32"))  # (K,d)
        all_struct.append(np.array(ids, dtype=np.int32))

        if i % 1000 == 0:
            print(f"  [{i}/{N}]")

    # -----------------------------
    # Save cache
    # -----------------------------
    torch.save(
        {
            "prefix": all_prefix,     # list of (K,d)
            "struct": all_struct,     # list of id[]
            "vocab_size": vocab_size,
            "d_model": d_model,
            "prefix_len": prefix_len,
        },
        OUT_FILE
    )

    print("[Stage2-Cache] Saved ->", OUT_FILE)


# ================================================================
if __name__ == "__main__":
    build_stage2_cache(
        stage1_ckpt="./ckpt_stage1/stage1_step50000.pt",
        N=20000,
        prefix_len=16,
        spec_dim=256,
    )
