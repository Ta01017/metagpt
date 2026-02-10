#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
train_stage2.py
---------------------
Stage-2 (MetaGPT SFT):
  光谱 → SpectrumEncoder → prefix → Transformer 生成结构 tokens
"""

import argparse
import os
import pickle
import numpy as np
import torch
import torch.nn.functional as F
from torch.cuda.amp import autocast, GradScaler
from torch.utils.data import Dataset, DataLoader
from torch.optim import AdamW

from models.metagpt import MetaGPT
from models.transformer_sdpa import TransformerConfig
from models.spectrum_encoder import SpectrumEncoder
from structure_lang.tokenizer import StructureTokenizer
from config_stage2 import TrainConfig



def classify_token(token: str):
    if token.startswith("PX_"):
        return "PX"
    if token.startswith("PY_"):
        return "PY"
    if token.startswith("SUB_"):
        return "SUB"
    if token.startswith("L1_MAT_"):
        return "MAT"
    if token.startswith("L1_SHAPE_"):
        return "SHAPE"
    if token.startswith("L1_H_"):
        return "H"
    if token.startswith("L1_R_"):
        return "R"
    if token.startswith("L1_W_"):
        return "W"
    if token.startswith("L1_L_"):
        return "L"
    return "OTHER"


def extract_num(token: str):
    return int(token.split("_")[-1])


def apply_constraints(logits, token_strings, generated_tokens, px=None, py=None, shape=None, margin=30):
    vocab = len(token_strings)
    mask = torch.zeros(vocab, dtype=torch.bool, device=logits.device)

    if px is None:
        need = "PX"
    elif py is None:
        need = "PY"
    elif shape is None:
        if not any(tok.startswith("SUB_") for tok in generated_tokens):
            need = "SUB"
        elif not any(tok.startswith("L1_MAT_") for tok in generated_tokens):
            need = "MAT"
        else:
            need = "SHAPE"
    else:
        if shape == "CYL":
            if not any(tok.startswith("L1_H_") for tok in generated_tokens):
                need = "H"
            elif not any(tok.startswith("L1_R_") for tok in generated_tokens):
                need = "R"
            else:
                need = "EOS"
        else:
            if not any(tok.startswith("L1_H_") for tok in generated_tokens):
                need = "H"
            elif not any(tok.startswith("L1_W_") for tok in generated_tokens):
                need = "W"
            elif not any(tok.startswith("L1_L_") for tok in generated_tokens):
                need = "L"
            else:
                need = "EOS"

    for tid, token in enumerate(token_strings):
        cls = classify_token(token)
        if need == "EOS" and token == "[EOS]":
            mask[tid] = True
        elif need == cls:
            mask[tid] = True

    for tid, tok in enumerate(token_strings):
        if tok in ("[PAD]", "[BOS]"):
            mask[tid] = False

    if px is not None and py is not None:
        Pmin = min(px, py)
        if need == "R":
            limit = Pmin / 2 - margin
            for tid, token in enumerate(token_strings):
                if token.startswith("L1_R_"):
                    val = extract_num(token)
                    mask[tid] = val <= limit
        if need in ("W", "L"):
            limit = Pmin - margin
            for tid, token in enumerate(token_strings):
                if token.startswith(f"L1_{need}_"):
                    val = extract_num(token)
                    mask[tid] = val <= limit

    if mask.sum() == 0:
        return logits

    # avoid fp16 overflow under AMP
    logits_fp = logits.float().clone()
    logits_fp[~mask] = -1e9
    return logits_fp.to(dtype=logits.dtype)


# ==========================================================
# Stage-2 Dataset
# ==========================================================
class Stage2Dataset(Dataset):
    def __init__(self, spec_arr, struct_list, tokenizer, max_len, use_cot=False, cot_dropout=0.0):
        self.spec = spec_arr              # (N, spec_dim)
        self.struct = struct_list         # list of id lists
        self.tk = tokenizer
        self.max_len = max_len
        self.pad = tokenizer.pad_id
        self.bos = tokenizer.bos_id
        self.eos = tokenizer.eos_id
        self.use_cot = bool(use_cot)
        self.cot_dropout = float(cot_dropout)
        # fixed CoT length: [COT], COT_MAT_*, COT_SHAPE_* (max 3)
        self.cot_max = 3 if self.use_cot else 0

    def build_cot_ids(self, tokens):
        if not self.use_cot:
            return []
        mat = None
        shape = None
        for t in tokens:
            if t.startswith("L1_MAT_"):
                mat = t.split("_", 2)[2]
            if t.startswith("L1_SHAPE_"):
                shape = t.split("_", 2)[2]
        cot = ["[COT]"]
        if mat is not None:
            cot.append(f"COT_MAT_{mat}")
        if shape is not None:
            cot.append(f"COT_SHAPE_{shape}")

        # optional dropout / truncation
        if self.cot_dropout > 0:
            # randomly drop last token
            if len(cot) > 1 and np.random.rand() < self.cot_dropout:
                cot = cot[:-1]
            if len(cot) > 1 and np.random.rand() < self.cot_dropout:
                cot = cot[:-1]

        cot_ids = [self.tk.vocab[t] for t in cot if t in self.tk.vocab]
        return cot_ids[: self.cot_max]

    def pad_ids(self, ids):
        if isinstance(ids, np.ndarray):
            ids = ids.tolist()
        seq = [self.bos] + ids + [self.eos]
        if len(seq) > self.max_len:
            seq = seq[:self.max_len]
        pad_len = self.max_len - len(seq)
        seq = seq + [self.pad]*pad_len
        return np.array(seq, dtype=np.int64)

    def __len__(self):
        return len(self.struct)

    def __getitem__(self, idx):
        spec = self.spec[idx]  # (spec_dim,)
        ids = self.struct[idx]

        # build CoT prefix
        toks = [self.tk.inv_vocab[i] for i in ids]
        cot_ids = self.build_cot_ids(toks)

        seq = cot_ids + [self.bos] + ids + [self.eos]
        max_len_total = self.max_len + self.cot_max
        if len(seq) > max_len_total:
            seq = seq[:max_len_total]
            seq[-1] = self.eos

        inp = seq[:-1]
        tgt = seq[1:]

        pad_len = max_len_total - len(inp)
        if pad_len > 0:
            inp = inp + [self.pad] * pad_len
            tgt = tgt + [-100] * pad_len

        # ignore loss on CoT tokens (and BOS)
        ignore_until = len(cot_ids)
        for i in range(ignore_until):
            if i < len(tgt):
                tgt[i] = -100

        return (
            torch.tensor(spec, dtype=torch.float32),
            torch.tensor(inp, dtype=torch.long),
            torch.tensor(tgt, dtype=torch.long),
        )


# ==========================================================
# Load Stage-1 weights into Stage-2
# ==========================================================
def load_stage1(model: MetaGPT, ckpt_path: str):
    print("[Stage2] Loading Stage-1 checkpoint:", ckpt_path)
    ckpt = torch.load(ckpt_path, map_location="cpu")
    state = ckpt["model"]
    state = {k: v for k, v in state.items() if not k.startswith("prefix.")}

    # handle vocab size mismatch (CoT tokens add new rows)
    model_state = model.state_dict()
    for key in ("tok_embed.lut.weight", "lm_head.weight"):
        if key in state and key in model_state:
            if state[key].shape != model_state[key].shape:
                new_w = model_state[key].clone()
                n = min(state[key].shape[0], new_w.shape[0])
                new_w[:n] = state[key][:n]
                state[key] = new_w

    # Stage-2 比 Stage-1 多 SpectrumEncoder，不 strict
    missing, unexpected = model.load_state_dict(state, strict=False)

    print("Missing:", missing)
    print("Unexpected:", unexpected)


# ==========================================================
# Train Stage-2
# ==========================================================
def train_stage2(args=None):
    cfg = TrainConfig()
    if args is not None:
        if getattr(args, "no_grammar", False):
            cfg.use_grammar = False
        elif getattr(args, "use_grammar", False):
            cfg.use_grammar = True
        if getattr(args, "no_cot", False):
            cfg.use_cot = False
        elif getattr(args, "use_cot", False):
            cfg.use_cot = True
        if hasattr(args, "cot_dropout") and args.cot_dropout is not None:
            cfg.cot_dropout = float(args.cot_dropout)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print("DEVICE =", device)

    os.makedirs(cfg.ckpt_dir, exist_ok=True)

    # ------------------------------------------------------
    # Load data
    # ------------------------------------------------------
    spec_arr = np.load(cfg.spec_file)  # (N, spec_dim)
    with open(cfg.struct_file, "rb") as f:
        struct_list = pickle.load(f)

    tk = StructureTokenizer()
    tk.id_to_token = [tk.inv_vocab[i] for i in range(tk.vocab_size)]
    vocab_size = tk.vocab_size

    print(f"[Data] Loaded {len(struct_list)} samples")

    dataset = Stage2Dataset(
        spec_arr=spec_arr,
        struct_list=struct_list,
        tokenizer=tk,
        max_len=cfg.max_len,
        use_cot=cfg.use_cot,
        cot_dropout=cfg.cot_dropout,
    )
    loader = DataLoader(
        dataset,
        batch_size=cfg.batch_size,
        shuffle=True,
        num_workers=cfg.num_workers,
        pin_memory=True,
        persistent_workers=True if cfg.num_workers > 0 else False,
    )

    # ------------------------------------------------------
    # Build model
    # ------------------------------------------------------
    model_cfg = TransformerConfig(
        vocab_size=vocab_size,
        d_model=cfg.d_model,
        d_ff=cfg.d_ff,
        n_heads=cfg.n_heads,
        n_layers=cfg.n_layers,
        max_len=cfg.max_len + (3 if cfg.use_cot else 0) + cfg.prefix_len + 4,
        dropout=cfg.dropout
    )

    # MetaGPT + SpectrumEncoder
    model = MetaGPT(
        cfg=model_cfg,
        spec_dim=cfg.spec_dim,
        prefix_len=cfg.prefix_len,
        pad_id=tk.pad_id,
    ).to(device)

    # 加载 Stage-1 权重
    load_stage1(model, cfg.stage1_ckpt)

    # 手动 attach encoder，MetaGPT 会使用 self.encoder？
    model.encoder = SpectrumEncoder(
        spec_dim=cfg.spec_dim,
        d_model=cfg.d_model,
        prefix_len=cfg.prefix_len
    ).to(device)

    optim = AdamW(model.parameters(), lr=cfg.lr)
    scaler = GradScaler(enabled=(cfg.use_amp and device == "cuda"))

    # ------------------------------------------------------
    # Training Loop
    # ------------------------------------------------------
    step = 0
    while step < cfg.total_steps:
        for spec, inp, tgt in loader:
            step += 1
            spec = spec.to(device)
            inp = inp.to(device)
            tgt = tgt.to(device)

            with autocast(enabled=(cfg.use_amp and device == "cuda")):
                logits, loss, aux = model(
                    input_ids=inp,
                    spectra=spec,
                    labels=tgt
                )

            if cfg.use_grammar:
                token_strings = tk.id_to_token
                B, T, V = logits.shape
                for b in range(B):
                    in_struct = False
                    generated_tokens = []
                    px = None
                    py = None
                    shape = None
                    for t in range(T):
                        if tgt[b, t].item() == -100:
                            continue
                        tok = token_strings[inp[b, t].item()]
                        if tok == "[BOS]":
                            in_struct = True
                            generated_tokens = []
                            px = None
                            py = None
                            shape = None
                        elif in_struct and tok not in ("[PAD]", "[EOS]"):
                            generated_tokens.append(tok)
                            if tok.startswith("PX_"):
                                px = extract_num(tok)
                            elif tok.startswith("PY_"):
                                py = extract_num(tok)
                            elif tok.startswith("L1_SHAPE_"):
                                shape = tok.split("_")[-1]

                        if in_struct:
                            logits[b, t] = apply_constraints(
                                logits[b, t],
                                token_strings=token_strings,
                                generated_tokens=generated_tokens,
                                px=px,
                                py=py,
                                shape=shape,
                                margin=cfg.grammar_margin,
                            )

            optim.zero_grad()
            if scaler.is_enabled():
                scaler.scale(loss).backward()
                scaler.step(optim)
                scaler.update()
            else:
                loss.backward()
                optim.step()

            if step % cfg.log_every == 0:
                print(f"[step {step}] loss={loss.item():.4f}")

            if step % cfg.save_every == 0:
                path = f"{cfg.ckpt_dir}/stage2_step{step}.pt"
                torch.save({
                    "model": model.state_dict(),
                    "model_cfg": model_cfg,
                    "meta": {
                        "vocab_size": vocab_size,
                        "d_model": model_cfg.d_model,
                        "prefix_len": cfg.prefix_len,
                        "spec_dim": cfg.spec_dim,
                        "bos_id": tk.bos_id,
                        "eos_id": tk.eos_id,
                        "pad_id": tk.pad_id,
                    }
                }, path)
                print("[Save]", path)

            if step >= cfg.total_steps:
                break

    print("[Stage2] Training done.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--use_grammar", action="store_true", help="enable grammar constraints")
    parser.add_argument("--no_grammar", action="store_true", help="disable grammar constraints")
    parser.add_argument("--use_cot", action="store_true", help="enable CoT tokens")
    parser.add_argument("--no_cot", action="store_true", help="disable CoT tokens")
    parser.add_argument("--cot_dropout", type=float, default=None, help="override CoT dropout")
    args = parser.parse_args()
    train_stage2(args)
