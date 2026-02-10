#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
verify_stage2.py
使用真实光谱 -> SpectrumEncoder -> prefix -> 生成结构
"""

import argparse
import random
import pickle
import os
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from models.metagpt import MetaGPT
from models.spectrum_encoder import SpectrumEncoder
from structure_lang.parser import StructureParser
from structure_lang.validator import StructureValidator
from structure_lang.tokenizer import StructureTokenizer
from data.gen_stage2_fake_dataset import fake_spectrum_from_structure


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


def apply_constraints(
    logits: torch.Tensor,
    token_strings,
    generated_tokens,
    px=None,
    py=None,
    shape=None,
    margin=30,
):
    vocab = len(token_strings)
    mask = torch.zeros(vocab, dtype=torch.bool, device=logits.device)

    # determine next field
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

    # allow tokens by class
    for tid, token in enumerate(token_strings):
        cls = classify_token(token)
        if need == "EOS" and token == "[EOS]":
            mask[tid] = True
        elif need == cls:
            mask[tid] = True

    # disallow PAD/BOS
    for tid, tok in enumerate(token_strings):
        if tok in ("[PAD]", "[BOS]"):
            mask[tid] = False

    # geometry constraints
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

    logits = logits.clone()
    logits[~mask] = -1e10
    return logits


def sample_one(model, spectrum, tk, device, max_len, top_k=30, top_p=0.95):
    generated = [tk.bos_id]
    token_strings = tk.id_to_token
    generated_tokens = []
    px = None
    py = None
    shape = None

    for step in range(max_len):
        inp = torch.tensor([generated], dtype=torch.long, device=device)
        if isinstance(spectrum, np.ndarray):
            spectrum = torch.from_numpy(spectrum)
        if isinstance(spectrum, torch.Tensor):
            spectrum = spectrum.to(device=device, dtype=torch.float32)
        else:
            spectrum = torch.tensor(spectrum, dtype=torch.float32, device=device)

        if spectrum.dim() == 1:
            spectrum = spectrum.unsqueeze(0)
        elif spectrum.dim() == 2:
            spectrum = spectrum.unsqueeze(0)

        logits, _, _ = model(input_ids=inp, spectra=spectrum)
        logits = logits[0, -1]

        logits = apply_constraints(
            logits,
            token_strings=token_strings,
            generated_tokens=generated_tokens,
            px=px,
            py=py,
            shape=shape,
            margin=30,
        )

        # top-k top-p sampling
        probs = F.softmax(logits, dim=-1)
        if top_k > 0:
            vals, inds = torch.topk(probs, top_k)
            probs = torch.zeros_like(probs).scatter_(0, inds, vals)

        sorted_probs, sorted_idx = torch.sort(probs, descending=True)
        cumsum = torch.cumsum(sorted_probs, dim=-1)
        keep = cumsum <= top_p
        if keep.sum() == 0:
            keep[0] = True
        filtered = torch.zeros_like(probs)
        filtered[sorted_idx[keep]] = probs[sorted_idx[keep]]
        probs = filtered / filtered.sum()

        next_id = torch.multinomial(probs, 1).item()
        next_token = token_strings[next_id]
        generated.append(next_id)

        if next_id == tk.eos_id:
            break

        generated_tokens.append(next_token)
        if next_token.startswith("PX_"):
            px = extract_num(next_token)
        elif next_token.startswith("PY_"):
            py = extract_num(next_token)
        elif next_token.startswith("L1_SHAPE_"):
            shape = next_token.split("_")[-1]

    return generated


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--ckpt", required=True)
    parser.add_argument("--base_ckpt", type=str, default="", help="stage2 ckpt for model_cfg/meta if rl ckpt lacks them")
    parser.add_argument("--spec_file", type=str, default="./dataset_stage2/spec_train.npy")
    parser.add_argument("--cache", type=str, default="./cache_stage2/stage2_latent_auto_cache.pt")
    parser.add_argument("--use_cache", action="store_true", help="use cached prefix instead of spectra")
    parser.add_argument("--n", type=int, default=20)
    parser.add_argument("--max_len", type=int, default=32)
    parser.add_argument("--spec_metric", action="store_true", help="compute spectrum reconstruction error")
    args = parser.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"

    print("[Verify] Loading model:", args.ckpt)
    ckpt = torch.load(args.ckpt, map_location="cpu")

    if "model_cfg" in ckpt and "meta" in ckpt:
        model_cfg = ckpt["model_cfg"]
        meta = ckpt["meta"]
    elif args.base_ckpt:
        base = torch.load(args.base_ckpt, map_location="cpu")
        model_cfg = base["model_cfg"]
        meta = base["meta"]
        print("[Verify] Using model_cfg/meta from base_ckpt:", args.base_ckpt)
    else:
        raise KeyError("model_cfg/meta missing. Provide --base_ckpt pointing to a stage2 checkpoint.")

    model = MetaGPT(
        cfg=model_cfg,
        spec_dim=meta["spec_dim"],
        prefix_len=meta["prefix_len"],
        pad_id=meta["pad_id"]
    )

    state = ckpt["model"]
    # If training used precomputed prefix (no prefix weights), ignore missing prefix.*
    state = {k: v for k, v in state.items() if not k.startswith("prefix.")}
    model.load_state_dict(state, strict=False)

    model.to(device).eval()

    tk = StructureTokenizer()
    tk.id_to_token = [tk.inv_vocab[i] for i in range(tk.vocab_size)]
    parser = StructureParser()
    validator = StructureValidator(min_feature_nm=20, margin_nm=30)

    if args.use_cache:
        cache = torch.load(args.cache, map_location="cpu")
        prefix_list = cache["prefix"]
        model.encoder = nn.Identity()
    else:
        spec_arr = np.load(args.spec_file)
        model.encoder = SpectrumEncoder(
            spec_dim=meta["spec_dim"],
            d_model=meta["d_model"],
            prefix_len=meta["prefix_len"]
        )
    model.encoder = model.encoder.to(device)

    valid_cnt = 0
    reasons = {}
    mse_list = []
    mae_list = []
    corr_list = []

    print("===================================================")
    for i in range(args.n):
        if args.use_cache:
            idx = random.randint(0, len(prefix_list) - 1)
            prefix_vec = prefix_list[idx]
            if isinstance(prefix_vec, np.ndarray):
                prefix_vec = torch.from_numpy(prefix_vec)
            ids = sample_one(
                model=model,
                spectrum=prefix_vec,
                tk=tk,
                device=device,
                max_len=args.max_len
            )
        else:
            idx = random.randint(0, len(spec_arr) - 1)
            s = spec_arr[idx]
            ids = sample_one(
                model=model,
                spectrum=s,
                tk=tk,
                device=device,
                max_len=args.max_len
            )

        toks = [tk.id_to_token[x] for x in ids]

        # validate
        struct = parser.parse(["[BOS]"] + toks[1:-1] + ["[EOS]"])
        ok, reason = validator.validate(struct)
        if ok:
            valid_cnt += 1
        else:
            reasons[reason] = reasons.get(reason, 0) + 1

        # spectrum metric (fake forward)
        if args.spec_metric and not args.use_cache:
            pred_spec = fake_spectrum_from_structure(toks[1:-1], spec_dim=len(s))
            mse = float(np.mean((pred_spec - s) ** 2))
            mae = float(np.mean(np.abs(pred_spec - s)))
            # correlation (safe)
            denom = (np.linalg.norm(pred_spec) * np.linalg.norm(s)) + 1e-8
            corr = float(np.dot(pred_spec, s) / denom)
            mse_list.append(mse)
            mae_list.append(mae)
            corr_list.append(corr)

        print(f"[Sample {i}] idx={idx}")
        print("IDs:", ids)
        print("Tokens:", toks)
        print("Valid:", ok, "Reason:", reason)
        print("---------------------------------------------------")

    print(f"Valid rate: {valid_cnt}/{args.n} = {100*valid_cnt/args.n:.2f}%")
    if reasons:
        print("Top invalid reasons:")
        for k, v in sorted(reasons.items(), key=lambda x: x[1], reverse=True):
            print(f"  {k}: {v}")

    if args.spec_metric and mse_list:
        print("Spectrum reconstruction (fake forward) metrics:")
        print(f"  MSE:  {np.mean(mse_list):.6f}")
        print(f"  MAE:  {np.mean(mae_list):.6f}")
        print(f"  Corr: {np.mean(corr_list):.6f}")


if __name__ == "__main__":
    main()
