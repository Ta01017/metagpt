#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
compare_stage2_stage3.py
对比 Stage-2 与 Stage-3（RL）checkpoint 在同一批光谱上的生成质量。

输出：
  - 合法率与主要非法原因
  - 结构长度统计
  - 可选的光谱重建（fake forward）指标
  - 可选样例展示
"""

import argparse
import random
import numpy as np
import torch
import torch.nn.functional as F

from models.metagpt import MetaGPT
from models.transformer_sdpa import TransformerConfig
from models.spectrum_encoder import SpectrumEncoder
from structure_lang.tokenizer import StructureTokenizer
from structure_lang.parser import StructureParser
from structure_lang.validator import StructureValidator
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

    logits = logits.clone()
    logits[~mask] = -1e10
    return logits


@torch.no_grad()
def sample_one(model, spectrum, tk, device, max_len, top_k=30, top_p=0.95, greedy=False):
    generated = [tk.bos_id]
    token_strings = tk.id_to_token
    generated_tokens = []
    px = None
    py = None
    shape = None

    for _ in range(max_len):
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

        if greedy:
            next_id = torch.argmax(logits).item()
        else:
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

        generated.append(next_id)
        if next_id == tk.eos_id:
            break

        next_token = token_strings[next_id]
        generated_tokens.append(next_token)
        if next_token.startswith("PX_"):
            px = extract_num(next_token)
        elif next_token.startswith("PY_"):
            py = extract_num(next_token)
        elif next_token.startswith("L1_SHAPE_"):
            shape = next_token.split("_")[-1]

    return generated


def _cfg_from_ckpt(ckpt):
    cfg = ckpt.get("model_cfg", None)
    if cfg is None:
        raise KeyError("Checkpoint missing model_cfg")
    if isinstance(cfg, dict):
        return TransformerConfig(**cfg)
    return cfg


def _align_state_dict(state, model):
    model_state = model.state_dict()
    for key in ("tok_embed.lut.weight", "lm_head.weight"):
        if key in state and key in model_state:
            if state[key].shape != model_state[key].shape:
                new_w = model_state[key].clone()
                n = min(state[key].shape[0], new_w.shape[0])
                new_w[:n] = state[key][:n]
                state[key] = new_w
    return state


def load_model(ckpt_path, base_ckpt, tk, spec_dim_fallback, device):
    ckpt = torch.load(ckpt_path, map_location="cpu")

    if "model_cfg" in ckpt and "meta" in ckpt:
        model_cfg = _cfg_from_ckpt(ckpt)
        meta = ckpt["meta"]
    elif base_ckpt:
        base = torch.load(base_ckpt, map_location="cpu")
        model_cfg = _cfg_from_ckpt(base)
        meta = base["meta"]
        print(f"[Load] {ckpt_path} 使用 base_ckpt: {base_ckpt}")
    else:
        raise KeyError("model_cfg/meta missing. Provide --base_ckpt for this checkpoint.")

    spec_dim = int(meta.get("spec_dim", spec_dim_fallback))
    prefix_len = int(meta.get("prefix_len", 16))
    pad_id = int(meta.get("pad_id", tk.pad_id))
    d_model = int(meta.get("d_model", model_cfg.d_model))

    model = MetaGPT(cfg=model_cfg, spec_dim=spec_dim, prefix_len=prefix_len, pad_id=pad_id).to(device)
    model.encoder = SpectrumEncoder(spec_dim=spec_dim, d_model=d_model, prefix_len=prefix_len).to(device)

    state = _align_state_dict(ckpt["model"], model)
    missing, unexpected = model.load_state_dict(state, strict=False)
    if missing or unexpected:
        print(f"[Load] {ckpt_path} missing={len(missing)} unexpected={len(unexpected)}")

    model.eval()
    return model, spec_dim


def update_stats(stats, ok, reason, ids, toks, spec, spec_metric):
    stats["count"] += 1
    stats["lens"].append(len(ids))
    if ok:
        stats["valid"] += 1
    else:
        stats["reasons"][reason] = stats["reasons"].get(reason, 0) + 1

    if spec_metric:
        pred = fake_spectrum_from_structure(toks[1:-1], spec_dim=len(spec))
        mse = float(np.mean((pred - spec) ** 2))
        mae = float(np.mean(np.abs(pred - spec)))
        denom = (np.linalg.norm(pred) * np.linalg.norm(spec)) + 1e-8
        corr = float(np.dot(pred, spec) / denom)
        stats["mse"].append(mse)
        stats["mae"].append(mae)
        stats["corr"].append(corr)


def print_report(name, stats, spec_metric):
    valid = stats["valid"]
    total = stats["count"]
    print(f"=== {name} ===")
    print(f"Valid rate: {valid}/{total} = {100*valid/total:.2f}%")
    print(f"Avg length: {np.mean(stats['lens']):.2f}")
    if stats["reasons"]:
        print("Top invalid reasons:")
        for k, v in sorted(stats["reasons"].items(), key=lambda x: x[1], reverse=True):
            print(f"  {k}: {v}")
    if spec_metric and stats["mse"]:
        print("Spectrum reconstruction (fake forward) metrics:")
        print(f"  MSE:  {np.mean(stats['mse']):.6f}")
        print(f"  MAE:  {np.mean(stats['mae']):.6f}")
        print(f"  Corr: {np.mean(stats['corr']):.6f}")
    print()


def compute_oracle_metrics(struct_list, idxs, tk, spec_arr):
    mse_list = []
    mae_list = []
    corr_list = []
    for idx in idxs:
        ids = struct_list[idx]
        toks = [tk.inv_vocab[i] for i in ids]
        pred = fake_spectrum_from_structure(toks, spec_dim=len(spec_arr[idx]))
        s = spec_arr[idx]
        mse = float(np.mean((pred - s) ** 2))
        mae = float(np.mean(np.abs(pred - s)))
        denom = (np.linalg.norm(pred) * np.linalg.norm(s)) + 1e-8
        corr = float(np.dot(pred, s) / denom)
        mse_list.append(mse)
        mae_list.append(mae)
        corr_list.append(corr)
    return mse_list, mae_list, corr_list


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--ckpt2", required=True, help="stage2 ckpt")
    parser.add_argument("--ckpt3", required=True, help="stage3 rl ckpt")
    parser.add_argument("--base_ckpt2", default="", help="optional base ckpt for stage2 if missing meta")
    parser.add_argument("--base_ckpt3", default="", help="optional base ckpt for stage3 if missing meta")
    parser.add_argument("--spec_file", default="./dataset_stage2/spec_train.npy")
    parser.add_argument("--struct_file", default="./dataset_stage2/struct_train.pkl")
    parser.add_argument("--n", type=int, default=50)
    parser.add_argument("--max_len", type=int, default=32)
    parser.add_argument("--seed", type=int, default=1234)
    parser.add_argument("--top_k", type=int, default=30)
    parser.add_argument("--top_p", type=float, default=0.95)
    parser.add_argument("--greedy", action="store_true", help="use greedy decoding for deterministic compare")
    parser.add_argument("--compare_greedy", action="store_true", help="compare sampling vs greedy")
    parser.add_argument("--spec_metric", action="store_true", help="compute fake spectrum metrics")
    parser.add_argument("--oracle", action="store_true", help="compute oracle metrics using dataset structures")
    parser.add_argument("--show", type=int, default=0, help="print first K samples for both models")
    args = parser.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)

    spec_arr = np.load(args.spec_file)
    tk = StructureTokenizer()
    tk.id_to_token = [tk.inv_vocab[i] for i in range(tk.vocab_size)]
    parser = StructureParser()
    validator = StructureValidator(min_feature_nm=20, margin_nm=30)

    model2, spec_dim2 = load_model(args.ckpt2, args.base_ckpt2, tk, spec_arr.shape[1], device)
    model3, spec_dim3 = load_model(args.ckpt3, args.base_ckpt3, tk, spec_arr.shape[1], device)
    if spec_dim2 != spec_dim3:
        print(f"[Warn] spec_dim mismatch: stage2={spec_dim2} stage3={spec_dim3}")

    idxs = np.random.randint(0, len(spec_arr), size=args.n)

    def run_one_pass(greedy_flag, label_suffix=""):
        stats2 = {"count": 0, "valid": 0, "reasons": {}, "lens": [], "mse": [], "mae": [], "corr": []}
        stats3 = {"count": 0, "valid": 0, "reasons": {}, "lens": [], "mse": [], "mae": [], "corr": []}

        for i, idx in enumerate(idxs):
            spec = spec_arr[idx]
            seed_i = args.seed + i
            torch.manual_seed(seed_i)
            if torch.cuda.is_available():
                torch.cuda.manual_seed_all(seed_i)

            ids2 = sample_one(model2, spec, tk, device, args.max_len, args.top_k, args.top_p, greedy_flag)
            toks2 = [tk.id_to_token[x] for x in ids2]
            struct2 = parser.parse(["[BOS]"] + toks2[1:-1] + ["[EOS]"])
            ok2, reason2 = validator.validate(struct2)
            update_stats(stats2, ok2, reason2, ids2, toks2, spec, args.spec_metric)

            torch.manual_seed(seed_i)
            if torch.cuda.is_available():
                torch.cuda.manual_seed_all(seed_i)

            ids3 = sample_one(model3, spec, tk, device, args.max_len, args.top_k, args.top_p, greedy_flag)
            toks3 = [tk.id_to_token[x] for x in ids3]
            struct3 = parser.parse(["[BOS]"] + toks3[1:-1] + ["[EOS]"])
            ok3, reason3 = validator.validate(struct3)
            update_stats(stats3, ok3, reason3, ids3, toks3, spec, args.spec_metric)

            if args.show > 0 and i < args.show:
                print(f"[Sample {i}] idx={idx}")
                print("S2 IDs:", ids2)
                print("S2 Tok:", toks2)
                print("S2 Valid:", ok2, "Reason:", reason2)
                print("S3 IDs:", ids3)
                print("S3 Tok:", toks3)
                print("S3 Valid:", ok3, "Reason:", reason3)
                print("---------------------------------------------------")

        print_report(f"Stage-2{label_suffix}", stats2, args.spec_metric)
        print_report(f"Stage-3{label_suffix}", stats3, args.spec_metric)

    if args.compare_greedy:
        run_one_pass(False, label_suffix=" (sample)")
        run_one_pass(True, label_suffix=" (greedy)")
    else:
        run_one_pass(args.greedy, label_suffix=" (greedy)" if args.greedy else "")

    if args.oracle:
        try:
            import pickle
            with open(args.struct_file, "rb") as f:
                struct_list = pickle.load(f)
            mse_list, mae_list, corr_list = compute_oracle_metrics(struct_list, idxs, tk, spec_arr)
            print("=== Oracle (GT Structure -> fake spectrum) ===")
            print(f"MSE:  {np.mean(mse_list):.6f}")
            print(f"MAE:  {np.mean(mae_list):.6f}")
            print(f"Corr: {np.mean(corr_list):.6f}")
            print()
        except Exception as e:
            print("[Oracle] Failed to compute:", e)


if __name__ == "__main__":
    main()
