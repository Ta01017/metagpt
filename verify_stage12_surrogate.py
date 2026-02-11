#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
verify_stage12_surrogate.py
Evaluate a Stage12 model using a learned forward surrogate (MSE/MAE/Corr).
"""

import argparse
import numpy as np
import torch

from models.forward_surrogate import build_surrogate_from_ckpt

# optogpt_new
import os
import sys

ROOT = os.path.dirname(os.path.abspath(__file__))
OPTOGPT_ROOT = os.path.join(ROOT, "optogpt_new")
if OPTOGPT_ROOT not in sys.path:
    sys.path.append(OPTOGPT_ROOT)

from core.models.transformer import make_model_I, subsequent_mask  # noqa: E402


def load_stage12(ckpt_path, device):
    ckpt = torch.load(ckpt_path, map_location="cpu")
    cfg = ckpt["model_cfg"]
    meta = ckpt["meta"]
    word_dict = meta["word_dict"]
    index_dict = meta["index_dict"]
    spec_dim = meta["spec_dim"]

    model = make_model_I(
        src_vocab=spec_dim,
        tgt_vocab=len(word_dict),
        N=cfg["n_layers"],
        d_model=cfg["d_model"],
        d_ff=cfg["d_ff"],
        h=cfg["n_heads"],
        dropout=cfg["dropout"],
    ).to(device)
    model.load_state_dict(ckpt["model"], strict=True)
    model.eval()
    return model, word_dict, index_dict, meta


@torch.no_grad()
def generate(model, spec, word_dict, max_len, device, top_k=30, top_p=0.95, greedy=False):
    bos_id = word_dict.get("BOS", 2)
    eos_id = word_dict.get("EOS", 3)
    pad_id = word_dict.get("PAD", 1)

    seq = [bos_id]
    for _ in range(max_len - 1):
        trg = torch.tensor([seq], dtype=torch.long, device=device)
        trg_mask = (trg != pad_id).unsqueeze(1) & subsequent_mask(trg.size(-1), device=trg.device)
        src = torch.tensor(spec[None, None, :], dtype=torch.float32, device=device)

        out = model(src, trg, src_mask=None, tgt_mask=trg_mask)
        log_probs = model.generator(out[:, -1, :])

        if greedy:
            next_id = torch.argmax(log_probs, dim=-1).item()
            seq.append(next_id)
            if next_id == eos_id:
                break
            continue

        probs = torch.exp(log_probs).squeeze(0)
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
        seq.append(next_id)
        if next_id == eos_id:
            break
    return seq


@torch.no_grad()
def eval_model(model, word_dict, spec_arr, surrogate, pad_id, n, max_len, greedy=False, n_candidates=1):
    mse_list, mae_list, corr_list = [], [], []
    rng = np.random.default_rng(0)
    idxs = rng.choice(len(spec_arr), size=min(n, len(spec_arr)), replace=False)

    for idx in idxs:
        spec = spec_arr[idx]
        best_pred = None
        best_mse = 1e9

        for _c in range(max(1, n_candidates)):
            ids = generate(model, spec, word_dict, max_len, device, greedy=greedy)
            if len(ids) > max_len:
                ids = ids[:max_len]
            if len(ids) < max_len:
                ids = ids + [pad_id] * (max_len - len(ids))

            tokens = torch.tensor(ids, dtype=torch.long, device=device).unsqueeze(0)
            pred = surrogate(tokens, pad_id).squeeze(0).detach().cpu().numpy()
            mse = float(np.mean((pred - spec) ** 2))
            if mse < best_mse:
                best_mse = mse
                best_pred = pred

        pred = best_pred
        mse = float(np.mean((pred - spec) ** 2))
        mae = float(np.mean(np.abs(pred - spec)))
        denom = (np.linalg.norm(pred) * np.linalg.norm(spec)) + 1e-8
        corr = float(np.dot(pred, spec) / denom)
        mse_list.append(mse)
        mae_list.append(mae)
        corr_list.append(corr)

    return float(np.mean(mse_list)), float(np.mean(mae_list)), float(np.mean(corr_list))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--ckpt", required=True)
    parser.add_argument("--surrogate_ckpt", required=True)
    parser.add_argument("--spec_file", required=True)
    parser.add_argument("--n", type=int, default=50)
    parser.add_argument("--max_len", type=int, default=None)
    parser.add_argument("--greedy", action="store_true")
    parser.add_argument("--n_candidates", type=int, default=1)
    args = parser.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    spec_arr = np.load(args.spec_file)

    model, word_dict, _, meta = load_stage12(args.ckpt, device)

    s_ckpt = torch.load(args.surrogate_ckpt, map_location="cpu")
    surrogate, s_meta = build_surrogate_from_ckpt(s_ckpt, torch.device(device))
    surrogate.eval()

    if s_meta.get("word_dict") is not None and s_meta["word_dict"] != word_dict:
        raise ValueError("Surrogate vocab != Stage12 vocab. Check ckpt alignment.")

    pad_id = s_meta.get("pad_id", 1)
    max_len = args.max_len if args.max_len is not None else s_meta.get("max_len", meta.get("max_len", 128))

    mse, mae, corr = eval_model(
        model, word_dict, spec_arr, surrogate, pad_id, args.n, max_len, args.greedy, args.n_candidates
    )

    tag = "greedy" if args.greedy else "sample"
    if args.n_candidates > 1:
        tag += f" best-of-{args.n_candidates}"

    print(f"Stage12 surrogate eval [{tag}]")
    print(f"MSE : {mse:.6f}")
    print(f"MAE : {mae:.6f}")
    print(f"Corr: {corr:.6f}")

"""
python .\verify_stage12_surrogate.py ^
  --ckpt .\ckpt_stage12_optogpt_real\stage12_best.pt ^
  --surrogate_ckpt .\ckpt_forward_surrogate_real\surrogate_best.pt ^
  --spec_file <你的spec.npy> ^
  --n 50 --max_len 22 --greedy
"""