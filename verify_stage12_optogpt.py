#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
verify_stage12_optogpt.py
快速验证 Stage12 OptoGPT 模型：生成结构 -> fake forward 计算 MSE/MAE/Corr
"""

import argparse
import numpy as np
import torch
import torch.nn.functional as F

from data.gen_stage2_fake_dataset import fake_spectrum_from_structure

# optogpt_new
import os
import sys

ROOT = os.path.dirname(os.path.abspath(__file__))
OPTOGPT_ROOT = os.path.join(ROOT, "optogpt_new")
if OPTOGPT_ROOT not in sys.path:
    sys.path.append(OPTOGPT_ROOT)

from core.models.transformer import make_model_I, subsequent_mask  # noqa: E402


def load_ckpt(ckpt_path, device):
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
def generate(model, spec, word_dict, index_dict, max_len, device, top_k=30, top_p=0.95, greedy=False):
    # start with BOS
    bos_id = word_dict.get("BOS", 2)
    eos_id = word_dict.get("EOS", 3)
    pad_id = word_dict.get("PAD", 1)

    seq = [bos_id]
    for _ in range(max_len - 1):
        trg = torch.tensor([seq], dtype=torch.long, device=device)
        trg_mask = (trg != pad_id).unsqueeze(1) & subsequent_mask(trg.size(-1), device=trg.device)
        src = torch.tensor(spec[None, None, :], dtype=torch.float32, device=device)

        out = model(src, trg, src_mask=None, tgt_mask=trg_mask)
        # use generator to project to vocab (OptoGPT uses generator outside model.forward)
        log_probs = model.generator(out[:, -1, :])
        logits = log_probs  # log-probabilities
        if greedy:
            next_id = torch.argmax(logits, dim=-1).item()
            seq.append(next_id)
            if next_id == eos_id:
                break
            continue

        probs = torch.exp(logits).squeeze(0)

        # top-k/top-p sampling
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


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--ckpt", required=True)
    parser.add_argument("--spec_file", default="./dataset_stage2_toy/spec_train.npy")
    parser.add_argument("--n", type=int, default=50)
    parser.add_argument("--max_len", type=int, default=10)
    parser.add_argument("--greedy", action="store_true", help="use greedy decoding")
    parser.add_argument("--n_candidates", type=int, default=1, help="best-of-N rerank by fake MSE")
    args = parser.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    spec_arr = np.load(args.spec_file)

    model, word_dict, index_dict, meta = load_ckpt(args.ckpt, device)
    max_len = args.max_len

    mse_list, mae_list, corr_list = [], [], []

    for _ in range(args.n):
        idx = np.random.randint(0, len(spec_arr))
        spec = spec_arr[idx]
        best = None
        best_mse = 1e9

        cands = max(1, args.n_candidates)
        for _c in range(cands):
            ids = generate(
                model, spec, word_dict, index_dict, max_len, device,
                greedy=args.greedy, top_k=30, top_p=0.95
            )
            toks = [index_dict.get(i, "UNK") for i in ids]
            toks = [t for t in toks if t not in ("BOS", "EOS", "PAD")]

            pred = fake_spectrum_from_structure(toks, spec_dim=spec.shape[0])
            mse = float(np.mean((pred - spec) ** 2))
            if mse < best_mse:
                best_mse = mse
                best = (pred, spec)

        pred, spec = best
        mse = float(np.mean((pred - spec) ** 2))
        mae = float(np.mean(np.abs(pred - spec)))
        denom = (np.linalg.norm(pred) * np.linalg.norm(spec)) + 1e-8
        corr = float(np.dot(pred, spec) / denom)
        mse_list.append(mse)
        mae_list.append(mae)
        corr_list.append(corr)

    tag = "best-of-N" if args.n_candidates > 1 else ("greedy" if args.greedy else "sample")
    print(f"Stage12 OptoGPT validation (fake forward) [{tag}]:")
    print(f"MSE : {np.mean(mse_list):.6f}")
    print(f"MAE : {np.mean(mae_list):.6f}")
    print(f"Corr: {np.mean(corr_list):.6f}")


if __name__ == "__main__":
    main()
