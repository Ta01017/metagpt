#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
verify_forward_surrogate.py
Evaluate forward surrogate on paired (struct, spec).
"""

import argparse
import pickle
import numpy as np
import torch

from structure_lang.tokenizer import StructureTokenizer
from models.forward_surrogate import build_surrogate_from_ckpt


def build_seq_ids(struct_list, word_dict):
    tk = StructureTokenizer()
    unk_id = word_dict.get("UNK", 0)
    bos_id = word_dict.get("BOS")
    eos_id = word_dict.get("EOS")
    seq_ids = []
    for ids in struct_list:
        toks = [tk.inv_vocab[i] for i in ids]
        toks = ["BOS"] + toks + ["EOS"]
        mapped = [word_dict.get(t, unk_id) for t in toks]
        if bos_id is not None:
            mapped[0] = bos_id
        if eos_id is not None:
            mapped[-1] = eos_id
        seq_ids.append(mapped)
    return seq_ids


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--ckpt", required=True)
    parser.add_argument("--spec_file", required=True)
    parser.add_argument("--struct_file", required=True)
    parser.add_argument("--n", type=int, default=50)
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    ckpt = torch.load(args.ckpt, map_location="cpu")
    model, meta = build_surrogate_from_ckpt(ckpt, device)
    model.eval()

    spec_arr = np.load(args.spec_file).astype("float32")
    with open(args.struct_file, "rb") as f:
        struct_list = pickle.load(f)

    word_dict = meta["word_dict"]
    pad_id = meta["pad_id"]
    seq_ids = build_seq_ids(struct_list, word_dict)
    max_len = meta.get("max_len", max(len(s) for s in seq_ids))

    mse_list, mae_list, corr_list = [], [], []
    rng = np.random.default_rng(0)
    idxs = rng.choice(len(seq_ids), size=min(args.n, len(seq_ids)), replace=False)

    with torch.no_grad():
        for idx in idxs:
            ids = seq_ids[idx]
            if len(ids) > max_len:
                ids = ids[:max_len]
            if len(ids) < max_len:
                ids = ids + [pad_id] * (max_len - len(ids))
            tokens = torch.tensor(ids, dtype=torch.long, device=device).unsqueeze(0)
            spec = torch.tensor(spec_arr[idx], dtype=torch.float32, device=device).unsqueeze(0)

            pred = model(tokens, pad_id)
            mse = torch.mean((pred - spec) ** 2).item()
            mae = torch.mean(torch.abs(pred - spec)).item()
            denom = (pred.norm(dim=1) * spec.norm(dim=1) + 1e-8).item()
            corr = (pred.mul(spec).sum(dim=1).item()) / denom
            mse_list.append(mse)
            mae_list.append(mae)
            corr_list.append(corr)

    print("Forward surrogate eval:")
    print(f"MSE : {np.mean(mse_list):.6f}")
    print(f"MAE : {np.mean(mae_list):.6f}")
    print(f"Corr: {np.mean(corr_list):.6f}")


if __name__ == "__main__":
    main()
