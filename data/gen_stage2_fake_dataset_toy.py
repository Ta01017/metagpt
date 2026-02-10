#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Generate a toy Stage2 dataset with a very small token set.
Goal: verify pipeline correctness with easier mapping and fewer tokens.
"""

import os
import pickle
import numpy as np
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from structure_lang.tokenizer import StructureTokenizer
from structure_lang.parser import StructureParser
from structure_lang.validator import StructureValidator
from data.gen_stage2_fake_dataset import fake_spectrum_from_structure


def parse_val(tok: str) -> int:
    return int(tok.split("_")[-1])


def choose_k(tokens, k):
    tokens = sorted(tokens, key=parse_val)
    if len(tokens) <= k:
        return tokens
    idxs = np.linspace(0, len(tokens) - 1, k, dtype=int)
    return [tokens[i] for i in idxs]


def main(
    out_dir="./dataset_stage2_toy",
    N=5000,
    spec_dim=128,
    seed=0,
    k_px=5,
    k_py=5,
    k_h=5,
    k_w=5,
    k_l=5,
    materials=("SiO2", "TiO2"),
    shape="RECT",
):
    os.makedirs(out_dir, exist_ok=True)
    rng = np.random.default_rng(seed)

    tk = StructureTokenizer()
    parser = StructureParser()
    val = StructureValidator(min_feature_nm=20, margin_nm=30)

    vocab = list(tk.vocab.keys())
    px_tokens = [t for t in vocab if t.startswith("PX_")]
    py_tokens = [t for t in vocab if t.startswith("PY_")]
    h_tokens = [t for t in vocab if t.startswith("L1_H_")]
    w_tokens = [t for t in vocab if t.startswith("L1_W_")]
    l_tokens = [t for t in vocab if t.startswith("L1_L_")]
    mat_tokens = [f"L1_MAT_{m}" for m in materials if f"L1_MAT_{m}" in tk.vocab]

    px_tokens = choose_k(px_tokens, k_px)
    py_tokens = choose_k(py_tokens, k_py)
    h_tokens = choose_k(h_tokens, k_h)
    w_tokens = choose_k(w_tokens, k_w)
    l_tokens = choose_k(l_tokens, k_l)

    specs = []
    structs = []

    for i in range(N):
        for _ in range(100):  # resample attempts
            px = rng.choice(px_tokens)
            py = rng.choice(py_tokens)
            pmin = min(parse_val(px), parse_val(py)) - 30

            # filter dims by pitch constraint
            w_candidates = [t for t in w_tokens if parse_val(t) <= pmin]
            l_candidates = [t for t in l_tokens if parse_val(t) <= pmin]
            if not w_candidates or not l_candidates:
                continue

            h = rng.choice(h_tokens)
            w = rng.choice(w_candidates)
            l = rng.choice(l_candidates)
            mat = rng.choice(mat_tokens)

            toks = [
                px, py, "SUB_Glass_Substrate",
                mat, f"L1_SHAPE_{shape}",
                h, w, l,
            ]
            ids = [tk.vocab[t] for t in toks]

            # validate
            ok, _ = val.validate(parser.parse(["[BOS]"] + toks + ["[EOS]"]))
            if not ok:
                continue

            spec = fake_spectrum_from_structure(toks, spec_dim=spec_dim, rng=rng)
            specs.append(spec)
            structs.append(ids)
            break

        if i % 500 == 0:
            print(f"[toy] {i}/{N}")

    spec_arr = np.stack(specs, axis=0).astype("float32")
    np.save(os.path.join(out_dir, "spec_train.npy"), spec_arr)
    with open(os.path.join(out_dir, "struct_train.pkl"), "wb") as f:
        pickle.dump(structs, f)

    print("[toy] Saved:")
    print("  spec_train.npy:", spec_arr.shape)
    print("  struct_train.pkl:", len(structs))


if __name__ == "__main__":
    main()
