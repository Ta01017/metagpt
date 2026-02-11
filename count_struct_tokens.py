#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
count_struct_tokens.py
Count token types/frequencies in struct_train.pkl (StructureTokenizer ids).
"""

import argparse
import pickle
from collections import Counter, defaultdict

from structure_lang.tokenizer import StructureTokenizer


def classify(tok: str) -> str:
    if tok.startswith("PX_"):
        return "PX"
    if tok.startswith("PY_"):
        return "PY"
    if tok.startswith("SUB_"):
        return "SUB"
    if tok.startswith("L1_MAT_"):
        return "L1_MAT"
    if tok.startswith("L1_SHAPE_"):
        return "L1_SHAPE"
    if tok.startswith("L1_H_"):
        return "L1_H"
    if tok.startswith("L1_W_"):
        return "L1_W"
    if tok.startswith("L1_L_"):
        return "L1_L"
    if tok.startswith("L1_R_"):
        return "L1_R"
    if tok.startswith("COT_") or tok == "[COT]":
        return "COT"
    if tok in ("[BOS]", "[EOS]", "[PAD]"):
        return "SPECIAL"
    return "OTHER"


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--struct_file", default="./dataset_stage2/struct_train.pkl")
    parser.add_argument("--top_k", type=int, default=30)
    parser.add_argument("--show_all", action="store_true")
    args = parser.parse_args()

    with open(args.struct_file, "rb") as f:
        struct_list = pickle.load(f)

    tk = StructureTokenizer()
    counter = Counter()
    cat_counter = defaultdict(int)
    total = 0

    for ids in struct_list:
        for i in ids:
            tok = tk.inv_vocab.get(i, f"UNK_ID_{i}")
            counter[tok] += 1
            cat_counter[classify(tok)] += 1
            total += 1

    unique = len(counter)
    print(f"[Token Stats] total_tokens={total} unique_tokens={unique}")

    print("\n[By Category]")
    for k in sorted(cat_counter.keys()):
        v = cat_counter[k]
        pct = 100.0 * v / max(total, 1)
        print(f"{k:10s} {v:8d}  ({pct:6.2f}%)")

    items = counter.most_common() if args.show_all else counter.most_common(args.top_k)
    print("\n[Top Tokens]")
    for tok, cnt in items:
        pct = 100.0 * cnt / max(total, 1)
        print(f"{tok:20s} {cnt:8d}  ({pct:6.2f}%)")


if __name__ == "__main__":
    main()

"""
python .\count_struct_tokens.py --struct_file <你的struct_train.pkl> --show_all
python .\count_struct_tokens.py --struct_file <你的struct_train.pkl> --show_all
"""