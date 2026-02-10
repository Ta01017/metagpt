#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
统计 fake 数据（struct_train.pkl）中的 token 种类与频次。
"""

import argparse
import pickle
from collections import Counter

from structure_lang.tokenizer import StructureTokenizer


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--struct_file", default="./dataset_stage2/struct_train.pkl")
    parser.add_argument("--topk", type=int, default=30, help="show top-k frequent tokens")
    args = parser.parse_args()

    with open(args.struct_file, "rb") as f:
        struct_list = pickle.load(f)

    tk = StructureTokenizer()
    id_to_token = tk.inv_vocab

    cnt = Counter()
    for ids in struct_list:
        for tid in ids:
            tok = id_to_token.get(tid, f"<UNK:{tid}>")
            cnt[tok] += 1

    total_tokens = sum(cnt.values())
    unique_tokens = len(cnt)

    print(f"[Token Stats] total_tokens={total_tokens} unique_tokens={unique_tokens}")
    print(f"[Top {args.topk}]")
    for tok, c in cnt.most_common(args.topk):
        print(f"{tok}\t{c}")


if __name__ == "__main__":
    main()
