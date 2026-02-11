#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
suggest_max_len.py
Analyze struct length distribution and suggest max_len choices.
"""

import argparse
import pickle
import numpy as np


def summarize(lengths, label):
    arr = np.array(lengths, dtype=np.int32)
    print(f"[{label}] count={arr.size}")
    print(f"min={arr.min()}  max={arr.max()}  mean={arr.mean():.2f}  median={np.median(arr):.2f}")
    for p in [90, 95, 97, 99]:
        print(f"p{p}={np.percentile(arr, p):.0f}", end="  ")
    print("\n")


def truncation_rate(lengths, max_len):
    arr = np.array(lengths, dtype=np.int32)
    rate = float(np.mean(arr > max_len))
    avg_over = float(np.mean(np.maximum(arr - max_len, 0)))
    return rate, avg_over


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--struct_file", required=True, help="path to struct_train.pkl")
    parser.add_argument("--add_bos_eos", action="store_true", help="add +2 length for BOS/EOS")
    parser.add_argument("--percentile", type=float, default=99.0, help="suggested percentile (e.g., 95/99)")
    parser.add_argument("--show_candidates", action="store_true", help="print truncation for p90/p95/p99/max")
    args = parser.parse_args()

    with open(args.struct_file, "rb") as f:
        struct_list = pickle.load(f)

    raw_lengths = [len(s) for s in struct_list]
    summarize(raw_lengths, "raw")

    if args.add_bos_eos:
        lengths = [l + 2 for l in raw_lengths]
        summarize(lengths, "raw + BOS/EOS")
    else:
        lengths = raw_lengths

    # percentile suggestion
    p = args.percentile
    p_len = int(np.percentile(lengths, p))
    max_len = int(np.max(lengths))
    print(f"Suggested max_len @ p{p:.0f}: {p_len}")
    print(f"Max (no truncation): {max_len}")

    if args.show_candidates:
        for cand in [int(np.percentile(lengths, 90)),
                     int(np.percentile(lengths, 95)),
                     int(np.percentile(lengths, 99)),
                     max_len]:
            rate, avg_over = truncation_rate(lengths, cand)
            print(f"max_len={cand}  trunc_rate={rate*100:.2f}%  avg_trunc={avg_over:.2f}")


if __name__ == "__main__":
    main()

"""
python .\suggest_max_len.py --struct_file .\dataset_stage2\struct_train.pkl --add_bos_eos --percentile 99 --show_candidates
"""