#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
check_spec_type.py
Quickly verify whether spec contains only T or both R+T.
"""

import argparse
import numpy as np


def stats(name, arr):
    return {
        "name": name,
        "mean_abs": float(np.mean(np.abs(arr))),
        "std": float(np.std(arr)),
        "min": float(np.min(arr)),
        "max": float(np.max(arr)),
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--spec_file", required=True)
    parser.add_argument("--sample", type=int, default=0, help="use first N samples (0=all)")
    args = parser.parse_args()

    spec = np.load(args.spec_file)
    if args.sample and args.sample < spec.shape[0]:
        spec = spec[: args.sample]

    N, D = spec.shape
    print(f"[Spec] shape=({N}, {D}) dtype={spec.dtype}")

    if D % 2 != 0:
        print("Spec dim is odd. Cannot split into R/T halves.")
        return

    half = D // 2
    a = spec[:, :half]
    b = spec[:, half:]

    sa = stats("first_half", a)
    sb = stats("second_half", b)

    print(f"[first_half] mean_abs={sa['mean_abs']:.6f} std={sa['std']:.6f} min={sa['min']:.6f} max={sa['max']:.6f}")
    print(f"[second_half] mean_abs={sb['mean_abs']:.6f} std={sb['std']:.6f} min={sb['min']:.6f} max={sb['max']:.6f}")

    # ratio check
    ratio = sa["mean_abs"] / (sb["mean_abs"] + 1e-12)
    print(f"[ratio] mean_abs(first/second)={ratio:.4f}")

    # similarity check
    diff = np.mean(np.abs(a - b))
    print(f"[diff] mean_abs(first-second)={diff:.6f}")

    # quick heuristic
    if sa["std"] < 1e-4 and sb["std"] >= 1e-4:
        print("=> first_half is near-constant/zero; likely only second_half is real.")
    elif sb["std"] < 1e-4 and sa["std"] >= 1e-4:
        print("=> second_half is near-constant/zero; likely only first_half is real.")
    else:
        print("=> both halves show variance; likely contains both R and T.")


if __name__ == "__main__":
    main()
