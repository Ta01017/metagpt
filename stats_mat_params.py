#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
stats_mat_params.py
Scan all .mat files and report distribution of P/D/H (and derived R).
Also reports how many samples fall into tokenizer value grids.
"""

import argparse
import glob
import numpy as np
import scipy.io
from structure_lang.tokenizer import StructureTokenizerExtended


def load_param(mat_path):
    data = scipy.io.loadmat(mat_path)
    P = float(data["P"].squeeze()) if "P" in data else None
    D = float(data["D"].squeeze()) if "D" in data else None
    H = float(data["H"].squeeze()) if "H" in data else None
    return P, D, H


def summarize(name, arr_nm):
    arr = np.array(arr_nm)
    print(f"[{name}] count={arr.size} min={arr.min():.2f} max={arr.max():.2f} mean={arr.mean():.2f}")
    for p in [1, 5, 50, 95, 99]:
        print(f"  p{p}: {np.percentile(arr, p):.2f}")


def in_grid(values, grid):
    grid = np.array(sorted(grid))
    # exact match check (values are continuous; we measure nearest)
    nearest = np.array([grid[np.argmin(np.abs(grid - v))] for v in values])
    return nearest


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--glob", required=True, help="glob pattern for .mat files")
    parser.add_argument("--limit", type=int, default=0, help="limit number of files (0=all)")
    args = parser.parse_args()

    files = glob.glob(args.glob)
    if args.limit and args.limit < len(files):
        files = files[: args.limit]
    if not files:
        print("No files matched.")
        return

    P_list, D_list, H_list, R_list = [], [], [], []
    for fp in files:
        P, D, H = load_param(fp)
        if P is None or D is None or H is None:
            continue
        P_nm = P * 1e9
        D_nm = D * 1e9
        H_nm = H * 1e9
        R_nm = D_nm / 2.0
        P_list.append(P_nm)
        D_list.append(D_nm)
        H_list.append(H_nm)
        R_list.append(R_nm)

    print(f"Files: {len(files)}  Parsed: {len(P_list)}")
    summarize("P (nm)", P_list)
    summarize("D (nm)", D_list)
    summarize("H (nm)", H_list)
    summarize("R (nm)", R_list)

    tk = StructureTokenizerExtended()
    P_q = in_grid(P_list, tk.P_vals)
    H_q = in_grid(H_list, tk.H_vals)
    R_q = in_grid(R_list, tk.R_vals)

    def report_quant(name, raw, q):
        raw = np.array(raw)
        q = np.array(q)
        err = np.abs(raw - q) / np.maximum(raw, 1e-9) * 100.0
        print(f"[Quantize {name}] mean_err={err.mean():.2f}%  p95={np.percentile(err,95):.2f}%  max={err.max():.2f}%")

    report_quant("P", P_list, P_q)
    report_quant("H", H_list, H_q)
    report_quant("R", R_list, R_q)


if __name__ == "__main__":
    main()
