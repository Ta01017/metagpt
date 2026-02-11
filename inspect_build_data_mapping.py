#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
inspect_build_data_mapping.py
Check consistency between filename params, MAT params, and token quantization.
"""

import argparse
import re
import numpy as np
import scipy.io

from structure_lang.tokenizer import StructureTokenizerExtended


def parse_filename(filename: str):
    pattern = r'T_P([\d\.e+-]+)_D([\d\.e+-]+)_H([\d\.e+-]+)_num-idx(\d+)\.mat'
    m = re.match(pattern, filename)
    if not m:
        raise ValueError(f"Cannot parse filename: {filename}")
    P, D, H, idx = m.groups()
    return float(P), float(D), float(H), int(idx)


def quantize_to_nearest(val, allowed):
    return min(allowed, key=lambda x: abs(x - val))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--mat_file", required=True)
    args = parser.parse_args()

    fname = args.mat_file.split("/")[-1].split("\\")[-1]
    P_fn, D_fn, H_fn, idx = parse_filename(fname)

    mat = scipy.io.loadmat(args.mat_file)
    P_mat = float(mat["P"].squeeze()) if "P" in mat else None
    D_mat = float(mat["D"].squeeze()) if "D" in mat else None
    H_mat = float(mat["H"].squeeze()) if "H" in mat else None
    T = mat["T"].squeeze() if "T" in mat else None

    print(f"[File] {fname} (idx={idx})")
    print(f"[Filename] P={P_fn:.10e} D={D_fn:.10e} H={H_fn:.10e}")
    if P_mat is not None:
        print(f"[MAT]      P={P_mat:.10e} D={D_mat:.10e} H={H_mat:.10e}")
        print(f"[Diff]     dP={abs(P_fn-P_mat):.3e} dD={abs(D_fn-D_mat):.3e} dH={abs(H_fn-H_mat):.3e}")
    else:
        print("[MAT] no P/D/H fields found.")

    if T is not None:
        print(f"[T] shape={T.shape} min={T.min():.6f} max={T.max():.6f} mean={T.mean():.6f}")

    tk = StructureTokenizerExtended()
    P_nm = P_fn * 1e9
    H_nm = H_fn * 1e9
    R_nm = (D_fn / 2.0) * 1e9

    P_q = quantize_to_nearest(P_nm, tk.P_vals)
    H_q = quantize_to_nearest(H_nm, tk.H_vals)
    R_q = quantize_to_nearest(R_nm, tk.R_vals)

    p_err = abs(P_nm - P_q) / P_nm * 100 if P_nm > 0 else 0
    h_err = abs(H_nm - H_q) / H_nm * 100 if H_nm > 0 else 0
    r_err = abs(R_nm - R_q) / R_nm * 100 if R_nm > 0 else 0

    print(f"[Quantize] P_nm={P_nm:.2f} -> {P_q} (err={p_err:.2f}%)")
    print(f"[Quantize] H_nm={H_nm:.2f} -> {H_q} (err={h_err:.2f}%)")
    print(f"[Quantize] R_nm={R_nm:.2f} -> {R_q} (err={r_err:.2f}%)")

    tokens = [
        f"PX_{P_q}",
        f"PY_{P_q}",
        "SUB_Glass_Substrate",
        "L1_MAT_Si-Alpha",
        "L1_SHAPE_CYL",
        f"L1_H_{H_q}",
        f"L1_R_{R_q}",
    ]
    missing = [t for t in tokens if t not in tk.vocab]
    print(f"[Tokens] {tokens}")
    if missing:
        print(f"[Tokens] missing in vocab: {missing}")
    else:
        print("[Tokens] all exist in vocab")


if __name__ == "__main__":
    main()
