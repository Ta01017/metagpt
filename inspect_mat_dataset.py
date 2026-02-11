#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
inspect_mat_dataset.py
Inspect .mat dataset structure (keys, shapes, dtype, basic stats).
Supports both v7.3 (HDF5) and older MAT formats.
"""

import argparse
import numpy as np


def _is_numeric(arr):
    return np.issubdtype(arr.dtype, np.number)


def _sample_stats(arr, max_elems=200000):
    flat = arr.reshape(-1)
    if flat.size > max_elems:
        rng = np.random.default_rng(0)
        idx = rng.choice(flat.size, size=max_elems, replace=False)
        flat = flat[idx]
    return float(np.min(flat)), float(np.max(flat)), float(np.mean(flat)), float(np.std(flat))


def _print_array_info(name, arr):
    shape = arr.shape
    dtype = arr.dtype
    print(f"{name}: shape={shape}, dtype={dtype}")
    if _is_numeric(arr) and arr.size > 0:
        mn, mx, mean, std = _sample_stats(arr)
        print(f"  stats(sample): min={mn:.6f} max={mx:.6f} mean={mean:.6f} std={std:.6f}")


def inspect_h5(mat_file):
    import h5py

    def visit(name, obj):
        if isinstance(obj, h5py.Dataset):
            arr = obj[()]
            _print_array_info(name, np.array(arr))

    with h5py.File(mat_file, "r") as f:
        print("[HDF5 .mat] keys:")
        f.visititems(visit)


def inspect_mat(mat_file):
    from scipy.io import loadmat

    data = loadmat(mat_file)
    keys = [k for k in data.keys() if not k.startswith("__")]
    print("[MAT] keys:", keys)

    for k in keys:
        v = data[k]
        if isinstance(v, np.ndarray):
            if v.dtype.names:  # struct array
                print(f"{k}: struct with fields={v.dtype.names}, shape={v.shape}")
            else:
                _print_array_info(k, v)
        else:
            print(f"{k}: type={type(v)}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--mat_file", required=True)
    args = parser.parse_args()

    # try h5py first (v7.3), fallback to scipy
    try:
        import h5py  # noqa: F401
        if h5py.is_hdf5(args.mat_file):
            inspect_h5(args.mat_file)
            return
    except Exception:
        pass

    inspect_mat(args.mat_file)


if __name__ == "__main__":
    main()
