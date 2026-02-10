#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Generate fake Stage2 dataset:
  - spec_train.npy: (N, spec_dim)
  - struct_train.pkl: list of token-id lists (no BOS/EOS)
"""

import os
import sys
import pickle
import numpy as np
from pathlib import Path

# ---- auto-add project root ----
ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from structure_lang.tokenizer import StructureTokenizer
from structure_lang.parser import StructureParser
from structure_lang.validator import StructureValidator
from data.gen_stage1_synth import Stage1SynthGenerator


def fake_spectrum_from_structure(toks, spec_dim=128, rng: np.random.Generator | None = None):
    if rng is None:
        rng = np.random.default_rng()
    spec = np.zeros(spec_dim, dtype="float32")

    # simple geometric feature extraction
    P = 500
    H = 300
    W = 200
    L = 200
    R = 100
    mat = "SiO2"
    shape = "RECT"
    for t in toks:
        if t.startswith("PX_"):
            P = int(t.split("_")[1])
        if t.startswith("L1_H_"):
            H = int(t.split("_")[2])
        if t.startswith("L1_R_"):
            R = int(t.split("_")[2])
        if t.startswith("L1_W_"):
            W = int(t.split("_")[2])
        if t.startswith("L1_L_"):
            L = int(t.split("_")[2])
        if t.startswith("L1_MAT_"):
            mat = t.split("_", 2)[2]
        if t.startswith("L1_SHAPE_"):
            shape = t.split("_", 2)[2]

    xs = np.linspace(0, 1, spec_dim, dtype="float32")
    mat_phase = {
        "SiO2": 0.1,
        "TiO2": 0.25,
        "Ta2O5": 0.35,
        "HfO2": 0.45,
        "Si3N4": 0.55,
    }.get(mat, 0.2)

    base = np.sin(xs * np.pi * H / 380 + mat_phase)
    geom = (W + L + R) / 900.0
    shape_mod = np.cos(xs * np.pi * (1.0 + geom) + 0.3 * mat_phase)
    period = np.cos(xs * np.pi * P / 850 + 0.5 * mat_phase)

    peak1 = np.exp(-((xs - (0.25 + 0.15 * geom)) ** 2) / (0.01 + 0.02 * geom))
    peak2 = np.exp(-((xs - (0.65 - 0.1 * geom)) ** 2) / (0.015 + 0.01 * geom))
    if shape == "CYL":
        peak2 *= 1.15
    else:
        peak1 *= 1.10

    spec = 0.35 * base + 0.25 * shape_mod + 0.2 * period + 0.2 * (peak1 + peak2)

    noise = rng.normal(0, 0.03, size=spec_dim).astype("float32")
    drift = (xs - 0.5) * rng.normal(0, 0.08)
    scale = rng.uniform(0.85, 1.15)
    spec = scale * spec + drift + noise

    spec = (spec - spec.min()) / (spec.max() - spec.min() + 1e-6)
    return spec.astype("float32")


def main(
    out_dir="./dataset_stage2",
    N=20000,
    spec_dim=128,
    seed=0,
):
    os.makedirs(out_dir, exist_ok=True)

    rng = np.random.default_rng(seed)
    tk = StructureTokenizer()
    parser = StructureParser()
    val = StructureValidator(min_feature_nm=20, margin_nm=30)

    gen = Stage1SynthGenerator(
        tokenizer=tk,
        parser=parser,
        validator=val,
        use_materials=["SiO2", "TiO2", "Ta2O5", "HfO2"],
        seed=seed,
    )

    specs = []
    structs = []

    for i in range(N):
        ids = gen.sample_one_ids()
        toks = tk.decode(ids)
        spec = fake_spectrum_from_structure(toks, spec_dim=spec_dim, rng=rng)

        specs.append(spec)
        structs.append(ids)

        if i % 2000 == 0:
            print(f"[fake-stage2] {i}/{N}")

    spec_arr = np.stack(specs, axis=0).astype("float32")
    np.save(os.path.join(out_dir, "spec_train.npy"), spec_arr)

    with open(os.path.join(out_dir, "struct_train.pkl"), "wb") as f:
        pickle.dump(structs, f)

    print("[fake-stage2] Saved:")
    print("  spec_train.npy:", spec_arr.shape)
    print("  struct_train.pkl:", len(structs))


if __name__ == "__main__":
    main()
