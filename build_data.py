# fixed_dataset_builder.py
from __future__ import annotations

import argparse
import glob
import os
import pickle
import re
from pathlib import Path
from typing import List, Tuple, Optional

import numpy as np
import scipy.io

from structure_lang.tokenizer import StructureTokenizerExtended


class FixedSiAlphaDatasetBuilder:
    """
    Build spec/struct dataset from .mat files.
    - spec: interpolated to target_spec_dim and normalized per-sample
    - struct: token ids from P/D/H (converted to nm and quantized)
    """

    def __init__(self, target_spec_dim: int = 128, use_mat_params: bool = False):
        self.target_spec_dim = target_spec_dim
        self.tokenizer = StructureTokenizerExtended()
        self.material = "Si-Alpha"
        self.use_mat_params = use_mat_params

        print("[Builder] Using StructureTokenizerExtended")
        print(f"  material={self.material}")
        print(f"  vocab_size={self.tokenizer.vocab_size}")
        print(f"  target_spec_dim={self.target_spec_dim}")
        print(f"  use_mat_params={self.use_mat_params}")

    def parse_filename(self, filename: str) -> dict:
        """Parse P/D/H from filename."""
        pattern = r"T_P([\d\.e+-]+)_D([\d\.e+-]+)_H([\d\.e+-]+)_num-idx(\d+)\.mat"
        match = re.match(pattern, filename)
        if not match:
            raise ValueError(f"Cannot parse filename: {filename}")
        P, D, H, idx = match.groups()
        return {"P": float(P), "D": float(D), "H": float(H), "idx": int(idx)}

    def process_spectrum(self, spectrum: np.ndarray) -> np.ndarray:
        """Interpolate spectrum to target dim and normalize to [0,1]."""
        spectrum = spectrum.flatten()
        original_dim = spectrum.shape[0]

        if original_dim == self.target_spec_dim:
            interpolated = spectrum
        else:
            original_x = np.linspace(0, 1, original_dim)
            target_x = np.linspace(0, 1, self.target_spec_dim)
            interpolated = np.interp(target_x, original_x, spectrum)

        spec_min, spec_max = float(interpolated.min()), float(interpolated.max())
        if spec_max - spec_min > 1e-10:
            normalized = (interpolated - spec_min) / (spec_max - spec_min)
        else:
            normalized = np.zeros_like(interpolated)

        return normalized.astype(np.float32)

    def _quantize_to_nearest(self, value: float, allowed_values: List[int]) -> int:
        """Quantize a continuous value to nearest allowed discrete value."""
        return min(allowed_values, key=lambda x: abs(x - value))

    def parameters_to_tokens(self, P: float, D: float, H: float) -> List[int]:
        """Convert P/D/H (meters) to token ids."""
        P_nm = P * 1e9
        H_nm = H * 1e9
        R_nm = (D / 2.0) * 1e9

        P_q = self._quantize_to_nearest(P_nm, self.tokenizer.P_vals)
        H_q = self._quantize_to_nearest(H_nm, self.tokenizer.H_vals)
        R_q = self._quantize_to_nearest(R_nm, self.tokenizer.R_vals)

        tokens = [
            f"PX_{P_q}",
            f"PY_{P_q}",
            "SUB_Glass_Substrate",
            f"L1_MAT_{self.material}",
            "L1_SHAPE_CYL",
            f"L1_H_{H_q}",
            f"L1_R_{R_q}",
        ]
        token_ids = self.tokenizer.encode(tokens)

        # warn if quantization error too large
        p_err = abs(P_nm - P_q) / P_nm * 100 if P_nm > 0 else 0
        r_err = abs(R_nm - R_q) / R_nm * 100 if R_nm > 0 else 0
        h_err = abs(H_nm - H_q) / H_nm * 100 if H_nm > 0 else 0
        if max(p_err, r_err, h_err) > 20:
            print(
                f"[Warn] Large quant error: P={p_err:.1f}%, R={r_err:.1f}%, H={h_err:.1f}%"
            )

        return token_ids

    def validate_parameters(self, P: float, D: float, H: float) -> bool:
        """Check if P/D/H within tokenizer ranges."""
        P_nm = P * 1e9
        H_nm = H * 1e9
        R_nm = (D / 2.0) * 1e9
        P_ok = min(self.tokenizer.P_vals) <= P_nm <= max(self.tokenizer.P_vals)
        H_ok = min(self.tokenizer.H_vals) <= H_nm <= max(self.tokenizer.H_vals)
        R_ok = min(self.tokenizer.R_vals) <= R_nm <= max(self.tokenizer.R_vals)
        return P_ok and H_ok and R_ok

    def _get_params(self, mat_data: dict, filename: str) -> dict:
        if self.use_mat_params and all(k in mat_data for k in ["P", "D", "H"]):
            return {
                "P": float(mat_data["P"].squeeze()),
                "D": float(mat_data["D"].squeeze()),
                "H": float(mat_data["H"].squeeze()),
                "idx": None,
            }
        return self.parse_filename(filename)

    def build_dataset(
        self,
        folder_path: str,
        output_prefix: str,
        max_samples: Optional[int] = None,
    ) -> Tuple[np.ndarray, List[List[int]], List[Tuple[str, str]], int]:
        """Build dataset from a folder of .mat files."""
        mat_files = sorted(glob.glob(os.path.join(folder_path, "*.mat")))
        if max_samples:
            mat_files = mat_files[:max_samples]

        print(f"\n[Build] folder={folder_path}")
        print(f"[Build] found {len(mat_files)} mat files")

        all_spectra: List[np.ndarray] = []
        all_tokens: List[List[int]] = []
        failed_files: List[Tuple[str, str]] = []

        for i, file_path in enumerate(mat_files):
            if i > 0 and i % 1000 == 0:
                print(f"[Build] progress {i}/{len(mat_files)}")
            try:
                filename = Path(file_path).name
                mat_data = scipy.io.loadmat(file_path)
                params = self._get_params(mat_data, filename)

                if not self.validate_parameters(params["P"], params["D"], params["H"]):
                    P_nm = params["P"] * 1e9
                    R_nm = (params["D"] / 2.0) * 1e9
                    H_nm = params["H"] * 1e9
                    msg = f"out_of_range: P={P_nm:.1f}nm R={R_nm:.1f}nm H={H_nm:.1f}nm"
                    failed_files.append((filename, msg))
                    continue

                if "T" not in mat_data:
                    failed_files.append((filename, "missing key T"))
                    continue

                spectrum = mat_data["T"].flatten()
                processed_spec = self.process_spectrum(spectrum)
                tokens = self.parameters_to_tokens(params["P"], params["D"], params["H"])

                all_spectra.append(processed_spec)
                all_tokens.append(tokens)
            except Exception as e:
                failed_files.append((Path(file_path).name, str(e)))

        spectra_array = np.array(all_spectra, dtype=np.float32)
        np.save(f"{output_prefix}_spec.npy", spectra_array)
        with open(f"{output_prefix}_struct.pkl", "wb") as f:
            pickle.dump(all_tokens, f)

        print("\n[Build] done")
        print(f"  success: {len(all_spectra)}")
        print(f"  failed : {len(failed_files)}")
        if failed_files:
            print("[Build] failed samples (first 5)")
            for name, err in failed_files[:5]:
                print(f"  {name}: {err}")

        return spectra_array, all_tokens, failed_files, len(mat_files)


def build_fixed_datasets(
    train_folder: str,
    val_folder: str,
    output_dir: str = ".",
    spec_dim: int = 128,
    test_mode: bool = False,
    use_mat_params: bool = False,
):
    os.makedirs(output_dir, exist_ok=True)
    builder = FixedSiAlphaDatasetBuilder(
        target_spec_dim=spec_dim, use_mat_params=use_mat_params
    )
    max_samples = 100 if test_mode else None

    print("=" * 60)
    print("[Build] Start building datasets")
    print("=" * 60)

    train_spec, train_struct, train_failed, train_total = builder.build_dataset(
        train_folder,
        os.path.join(output_dir, "spec_train"),
        max_samples=max_samples,
    )

    val_spec, val_struct, val_failed, val_total = builder.build_dataset(
        val_folder,
        os.path.join(output_dir, "spec_val"),
        max_samples=max_samples,
    )

    print("\n[Build] Summary")
    print(f"  train: {train_spec.shape[0]:,} samples")
    print(f"  val  : {val_spec.shape[0]:,} samples")
    print(f"  train fail rate: {len(train_failed) / max(1, train_total) * 100:.1f}%")
    print(f"  val   fail rate: {len(val_failed) / max(1, val_total) * 100:.1f}%")

    return train_spec, train_struct, val_spec, val_struct


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--train_dir", required=True, help="folder with train .mat files")
    parser.add_argument("--val_dir", required=True, help="folder with val .mat files")
    parser.add_argument("--out_dir", default=".", help="output dir")
    parser.add_argument("--spec_dim", type=int, default=128)
    parser.add_argument("--test_mode", action="store_true")
    parser.add_argument("--use_mat_params", action="store_true")
    args = parser.parse_args()

    build_fixed_datasets(
        train_folder=args.train_dir,
        val_folder=args.val_dir,
        output_dir=args.out_dir,
        spec_dim=args.spec_dim,
        test_mode=args.test_mode,
        use_mat_params=args.use_mat_params,
    )


if __name__ == "__main__":
    main()
