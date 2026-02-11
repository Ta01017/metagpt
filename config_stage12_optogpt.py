# config_stage12_optogpt.py
from dataclasses import dataclass


@dataclass
class TrainConfig:
    # ===== paths =====
    spec_file: str = "./dataset_stage2_toy/spec_train.npy"
    struct_file: str = "./dataset_stage2_toy/struct_train.pkl"
    save_dir: str = "./ckpt_stage12_optogpt_toy"

    # ===== model (OptoGPT fp_100k_full defaults) =====
    d_model: int = 256
    d_ff: int = 1024
    n_heads: int = 8
    n_layers: int = 1
    dropout: float = 0.05
    max_len: int = 10

    # ===== training =====
    epochs: int = 500
    batch_size: int = 256
    max_lr: float = 1e-3
    warm_steps: int = 16000
    smoothing: float = 0.1
    log_every: int = 50
    save_every: int = 0  # 0=disable periodic saves
    seed: int = 42

    # ===== imbalance handling =====
    token_reweight: bool = True
    reweight_alpha: float = 0.5  # 0=off, 0.5~1.0 stronger
    reweight_min_freq: int = 1

    # ===== spec type =====
    # 'R' / 'T' / 'R_T' (default keep full)
    spec_type: str = "R_T"

    # ===== split & save =====
    dev_ratio: float = 0.1  # use a small dev split to track best
    save_best: bool = True
    save_recent: bool = True
