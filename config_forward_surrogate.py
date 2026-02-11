# config_forward_surrogate.py
from dataclasses import dataclass


@dataclass
class TrainConfig:
    # ===== data =====
    spec_file: str = "./dataset_stage2_toy/spec_train.npy"
    struct_file: str = "./dataset_stage2_toy/struct_train.pkl"
    stage12_ckpt: str = "./ckpt_stage12_optogpt_toy/stage12_best.pt"  # to align vocab
    save_dir: str = "./ckpt_forward_surrogate_toy"

    # ===== model =====
    d_model: int = 256
    n_heads: int = 8
    n_layers: int = 2
    d_ff: int = 512
    dropout: float = 0.1
    max_len: int = 10

    # ===== train =====
    batch_size: int = 256
    epochs: int = 200
    lr: float = 1e-3
    weight_decay: float = 1e-4
    dev_ratio: float = 0.1
    seed: int = 42

    # ===== logging =====
    log_every: int = 50
