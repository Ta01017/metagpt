# config_stage1.py
from dataclasses import dataclass


@dataclass
class TrainConfig:
    # dataset
    num_samples: int = 20000
    seed: int = 0

    # data
    vocab_size: int = 512
    max_len: int = 64
    pad_id: int = 0
    bos_id: int = 1
    eos_id: int = 2

    # model
    d_model: int = 256
    d_ff: int = 1024
    n_heads: int = 8
    n_layers: int = 6
    dropout: float = 0.1

    # training
    batch_size: int = 64
    lr: float = 3e-4
    warmup: int = 4000
    total_steps: int = 50000
    log_every: int = 50
    save_every: int = 1000

    # ckpt
    cache_dir: str = "./cache_stage1"
    ckpt_dir: str = "./ckpt_stage1"
