# config_stage12_rl_surrogate.py
from dataclasses import dataclass


@dataclass
class TrainConfig:
    # ===== paths =====
    stage12_ckpt: str = "D:\work\hikvision\metagpt_old/ckpt_stage12_optogpt_toy/stage12_best.pt"
    surrogate_ckpt: str = "./ckpt_forward_surrogate_real/surrogate_best.pt"
    spec_file: str = "D:\work\hikvision\metagpt_old\dataset_stage2_toy\spec_train.npy"
    save_dir: str = "./ckpt_stage12_rl_surrogate_toy"

    # ===== rl =====
    steps: int = 2000
    batch_size: int = 16
    lr: float = 1e-6
    grad_clip: float = 1.0
    temperature: float = 1.0
    top_k: int = 30
    top_p: float = 0.95
    entropy_beta: float = 0.01
    invalid_penalty: float = -1.0

    reward_mse_weight: float = 1.0
    reward_corr_weight: float = 0.2
    reward_len_penalty: float = 0.001
    reward_repeat_penalty: float = 0.005
    reward_missing_eos_penalty: float = 0.2

    # ===== generation =====
    max_len: int = 10

    # ===== grammar =====
    use_grammar: bool = True
    grammar_margin: int = 30

    # ===== logging =====
    log_every: int = 50
    save_every: int = 500
    seed: int = 42
