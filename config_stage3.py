# config_stage3.py
from dataclasses import dataclass


@dataclass
class TrainConfig:
    # ===== paths =====
    stage2_ckpt: str = "./ckpt_stage2/stage2_step20000.pt"
    spec_file: str = "./dataset_stage2/spec_train.npy"
    save_dir: str = "./ckpt_stage3_rl"
    resume: str = ""

    # ===== rl =====
    steps: int = 4000
    batch_size: int = 8
    lr: float = 1e-5
    grad_clip: float = 1.0
    temperature: float = 1.0
    top_k: int = 30
    top_p: float = 0.95
    entropy_beta: float = 0.0
    invalid_penalty: float = -1.0
    reward_mse_weight: float = 1.0
    reward_corr_weight: float = 0.2
    reward_len_penalty: float = 0.002
    reward_repeat_penalty: float = 0.01
    reward_missing_eos_penalty: float = 0.2

    # ===== generation =====
    max_new: int = 32

    # ===== grammar =====
    use_grammar: bool = True
    grammar_margin: int = 30

    # ===== logging =====
    log_every: int = 50
    save_every: int = 500
    seed: int = 42
