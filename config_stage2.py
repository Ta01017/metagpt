# config_stage2.py
from dataclasses import dataclass

@dataclass
class TrainConfig:
    # ===== paths =====
    stage1_ckpt: str = "./ckpt_stage1/stage1_step50000.pt"
    cache_dir: str = "./cache_stage2"
    ckpt_dir: str = "./ckpt_stage2"

    # ===== model =====
    # 这些要和 Stage1 一致（d_model/n_layers/n_heads/d_ff）
    d_model: int = 256
    d_ff: int = 1024
    n_heads: int = 8
    n_layers: int = 4
    dropout: float = 0.05

    # Stage2 新增：prefix
    prefix_len: int = 32          # 增强光谱条件表达
    spec_dim: int = 128           # 伪光谱/特征维度（你可改，和 cache 对齐）

    # token sequence length（不含 prefix）
    max_len: int = 32             # token序列长度（含 BOS/EOS/PAD 的总长度，缓存会按这个做）

    pad_id: int = 0
    bos_id: int = 1
    eos_id: int = 2

    # ===== data files =====
    spec_file: str = "./dataset_stage2/spec_train.npy"
    struct_file: str = "./dataset_stage2/struct_train.pkl"

    # ===== train =====
    batch_size: int = 32
    num_workers: int = 2
    use_amp: bool = True
    lr: float = 3e-4
    total_steps: int = 50000
    log_every: int = 50
    save_every: int = 2000

    # ===== cot & grammar =====
    use_cot: bool = True
    cot_dropout: float = 0.2
    use_grammar: bool = True
    grammar_margin: int = 30

    # optional: 先只训 prefix，过几步再全量 finetune
    freeze_stage1_steps: int = 0  # 0=不冻结；比如 1000 表示前1000步只训 prefix
