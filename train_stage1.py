# train_stage1.py
import os
import sys
import numpy as np
from pathlib import Path

# ---- auto-add project root ----
ROOT = Path(__file__).resolve().parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import torch
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from torch.optim import AdamW

from config_stage1 import TrainConfig
from models.metagpt import MetaGPT
from models.transformer_sdpa import TransformerConfig


# -------------------------------
# Dataset using cached npz files
# -------------------------------
class Stage1CacheDataset(Dataset):
    def __init__(self, path_npz):
        data = np.load(path_npz)
        self.seq = torch.from_numpy(data["seq"]).long()

    def __len__(self):
        return self.seq.shape[0]

    def __getitem__(self, idx):
        x = self.seq[idx]
        # next-token labels（完全语言模型）
        inp = x[:-1]
        tgt = x[1:]
        return inp, tgt


def train_stage1():
    cfg = TrainConfig()
    device = "cuda" if torch.cuda.is_available() else "cpu"

    # ---- check cache ----
    train_npz = os.path.join(cfg.cache_dir, "train.npz")
    dev_npz = os.path.join(cfg.cache_dir, "dev.npz")

    if not os.path.exists(train_npz):
        raise RuntimeError(
            f"Stage1 cache not found. 请先运行:\n  python data/gen_stage1_synth_cache.py"
        )

    # ---- load datasets ----
    train_ds = Stage1CacheDataset(train_npz)
    dev_ds = Stage1CacheDataset(dev_npz)

    train_loader = DataLoader(train_ds, batch_size=cfg.batch_size, shuffle=True)
    dev_loader = DataLoader(dev_ds, batch_size=cfg.batch_size, shuffle=False)

    # ---- load tokenizer metadata ----
    import json
    meta = json.load(open(os.path.join(cfg.cache_dir, "meta.json"), "r"))
    vocab_size = meta["vocab_size"]
    bos_id = meta["bos_id"]
    eos_id = meta["eos_id"]
    pad_id = meta["pad_id"]

    # ---- build model ----
    model_cfg = TransformerConfig(
        vocab_size=vocab_size,
        d_model=cfg.d_model,
        d_ff=cfg.d_ff,
        n_heads=cfg.n_heads,
        n_layers=cfg.n_layers,
        max_len=64,          # 全局模型位置长度（你最后确认的）
        dropout=cfg.dropout,
    )

    model = MetaGPT(
        cfg=model_cfg,
        spec_dim=1,        # Stage1 不使用
        prefix_len=0,      # Stage1 没有 prefix
        pad_id=pad_id,
    ).to(device)

    optim = AdamW(model.parameters(), lr=cfg.lr)

    step = 0
    for epoch in range(9999999):  # 由 total_steps 控制退出
        for inp, tgt in train_loader:
            step += 1
            inp = inp.to(device)
            tgt = tgt.to(device)

            logits, loss, aux = model(
                input_ids=inp,
                spectra=None,
                labels=tgt,
            )

            optim.zero_grad()
            loss.backward()
            optim.step()

            if step % cfg.log_every == 0:
                print(f"[Stage1][step {step}] loss={loss.item():.4f}")

            if step % cfg.save_every == 0:
                os.makedirs(cfg.ckpt_dir, exist_ok=True)
                torch.save(
                    {
                        "step": step,
                        "model": model.state_dict(),
                        "config": cfg,
                    },
                    os.path.join(cfg.ckpt_dir, f"stage1_step{step}.pt")
                )

            if step >= cfg.total_steps:
                return


if __name__ == "__main__":
    train_stage1()
