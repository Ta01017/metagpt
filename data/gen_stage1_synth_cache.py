# data/gen_stage1_synth_cache.py
import os
import sys
import json
import numpy as np
from pathlib import Path

# ---- auto-add project root ----
ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from config_stage1 import TrainConfig
from structure_lang.tokenizer import StructureTokenizer
from structure_lang.parser import StructureParser
from structure_lang.validator import StructureValidator

from data.gen_stage1_synth import Stage1SynthGenerator   # 你已有的生成器


def main():
    cfg = TrainConfig()

    # cache dir
    os.makedirs(cfg.cache_dir, exist_ok=True)

    # ---- load tokenizer ----
    tk = StructureTokenizer()
    parser = StructureParser()
    val = StructureValidator(min_feature_nm=20, margin_nm=30)

    # ---- align metadata ----
    vocab_size = tk.vocab_size
    bos_id = tk.bos_id
    eos_id = tk.eos_id
    pad_id = tk.pad_id

    print(f"[Stage1Cache] vocab={vocab_size}, bos/eos/pad={bos_id}/{eos_id}/{pad_id}")

    # ---- build generator ----
    gen = Stage1SynthGenerator(
        tokenizer=tk,
        parser=parser,
        validator=val,
        P_range_nm=(400, 900),
        H_range_nm=(80, 1000),
        R_range_nm=(40, 250),
        W_range_nm=(60, 550),
        use_materials=["SiO2", "TiO2", "Ta2O5", "HfO2"],
        p_cyl=0.5,
        seed=cfg.seed,
    )

    # ---- generate dataset ----
    N = cfg.num_samples
    max_len = cfg.max_len

    seqs = []
    for _ in range(N):
        ids = gen.sample_one_ids()                # 无BOS/EOS
        ids = [bos_id] + ids + [eos_id]           # 加 BOS、EOS
        if len(ids) > max_len:
            ids = ids[:max_len]                   # 截断
            ids[-1] = eos_id
        seq_pad = ids + [pad_id] * (max_len - len(ids))
        seqs.append(seq_pad)

    seqs = np.array(seqs, dtype=np.int64)

    # ---- split train/dev ----
    n_train = int(N * 0.95)
    train = seqs[:n_train]
    dev = seqs[n_train:]

    # ---- save ----
    np.savez_compressed(os.path.join(cfg.cache_dir, "train.npz"), seq=train)
    np.savez_compressed(os.path.join(cfg.cache_dir, "dev.npz"), seq=dev)

    meta = {
        "stage": "stage1",
        "vocab_size": vocab_size,
        "bos_id": bos_id,
        "eos_id": eos_id,
        "pad_id": pad_id,
        "max_len": max_len,
        "num_train": train.shape[0],
        "num_dev": dev.shape[0],
        "seed": cfg.seed,
    }
    with open(os.path.join(cfg.cache_dir, "meta.json"), "w", encoding="utf-8") as f:
        json.dump(meta, f, indent=2, ensure_ascii=False)

    print(f"[Stage1Cache] Saved train/dev/meta to {cfg.cache_dir}")
    print("[Stage1Cache] Done.")


if __name__ == "__main__":
    main()
