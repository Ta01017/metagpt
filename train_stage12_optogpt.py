#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
train_stage12_optogpt.py
合并 Stage-1 + Stage-2：直接用光谱条件训练结构序列（OptoGPT 风格）
参考 optogpt_new/core/models/transformer.py 的 Transformer_I
不修改现有文件，单独训练并保存 checkpoint。
"""

import os
import sys
import time
import pickle
import argparse
import numpy as np
import torch
from collections import Counter
from torch.utils.data import Dataset, DataLoader

from config_stage12_optogpt import TrainConfig
from structure_lang.tokenizer import StructureTokenizer


# --- import optogpt_new modules ---
# ROOT = os.path.dirname(os.path.abspath(__file__))
# OPTOGPT_ROOT = os.path.join(ROOT, "optogpt_new")
# if OPTOGPT_ROOT not in sys.path:
#     sys.path.append(OPTOGPT_ROOT)

from core.models.transformer import make_model_I, subsequent_mask  # noqa: E402
from core.trains.train import LabelSmoothing, NoamOpt, SimpleLossCompute, count_params  # noqa: E402


class Stage12Dataset(Dataset):
    def __init__(self, spec_arr, seq_list, max_len: int, pad_id: int, bos_id: int, eos_id: int):
        self.spec = spec_arr
        self.seq = seq_list
        self.max_len = max_len
        self.pad = pad_id
        self.bos = bos_id
        self.eos = eos_id

    def __len__(self):
        return len(self.seq)

    def __getitem__(self, idx):
        spec = self.spec[idx]
        ids = self.seq[idx]

        # truncate if needed
        if len(ids) > self.max_len:
            ids = ids[: self.max_len]
            ids[-1] = self.eos

        return spec, ids


def collate_fn(batch, pad_id):
    specs, seqs = zip(*batch)
    max_len = max(len(s) for s in seqs)
    padded = []
    for s in seqs:
        if len(s) < max_len:
            s = s + [pad_id] * (max_len - len(s))
        padded.append(s)
    seq = torch.tensor(padded, dtype=torch.long)
    spec = torch.tensor(np.stack(specs), dtype=torch.float32)

    # decoder input / target
    trg = seq[:, :-1]
    trg_y = seq[:, 1:]
    trg_mask = (trg != pad_id).unsqueeze(1) & subsequent_mask(trg.size(-1), device=trg.device)
    ntokens = (trg_y != pad_id).sum()

    # src for Transformer_I expects (B, 1, spec_dim)
    src = spec.unsqueeze(1)
    return src, trg, trg_y, trg_mask, ntokens


def save_checkpoint(path, model, epoch, cfg, word_dict, index_dict, spec_dim, metrics=None):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    payload = {
        "epoch": epoch,
        "model": model.state_dict(),
        "model_cfg": {
            "d_model": cfg.d_model,
            "d_ff": cfg.d_ff,
            "n_heads": cfg.n_heads,
            "n_layers": cfg.n_layers,
            "dropout": cfg.dropout,
            "max_len": cfg.max_len,
        },
        "meta": {
            "spec_dim": spec_dim,
            "vocab_size": len(word_dict),
            "pad_id": 1,
            "bos_id": word_dict.get("BOS", 2),
            "eos_id": word_dict.get("EOS", 3),
            "word_dict": word_dict,
            "index_dict": index_dict,
        },
        "metrics": metrics or {},
    }
    torch.save(payload, path)


def build_vocab(seqs, max_words=100000):
    word_count = Counter()
    for s in seqs:
        for w in s:
            word_count[w] += 1
    ls = word_count.most_common(max_words)
    word_dict = {w[0]: idx + 2 for idx, w in enumerate(ls)}
    word_dict["UNK"] = 0
    word_dict["PAD"] = 1
    index_dict = {v: k for k, v in word_dict.items()}
    return word_dict, index_dict


def run_epoch(loader, model, loss_compute, device, log_every=None, tag="train"):
    total_loss = 0.0
    total_tokens = 0
    t0 = time.time()
    for i, (src, trg, trg_y, trg_mask, ntokens) in enumerate(loader, start=1):
        out = model(src.to(device), trg.to(device), src_mask=None, tgt_mask=trg_mask.to(device))
        loss = loss_compute(out, trg_y.to(device), ntokens.to(device))
        total_loss += float(loss)
        total_tokens += int(ntokens)
        if log_every and (i % log_every == 0):
            avg = total_loss / max(total_tokens, 1)
            print(f"[{tag}] step={i} avg_loss={avg:.6f} time={time.time()-t0:.1f}s")
    return total_loss / max(total_tokens, 1)


def eval_loss_compute(generator, criterion):
    def _fn(x, y, norm):
        x = generator(x)
        loss = criterion(x.contiguous().view(-1, x.size(-1)), y.contiguous().view(-1)) / norm
        return loss.data.item() * norm.float()
    return _fn


def train():
    cfg = TrainConfig()
    parser = argparse.ArgumentParser()
    parser.add_argument("--spec_file", type=str, default="")
    parser.add_argument("--struct_file", type=str, default="")
    parser.add_argument("--save_dir", type=str, default="")
    parser.add_argument("--epochs", type=int, default=0)
    parser.add_argument("--batch_size", type=int, default=0)
    parser.add_argument("--smoothing", type=float, default=-1.0)
    parser.add_argument("--spec_type", type=str, default="")
    args = parser.parse_args()

    if args.spec_file:
        cfg.spec_file = args.spec_file
    if args.struct_file:
        cfg.struct_file = args.struct_file
    if args.save_dir:
        cfg.save_dir = args.save_dir
    if args.epochs > 0:
        cfg.epochs = args.epochs
    if args.batch_size > 0:
        cfg.batch_size = args.batch_size
    if args.smoothing >= 0:
        cfg.smoothing = args.smoothing
    if args.spec_type:
        cfg.spec_type = args.spec_type
    torch.manual_seed(cfg.seed)
    np.random.seed(cfg.seed)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("DEVICE =", device)

    # load data
    spec_arr = np.load(cfg.spec_file).astype("float32")
    if cfg.spec_type in ("R", "T") and spec_arr.shape[1] % 2 == 0:
        half = spec_arr.shape[1] // 2
        if cfg.spec_type == "R":
            spec_arr = spec_arr[:, :half]
        elif cfg.spec_type == "T":
            spec_arr = spec_arr[:, half:]

    with open(cfg.struct_file, "rb") as f:
        struct_list = pickle.load(f)

    # build optogpt-style vocab from token strings
    tk = StructureTokenizer()
    seq_tokens = []
    for ids in struct_list:
        toks = [tk.inv_vocab[i] for i in ids]
        seq_tokens.append(["BOS"] + toks + ["EOS"])

    word_dict, index_dict = build_vocab(seq_tokens)
    bos_id = word_dict["BOS"]
    eos_id = word_dict["EOS"]
    pad_id = word_dict["PAD"]

    seq_ids = [[word_dict.get(t, 0) for t in s] for s in seq_tokens]

    # max_len from dataset (include BOS/EOS)
    data_max_len = max(len(s) for s in seq_ids)
    if cfg.max_len != data_max_len:
        print(f"[Info] max_len override: cfg={cfg.max_len} -> data={data_max_len}")
    cfg.max_len = data_max_len

    dataset = Stage12Dataset(spec_arr, seq_ids, cfg.max_len, pad_id, bos_id, eos_id)

    # split train/dev
    n = len(dataset)
    idx = np.arange(n)
    np.random.shuffle(idx)
    dev_n = int(n * cfg.dev_ratio) if cfg.dev_ratio > 0 else 0
    dev_idx = idx[:dev_n]
    train_idx = idx[dev_n:]

    train_set = torch.utils.data.Subset(dataset, train_idx)
    dev_set = torch.utils.data.Subset(dataset, dev_idx) if dev_n > 0 else None

    train_loader = DataLoader(
        train_set,
        batch_size=cfg.batch_size,
        shuffle=True,
        collate_fn=lambda b: collate_fn(b, pad_id),
    )
    dev_loader = None
    if dev_set is not None:
        dev_loader = DataLoader(
            dev_set,
            batch_size=cfg.batch_size,
            shuffle=False,
            collate_fn=lambda b: collate_fn(b, pad_id),
        )

    # build model (OptoGPT Transformer_I)
    spec_dim = spec_arr.shape[1]
    model = make_model_I(
        src_vocab=spec_dim,
        tgt_vocab=len(word_dict),
        N=cfg.n_layers,
        d_model=cfg.d_model,
        d_ff=cfg.d_ff,
        h=cfg.n_heads,
        dropout=cfg.dropout,
    ).to(device)

    print("Model params:", count_params(model))

    criterion = LabelSmoothing(len(word_dict), padding_idx=pad_id, smoothing=cfg.smoothing)
    optimizer = NoamOpt(
        cfg.d_model,
        cfg.max_lr,
        cfg.warm_steps,
        torch.optim.Adam(model.parameters(), lr=0, betas=(0.9, 0.98), eps=1e-9),
    )

    loss_compute = SimpleLossCompute(model.generator, criterion, optimizer)
    loss_compute_eval = eval_loss_compute(model.generator, criterion)

    # training loop
    best_loss = float("inf")

    for epoch in range(1, cfg.epochs + 1):
        model.train()
        train_loss = run_epoch(train_loader, model, loss_compute, device, log_every=cfg.log_every, tag="train")
        print(f"[Epoch {epoch}] train_loss={train_loss:.6f}")

        dev_loss = train_loss
        if dev_loader is not None:
            model.eval()
            with torch.no_grad():
                dev_loss = run_epoch(dev_loader, model, loss_compute_eval, device, log_every=None, tag="dev")
            print(f"[Epoch {epoch}] dev_loss={dev_loss:.6f}")

        metrics = {"train_loss": float(train_loss), "dev_loss": float(dev_loss)}

        if cfg.save_best and dev_loss < best_loss:
            best_loss = dev_loss
            ck = os.path.join(cfg.save_dir, "stage12_best.pt")
            save_checkpoint(ck, model, epoch, cfg, word_dict, index_dict, spec_dim, metrics=metrics)
            print("[Save] best", ck)

        if cfg.save_recent:
            ck = os.path.join(cfg.save_dir, "stage12_recent.pt")
            save_checkpoint(ck, model, epoch, cfg, word_dict, index_dict, spec_dim, metrics=metrics)
            print("[Save] recent", ck)

        if cfg.save_every and (epoch % cfg.save_every == 0):
            ck = os.path.join(cfg.save_dir, f"stage12_epoch{epoch}.pt")
            save_checkpoint(ck, model, epoch, cfg, word_dict, index_dict, spec_dim, metrics=metrics)
            print("[Save] epoch", ck)


if __name__ == "__main__":
    train()
