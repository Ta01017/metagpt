#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
train_forward_surrogate.py
Train a forward surrogate: structure tokens -> spectrum vector.
Does not modify existing code.
"""

from __future__ import annotations

import os
import argparse
import pickle
import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader

from config_forward_surrogate import TrainConfig
from structure_lang.tokenizer import StructureTokenizer
from models.forward_surrogate import ForwardSurrogate


class SurrogateDataset(Dataset):
    def __init__(self, spec_arr, seq_ids, max_len: int, pad_id: int):
        self.spec = spec_arr
        self.seq = seq_ids
        self.max_len = max_len
        self.pad = pad_id

    def __len__(self):
        return len(self.seq)

    def __getitem__(self, idx):
        spec = self.spec[idx]
        ids = self.seq[idx]
        if len(ids) > self.max_len:
            ids = ids[: self.max_len]
        if len(ids) < self.max_len:
            ids = ids + [self.pad] * (self.max_len - len(ids))
        return torch.tensor(ids, dtype=torch.long), torch.tensor(spec, dtype=torch.float32)


def load_stage12_vocab(ckpt_path: str):
    ckpt = torch.load(ckpt_path, map_location="cpu")
    meta = ckpt["meta"]
    word_dict = meta["word_dict"]
    index_dict = meta["index_dict"]
    return word_dict, index_dict


def build_seq_ids(struct_list, word_dict):
    tk = StructureTokenizer()
    unk_id = word_dict.get("UNK", 0)
    bos_id = word_dict.get("BOS")
    eos_id = word_dict.get("EOS")

    seq_ids = []
    for ids in struct_list:
        toks = [tk.inv_vocab[i] for i in ids]
        toks = ["BOS"] + toks + ["EOS"]
        mapped = [word_dict.get(t, unk_id) for t in toks]
        if bos_id is not None:
            mapped[0] = bos_id
        if eos_id is not None:
            mapped[-1] = eos_id
        seq_ids.append(mapped)
    return seq_ids


def evaluate(model, loader, pad_id, device):
    model.eval()
    mse_list, mae_list, corr_list = [], [], []
    with torch.no_grad():
        for tokens, spec in loader:
            tokens = tokens.to(device)
            spec = spec.to(device)
            pred = model(tokens, pad_id)
            mse = torch.mean((pred - spec) ** 2, dim=1)
            mae = torch.mean(torch.abs(pred - spec), dim=1)
            denom = (pred.norm(dim=1) * spec.norm(dim=1)) + 1e-8
            corr = torch.sum(pred * spec, dim=1) / denom
            mse_list.append(mse.detach().cpu())
            mae_list.append(mae.detach().cpu())
            corr_list.append(corr.detach().cpu())
    mse = torch.cat(mse_list).mean().item()
    mae = torch.cat(mae_list).mean().item()
    corr = torch.cat(corr_list).mean().item()
    return mse, mae, corr


def save_ckpt(path, model, cfg: TrainConfig, meta):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    payload = {
        "model": model.state_dict(),
        "model_cfg": {
            "d_model": cfg.d_model,
            "n_heads": cfg.n_heads,
            "n_layers": cfg.n_layers,
            "d_ff": cfg.d_ff,
            "dropout": cfg.dropout,
            "max_len": cfg.max_len,
        },
        "meta": meta,
    }
    torch.save(payload, path)


def train(args):
    cfg = TrainConfig()
    for k, v in vars(args).items():
        if v is not None and hasattr(cfg, k):
            setattr(cfg, k, v)

    torch.manual_seed(cfg.seed)
    np.random.seed(cfg.seed)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("DEVICE =", device)

    spec_arr = np.load(cfg.spec_file).astype("float32")
    with open(cfg.struct_file, "rb") as f:
        struct_list = pickle.load(f)

    word_dict, index_dict = load_stage12_vocab(cfg.stage12_ckpt)
    seq_ids = build_seq_ids(struct_list, word_dict)

    data_max_len = max(len(s) for s in seq_ids)
    if cfg.max_len != data_max_len:
        print(f"[Info] max_len override: cfg={cfg.max_len} -> data={data_max_len}")
    cfg.max_len = data_max_len

    pad_id = word_dict.get("PAD", 1)
    dataset = SurrogateDataset(spec_arr, seq_ids, cfg.max_len, pad_id)

    n = len(dataset)
    idx = np.arange(n)
    np.random.shuffle(idx)
    dev_n = int(n * cfg.dev_ratio) if cfg.dev_ratio > 0 else 0
    dev_idx = idx[:dev_n]
    train_idx = idx[dev_n:]

    train_set = torch.utils.data.Subset(dataset, train_idx)
    dev_set = torch.utils.data.Subset(dataset, dev_idx) if dev_n > 0 else None

    train_loader = DataLoader(train_set, batch_size=cfg.batch_size, shuffle=True)
    dev_loader = DataLoader(dev_set, batch_size=cfg.batch_size, shuffle=False) if dev_set else None

    spec_dim = spec_arr.shape[1]
    model = ForwardSurrogate(
        vocab_size=len(word_dict),
        spec_dim=spec_dim,
        d_model=cfg.d_model,
        n_heads=cfg.n_heads,
        n_layers=cfg.n_layers,
        d_ff=cfg.d_ff,
        dropout=cfg.dropout,
        max_len=cfg.max_len,
    ).to(device)

    optim = torch.optim.AdamW(model.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay)

    meta = {
        "spec_dim": spec_dim,
        "vocab_size": len(word_dict),
        "pad_id": pad_id,
        "word_dict": word_dict,
        "index_dict": index_dict,
        "max_len": cfg.max_len,
    }

    best_mse = 1e9
    for epoch in range(1, cfg.epochs + 1):
        model.train()
        total = 0.0
        count = 0
        for i, (tokens, spec) in enumerate(train_loader, start=1):
            tokens = tokens.to(device)
            spec = spec.to(device)
            pred = model(tokens, pad_id)
            loss = F.mse_loss(pred, spec)
            optim.zero_grad()
            loss.backward()
            optim.step()
            total += loss.item()
            count += 1
            if cfg.log_every and (i % cfg.log_every == 0):
                print(f"[train] epoch={epoch} step={i} loss={total/max(count,1):.6f}")

        if dev_loader is not None:
            mse, mae, corr = evaluate(model, dev_loader, pad_id, device)
            print(f"[dev] epoch={epoch} MSE={mse:.6f} MAE={mae:.6f} Corr={corr:.6f}")
            if mse < best_mse:
                best_mse = mse
                save_ckpt(os.path.join(cfg.save_dir, "surrogate_best.pt"), model, cfg, meta)
        else:
            print(f"[train] epoch={epoch} avg_loss={total/max(count,1):.6f}")

    save_ckpt(os.path.join(cfg.save_dir, "surrogate_last.pt"), model, cfg, meta)
    print("[Save] last ->", os.path.join(cfg.save_dir, "surrogate_last.pt"))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--spec_file")
    parser.add_argument("--struct_file")
    parser.add_argument("--stage12_ckpt")
    parser.add_argument("--save_dir")
    parser.add_argument("--epochs", type=int)
    parser.add_argument("--batch_size", type=int)
    parser.add_argument("--lr", type=float)
    parser.add_argument("--weight_decay", type=float)
    parser.add_argument("--dev_ratio", type=float)
    parser.add_argument("--seed", type=int)
    args = parser.parse_args()
    train(args)
