#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
train_stage12_rl_optogpt.py
在 Stage12 OptoGPT 模型基础上进行 RL 微调（不修改原代码）。
"""

import os
import sys
import time
import numpy as np
import torch
import torch.nn.functional as F
from torch.optim import AdamW

from config_stage12_rl_optogpt import TrainConfig
from data.gen_stage2_fake_dataset import fake_spectrum_from_structure
from structure_lang.parser import StructureParser
from structure_lang.validator import StructureValidator

# optogpt_new
ROOT = os.path.dirname(os.path.abspath(__file__))
OPTOGPT_ROOT = os.path.join(ROOT, "optogpt_new")
if OPTOGPT_ROOT not in sys.path:
    sys.path.append(OPTOGPT_ROOT)

from core.models.transformer import make_model_I, subsequent_mask  # noqa: E402


def classify_token(token: str):
    if token.startswith("PX_"):
        return "PX"
    if token.startswith("PY_"):
        return "PY"
    if token.startswith("SUB_"):
        return "SUB"
    if token.startswith("L1_MAT_"):
        return "MAT"
    if token.startswith("L1_SHAPE_"):
        return "SHAPE"
    if token.startswith("L1_H_"):
        return "H"
    if token.startswith("L1_R_"):
        return "R"
    if token.startswith("L1_W_"):
        return "W"
    if token.startswith("L1_L_"):
        return "L"
    return "OTHER"


def extract_num(token: str):
    return int(token.split("_")[-1])


def apply_constraints(logits, token_strings, generated_tokens, px=None, py=None, shape=None, margin=30):
    vocab = len(token_strings)
    mask = torch.zeros(vocab, dtype=torch.bool, device=logits.device)

    if px is None:
        need = "PX"
    elif py is None:
        need = "PY"
    elif shape is None:
        if not any(tok.startswith("SUB_") for tok in generated_tokens):
            need = "SUB"
        elif not any(tok.startswith("L1_MAT_") for tok in generated_tokens):
            need = "MAT"
        else:
            need = "SHAPE"
    else:
        if shape == "CYL":
            if not any(tok.startswith("L1_H_") for tok in generated_tokens):
                need = "H"
            elif not any(tok.startswith("L1_R_") for tok in generated_tokens):
                need = "R"
            else:
                need = "EOS"
        else:
            if not any(tok.startswith("L1_H_") for tok in generated_tokens):
                need = "H"
            elif not any(tok.startswith("L1_W_") for tok in generated_tokens):
                need = "W"
            elif not any(tok.startswith("L1_L_") for tok in generated_tokens):
                need = "L"
            else:
                need = "EOS"

    for tid, token in enumerate(token_strings):
        cls = classify_token(token)
        if need == "EOS" and token == "EOS":
            mask[tid] = True
        elif need == cls:
            mask[tid] = True

    for tid, tok in enumerate(token_strings):
        if tok in ("PAD", "BOS", "UNK"):
            mask[tid] = False

    if px is not None and py is not None:
        Pmin = min(px, py)
        if need == "R":
            limit = Pmin / 2 - margin
            for tid, token in enumerate(token_strings):
                if token.startswith("L1_R_"):
                    val = extract_num(token)
                    mask[tid] = val <= limit
        if need in ("W", "L"):
            limit = Pmin - margin
            for tid, token in enumerate(token_strings):
                if token.startswith(f"L1_{need}_"):
                    val = extract_num(token)
                    mask[tid] = val <= limit

    if mask.sum() == 0:
        return logits

    logits = logits.clone()
    logits[~mask] = -1e9
    return logits


def tokens_to_struct(tokens):
    keep = []
    for t in tokens:
        if t.startswith("PX_") or t.startswith("PY_") or t.startswith("SUB_") or t.startswith("L1_"):
            keep.append(t)
    return keep


def load_stage12(ckpt_path, device):
    ckpt = torch.load(ckpt_path, map_location="cpu")
    cfg = ckpt["model_cfg"]
    meta = ckpt["meta"]
    word_dict = meta["word_dict"]
    index_dict = meta["index_dict"]
    spec_dim = meta["spec_dim"]

    model = make_model_I(
        src_vocab=spec_dim,
        tgt_vocab=len(word_dict),
        N=cfg["n_layers"],
        d_model=cfg["d_model"],
        d_ff=cfg["d_ff"],
        h=cfg["n_heads"],
        dropout=cfg["dropout"],
    ).to(device)
    model.load_state_dict(ckpt["model"], strict=True)
    # build index->token list for fast lookup
    token_strings = ["UNK"] * len(word_dict)
    for k, v in word_dict.items():
        if v < len(token_strings):
            token_strings[v] = k
    return model, cfg, meta, word_dict, index_dict, token_strings


@torch.no_grad()
def generate_rollout(model, spec_batch, word_dict, token_strings, cfg: TrainConfig, greedy: bool):
    device = spec_batch.device
    B = spec_batch.size(0)
    bos_id = word_dict.get("BOS", 2)
    eos_id = word_dict.get("EOS", 3)
    pad_id = word_dict.get("PAD", 1)

    seqs = [[bos_id] for _ in range(B)]
    finished = torch.zeros(B, dtype=torch.bool, device=device)
    generated_tokens = [[] for _ in range(B)]
    px = [None] * B
    py = [None] * B
    shape = [None] * B

    for _ in range(cfg.max_len - 1):
        # pad to current max len
        max_t = max(len(s) for s in seqs)
        trg = torch.tensor(
            [s + [pad_id] * (max_t - len(s)) for s in seqs],
            dtype=torch.long,
            device=device
        )
        trg_mask = (trg != pad_id).unsqueeze(1) & subsequent_mask(trg.size(-1), device=trg.device)
        src = spec_batch.unsqueeze(1)

        out = model(src, trg, src_mask=None, tgt_mask=trg_mask)
        log_probs = model.generator(out[:, -1, :])  # (B, V)

        if cfg.use_grammar:
            for b in range(B):
                if finished[b]:
                    continue
                log_probs[b] = apply_constraints(
                    log_probs[b],
                    token_strings=token_strings,
                    generated_tokens=generated_tokens[b],
                    px=px[b],
                    py=py[b],
                    shape=shape[b],
                    margin=cfg.grammar_margin,
                )

        if greedy:
            next_id = torch.argmax(log_probs, dim=-1)
        else:
            logits = log_probs / max(cfg.temperature, 1e-8)
            if cfg.top_k > 0:
                v, _ = torch.topk(logits, k=min(cfg.top_k, logits.size(-1)))
                kth = v[:, -1].unsqueeze(-1)
                logits = torch.where(logits < kth, torch.full_like(logits, -1e9), logits)
            if cfg.top_p < 1.0:
                sorted_logits, sorted_idx = torch.sort(logits, descending=True)
                probs = torch.softmax(sorted_logits, dim=-1)
                cdf = probs.cumsum(dim=-1)
                cutoff = cdf > cfg.top_p
                cutoff[:, 0] = False
                sorted_logits = torch.where(cutoff, torch.full_like(sorted_logits, -1e9), sorted_logits)
                logits = torch.zeros_like(logits).scatter(-1, sorted_idx, sorted_logits)
            probs = torch.softmax(logits, dim=-1)
            next_id = torch.multinomial(probs, 1).squeeze(-1)

        next_id = torch.where(finished, torch.full_like(next_id, eos_id), next_id)

        for b in range(B):
            if finished[b]:
                continue
            nid = int(next_id[b].item())
            seqs[b].append(nid)
            tok = token_strings[nid] if nid < len(token_strings) else "UNK"
            if tok not in ("BOS", "EOS", "PAD", "UNK"):
                generated_tokens[b].append(tok)
                if tok.startswith("PX_"):
                    px[b] = extract_num(tok)
                elif tok.startswith("PY_"):
                    py[b] = extract_num(tok)
                elif tok.startswith("L1_SHAPE_"):
                    shape[b] = tok.split("_")[-1]
            if nid == eos_id:
                finished[b] = True

        if finished.all():
            break

    # pad to max_len
    padded = []
    for s in seqs:
        if len(s) < cfg.max_len:
            s = s + [pad_id] * (cfg.max_len - len(s))
        padded.append(s)
    return torch.tensor(padded, dtype=torch.long, device=device)


def sequence_logprob(model, spec_batch, tokens, word_dict):
    pad_id = word_dict.get("PAD", 1)
    trg = tokens[:, :-1].contiguous()
    trg_y = tokens[:, 1:].contiguous()
    trg_mask = (trg != pad_id).unsqueeze(1) & subsequent_mask(trg.size(-1), device=trg.device)
    src = spec_batch.unsqueeze(1)

    out = model(src, trg, src_mask=None, tgt_mask=trg_mask)
    log_probs = model.generator(out)  # (B, T, V) log_softmax
    token_logp = log_probs.gather(-1, trg_y.unsqueeze(-1)).squeeze(-1)
    mask = (trg_y != pad_id).float()

    # ignore tokens after EOS
    eos_id = word_dict.get("EOS", 3)
    for b in range(mask.size(0)):
        eos_pos = (trg_y[b] == eos_id).nonzero(as_tuple=False)
        if eos_pos.numel() > 0:
            first = int(eos_pos[0].item())
            if first + 1 < mask.size(1):
                mask[b, first + 1:] = 0.0

    seq_logp = (token_logp * mask).sum(dim=-1)

    probs = log_probs.exp()
    entropy = -(probs * log_probs).sum(dim=-1)
    entropy = (entropy * mask).sum(dim=-1) / (mask.sum(dim=-1) + 1e-8)
    return seq_logp, entropy


def compute_reward(tokens, spec_batch, word_dict, index_dict, cfg: TrainConfig, parser, validator):
    rewards = []
    for b in range(tokens.size(0)):
        ids = tokens[b].tolist()
        eos_id = word_dict.get("EOS", 3)
        pad_id = word_dict.get("PAD", 1)
        has_eos = eos_id in ids
        if has_eos:
            ids = ids[: ids.index(eos_id) + 1]

        toks = [index_dict.get(i, "UNK") for i in ids]
        toks = [t for t in toks if t not in ("BOS", "EOS", "PAD", "UNK")]
        struct_toks = tokens_to_struct(toks)

        ok, reason = validator.validate(parser.parse(["[BOS]"] + struct_toks + ["[EOS]"]))
        if not ok:
            r = torch.tensor(cfg.invalid_penalty, device=spec_batch.device)
        else:
            pred = fake_spectrum_from_structure(struct_toks, spec_dim=spec_batch.size(1))
            pred_spec = torch.tensor(pred, device=spec_batch.device)
            mse = torch.mean((pred_spec - spec_batch[b]) ** 2)
            denom = (torch.norm(pred_spec) * torch.norm(spec_batch[b])) + 1e-8
            corr = torch.dot(pred_spec, spec_batch[b]) / denom
            r = -cfg.reward_mse_weight * mse + cfg.reward_corr_weight * corr

        # length penalty
        if len(struct_toks) > 0:
            r = r - cfg.reward_len_penalty * len(struct_toks)

        # repeat penalty
        max_run = 1
        run = 1
        last = None
        for t in toks:
            if t == last:
                run += 1
                max_run = max(max_run, run)
            else:
                run = 1
                last = t
        if max_run > 3:
            r = r - cfg.reward_repeat_penalty * float(max_run - 3)

        if not has_eos:
            r = r - cfg.reward_missing_eos_penalty

        rewards.append(r)
    return torch.stack(rewards, dim=0)


def train():
    cfg = TrainConfig()
    torch.manual_seed(cfg.seed)
    np.random.seed(cfg.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(cfg.seed)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("DEVICE =", device)

    spec_arr = np.load(cfg.spec_file).astype("float32")
    model, model_cfg, meta, word_dict, index_dict, token_strings = load_stage12(cfg.stage12_ckpt, device)
    model.train()

    parser = StructureParser()
    validator = StructureValidator(min_feature_nm=20, margin_nm=30)

    optim = AdamW(model.parameters(), lr=cfg.lr)

    os.makedirs(cfg.save_dir, exist_ok=True)

    for step in range(1, cfg.steps + 1):
        idx = np.random.randint(0, spec_arr.shape[0], size=cfg.batch_size)
        spec_batch = torch.tensor(spec_arr[idx], device=device)

        # sample rollout
        sample_tokens = generate_rollout(model, spec_batch, word_dict, token_strings, cfg, greedy=False)
        r_sample = compute_reward(sample_tokens, spec_batch, word_dict, index_dict, cfg, parser, validator)

        # greedy baseline
        base_tokens = generate_rollout(model, spec_batch, word_dict, token_strings, cfg, greedy=True)
        r_base = compute_reward(base_tokens, spec_batch, word_dict, index_dict, cfg, parser, validator)

        adv = (r_sample - r_base).detach()

        seq_logp, entropy = sequence_logprob(model, spec_batch, sample_tokens, word_dict)
        loss_pg = -(adv * seq_logp).mean()
        loss = loss_pg - cfg.entropy_beta * entropy.mean()

        optim.zero_grad()
        loss.backward()
        if cfg.grad_clip > 0:
            torch.nn.utils.clip_grad_norm_(model.parameters(), cfg.grad_clip)
        optim.step()

        if step % cfg.log_every == 0:
            print(
                f"[step {step}] loss={loss.item():.4f} "
                f"r_sample={r_sample.mean().item():.4f} r_base={r_base.mean().item():.4f} "
                f"adv={adv.mean().item():.4f}"
            )

        if step % cfg.save_every == 0:
            ck = os.path.join(cfg.save_dir, f"stage12_rl_step{step}.pt")
            payload = {
                "model": model.state_dict(),
                "model_cfg": model_cfg,
                "meta": meta,
                "step": step,
            }
            torch.save(payload, ck)
            print("[Save]", ck)


if __name__ == "__main__":
    train()
