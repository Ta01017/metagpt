# train_rl_scst.py
import os
import argparse
from typing import List

import numpy as np
import torch
import torch.nn.functional as F

from models.transformer_sdpa import TransformerConfig
from models.metagpt import MetaGPT
from models.spectrum_encoder import SpectrumEncoder
from structure_lang.tokenizer import StructureTokenizer
from structure_lang.parser import StructureParser
from structure_lang.validator import StructureValidator
from data.gen_stage2_fake_dataset import fake_spectrum_from_structure
from utils import save_checkpoint, load_checkpoint, save_json, SimpleLogger, set_seed


def _cfg_from_ckpt(ckpt):
    if "model_cfg" in ckpt:
        cfg = ckpt["model_cfg"]
        if isinstance(cfg, dict):
            return TransformerConfig(**cfg)
        return cfg
    raise KeyError("Checkpoint missing model_cfg")


def tokens_to_struct(tokens: List[str]) -> List[str]:
    keep = []
    for t in tokens:
        if t.startswith("PX_") or t.startswith("PY_") or t.startswith("SUB_") or t.startswith("L1_"):
            keep.append(t)
    return keep


def reward_fn(pred_spec: torch.Tensor, tgt_spec: torch.Tensor) -> torch.Tensor:
    # reward = -MSE (paper)
    return -torch.mean((pred_spec - tgt_spec) ** 2)


@torch.no_grad()
def rollout_and_score(
    model: MetaGPT,
    spectra: torch.Tensor,
    bos_id: int,
    eos_id: int,
    max_new: int,
    temperature: float,
    top_k: int,
    top_p: float,
    greedy: bool,
    tk: StructureTokenizer,
    parser: StructureParser,
    validator: StructureValidator,
    invalid_penalty: float = -1.0,
):
    """
    Generate tokens and compute reward via surrogate simulator.
    Returns:
      tokens: (B, L)
      rewards: (B,)
    """
    B = spectra.size(0)
    tokens = model.generate(
        spectra=spectra,
        bos_id=bos_id,
        eos_id=eos_id,
        max_new_tokens=max_new,
        temperature=temperature,
        top_k=top_k,
        top_p=top_p,
        greedy=greedy,
    )  # (B, L)

    rewards = []
    for b in range(B):
        ids = tokens[b].tolist()
        toks = tk.decode([i for i in ids if i in tk.inv_vocab])
        struct_toks = tokens_to_struct(toks)
        ok, _ = validator.validate(parser.parse(["[BOS]"] + struct_toks + ["[EOS]"]))
        if not ok:
            r = torch.tensor(invalid_penalty, device=spectra.device)
        else:
            pred = fake_spectrum_from_structure(struct_toks, spec_dim=spectra.size(1))
            pred_spec = torch.tensor(pred, device=spectra.device)
            r = reward_fn(pred_spec, spectra[b])
        rewards.append(r)

    rewards = torch.stack(rewards, dim=0)
    return tokens, rewards


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--save_dir", default="ckpt_stage3_rl")
    ap.add_argument("--resume", default="")
    ap.add_argument("--init_from_sft", default="")
    ap.add_argument("--seed", type=int, default=42)

    ap.add_argument("--spec_file", default="./dataset_stage2/spec_train.npy")
    ap.add_argument("--spec_dim", type=int, default=128)
    ap.add_argument("--prefix_len", type=int, default=16)
    ap.add_argument("--max_new", type=int, default=32)

    ap.add_argument("--batch", type=int, default=8)
    ap.add_argument("--steps", type=int, default=2000)
    ap.add_argument("--lr", type=float, default=1e-5)
    ap.add_argument("--grad_clip", type=float, default=1.0)

    ap.add_argument("--temperature", type=float, default=1.0)
    ap.add_argument("--top_k", type=int, default=30)
    ap.add_argument("--top_p", type=float, default=0.95)

    ap.add_argument("--entropy_beta", type=float, default=0.0)
    ap.add_argument("--invalid_penalty", type=float, default=-1.0)
    args = ap.parse_args()

    set_seed(args.seed)
    log = SimpleLogger()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    os.makedirs(args.save_dir, exist_ok=True)

    tk = StructureTokenizer()
    tk.id_to_token = [tk.inv_vocab[i] for i in range(tk.vocab_size)]
    parser = StructureParser()
    validator = StructureValidator(min_feature_nm=20, margin_nm=30)

    if args.init_from_sft:
        ckpt = torch.load(args.init_from_sft, map_location="cpu")
        cfg = _cfg_from_ckpt(ckpt)
        meta = ckpt.get("meta", {})
        spec_dim = int(meta.get("spec_dim", args.spec_dim))
        prefix_len = int(meta.get("prefix_len", args.prefix_len))
        pad_id = int(meta.get("pad_id", tk.pad_id))
        bos_id = int(meta.get("bos_id", tk.bos_id))
        eos_id = int(meta.get("eos_id", tk.eos_id))
    else:
        cfg = TransformerConfig(
            vocab_size=tk.vocab_size,
            d_model=256,
            n_layers=6,
            n_heads=8,
            d_ff=1024,
            dropout=0.1,
            max_len=args.max_new + args.prefix_len + 8,
        )
        spec_dim = args.spec_dim
        prefix_len = args.prefix_len
        pad_id = tk.pad_id
        bos_id = tk.bos_id
        eos_id = tk.eos_id

    model = MetaGPT(cfg=cfg, spec_dim=spec_dim, prefix_len=prefix_len, pad_id=pad_id).to(device)
    model.encoder = SpectrumEncoder(spec_dim=spec_dim, d_model=cfg.d_model, prefix_len=prefix_len).to(device)
    save_json(os.path.join(args.save_dir, "model_config.json"), model.get_config())

    if args.init_from_sft:
        load_checkpoint(args.init_from_sft, model, optimizer=None, scaler=None, strict=False)
        log.log(f"Loaded SFT weights from {args.init_from_sft}")

    optim = torch.optim.AdamW(model.parameters(), lr=args.lr)

    start_step = 0
    if args.resume:
        start_step, _ = load_checkpoint(args.resume, model, optim, scaler=None, strict=True)
        log.log(f"Resumed from {args.resume} at step={start_step}")

    model.train()

    spec_arr = np.load(args.spec_file).astype("float32")
    log.log(f"Loaded spectra: {spec_arr.shape}")

    for step in range(start_step, args.steps):
        idx = np.random.randint(0, spec_arr.shape[0], size=args.batch)
        spectra = torch.tensor(spec_arr[idx], device=device)

        sample_tokens, r_sample = rollout_and_score(
            model, spectra, bos_id, eos_id, args.max_new,
            temperature=args.temperature, top_k=args.top_k, top_p=args.top_p, greedy=False,
            tk=tk, parser=parser, validator=validator, invalid_penalty=args.invalid_penalty
        )

        base_tokens, r_base = rollout_and_score(
            model, spectra, bos_id, eos_id, args.max_new,
            temperature=1.0, top_k=0, top_p=1.0, greedy=True,
            tk=tk, parser=parser, validator=validator, invalid_penalty=args.invalid_penalty
        )

        adv = (r_sample - r_base).detach()

        inp = sample_tokens[:, :-1].contiguous()
        lab = sample_tokens[:, 1:].contiguous()

        logits, _, _ = model(input_ids=inp, spectra=spectra, labels=None)
        logp = F.log_softmax(logits, dim=-1)
        token_logp = logp.gather(-1, lab.unsqueeze(-1)).squeeze(-1)
        seq_logp = token_logp.sum(dim=-1)

        loss_pg = -(adv * seq_logp).mean()

        if args.entropy_beta > 0:
            probs = torch.softmax(logits, dim=-1)
            entropy = -(probs * torch.log(probs.clamp(min=1e-12))).sum(dim=-1).mean()
            loss = loss_pg - args.entropy_beta * entropy
        else:
            loss = loss_pg

        optim.zero_grad(set_to_none=True)
        loss.backward()
        if args.grad_clip > 0:
            torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)
        optim.step()

        if (step + 1) % 50 == 0:
            log.log(f"step={step+1} loss={loss.item():.4f} r_sample={r_sample.mean().item():.4f} r_base={r_base.mean().item():.4f} adv={adv.mean().item():.4f}")

        if (step + 1) % 500 == 0:
            ck = os.path.join(args.save_dir, f"rl_step{step+1}.pt")
            save_checkpoint(ck, model, optim, scaler=None, step=step+1, extra={})
            log.log(f"saved {ck}")


if __name__ == "__main__":
    main()
