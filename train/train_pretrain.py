# train_pretrain.py
import os
import argparse
import math

import torch
from torch.utils.data import DataLoader

from models.transformer_sdpa import TransformerConfig
from models.metagpt import MetaGPT
from data.datasets import PretrainStructureDataset
from utils import save_checkpoint, load_checkpoint, save_json, SimpleLogger, set_seed


def build_fake_sequences(num=50000, vocab=500, min_len=8, max_len=40, bos=2, eos=3):
    # 你后面换成真实数据加载即可
    import random
    seqs = []
    for _ in range(num):
        L = random.randint(min_len, max_len)
        x = [bos] + [random.randint(4, vocab - 1) for _ in range(L)] + [eos]
        seqs.append(x)
    return seqs


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--save_dir", default="ckpt_stage1_pretrain")
    ap.add_argument("--resume", default="")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--vocab", type=int, default=500)
    ap.add_argument("--pad_id", type=int, default=0)
    ap.add_argument("--bos_id", type=int, default=2)
    ap.add_argument("--eos_id", type=int, default=3)

    ap.add_argument("--max_len", type=int, default=64)   # includes BOS/EOS; dataset uses max_len then shift
    ap.add_argument("--batch", type=int, default=64)
    ap.add_argument("--epochs", type=int, default=5)
    ap.add_argument("--max_steps", type=int, default=0)  # 0 means no limit
    ap.add_argument("--lr", type=float, default=3e-4)
    ap.add_argument("--grad_clip", type=float, default=1.0)
    ap.add_argument("--amp", action="store_true")

    ap.add_argument("--d_model", type=int, default=512)
    ap.add_argument("--n_layers", type=int, default=8)
    ap.add_argument("--n_heads", type=int, default=8)
    ap.add_argument("--d_ff", type=int, default=2048)
    ap.add_argument("--dropout", type=float, default=0.1)
    args = ap.parse_args()

    set_seed(args.seed)
    log = SimpleLogger()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    os.makedirs(args.save_dir, exist_ok=True)

    # Stage1: no prefix used during forward; but model still has prefix module (we won't call it)
    cfg = TransformerConfig(
        vocab_size=args.vocab,
        d_model=args.d_model,
        n_layers=args.n_layers,
        n_heads=args.n_heads,
        d_ff=args.d_ff,
        dropout=args.dropout,
        max_len=args.max_len + 16,  # room
    )
    model = MetaGPT(cfg=cfg, spec_dim=322, prefix_len=16, pad_id=args.pad_id).to(device)

    # save config
    save_json(os.path.join(args.save_dir, "model_config.json"), model.get_config())

    # data (replace with your real sequences)
    seqs = build_fake_sequences(num=20000, vocab=args.vocab, bos=args.bos_id, eos=args.eos_id)
    ds = PretrainStructureDataset(sequences=seqs, pad_id=args.pad_id, max_len=args.max_len)
    dl = DataLoader(ds, batch_size=args.batch, shuffle=True, num_workers=4, pin_memory=True, drop_last=True)

    optim = torch.optim.AdamW(model.parameters(), lr=args.lr)
    scaler = torch.cuda.amp.GradScaler(enabled=args.amp)

    start_step = 0
    if args.resume:
        start_step, _ = load_checkpoint(args.resume, model, optim, scaler, strict=True)
        log.log(f"Resumed from {args.resume} at step={start_step}")

    step = start_step
    model.train()

    for ep in range(args.epochs):
        for inp, lab in dl:
            inp = inp.to(device, non_blocking=True)  # (B,T-1)
            lab = lab.to(device, non_blocking=True)

            with torch.cuda.amp.autocast(enabled=args.amp):
                # Stage1: spectra=None
                logits, loss, _ = model(input_ids=inp, spectra=None, labels=lab)

            scaler.scale(loss).backward()
            if args.grad_clip > 0:
                scaler.unscale_(optim)
                torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)

            scaler.step(optim)
            scaler.update()
            optim.zero_grad(set_to_none=True)

            step += 1
            if step % 100 == 0:
                ppl = math.exp(float(loss.detach().cpu().clamp(max=20)))
                log.log(f"ep={ep} step={step} loss={loss.item():.4f} ppl={ppl:.2f}")

            if step % 1000 == 0:
                ck = os.path.join(args.save_dir, f"pretrain_step{step}.pt")
                save_checkpoint(ck, model, optim, scaler, step=step, extra={"epoch": ep})
                log.log(f"saved {ck}")

                # quick generation sanity
                gen = model.generate(
                    spectra=None,
                    bos_id=args.bos_id,
                    eos_id=args.eos_id,
                    max_new_tokens=24,
                    temperature=1.0,
                    top_k=20,
                    top_p=0.95,
                    greedy=False
                )
                log.log(f"gen sample ids: {gen[0].tolist()[:32]}")

            if args.max_steps and step >= args.max_steps:
                break
        if args.max_steps and step >= args.max_steps:
            break

    final_ck = os.path.join(args.save_dir, "pretrain_final.pt")
    save_checkpoint(final_ck, model, optim, scaler, step=step, extra={"done": True})
    log.log(f"saved final {final_ck}")


if __name__ == "__main__":
    main()
