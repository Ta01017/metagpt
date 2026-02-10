# train_sft.py
import os
import argparse
import math

import torch
from torch.utils.data import DataLoader

from models.transformer_sdpa import TransformerConfig
from models.metagpt import MetaGPT
from data.datasets import SFTSpectraStructureDataset
from utils import save_checkpoint, load_checkpoint, save_json, SimpleLogger, set_seed


def build_fake_sft(num=10000, spec_dim=322, vocab=500, bos=2, eos=3):
    import random
    import numpy as np
    spectra = []
    seqs = []
    for _ in range(num):
        spectra.append(np.random.rand(spec_dim).astype("float32"))
        L = random.randint(8, 40)
        seqs.append([bos] + [random.randint(4, vocab - 1) for _ in range(L)] + [eos])
    return spectra, seqs


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--save_dir", default="ckpt_stage2_sft")
    ap.add_argument("--resume", default="")
    ap.add_argument("--init_from_pretrain", default="")  # stage1 ckpt
    ap.add_argument("--seed", type=int, default=42)

    ap.add_argument("--vocab", type=int, default=500)
    ap.add_argument("--pad_id", type=int, default=0)
    ap.add_argument("--bos_id", type=int, default=2)
    ap.add_argument("--eos_id", type=int, default=3)

    ap.add_argument("--spec_dim", type=int, default=322)
    ap.add_argument("--prefix_len", type=int, default=16)
    ap.add_argument("--max_len", type=int, default=64)

    ap.add_argument("--batch", type=int, default=64)
    ap.add_argument("--epochs", type=int, default=5)
    ap.add_argument("--max_steps", type=int, default=0)
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

    cfg = TransformerConfig(
        vocab_size=args.vocab,
        d_model=args.d_model,
        n_layers=args.n_layers,
        n_heads=args.n_heads,
        d_ff=args.d_ff,
        dropout=args.dropout,
        max_len=args.max_len + args.prefix_len + 8,
    )
    model = MetaGPT(cfg=cfg, spec_dim=args.spec_dim, prefix_len=args.prefix_len, pad_id=args.pad_id).to(device)
    save_json(os.path.join(args.save_dir, "model_config.json"), model.get_config())

    # load from stage1
    if args.init_from_pretrain:
        step0, extra0 = load_checkpoint(args.init_from_pretrain, model, optimizer=None, scaler=None, strict=False)
        log.log(f"Loaded pretrain weights (strict=False) from {args.init_from_pretrain} (step={step0}, extra={extra0})")

    # data (replace with your real spectra+seqs)
    spectra, seqs = build_fake_sft(num=10000, spec_dim=args.spec_dim, vocab=args.vocab, bos=args.bos_id, eos=args.eos_id)
    ds = SFTSpectraStructureDataset(spectra_list=spectra, sequences=seqs, pad_id=args.pad_id, max_len=args.max_len)
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
        for spectra_b, inp, lab, keep in dl:
            spectra_b = spectra_b.to(device, non_blocking=True)
            inp = inp.to(device, non_blocking=True)
            lab = lab.to(device, non_blocking=True)
            keep = keep.to(device, non_blocking=True)

            with torch.cuda.amp.autocast(enabled=args.amp):
                logits, loss, _ = model(input_ids=inp, spectra=spectra_b, labels=lab, attention_keep=keep)

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
                ck = os.path.join(args.save_dir, f"sft_step{step}.pt")
                save_checkpoint(ck, model, optim, scaler, step=step, extra={"epoch": ep})
                log.log(f"saved {ck}")

                # quick conditional generation sanity
                gen = model.generate(
                    spectra=spectra_b[:1],
                    bos_id=args.bos_id,
                    eos_id=args.eos_id,
                    max_new_tokens=32,
                    temperature=1.0,
                    top_k=30,
                    top_p=0.95,
                    greedy=False
                )
                log.log(f"gen(ids)={gen[0].tolist()[:48]}")

            if args.max_steps and step >= args.max_steps:
                break
        if args.max_steps and step >= args.max_steps:
            break

    final_ck = os.path.join(args.save_dir, "sft_final.pt")
    save_checkpoint(final_ck, model, optim, scaler, step=step, extra={"done": True})
    log.log(f"saved final {final_ck}")


if __name__ == "__main__":
    main()
