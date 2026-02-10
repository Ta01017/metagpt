# generate.py
import argparse
import torch

from models.transformer_sdpa import TransformerConfig
from models.metagpt import MetaGPT
from utils import load_checkpoint


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--vocab", type=int, default=500)
    ap.add_argument("--pad_id", type=int, default=0)
    ap.add_argument("--bos_id", type=int, default=2)
    ap.add_argument("--eos_id", type=int, default=3)

    ap.add_argument("--spec_dim", type=int, default=322)
    ap.add_argument("--prefix_len", type=int, default=16)
    ap.add_argument("--max_new", type=int, default=32)

    ap.add_argument("--temperature", type=float, default=1.0)
    ap.add_argument("--top_k", type=int, default=30)
    ap.add_argument("--top_p", type=float, default=0.95)
    ap.add_argument("--greedy", action="store_true")

    ap.add_argument("--d_model", type=int, default=512)
    ap.add_argument("--n_layers", type=int, default=8)
    ap.add_argument("--n_heads", type=int, default=8)
    ap.add_argument("--d_ff", type=int, default=2048)
    ap.add_argument("--dropout", type=float, default=0.1)
    args = ap.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    cfg = TransformerConfig(
        vocab_size=args.vocab,
        d_model=args.d_model,
        n_layers=args.n_layers,
        n_heads=args.n_heads,
        d_ff=args.d_ff,
        dropout=args.dropout,
        max_len=args.max_new + args.prefix_len + 8,
    )
    model = MetaGPT(cfg=cfg, spec_dim=args.spec_dim, prefix_len=args.prefix_len, pad_id=args.pad_id).to(device)
    load_checkpoint(args.ckpt, model, optimizer=None, scaler=None, strict=False)
    model.eval()

    # TODO: 替换成你的真实输入光谱
    spectra = torch.rand(1, args.spec_dim, device=device)

    out = model.generate(
        spectra=spectra,
        bos_id=args.bos_id,
        eos_id=args.eos_id,
        max_new_tokens=args.max_new,
        temperature=args.temperature,
        top_k=args.top_k,
        top_p=args.top_p,
        greedy=args.greedy
    )
    print("Generated token ids:", out[0].tolist())


if __name__ == "__main__":
    main()
