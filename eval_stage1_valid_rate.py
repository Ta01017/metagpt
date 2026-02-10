# eval_stage1_valid_rate.py
import argparse
import sys
from pathlib import Path

import torch

from models.metagpt import MetaGPT
from models.transformer_sdpa import TransformerConfig

from structure_lang.tokenizer import StructureTokenizer
from structure_lang.parser import StructureParser
from structure_lang.validator import StructureValidator

# Ensure project root is on sys.path so absolute imports work from any cwd.
_ROOT = Path(__file__).resolve().parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))


def _cfg_get(cfg, key):
    if isinstance(cfg, dict):
        return cfg[key]
    return getattr(cfg, key)


def _load_cfg_from_ckpt(ckpt):
    if "model_cfg" in ckpt:
        model_cfg = TransformerConfig(**ckpt["model_cfg"])
        special = ckpt.get("special_ids", {})
        pad_id = special.get("pad_id", 0)
        bos_id = special.get("bos_id", 1)
        eos_id = special.get("eos_id", 2)
        return model_cfg, pad_id, bos_id, eos_id

    if "config" in ckpt:
        cfg = ckpt["config"]
        model_cfg = TransformerConfig(
            vocab_size=_cfg_get(cfg, "vocab_size"),
            d_model=_cfg_get(cfg, "d_model"),
            d_ff=_cfg_get(cfg, "d_ff"),
            n_heads=_cfg_get(cfg, "n_heads"),
            n_layers=_cfg_get(cfg, "n_layers"),
            max_len=_cfg_get(cfg, "max_len") + 8,
            dropout=_cfg_get(cfg, "dropout"),
        )
        pad_id = _cfg_get(cfg, "pad_id")
        bos_id = _cfg_get(cfg, "bos_id")
        eos_id = _cfg_get(cfg, "eos_id")
        return model_cfg, pad_id, bos_id, eos_id

    raise KeyError("Checkpoint missing 'model_cfg' or 'config'.")


@torch.no_grad()
def generate_one(model, tk, bos_id, eos_id, max_len, device):
    out = torch.tensor([[bos_id]], dtype=torch.long, device=device)

    for _ in range(max_len):
        logits, _, _ = model(
            input_ids=out,
            spectra=None,
            labels=None
        )
        nxt = torch.softmax(logits[:, -1, :], dim=-1)
        nxt_id = torch.multinomial(nxt, num_samples=1)
        out = torch.cat([out, nxt_id], dim=1)

        if nxt_id.item() == eos_id:
            break

    seq = out[0].tolist()
    # remove BOS/EOS for decode
    seq = [x for x in seq if x not in (bos_id, eos_id)]
    return seq


def eval_stage1(checkpoint, num_gen=200):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print("DEVICE =", device)

    # -------- load checkpoint --------
    ckpt = torch.load(checkpoint, map_location=device)

    model_cfg, pad_id, bos_id, eos_id = _load_cfg_from_ckpt(ckpt)

    model = MetaGPT(
        cfg=model_cfg,
        spec_dim=1,
        prefix_len=0,
        pad_id=pad_id,
    ).to(device)

    model.load_state_dict(ckpt["model"])
    model.eval()

    # -------- build tokenizer + parser + validator --------
    tk = StructureTokenizer()
    parser = StructureParser()
    val = StructureValidator(min_feature_nm=20, margin_nm=30)

    # -------- generate + decode + validate --------
    valid_cnt = 0
    samples = []

    for i in range(num_gen):
        ids = generate_one(
            model=model,
            tk=tk,
            bos_id=bos_id,
            eos_id=eos_id,
            max_len=32,
            device=device
        )
        toks = tk.decode(ids)
        struct = parser.parse(["[BOS]"] + toks + ["[EOS]"])
        ok, reason = val.validate(struct)
        if ok:
            valid_cnt += 1

        if i < 10:  # 保存前 10 条示例
            samples.append((ids, toks, ok, reason))

    print(f"\n=== Stage1 Validity Evaluation ===")
    print(f"Generated: {num_gen}")
    print(f"Valid structures: {valid_cnt} ({100*valid_cnt/num_gen:.2f}%)\n")

    print("=== Example Generated Structures ===")
    for i, (ids, toks, ok, reason) in enumerate(samples):
        print(f"\n[{i}] {'OK' if ok else 'BAD'} reason={reason}")
        print("IDs: ", ids)
        print("TOKS:", toks)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--ckpt", type=str, required=True)
    parser.add_argument("--n", type=int, default=200)
    args = parser.parse_args()

    eval_stage1(args.ckpt, num_gen=args.n)
