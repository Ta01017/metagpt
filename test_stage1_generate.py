# test_stage1_generate.py
import torch
from config import TrainConfig
from models.metagpt import MetaGPT
from models.transformer_sdpa import TransformerConfig


def load_ckpt(path):
    ckpt = torch.load(path, map_location="cpu")
    return ckpt


def test_generate():
    ckpt = load_ckpt("./checkpoints_stage1/stage1_step1000.pt")
    cfg = ckpt["config"]

    model_cfg = TransformerConfig(
        vocab_size=cfg.vocab_size,
        d_model=cfg.d_model,
        d_ff=cfg.d_ff,
        n_heads=cfg.n_heads,
        n_layers=cfg.n_layers,
        max_len=cfg.max_len + 8,
        dropout=cfg.dropout
    )

    model = MetaGPT(
        cfg=model_cfg,
        spec_dim=1,
        prefix_len=0,
        pad_id=cfg.pad_id
    )
    model.load_state_dict(ckpt["model"])

    model.eval()

    bos = cfg.bos_id
    eos = cfg.eos_id

    out = model.generate(
        spectra=None,
        bos_id=bos,
        eos_id=eos,
        max_new_tokens=32,
        greedy=True
    )

    print("Generated:", out)


if __name__ == "__main__":
    test_generate()
