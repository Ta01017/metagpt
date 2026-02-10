import torch
from models.transformer_sdpa import TransformerConfig
from models.metagpt import MetaGPT

device = "cuda" if torch.cuda.is_available() else "cpu"

cfg = TransformerConfig(vocab_size=500, d_model=256, n_layers=4, n_heads=8, d_ff=1024, dropout=0.1, max_len=128)
model = MetaGPT(cfg=cfg, spec_dim=322, prefix_len=16, pad_id=0).to(device)

B, T = 4, 32
inp = torch.randint(0, 500, (B, T), device=device)
lab = torch.randint(0, 500, (B, T), device=device)
spec = torch.randn(B, 322, device=device)

# Stage1（无prefix）
logits1, loss1, aux1 = model(inp, spectra=None, labels=lab)
print("stage1 logits:", logits1.shape, "loss:", float(loss1), aux1)

# Stage2（有prefix）
logits2, loss2, aux2 = model(inp, spectra=spec, labels=lab)
print("stage2 logits:", logits2.shape, "loss:", float(loss2), aux2)

assert logits1.shape == (B, T, 500)
assert logits2.shape == (B, T, 500)
assert torch.isfinite(loss1) and torch.isfinite(loss2)
print("✅ forward sanity ok")
