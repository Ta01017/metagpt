# test_generation.py
import torch
from models.transformer_sdpa import MetaGPT
from train.train_pretrain import MetaGPT_Pretrain

def generate(model, vocab, T=20):
    cur = torch.randint(0, vocab, (1,1)).cuda()

    for _ in range(T):
        logits = model(cur)
        next_token = torch.argmax(logits[:, -1, :], dim=-1).unsqueeze(0)
        cur = torch.cat([cur, next_token.unsqueeze(0)], dim=1)
    return cur[0].tolist()




model = MetaGPT_Pretrain(tgt_vocab=500).cuda()
model.load_state_dict(torch.load("pretrain.pt"))

print(generate(model, vocab=500))
