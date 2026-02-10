# models/spectrum_encoder.py
import torch
import torch.nn as nn


class SpectrumEncoder(nn.Module):
    """
    从光谱 S(λ) -> prefix_len × d_model
    可替代 MetaGPT 的 prefix-generator
    """

    def __init__(self, spec_dim, d_model, prefix_len):
        super().__init__()
        self.prefix_len = prefix_len
        self.d_model = d_model

        self.net = nn.Sequential(
            nn.Linear(spec_dim, 512),
            nn.ReLU(),
            nn.Linear(512, 512),
            nn.ReLU(),
            nn.Linear(512, d_model * prefix_len)
        )

    def forward(self, spec):
        """
        spec: (B, spec_dim)
        return: (B, prefix_len, d_model)
        """
        B = spec.shape[0]
        x = self.net(spec)               # (B, prefix_len * d_model)
        x = x.view(B, self.prefix_len, self.d_model)
        return x
