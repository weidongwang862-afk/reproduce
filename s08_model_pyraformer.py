import torch
import torch.nn as nn


class PyraformerRegressor(nn.Module):
    def __init__(
        self,
        input_dim=1,
        d_model=32,
        dim_ff=64,
        nhead=2,
        num_blocks=2,
        num_levels=3,
        output_dim=1,
    ):
        super().__init__()
        self.input_proj = nn.Linear(input_dim, d_model)

        # TODO:
        # 1. 加 PositionalEncoding
        # 2. 加 CSCM
        # 3. 加 PAM
        # 4. 加 readout
        self.head = nn.Linear(d_model, output_dim)

    def forward(self, x):
        x = self.input_proj(x)
        feat = x[:, -1, :]
        y = self.head(feat)
        return y