import torch
import torch.nn as nn


class CNNLSTMRegressor(nn.Module):
    def __init__(
        self,
        input_dim=1,
        conv_out_channels=8,
        conv_kernel_size=8,
        pool_size=2,
        hidden_dim=64,
        num_layers=2,
        output_dim=1,
    ):
        super().__init__()
        self.conv = nn.Conv1d(
            in_channels=input_dim,
            out_channels=conv_out_channels,
            kernel_size=conv_kernel_size,
        )
        self.relu = nn.ReLU()
        self.pool = nn.MaxPool1d(kernel_size=pool_size)

        self.lstm = nn.LSTM(
            input_size=conv_out_channels,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            batch_first=True,
        )
        self.head = nn.Linear(hidden_dim, output_dim)

    def forward(self, x):
        # x: (B, L, 1)
        x = x.transpose(1, 2)         # (B, 1, L)
        x = self.conv(x)
        x = self.relu(x)
        x = self.pool(x)
        x = x.transpose(1, 2)         # (B, L', C)

        out, _ = self.lstm(x)
        feat = out[:, -1, :]
        y = self.head(feat)
        return y