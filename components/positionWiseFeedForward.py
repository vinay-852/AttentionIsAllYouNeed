import torch
import torch.nn as nn


class PositionwiseFeedForward(nn.Module):
    """
    Implements the position-wise feed-forward network in the Transformer architecture.
    """
    def __init__(self, d_in, d_hid, dropout=0.1):
        super().__init__()
        # Ensure d_in and d_hid are integers to avoid the TypeError
        self.w_1 = nn.Linear(int(d_in), int(d_hid)) # position-wise
        self.w_2 = nn.Linear(int(d_hid), int(d_in)) # position-wise
        self.layer_norm = nn.LayerNorm(int(d_in), eps=1e-6)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        """
        Applies the feed-forward network to the input.
        """
        residual = x
        x = self.w_2(torch.relu(self.w_1(x)))
        x = self.dropout(x)
        x += residual
        x = self.layer_norm(x)
        return x