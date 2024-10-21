import torch
import torch.nn as nn
from components.MultiHead import MultiHeadAttention
from components.positionWiseFeedForward import PositionwiseFeedForward


class EncoderLayer(nn.Module):
    """
    Encoder Layer
    Args:
        d_model: dimension of hidden layer
        d_inner: dimension of inner layer
        num_heads: number of heads
        Head: Attention head class (optional to define externally)
        d_in: input dimension
        d_hid: dimension of inner layer
        dropout: dropout rate
    """
    def __init__(self, d_model, num_heads, d_in, d_hid, dropout=0.1):
        super(EncoderLayer, self).__init__()
        self.mhsa = MultiHeadAttention(d_model, num_heads)
        self.pwff = PositionwiseFeedForward(d_in, d_hid, dropout)
        self.layer_norm1 = nn.LayerNorm(d_model, eps=1e-6)
        self.layer_norm2 = nn.LayerNorm(d_model, eps=1e-6)

    def forward(self, x, mask=None):
        """
        Forward pass through the encoder layer.

        Args:
        - x: input sequence
        - mask: mask for the input sequence (self-attention)

        Returns:
        - x: encoder output
        """
        # Multi-Head Self-Attention + residual connection + normalization
        residual = x
        x, _ = self.mhsa(x, x, x, mask)  # _ is the attention weights, which we can discard here
        x = residual + x  # Add the residual connection
        x = self.layer_norm1(x)  # Apply layer normalization

        # Position-wise Feed-Forward + residual connection + normalization
        residual = x
        x = self.pwff(x)
        x = residual + x  # Add the residual connection
        x = self.layer_norm2(x)  # Apply layer normalization

        return x
