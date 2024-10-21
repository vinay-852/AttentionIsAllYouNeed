import torch
import torch.nn as nn
from components.MultiHead import MultiHeadAttention
from components.positionWiseFeedForward import PositionwiseFeedForward


class DecoderLayer(nn.Module):
    """
    Decoder Layer for the Transformer model.

    Args:
    - d_model: dimension of hidden layer
    - num_heads: number of attention heads
    - d_in: input dimension
    - d_hid: dimension of feed-forward network's inner layer
    - dropout: dropout rate
    """
    def __init__(self, d_model, num_heads, d_in, d_hid, dropout=0.1):
        super(DecoderLayer, self).__init__()

        # Self-attention layer for the target sequence
        self.self_attn = MultiHeadAttention(d_model, num_heads)

        # Multi-head attention layer attending to the encoder's output
        self.mhsa = MultiHeadAttention(d_model, num_heads)

        # Position-wise feed-forward network
        self.pwff = PositionwiseFeedForward(d_in, d_hid, dropout)

        # Layer norms
        self.layer_norm1 = nn.LayerNorm(d_model, eps=1e-6)
        self.layer_norm2 = nn.LayerNorm(d_model, eps=1e-6)
        self.layer_norm3 = nn.LayerNorm(d_model, eps=1e-6)

    def forward(self, x, enc_output, mask=None, self_attn_mask=None):
        """
        Forward pass through the decoder layer.

        Args:
        - x: input target sequence
        - enc_output: encoder's output sequence
        - mask: mask for the encoder-decoder attention
        - self_attn_mask: mask for the self-attention mechanism

        Returns:
        - x: decoder output
        """
        # 1. Self-attention over the target sequence
        residual = x
        x, _ = self.self_attn(x, x, x, self_attn_mask)  # Self-attention
        x = residual + x  # Add residual connection
        x = self.layer_norm1(x)  # Layer norm

        # 2. Multi-head attention over the encoder's output
        residual = x
        x, _ = self.mhsa(x, enc_output, enc_output, mask)  # Encoder-decoder attention
        x = residual + x  # Add residual connection
        x = self.layer_norm2(x)  # Layer norm

        # 3. Position-wise feed-forward network
        residual = x
        x = self.pwff(x)  # Feed-forward
        x = residual + x  # Add residual connection
        x = self.layer_norm3(x)  # Layer norm

        return x
