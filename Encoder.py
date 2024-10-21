import torch
import torch.nn as nn
from layers.EncoderLayer import EncoderLayer


class Encoder(nn.Module):
    """
    Transformer Encoder consisting of N layers of EncoderLayer.
    """
    def __init__(self, N, d_model, num_heads, d_in, d_hid, dropout=0.1):
        super(Encoder, self).__init__()
        self.layers = nn.ModuleList([EncoderLayer(d_model, num_heads, d_in, d_hid, dropout) for _ in range(N)])
        self.norm = nn.LayerNorm(d_model)

    def forward(self, src, mask=None):
        """
        Forward pass through the encoder.

        Args:
        - src: input sequence
        - mask: mask for the input sequence (self-attention)

        Returns:
        - src: encoder output
        - attn_weights_all_layers: list of attention weights from all encoder layers
        """

        # Pass input through each encoder layer
        for layer in self.layers:
            src = layer(src, mask)

        # Final layer normalization
        src = self.norm(src)

        return src
