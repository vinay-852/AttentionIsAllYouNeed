import torch
import torch.nn as nn
from layers.DecoderLayer import DecoderLayer



class Decoder(nn.Module):
    """
    Transformer Decoder consisting of N DecoderLayers.
    """
    def __init__(self, N, d_model, num_heads, d_in, d_hid, dropout=0.1):
        super(Decoder, self).__init__()

        # Stack of decoder layers
        self.layers = nn.ModuleList([DecoderLayer(d_model, num_heads, d_in, d_hid, dropout) for _ in range(N)])

        # Final normalization layer
        self.norm = nn.LayerNorm(d_model)

    def forward(self, x, enc_output, tgt_mask=None, memory_mask=None):
        """
        Forward pass through the decoder.

        Args:
        - x: target sequence
        - enc_output: encoder output
        - tgt_mask: mask for the target sequence (self-attention)
        - memory_mask: mask for encoder-decoder attention

        Returns:
        - x: decoder output
        - attn_weights_all_layers: list of attention weights from all decoder layers
        """

        # Pass the target sequence through each decoder layer
        for layer in self.layers:
            x = layer(x, enc_output, tgt_mask, memory_mask)

        # Final layer normalization
        x = self.norm(x)

        return x