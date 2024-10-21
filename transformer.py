import torch
import torch.nn as nn
from Encoder import Encoder
from Decoder import Decoder
class Transformer(nn.Module):
    """
    Transformer model consisting of an Encoder, Decoder, and final linear projection layer.
    """
    def __init__(self, N, d_model, num_heads, d_in, d_hid, output_size, dropout=0.1):
        super(Transformer, self).__init__()

        # Encoder (source sequence processing)
        self.encoder = Encoder(N, d_model, num_heads, d_in, d_hid, dropout)

        # Decoder (target sequence generation)
        self.decoder = Decoder(N, d_model, num_heads, d_in, d_hid, dropout)

        # Final linear layer to map the decoder output to the target vocabulary size
        self.linear = nn.Linear(d_model, output_size)

        self.softmax = nn.Softmax(dim=-1)

    def forward(self, src_seq, tgt_seq, src_mask=None, tgt_mask=None, memory_mask=None):
        """
        Forward pass through the Transformer model.

        Args:
        - src_seq: source sequence (input)
        - tgt_seq: target sequence (output)
        - src_mask: mask for the source sequence (optional)
        - tgt_mask: mask for the target sequence (optional)
        - memory_mask: mask for encoder-decoder attention (optional)

        Returns:
        - output: the projected output from the decoder to target vocabulary size
        """
        # Encoder step: process the source sequence
        enc_output = self.encoder(src_seq, src_mask)

        # Decoder step: process the target sequence with the encoder output
        dec_output = self.decoder(tgt_seq, enc_output, tgt_mask, memory_mask)

        # Final linear projection to output size (e.g., vocabulary size in translation tasks)
        output = self.linear(dec_output)
        output = self.softmax(output)

        return output