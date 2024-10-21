import torch
import torch.nn as nn
import numpy as np

class PositionalEncoding(nn.Module):
    """
    Positional Encoding

    Args:
        d_hidden: dimension of hidden layer
        n_position: length of positions
        x: input embedding
    return:
        x: output embedding
    """
    def __init__(self, d_hidden, n_position=200):
        super(PositionalEncoding, self).__init__()
        self.register_buffer('pos_table', self._get_sinusoid_encoding_table(n_position, d_hidden))

    def _get_sinusoid_encoding_table(self, n_position, d_hidden):
        def get_position_angle_vec(position):
            return [position / np.power(10000, 2 * (hid_j // 2) / d_hidden) for hid_j in range(d_hidden)]

        sinusoid_table = np.array([get_position_angle_vec(pos_i) for pos_i in range(n_position)])
        sinusoid_table[:, 0::2] = np.sin(sinusoid_table[:, 0::2])  # Apply sin to even indices
        sinusoid_table[:, 1::2] = np.cos(sinusoid_table[:, 1::2])  # Apply cos to odd indices
        return torch.FloatTensor(sinusoid_table).unsqueeze(0)

    def forward(self, x):
        return x + self.pos_table[:, :x.size(1)].clone().detach()
