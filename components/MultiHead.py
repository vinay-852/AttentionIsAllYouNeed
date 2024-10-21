import torch
import torch.nn as nn
from Head import Head
class MultiHeadAttention(nn.Module):
    """
    Multi-Head Attention module that projects the queries, keys, and values h times with different,
    learned linear projections to d_k, d_k, and d_v dimensions.
    """
    def __init__(self, d_model, num_heads):
        super().__init__()
        self.num_heads = num_heads
        self.d_k = d_model // num_heads
        self.d_v = d_model // num_heads

        # Define multiple attention heads
        self.heads = nn.ModuleList([Head(d_model, self.d_k, self.d_v) for _ in range(num_heads)])

        # Linear layer to project the concatenated output of attention heads
        self.linear = nn.Linear(num_heads * self.d_v, d_model)

    def forward(self, query, key, value, mask=None):
      """
      Forward pass through the multi-head attention module.

      Args:
      - query: query tensor
      - key: key tensor
      - value: value tensor
      - mask: mask tensor (optional)

      Returns:
      - output: output tensor
      """
      head_outputs = []
      attn_weights_list = []

        # Iterate through each head and compute attention
      for head in self.heads:
          head_output, attn_weights = head(query, key, value, mask)
          head_outputs.append(head_output)
          attn_weights_list.append(attn_weights)

        # Concatenate the outputs of all heads along the last dimension
      concat = torch.cat(head_outputs, dim=-1)

        # Project concatenated outputs back to original d_model dimension
      output = self.linear(concat)
      attn_weights_stack = torch.stack(attn_weights_list, dim=1)
      return output, attn_weights_stack
