import torch
import torch.nn as nn

class ScalarDotProduct(nn.Module):
  """
  Scaled Dot Product Attention
  Args:
    drp_out: dropout rate
    q: query
    k: key
    v: value
  return:
    output: output of attention
    attention_weights: attention weights
  """
  def __init__(self,drp_out=0.1):
    super().__init__()
    self.dropout = nn.Dropout(drp_out)
  def forward(self,q,k,v,dk,mask=None):
    attention_weights = torch.matmul(q,k.transpose(-2,-1))
    attention_weights = attention_weights/torch.sqrt(torch.tensor(dk))
    if mask is not None:
      attention_weights = attention_weights.masked_fill(mask==0,-1e9)
    attention_weights = self.dropout(torch.softmax(attention_weights,dim=-1))
    output = torch.matmul(attention_weights,v)
    return output,attention_weights
