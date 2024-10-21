import torch
import torch.nn as nn
from scalarDotProduct import ScalarDotProduct
class Head(nn.Module):
  """
  Performing a single attention function with d_k-dimensional keys, d_v-dimensional values and d_k-dimensional queries
  """
  def __init__(self,d_model,d_k,d_v):
    super().__init__()
    self.lc1 = nn.Linear(d_model,d_k)
    self.lc2 = nn.Linear(d_model,d_k)
    self.lc3 = nn.Linear(d_model,d_v)
    self.sdpa = ScalarDotProduct()
  def forward(self,query,key,value,mask=None):
    q = self.lc1(query)
    k = self.lc2(key)
    v = self.lc3(value)
    attn_output,attn_weights = self.sdpa(q,k,v,dk=q.size(-1),mask=mask)
    return attn_output,attn_weights