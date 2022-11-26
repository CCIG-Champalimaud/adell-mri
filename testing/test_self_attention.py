import sys
import os
sys.path.append(os.path.join(os.path.dirname(os.path.abspath(__file__)),'..'))

import torch
from lib.modules.layers.linear_blocks import SelfAttention,MultiHeadAttention

input_dim_primary = 32
input_dim_context = 16
attention_dim = 64
hidden_dim = 96
output_dim = 128
batch_size = 4
token_size = 9
n_heads = 2

def test_self_attention():
    out = SelfAttention(input_dim_primary,attention_dim,hidden_dim)(
        torch.rand(size=[batch_size,token_size,input_dim_primary]))
    assert list(out.shape) == [batch_size,token_size,hidden_dim]

def test_multi_head_attention():
    out = MultiHeadAttention(input_dim_primary,attention_dim,
                             hidden_dim,output_dim,n_heads=n_heads)(
        torch.rand(size=[batch_size,token_size,input_dim_primary]))
    assert list(out.shape) == [batch_size,token_size,output_dim]
    