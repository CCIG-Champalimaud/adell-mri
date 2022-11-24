import sys
import os
sys.path.append(os.path.join(os.path.dirname(os.path.abspath(__file__)),'..'))

import torch
from lib.modules.layers.vit import TransformerBlock,TransformerBlockStack
from lib.modules.layers.adn_fn import get_adn_fn

input_dim_primary = 32
input_dim_context = 16
attention_dim = 64
hidden_dim = 96
batch_size = 4
token_size = 9
n_heads = 2
adn_fn = get_adn_fn(1,"identity","gelu",0.1)

def test_transformer():
    out = TransformerBlock(
        input_dim_primary,attention_dim,
        hidden_dim,4,[64,64],adn_fn)(
            torch.rand(size=[batch_size,token_size,input_dim_primary]))
    assert list(out.shape) == [batch_size,token_size,input_dim_primary]
    
def test_transformer_stack():
    out,_ = TransformerBlockStack(
        3,
        input_dim_primary,attention_dim,
        hidden_dim,4,[64,64],adn_fn)(
            torch.rand(size=[batch_size,token_size,input_dim_primary]))
    assert list(out.shape) == [batch_size,token_size,input_dim_primary]