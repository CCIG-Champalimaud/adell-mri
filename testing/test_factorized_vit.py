import sys
import os
sys.path.append(os.path.join(os.path.dirname(os.path.abspath(__file__)),'..'))
import pytest

import time
import torch
import numpy as np
from lib.modules.layers.vit import FactorizedViT
from lib.modules.layers.adn_fn import get_adn_fn

image_size = [32,32,32]
patch_size = [4,4,4]
window_size = [16,16,16]
n_channels = 2

attention_dim = 64
hidden_dim = 64
batch_size = 4
n_heads = 4
adn_fn = get_adn_fn(1,"identity","gelu",0.1)

def test_factorized_transformer():
    vit = FactorizedViT(
        image_size=image_size,patch_size=patch_size,n_channels=n_channels,
        number_of_blocks=4,attention_dim=attention_dim,hidden_dim=hidden_dim,
        n_heads=n_heads,dropout_rate=0.1,
        mlp_structure=[64,64],adn_fn=adn_fn)
    im_size = [batch_size] + [n_channels] + image_size
    im = torch.rand(im_size)
    out = vit(im)
    token_size = vit.input_dim_primary
    assert list(out.shape) == [batch_size,image_size[-1],token_size]

def test_factorized_transformer_token():
    vit = FactorizedViT(
        image_size=image_size,patch_size=patch_size,n_channels=n_channels,
        number_of_blocks=4,attention_dim=attention_dim,hidden_dim=hidden_dim,
        n_heads=n_heads,dropout_rate=0.1,
        mlp_structure=[64,64],adn_fn=adn_fn,
        use_class_token=True)
    im_size = [batch_size] + [n_channels] + image_size
    im = torch.rand(im_size)
    out = vit(im)
    token_size = vit.input_dim_primary
    assert list(out.shape) == [batch_size,image_size[-1]+1,token_size]
