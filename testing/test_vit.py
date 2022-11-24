import sys
import os
sys.path.append(os.path.join(os.path.dirname(os.path.abspath(__file__)),'..'))
import pytest

import torch
from lib.modules.layers.vit import ViT
from lib.modules.layers.adn_fn import get_adn_fn

image_size = [32,32,32]
patch_size = [8,8,8]
n_channels = 1

attention_dim = 64
hidden_dim = 96
batch_size = 4
n_heads = 2
adn_fn = get_adn_fn(1,"identity","gelu",0.1)

@pytest.mark.parametrize("embed_method",["linear","convolution"])
def test_transformer(embed_method):
    vit = ViT(
        image_size,patch_size,n_channels,4,
        attention_dim,hidden_dim,n_heads,dropout_rate=0.1,
        embed_method=embed_method,mlp_structure=[64,64],
        adn_fn=adn_fn)
    im = torch.rand([batch_size] + [n_channels] + image_size)
    out,_ = vit(im)
    token_size = vit.input_dim_primary
    n_patches = vit.embedding.n_patches
    assert list(out.shape) == [batch_size,n_patches,token_size]
