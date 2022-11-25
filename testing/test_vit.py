import sys
import os
sys.path.append(os.path.join(os.path.dirname(os.path.abspath(__file__)),'..'))
import pytest

import torch
import numpy as np
from lib.modules.layers.vit import ViT,downsample_ein_op_dict
from lib.modules.layers.adn_fn import get_adn_fn

image_size = [32,32,32]
patch_size = [4,4,4]
n_channels = 2

attention_dim = 64
hidden_dim = 96
batch_size = 4
n_heads = 2
adn_fn = get_adn_fn(1,"identity","gelu",0.1)

@pytest.mark.parametrize("embed_method,scale",[
    ("linear",1),
    ("convolution",1),
    ("linear",2),
    ("convolution",2),
    ("linear",4),
    ("convolution",4)])
def test_transformer(embed_method,scale):
    vit = ViT(
        image_size,patch_size,n_channels,4,
        attention_dim,hidden_dim,n_heads,dropout_rate=0.1,
        embed_method=embed_method,mlp_structure=[64,64],
        adn_fn=adn_fn)
    im_size = [batch_size] + [n_channels] + image_size
    output_image_size = [batch_size,n_channels*scale**len(image_size)] + [x//scale for x in image_size]
    im = torch.rand(im_size)
    out,_ = vit(im)
    token_size = vit.input_dim_primary
    n_patches = vit.embedding.n_patches
    assert list(out.shape) == [batch_size,n_patches,token_size]
    assert list(vit.embedding.rearrange_rescale(out,sscale).shape) == output_image_size
    assert list(vit.embedding.rearrange_inverse(out).shape) == im_size
