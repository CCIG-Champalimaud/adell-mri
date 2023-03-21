import sys
import os
sys.path.append(os.path.join(os.path.dirname(os.path.abspath(__file__)),'..'))
import pytest

import torch
import numpy as np
from lib.modules.layers.vit import ViT
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

@pytest.mark.parametrize("embed_method,scale",[
    ("linear",1),
    ("convolutional",1),
    ("linear",2),
    ("convolutional",2),
    ("linear",4),
    ("convolutional",4)])
def test_transformer(embed_method,scale):
    vit = ViT(
        image_size=image_size,patch_size=patch_size,n_channels=n_channels,
        number_of_blocks=4,attention_dim=attention_dim,hidden_dim=hidden_dim,
        n_heads=n_heads,dropout_rate=0.1,embed_method=embed_method,
        mlp_structure=[64,64],adn_fn=adn_fn)
    im_size = [batch_size] + [n_channels] + image_size
    output_image_size = [batch_size,n_channels*scale**len(image_size)] 
    output_image_size += [x//scale for x in image_size]
    im = torch.rand(im_size)
    out,_ = vit(im)
    token_size = vit.input_dim_primary
    n_patches = vit.embedding.n_patches
    assert list(out.shape) == [batch_size,n_patches,token_size]
    assert list(vit.embedding.rearrange_rescale(out,scale).shape) == output_image_size
    assert list(vit.embedding.rearrange_inverse(out).shape) == im_size

@pytest.mark.parametrize("embed_method,scale",[
    ("linear",1),
    ("convolutional",1),
    ("linear",2),
    ("convolutional",2),
    ("linear",4),
    ("convolutional",4)])
def test_transformer_windowed(embed_method,scale):
    vit = ViT(
        image_size=image_size,patch_size=patch_size,n_channels=n_channels,
        number_of_blocks=4,attention_dim=attention_dim,hidden_dim=hidden_dim,
        n_heads=n_heads,dropout_rate=0.1,embed_method=embed_method,
        mlp_structure=[64,64],window_size=window_size,adn_fn=adn_fn)
    im_size = [batch_size] + [n_channels] + image_size
    output_image_size = [batch_size,n_channels*scale**len(image_size)] 
    output_image_size += [x//scale for x in image_size]
    im = torch.rand(im_size)
    out,_ = vit(im)
    token_size = vit.input_dim_primary
    n_patches = vit.embedding.n_patches
    size = [batch_size,np.prod(vit.embedding.n_windows),n_patches,token_size]
    assert list(out.shape) == size
    assert list(vit.embedding.rearrange_rescale(out,scale).shape) == output_image_size
    assert list(vit.embedding.rearrange_inverse(out).shape) == im_size

@pytest.mark.parametrize("embed_method",[
    "linear","convolutional"])
def test_transformer_erase(embed_method):
    vit = ViT(
        image_size=image_size,patch_size=patch_size,n_channels=n_channels,
        number_of_blocks=4,attention_dim=attention_dim,hidden_dim=hidden_dim,
        n_heads=n_heads,dropout_rate=0.1,embed_method=embed_method,
        mlp_structure=[64,64],window_size=window_size,adn_fn=adn_fn,
        patch_erasing=0.1)
    im_size = [batch_size] + [n_channels] + image_size
    output_image_size = [batch_size,n_channels*1**len(image_size)] 
    output_image_size += [x//1 for x in image_size]
    im = torch.rand(im_size)
    out,_ = vit(im)
    token_size = vit.input_dim_primary
    n_patches = vit.embedding.n_patches
    size = [batch_size,np.prod(vit.embedding.n_windows),n_patches,token_size]
    assert list(out.shape) == size
    assert list(vit.embedding.rearrange_rescale(out,1).shape) == output_image_size
    assert list(vit.embedding.rearrange_inverse(out).shape) == im_size
