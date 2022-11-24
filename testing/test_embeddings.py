import sys
import os
sys.path.append(os.path.join(os.path.dirname(os.path.abspath(__file__)),'..'))

import torch
import numpy as np
from lib.modules.layers.vit import LinearEmbedding

image_size = [32,32,32]
patch_size = [4,4,4]
n_channels = 1

def test_linear_embedding_2d():
    i_s,p_s = image_size[:2],patch_size[:2]
    le = LinearEmbedding(i_s,p_s,n_channels)
    out = le(torch.rand(size=[1] + [n_channels] + i_s))
    assert list(out.shape) == [1,le.n_patches,le.n_tokens]
    
def test_linear_embedding_2d_reverse():
    i_s,p_s = image_size[:2],patch_size[:2]
    t = torch.rand(size=[1] + [n_channels] + i_s)
    le = LinearEmbedding(i_s,p_s,n_channels)
    out = le(t)
    rev = le.rearrange_inverse(out - le.positional_embedding)
    b = np.max(np.abs(t.detach().numpy()-rev.detach().numpy()))
    assert b < 1e-6

def test_linear_embedding_3d():
    i_s,p_s = image_size[:3],patch_size[:3]
    le = LinearEmbedding(i_s,p_s,n_channels)
    out = le(torch.rand(size=[1] + [n_channels] + i_s))
    assert list(out.shape) == [1,le.n_patches,le.n_tokens]
    
def test_linear_embedding_3d_reverse():
    i_s,p_s = image_size[:3],patch_size[:3]
    t = torch.rand(size=[1] + [n_channels] + i_s)
    le = LinearEmbedding(i_s,p_s,n_channels)
    out = le(t)
    rev = le.rearrange_inverse(out - le.positional_embedding)
    b = np.max(np.abs(t.detach().numpy()-rev.detach().numpy()))
    assert b < 1e-6

def test_conv_embedding_2d():
    i_s,p_s = image_size[:2],patch_size[:2]
    le = LinearEmbedding(i_s,p_s,n_channels,embed_method="convolutional")
    out = le(torch.rand(size=[1] + [n_channels] + i_s))
    assert list(out.shape) == [1,le.n_patches,le.n_tokens]

def test_conv_embedding_3d():
    i_s,p_s = image_size[:3],patch_size[:3]
    le = LinearEmbedding(i_s,p_s,n_channels,embed_method="convolutional")
    out = le(torch.rand(size=[1] + [n_channels] + i_s))
    assert list(out.shape) == [1,le.n_patches,le.n_tokens]
