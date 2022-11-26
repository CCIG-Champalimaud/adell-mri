import sys
import os
sys.path.append(os.path.join(os.path.dirname(os.path.abspath(__file__)),'..'))
import pytest

import torch
import numpy as np
from lib.modules.layers.vit import LinearEmbedding

image_size = [32,32,32]
patch_size = [4,4,4]
n_windows = [2,2,2]
n_channels = 1

@pytest.mark.parametrize("d",[2,3])
def test_linear_embedding(d):
    i_s,p_s = image_size[:d],patch_size[:d]
    t = torch.rand(size=[1] + [n_channels] + i_s)
    le = LinearEmbedding(i_s,p_s,n_channels)
    out = le(t)
    assert list(out.shape) == [1,le.n_patches,le.n_features]
    rev = le.rearrange_inverse(out - le.positional_embedding)
    b = np.max(np.abs(t.detach().numpy()-rev.detach().numpy()))
    assert b < 1e-6
    
@pytest.mark.parametrize("d",[2,3])
def test_conv_embedding(d):
    i_s,p_s = image_size[:d],patch_size[:d]
    le = LinearEmbedding(i_s,p_s,n_channels,embed_method="convolutional")
    out = le(torch.rand(size=[1] + [n_channels] + i_s))
    assert list(out.shape) == [1,le.n_patches,le.n_features]

@pytest.mark.parametrize("d",[2,3])
def test_windowed_linear_embedding(d):
    i_s,p_s,w_s = image_size[:d],patch_size[:d],n_windows[:d]
    t = torch.rand(size=[1] + [n_channels] + i_s)
    le = LinearEmbedding(i_s,p_s,n_channels,n_windows=w_s)
    out = le(t)
    assert list(out.shape) == [1,np.prod(w_s),le.n_patches,le.n_features]
