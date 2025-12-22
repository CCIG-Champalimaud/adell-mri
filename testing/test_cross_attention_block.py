import os
import sys

sys.path.append(os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))

import torch

from adell_mri.modules.layers.self_attention import SelfAttentionBlock

c, h, w, d = [16, 32, 32, 16]


def test_convnext_2d():
    cross_at = SelfAttentionBlock(3, c, c * 2, patch_size=[4, 4, 2])
    i = torch.rand([1, c, h, w, d])
    o = cross_at(i)
    assert list(o.shape) == [1, c, h, w, d]
