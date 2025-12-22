import os
import sys

sys.path.append(os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))

import torch

from adell_mri.modules.layers.res_blocks import ConvNeXtBlock2d, ConvNeXtBlock3d

c, h, w, d = [1, 32, 32, 16]
inter = 64
out = 32


def test_convnext_2d():
    convnext = ConvNeXtBlock2d(1, 5, out, out)
    i = torch.rand([1, c, h, w])
    o = convnext(i)
    assert list(o.shape) == [1, out, h, w]


def test_convnext_2d_different_inter():
    convnext = ConvNeXtBlock2d(1, 5, inter, out)
    i = torch.rand([1, c, h, w])
    o = convnext(i)
    assert list(o.shape) == [1, out, h, w]


def test_convnext_3d():
    convnext = ConvNeXtBlock3d(1, 5, out, out)
    i = torch.rand([1, c, h, w, d])
    o = convnext(i)
    assert list(o.shape) == [1, out, h, w, d]


def test_convnext_3d_different_inter():
    convnext = ConvNeXtBlock3d(1, 5, inter, out)
    i = torch.rand([1, c, h, w, d])
    o = convnext(i)
    assert list(o.shape) == [1, out, h, w, d]
