import os
import sys

sys.path.append(os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))

import torch

from adell_mri.modules.layers.multi_resolution import PyramidSpatialPooling3d


def test_psp():
    h, w, d, c = 128, 128, 32, 16
    i = torch.rand(size=[1, c, h, w, d])
    levels = [[4, 4, 2], [8, 8, 4], [16, 16, 8]]
    p = PyramidSpatialPooling3d(16, levels)
    o = p(i)
    assert list(o.shape) == [1, c * (len(levels) + 1), h, w, d]
