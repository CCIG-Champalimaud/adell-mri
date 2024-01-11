import sys
import os

sys.path.append(os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))

import torch
from adell_mri.modules.layers.conv_next import ConvNeXtBackbone

c, h, w = [1, 64, 64]
inter = 64
out = 32

structure = [(32, 32, 3, 1), (64, 64, 3, 1), (128, 128, 3, 1)]
maxpool_structure = [(2, 2), (2, 2), (2, 2)]


def test_convnext_backbone_2d():
    convnext = ConvNeXtBackbone(
        2, 1, structure=structure, maxpool_structure=maxpool_structure
    )
    i = torch.rand([1, c, h, w])
    o = convnext(i)
    d = 4 * 2 ** len(maxpool_structure)
    assert list(o.shape) == [1, structure[-1][0], h // d, w // d]
