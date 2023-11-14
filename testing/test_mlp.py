import sys
import os

sys.path.append(os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))

import torch
from lib.modules.layers.linear_blocks import MLP


def test_mlp():
    out = MLP(16, 32)(torch.rand(size=[16, 16]))
    assert list(out.shape) == [16, 32]


def test_mlp_hidden():
    out = MLP(16, 32, [32, 64, 16])(torch.rand(size=[16, 16]))
    assert list(out.shape) == [16, 32]
