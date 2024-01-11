import sys
import os

sys.path.append(os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))

from copy import deepcopy
import torch

from adell_mri.modules.callbacks import SpectralNorm


def test_spectral_norm():
    m1 = torch.nn.Linear(16, 32)
    m2 = deepcopy(m1)
    SpectralNorm(5)(m1)

    weights_1 = dict(m1.named_parameters())
    weights_2 = dict(m2.named_parameters())

    a = torch.norm(weights_1["weight"])
    b = torch.norm(weights_2["weight"])
    assert a < b
