import os
import sys
from copy import deepcopy

import torch

from adell_mri.utils.pl_callbacks import SpectralNorm


def test_spectral_norm():
    m1 = torch.nn.Linear(16, 32)
    m2 = deepcopy(m1)
    SpectralNorm()._apply_spectral_norm(m1)

    # simulate a train step to trigger the SN hook
    m1.train()
    m1(torch.rand([1, 16]))

    a = torch.norm(m1.weight)
    b = torch.norm(m2.weight)
    assert a < b
