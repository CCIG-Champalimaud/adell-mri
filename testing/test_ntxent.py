import sys
import os

sys.path.append(os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))

import torch
from adell_mri.modules.self_supervised import NTXentLoss


def test_ntxent():
    X1, X2 = torch.rand(4, 32), torch.rand(4, 32)
    ntxent = NTXentLoss()
    p = ntxent(X1, X2)
