import sys
import os

sys.path.append(os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))

import torch
from adell_mri.modules.layers.linear_blocks import SeqPool

batch_size = 4
n_tokens = 16
n_features = 32


def test_seq_pool():
    sp = SeqPool(n_features)
    t = torch.rand(batch_size, n_tokens, n_features)
    out = sp(t)
    assert list(out.shape) == [batch_size, 1, n_features]
