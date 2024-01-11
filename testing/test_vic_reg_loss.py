import os
import sys

sys.path.append(os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))

import numpy as np
import torch

from adell_mri.modules.self_supervised import VICRegLoss, VICRegLocalLoss

b, c, h, w, d = 4, 512, 4, 4, 2


def test_vic_reg_loss():
    vrl = VICRegLoss()
    inp1 = torch.rand(size=[b, c, h, w, d])
    inp2 = torch.rand(size=[b, c, h, w, d])
    out = [x.cpu().detach().numpy() for x in vrl(inp1, inp2)]
    assert np.all([np.isfinite(out)]), "non-finite values in loss"


def test_vic_reg_local_loss():
    def make_random_box():
        M = 256
        S = 128
        box = np.random.randint(0, M - S, 3).astype(np.float32)
        box = torch.as_tensor(np.concatenate([box, box + S]))
        box = torch.stack([box for _ in range(b)])
        return box

    vrl = VICRegLocalLoss()
    inp1 = torch.rand(size=[b, c, h, w, d])
    inp2 = torch.rand(size=[b, c, h, w, d])
    box1 = make_random_box()
    box2 = make_random_box()
    out = [x.cpu().detach().numpy() for x in vrl(inp1, inp2, box1, box2)]
    assert np.all([np.isfinite(out)]), "non-finite values in loss"
