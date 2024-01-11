import sys
import os

sys.path.append(os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))

import torch
import torch.nn.functional as F
import numpy as np
from adell_mri.modules.layers.gaussian_process import GaussianProcessLayer

i, o = 128, 8
a, b, c = 16, 32, 64
n = 4


def test_gp_1d():
    gp = GaussianProcessLayer(i, o)
    input_tensor = torch.as_tensor(
        np.random.normal(size=[n, i]), dtype=torch.float32
    )
    pred = F.softmax(torch.as_tensor(np.random.normal(size=[n, o])), 1)
    output = gp(input_tensor)
    gp.update_inv_cov(input_tensor, pred)
    gp.get_cov()
    assert list(output.shape) == [n, o]
    assert list(gp.cov.shape) == [1, o, o]


def test_gp_2d():
    gp = GaussianProcessLayer(i, o)
    input_tensor = torch.as_tensor(
        np.random.normal(size=[n, i, a]), dtype=torch.float32
    )
    pred = F.softmax(torch.as_tensor(np.random.normal(size=[n, o])), 1)
    output = gp(input_tensor)
    gp.update_inv_cov(input_tensor, pred)
    gp.get_cov()
    assert list(output.shape) == [n, o, a]
    assert list(gp.cov.shape) == [1, o, o]


def test_gp_3d():
    gp = GaussianProcessLayer(i, o)
    input_tensor = torch.as_tensor(
        np.random.normal(size=[n, i, a, b]), dtype=torch.float32
    )
    pred = F.softmax(torch.as_tensor(np.random.normal(size=[n, o])), 1)
    output = gp(input_tensor)
    gp.update_inv_cov(input_tensor, pred)
    gp.get_cov()
    assert list(output.shape) == [n, o, a, b]
    assert list(gp.cov.shape) == [1, o, o]


def test_gp_4d():
    gp = GaussianProcessLayer(i, o)
    input_tensor = torch.as_tensor(
        np.random.normal(size=[n, i, a, b, c]), dtype=torch.float32
    )
    pred = F.softmax(torch.as_tensor(np.random.normal(size=[n, o])), 1)
    output = gp(input_tensor)
    gp.update_inv_cov(input_tensor, pred)
    gp.get_cov()
    assert list(output.shape) == [n, o, a, b, c]
    assert list(gp.cov.shape) == [1, o, o]
