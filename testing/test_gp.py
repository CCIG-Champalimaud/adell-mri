import os
import sys

sys.path.append(os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
import numpy as np
import torch
import torch.nn.functional as F

from adell_mri.modules.layers.gaussian_process import GaussianProcessLayer

i, o = 128, 8
a, b, c = 16, 32, 64
n = 4


def test_gp_1d():
    gp = GaussianProcessLayer(i, o)
    input_tensor = torch.as_tensor(
        np.random.normal(size=[n, i]), dtype=torch.float32
    )
    labels = torch.randint(0, 2, [n]).float()
    output = gp(input_tensor)
    gp.update_inv_cov(input_tensor, torch.full_like(labels, 0.5))
    gp.get_cov()
    assert list(output.shape) == [n, o]
    assert list(gp.cov.shape) == [1, o, o]


def test_gp_2d():
    gp = GaussianProcessLayer(i, o)
    input_tensor = torch.as_tensor(
        np.random.normal(size=[n, i, a]), dtype=torch.float32
    )
    labels = torch.randint(0, 2, [n]).float()
    output = gp(input_tensor)
    gp.update_inv_cov(input_tensor, torch.full_like(labels, 0.5))
    gp.get_cov()
    assert list(output.shape) == [n, o, a]
    assert list(gp.cov.shape) == [1, o, o]


def test_gp_3d():
    gp = GaussianProcessLayer(i, o)
    input_tensor = torch.as_tensor(
        np.random.normal(size=[n, i, a, b]), dtype=torch.float32
    )
    labels = torch.randint(0, 2, [n]).float()
    output = gp(input_tensor)
    gp.update_inv_cov(input_tensor, torch.full_like(labels, 0.5))
    gp.get_cov()
    assert list(output.shape) == [n, o, a, b]
    assert list(gp.cov.shape) == [1, o, o]


def test_gp_4d():
    gp = GaussianProcessLayer(i, o)
    input_tensor = torch.as_tensor(
        np.random.normal(size=[n, i, a, b, c]), dtype=torch.float32
    )
    labels = torch.randint(0, 2, [n]).float()
    output = gp(input_tensor)
    gp.update_inv_cov(input_tensor, torch.full_like(labels, 0.5))
    gp.get_cov()
    assert list(output.shape) == [n, o, a, b, c]
    assert list(gp.cov.shape) == [1, o, o]


def test_gp_multiclass():
    """Test multiclass scenario with one-hot encoded labels"""
    n_classes = 4
    gp = GaussianProcessLayer(i, n_classes)
    input_tensor = torch.as_tensor(
        np.random.normal(size=[n, i]), dtype=torch.float32
    )
    labels = F.one_hot(torch.randint(0, n_classes, [n]), n_classes).float()
    output = gp(input_tensor)
    gp.update_inv_cov(input_tensor, torch.full_like(labels, 0.5))
    gp.get_cov()
    assert list(output.shape) == [n, n_classes]
    assert list(gp.cov.shape) == [1, n_classes, n_classes]


def test_gp_sampling():
    """Test GP sampling functionality"""
    gp = GaussianProcessLayer(i, o)
    input_tensor = torch.as_tensor(
        np.random.normal(size=[n, i]), dtype=torch.float32
    )
    labels = torch.randint(0, 2, [n]).float()

    gp.update_inv_cov(input_tensor, torch.full_like(labels, 0.5))
    gp.get_cov()

    samples = gp.rsample(input_tensor, n_samples=5)
    assert list(samples.shape) == [5, n, o]


def test_gp_numerical_stability():
    """Test numerical stability with edge cases"""
    gp = GaussianProcessLayer(i, o)
    input_tensor = torch.as_tensor(
        np.random.normal(size=[n, i]), dtype=torch.float32
    )
    labels = torch.ones([n]).float()
    output = gp(input_tensor)
    gp.update_inv_cov(input_tensor, torch.full_like(labels, 0.5))
    gp.get_cov()
    assert list(output.shape) == [n, o]
    assert list(gp.cov.shape) == [1, o, o]


if __name__ == "__main__":
    test_gp_1d()
    test_gp_2d()
    test_gp_3d()
    test_gp_4d()
    test_gp_multiclass()
    test_gp_sampling()
    test_gp_numerical_stability()
    print("All tests passed!")
