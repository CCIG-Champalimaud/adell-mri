import sys
import os

sys.path.append(os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))

import torch
from adell_mri.modules.layers.batch_ensemble import (
    BatchEnsemble,
    BatchEnsembleWrapper,
)
from adell_mri.modules.layers.adn_fn import get_adn_fn

c, h, w, d = [16, 32, 32, 16]
adn_fn = {
    0: get_adn_fn(1),
    1: get_adn_fn(1),
    2: get_adn_fn(2),
    3: get_adn_fn(3),
}
n = 10
out_channels = 16

op_kwargs_0d = {}
op_kwargs_nd = {"kernel_size": 3, "padding": 1}
size_list = [2, c, h, w, d]


def test_0d():
    op_kwargs = op_kwargs_0d
    i = 0
    size = size_list[0 : (i + 2)]
    D = BatchEnsemble(i, n, c, out_channels, adn_fn[i], op_kwargs)
    input_tensor = torch.rand(size=size)
    out = D(input_tensor)
    assert list(out.shape) == size, "normal forward failed"
    out = D(input_tensor, 1)
    assert list(out.shape) == size, "indexed forward failed"
    D.training = False
    out = D(input_tensor)
    assert list(out.shape) == size, "testing forward failed"


def test_1d():
    op_kwargs = op_kwargs_nd
    i = 1
    size = size_list[0 : (i + 2)]
    D = BatchEnsemble(i, n, c, out_channels, adn_fn[i], op_kwargs)
    input_tensor = torch.rand(size=size)
    out = D(input_tensor)
    assert list(out.shape) == size, "normal forward failed"
    out = D(input_tensor, 1)
    assert list(out.shape) == size, "indexed forward failed"
    D.training = False
    out = D(input_tensor)
    assert list(out.shape) == size, "testing forward failed"


def test_2d():
    op_kwargs = op_kwargs_nd
    i = 2
    size = size_list[0 : (i + 2)]
    D = BatchEnsemble(i, n, c, out_channels, adn_fn[i], op_kwargs)
    input_tensor = torch.rand(size=size)
    out = D(input_tensor)
    assert list(out.shape) == size, "normal forward failed"
    out = D(input_tensor, 1)
    assert list(out.shape) == size, "indexed forward failed"
    D.training = False
    out = D(input_tensor)
    assert list(out.shape) == size, "testing forward failed"


def test_3d():
    op_kwargs = op_kwargs_nd
    i = 3
    size = size_list[0 : (i + 2)]
    D = BatchEnsemble(i, n, c, out_channels, adn_fn[i], op_kwargs)
    input_tensor = torch.rand(size=size)
    out = D(input_tensor)
    assert list(out.shape) == size, "normal forward failed"
    out = D(input_tensor, 1)
    assert list(out.shape) == size, "indexed forward failed"
    D.training = False
    out = D(input_tensor)
    assert list(out.shape) == size, "testing forward failed"


def test_3d_red():
    op_kwargs = {"kernel_size": 3}
    i = 3
    size = size_list[0 : (i + 2)]
    D = BatchEnsemble(
        i, n, c, out_channels, adn_fn[i], op_kwargs, res_blocks=True
    )
    input_tensor = torch.rand(size=size)
    out = D(input_tensor)
    assert list(out.shape) == size, "normal forward failed"
    out = D(input_tensor, 1)
    assert list(out.shape) == size, "indexed forward failed"
    D.training = False
    out = D(input_tensor)
    assert list(out.shape) == size, "testing forward failed"


def test_3d_wrapper():
    i = 3
    size = size_list[0 : (i + 2)]
    mod = torch.nn.Conv3d(c, out_channels, 3, padding="same")
    D = BatchEnsembleWrapper(mod, n, c, out_channels, adn_fn[i])
    input_tensor = torch.rand(size=size)
    out = D(input_tensor)
    assert list(out.shape) == size, "normal forward failed"
    out = D(input_tensor, 1)
    assert list(out.shape) == size, "indexed forward failed"
    D.training = False
    out = D(input_tensor)
    assert list(out.shape) == size, "testing forward failed"


def test_3d_wrapper_with_idx_list():
    i = 3
    size = size_list[0 : (i + 2)]
    mod = torch.nn.Conv3d(c, out_channels, 3, padding="same")
    D = BatchEnsembleWrapper(mod, n, c, out_channels, adn_fn[i])
    input_tensor = torch.rand(size=size)
    out = D(input_tensor)
    assert list(out.shape) == size, "normal forward failed"
    out = D(input_tensor, 1)
    assert list(out.shape) == size, "indexed forward failed"
    out = D(input_tensor, [0, 1])
    assert list(out.shape) == size, "indexed forward failed"
    D.training = False
    out = D(input_tensor)
    assert list(out.shape) == size, "testing forward failed"
