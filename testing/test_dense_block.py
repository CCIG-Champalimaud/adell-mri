import sys
import os

sys.path.append(os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))

import torch
from adell_mri.modules.layers.standard_blocks import DenseBlock
from adell_mri.modules.layers.adn_fn import ActDropNorm
from adell_mri.utils.python_logging import get_logger

c, h, w, d = [1, 32, 32, 16]
adn_fn_2d = ActDropNorm

logger = get_logger(__name__)


def adn_fn_3d(s):
    return ActDropNorm(s, norm_fn=torch.nn.BatchNorm3d)


def test_dense_block_2d_1():
    logger.info("Testing case of a single convolution (no dense connections)")
    structure = [1, 16]
    D = DenseBlock(2, structure, 3, adn_fn_2d)
    input_tensor = torch.rand(size=[1, c, h, w])
    out = D(input_tensor)
    assert list(out.shape) == [1, structure[-1], h, w]


def test_dense_block_2d_normal():
    logger.info("Testing 2D case")
    structure = [1, 16, 16, 16]
    D = DenseBlock(2, structure, 3, adn_fn_2d)
    input_tensor = torch.rand(size=[1, c, h, w])
    out = D(input_tensor)
    assert list(out.shape) == [1, structure[-1], h, w]


def test_dense_block_2d_increasing():
    logger.info("Testing 2D case with increasing structure")
    structure = [1, 16, 32, 48]
    D = DenseBlock(2, structure, 3, adn_fn_2d)
    input_tensor = torch.rand(size=[1, c, h, w])
    out = D(input_tensor)
    assert list(out.shape) == [1, structure[-1], h, w]


def test_dense_block_3d_increasing():
    logger.info("Testing 3D case with increasing structure")
    structure = [1, 16, 32, 48]
    D = DenseBlock(3, structure, 3, adn_fn_3d)
    input_tensor = torch.rand(size=[1, c, h, w, d])
    out = D(input_tensor)
    assert list(out.shape) == [1, structure[-1], h, w, d]


def test_dense_block_3d_skip():
    logger.info("Testing skip connections")
    structure = [1, 16, 32, 16]
    structure_skip = [16, 16]
    D = DenseBlock(3, structure, 3, adn_fn_3d, structure_skip=structure_skip)
    input_tensor = torch.rand(size=[1, c, h, w, d])
    input_skips = [torch.rand(size=[1, s, h, w, d]) for s in structure_skip]
    out = D(input_tensor, input_skips)
    assert list(out.shape) == [1, structure[-1], h, w, d]
