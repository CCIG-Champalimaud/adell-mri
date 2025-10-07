import pytest
import torch
from adell_mri.modules.gan.gan.style import (
    ProgressiveGenerator,
    ProgressiveDiscriminator,
)
from copy import deepcopy


@pytest.fixture
def get_stylegan_params():
    return deepcopy(
        {
            "n_dim": 2,
            "input_channels": 1,
            "output_channels": 1,
            "depths": [32, 16, 8],
        }
    )


@pytest.fixture
def get_discriminator_params():
    return deepcopy(
        {
            "n_dim": 2,
            "input_channels": 1,
            "output_channels": 1,
            "depths": [4, 8, 16, 32],
        }
    )


def test_progressivegan(get_stylegan_params):
    params = get_stylegan_params
    input_tensor = torch.randn(2, 1, 2, 2)
    out_shape = [2, 1, 32, 32]
    pro_gan = ProgressiveGenerator(**params)
    out = pro_gan(input_tensor)
    assert list(out.shape) == out_shape
    out = pro_gan(input_tensor, level=1)
    assert list(out.shape) == [*out_shape[:2], *[x // 2 for x in out_shape[2:]]]
    out = pro_gan(input_tensor, level=1, prog_level=0, alpha=0.5)
    assert list(out.shape) == out_shape
    out = pro_gan(input_tensor, level=2, prog_level=1, alpha=0.5)
    assert list(out.shape) == [*out_shape[:2], *[x // 2 for x in out_shape[2:]]]


def test_discriminator_regular(get_discriminator_params):
    params = get_discriminator_params
    input_tensor = torch.randn(2, 1, 32, 32)
    discriminator = ProgressiveDiscriminator(**params)
    discriminator(input_tensor, level=0)
    discriminator(input_tensor, level=1)
    discriminator(input_tensor, level=1, prog_level=0, alpha=0.5)
    discriminator(input_tensor, level=2, prog_level=1, alpha=0.5)


def test_discriminator_minibatch_std(get_discriminator_params):
    params = get_discriminator_params
    input_tensor = torch.randn(2, 1, 32, 32)
    discriminator = ProgressiveDiscriminator(**params, minibatch_std=True)
    discriminator(input_tensor, level=0)
    discriminator(input_tensor, level=1)
    discriminator(input_tensor, level=1, prog_level=0, alpha=0.5)
    discriminator(input_tensor, level=2, prog_level=1, alpha=0.5)
