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
    pro_gan = ProgressiveGenerator(**params)
    pro_gan(torch.randn(2, 1, 2, 2))


def test_discriminator(get_discriminator_params):
    params = get_discriminator_params
    input_tensor = torch.randn(2, 1, 32, 32)
    discriminator = ProgressiveDiscriminator(**params)
    discriminator(input_tensor, level=0)
    discriminator(input_tensor, level=1)
    discriminator(input_tensor, level=1, progressive_level=0, alpha=0.5)
