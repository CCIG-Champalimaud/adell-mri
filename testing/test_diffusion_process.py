import os
import sys

sys.path.append(os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
import pytest

import torch
from adell_mri.modules.diffusion.diffusion_process import Diffusion
from adell_mri.modules.diffusion.unet import DiffusionUNet

BASE_SIZE = 32


@pytest.mark.parametrize("ndim", [2, 3])
def test_diffusion_process(ndim):
    if ndim == 2:
        image_size = [BASE_SIZE, BASE_SIZE]
        sh = [1, 1, BASE_SIZE, BASE_SIZE]
    if ndim == 3:
        image_size = [BASE_SIZE, BASE_SIZE, BASE_SIZE // 2]
        sh = [1, 1, BASE_SIZE, BASE_SIZE, BASE_SIZE // 2]
    diff_proc = Diffusion(10, 1e-4, 1e-2, image_size)

    image = torch.rand(*sh)
    epsilon = torch.randn_like(image)
    t = diff_proc.sample_timesteps(image.shape[0])
    noise_image, epsilon = diff_proc.noise_images(image, epsilon=epsilon, t=t)
    assert noise_image.shape == image.shape

    model = DiffusionUNet(
        n_channels=1,
        padding=1,
        spatial_dimensions=ndim,
        upscale_type="transpose",
    )
    random_sample = diff_proc.sample(model, 1)
    assert random_sample.shape == image.shape

    random_sample_input = diff_proc.sample(model, 1, x=torch.rand(sh))
    assert random_sample_input.shape == image.shape

    model = DiffusionUNet(
        n_channels=1,
        padding=1,
        spatial_dimensions=ndim,
        classifier_free_guidance=True,
        classifier_classes=5,
        upscale_type="transpose",
    )
    random_sample_class_input = diff_proc.sample(
        model, 1, x=torch.rand(sh), classification=torch.as_tensor([2])
    )
    assert random_sample_class_input.shape == image.shape
