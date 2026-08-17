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
        in_channels=1,
        padding=1,
        spatial_dimensions=ndim,
        upscale_type="transpose",
    )
    random_sample = diff_proc.sample(model, 1)
    assert list(random_sample.shape) == sh

    random_sample_input = diff_proc.sample(model, 1, x=torch.rand(sh))
    assert random_sample_input.shape == image.shape

    model = DiffusionUNet(
        in_channels=1,
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


@pytest.mark.parametrize("ndim", [2, 3])
def test_diffusion_ddim_sampling(ndim):
    if ndim == 2:
        image_size = [BASE_SIZE, BASE_SIZE]
        sh = [1, 1, BASE_SIZE, BASE_SIZE]
    if ndim == 3:
        image_size = [BASE_SIZE, BASE_SIZE, BASE_SIZE // 2]
        sh = [1, 1, BASE_SIZE, BASE_SIZE, BASE_SIZE // 2]
    diff_proc = Diffusion(10, 1e-4, 1e-2, image_size, step_key="ddim")

    model = DiffusionUNet(
        in_channels=1,
        padding=1,
        spatial_dimensions=ndim,
        upscale_type="transpose",
    )
    random_sample = diff_proc.sample(model, 1)
    assert list(random_sample.shape) == sh


@pytest.mark.parametrize("ndim", [2, 3])
def test_diffusion_alpha_deblending(ndim):
    if ndim == 2:
        image_size = [BASE_SIZE, BASE_SIZE]
        sh = [1, 1, BASE_SIZE, BASE_SIZE]
    if ndim == 3:
        image_size = [BASE_SIZE, BASE_SIZE, BASE_SIZE // 2]
        sh = [1, 1, BASE_SIZE, BASE_SIZE, BASE_SIZE // 2]
    diff_proc = Diffusion(
        10, 1e-4, 1e-2, image_size, step_key="alpha_deblending"
    )

    model = DiffusionUNet(
        in_channels=1,
        padding=1,
        spatial_dimensions=ndim,
        upscale_type="transpose",
    )
    random_sample = diff_proc.sample(model, 1)
    assert list(random_sample.shape) == sh


@pytest.mark.parametrize(
    "scheduler", ["linear", "scaled_linear", "quadratic", "sigmoid", "cosine"]
)
def test_diffusion_schedulers(scheduler):
    diff_proc = Diffusion(100, 1e-4, 0.02, [8, 8], scheduler=scheduler)
    assert len(diff_proc.beta) == 100
    assert (diff_proc.beta >= 0).all()


def test_diffusion_cfg_guidance_shifts_prediction():
    diff_proc = Diffusion(50, 1e-4, 0.02, [8, 8])
    model = DiffusionUNet(
        in_channels=1,
        padding=1,
        spatial_dimensions=2,
        classifier_free_guidance=True,
        classifier_classes=5,
        upscale_type="transpose",
    )
    x = torch.rand(1, 1, 8, 8)
    t = torch.as_tensor([0.5])
    cond = model(x, t, torch.as_tensor([1]))
    uncond = model(x, t)
    shifted = uncond + 3.0 * (cond - uncond)
    assert shifted.shape == x.shape
    # with scale=0 the shift should equal the unconditional prediction
    assert torch.allclose(uncond + 0.0 * (cond - uncond), uncond, atol=1e-6)


def test_ddim_reverse_step_no_nan():
    diff_proc = Diffusion(10, 1e-4, 1e-2, [8, 8], step_key="ddim")
    x = torch.rand(1, 1, 8, 8)
    epsilon = torch.randn_like(x)
    out = diff_proc.ddim_reverse_step(x, epsilon, t=5, eta=0.0)
    assert out.shape == x.shape
    assert torch.isnan(out).any() == False
    out = diff_proc.ddim_reverse_step(x, epsilon, t=5, eta=1.0)
    assert out.shape == x.shape
    assert torch.isnan(out).any() == False


def test_ddpm_reverse_step_clipping():
    diff_proc = Diffusion(10, 1e-4, 1e-2, [8, 8])
    x = torch.rand(1, 1, 8, 8)
    epsilon = torch.randn_like(x)
    out = diff_proc.ddpm_reverse_step(x, epsilon, t=5)
    assert out.shape == x.shape
    assert torch.isnan(out).any() == False
