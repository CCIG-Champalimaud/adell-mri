import os
import sys
sys.path.append(os.path.join(os.path.dirname(os.path.abspath(__file__)),'..'))
import pytest

import torch
from lib.modules.diffusion.diffusion_process import Diffusion
from lib.modules.diffusion.unet import DiffusionUNet


@pytest.mark.parametrize("ndim",[2,3])
def test_diffusion_process(ndim):
    if ndim == 2:
        image_size = [64,64]
        sh = [1,1,64,64]
    if ndim == 3:
        image_size = [64,64,32]
        sh = [1,1,64,64,32]
    diff_proc = Diffusion(10,1e-4,1e-2,image_size,"cpu")

    image = torch.rand(*sh)
    noise_image,epsilon = diff_proc.noise_images(image,0)
    assert noise_image.shape == image.shape

    model = DiffusionUNet(n_channels=1,padding=1,
                          spatial_dimensions=ndim,
                          upscale_type="transpose")
    random_sample = diff_proc.sample(model,1)
    assert random_sample.shape == image.shape

    random_sample_input = diff_proc.sample(
        model,1,x=torch.rand(sh))
    assert random_sample_input.shape == image.shape

    model = DiffusionUNet(n_channels=1,padding=1,
                          spatial_dimensions=ndim,
                          classifier_free_guidance=True,
                          classifier_classes=5,
                          upscale_type="transpose")
    random_sample_class_input = diff_proc.sample(
        model,1,x=torch.rand(sh),classification=torch.as_tensor([2]))
    assert random_sample_class_input.shape == image.shape