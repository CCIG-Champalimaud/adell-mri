import sys
import os

sys.path.append(os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
import pytest

import torch
from adell_mri.modules.classification import VGG

image_size = [32, 32, 32]
in_channels = 3
n_classes = 2


@pytest.mark.parametrize("n_dim,be", [[3, 0], [3, 1], [3, 2]])
def test_vgg(n_dim, be):
    vgg_mod = VGG(
        spatial_dimensions=n_dim, in_channels=in_channels, n_classes=n_classes
    )
    input_tensor = torch.ones([2, in_channels, *image_size[:n_dim]])
    assert list(vgg_mod(input_tensor).shape) == [2, 1]
    if be > 0:
        for i in range(be):
            assert list(vgg_mod(input_tensor, batch_idx=i).shape) == [2, 1]


def test_vgg_names():
    # tests whether parameter names remain unchanged (allows for an easier
    # extension of the module using batch ensemble)
    vgg_mod = VGG(
        spatial_dimensions=3, in_channels=in_channels, n_classes=n_classes
    )
    vgg_mod_be = VGG(
        spatial_dimensions=3,
        in_channels=in_channels,
        n_classes=n_classes,
        batch_ensemble=2,
    )

    vgg_mod_param_names = [k for k, _ in vgg_mod.named_parameters()]
    vgg_mod_be_param_names = [k for k, _ in vgg_mod_be.named_parameters()]

    set_diff = set.difference(
        set(vgg_mod_be_param_names), set(vgg_mod_param_names)
    )
    assert len(set_diff) == len(vgg_mod_be_param_names) - len(
        vgg_mod_param_names
    )
