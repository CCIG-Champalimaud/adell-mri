import os
import sys

sys.path.append(os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))

import torch
from adell_mri.modules.segmentation.mimunet import MIMUNet
from adell_mri.modules.layers.res_net import ResNetBackbone
from adell_mri.modules.layers.adn_fn import get_adn_fn

h, w, d, c = 32, 32, 20, 1

backbone_structure = [
    [16, 16, 3, 1],
    [32, 32, 3, 1],
    [64, 64, 3, 1],
    [128, 128, 3, 1],
]
backbone_args = {
    "spatial_dim": 2,
    "in_channels": 1,
    "structure": backbone_structure,
    "maxpool_structure": [2 for _ in backbone_structure],
    "res_type": "resnet",
    "adn_fn": get_adn_fn(3, "instance", "relu", dropout_param=0.0),
}


def test_unet_base():
    module = ResNetBackbone(**backbone_args)
    module.forward = module.forward_intermediate
    i = torch.rand(size=[1, c, h, w, d])
    output_size = [1, 1, h, w, d]
    a = MIMUNet(
        module=module,
        upscale_type="transpose",
        link_type="identity",
        n_classes=2,
        in_channels=c,
        n_slices=d,
    )
    o = a(i)
    assert list(o.shape) == output_size


def test_unet_base_more_channels():
    c = 2
    module = ResNetBackbone(**backbone_args)
    module.forward = module.forward_intermediate
    i = torch.rand(size=[1, c, h, w, d])
    output_size = [1, 1, h, w, d]
    a = MIMUNet(
        module=module,
        upscale_type="transpose",
        link_type="identity",
        n_classes=2,
        in_channels=c,
        n_slices=d,
    )
    o = a(i)
    assert list(o.shape) == output_size
