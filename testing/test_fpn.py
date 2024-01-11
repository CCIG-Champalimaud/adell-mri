import sys
import os

sys.path.append(os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))

from adell_mri.modules.layers.adn_fn import ActDropNorm
from adell_mri.modules.layers.multi_resolution import (
    FeaturePyramidNetworkBackbone,
)
from adell_mri.modules.layers.res_net import ResNetBackbone
import torch

resnet_structure = [(64, 128, 3, 3), (128, 256, 3, 3), (256, 512, 3, 3)]
maxpool_structure = [(2, 2, 2), (2, 2, 2), (2, 2, 1)]
h, w, d, c = 64, 64, 19, 1


def test_fpn_2d():
    def adn_fn(s):
        return ActDropNorm(s, norm_fn=torch.nn.BatchNorm2d)

    sd = 2
    i = torch.rand(size=[1, c, h, w])
    backbone = ResNetBackbone(
        sd, in_channels=1, structure=resnet_structure[:-1], adn_fn=adn_fn
    )
    a = FeaturePyramidNetworkBackbone(
        backbone, sd, structure=resnet_structure[:-1], adn_fn=adn_fn
    )
    o = a(i)
    assert list(o.shape) == [
        1,
        resnet_structure[0][0],
        h // 2,
        w // 2,
    ], "2d output shape is not correct"


def test_fpn_3d():
    def adn_fn(s):
        return ActDropNorm(s, norm_fn=torch.nn.BatchNorm3d)

    sd = 3
    i = torch.rand(size=[1, c, h, w, d])
    backbone = ResNetBackbone(
        sd, in_channels=1, structure=resnet_structure[:-1], adn_fn=adn_fn
    )
    a = FeaturePyramidNetworkBackbone(
        backbone, sd, structure=resnet_structure[:-1], adn_fn=adn_fn
    )
    o = a(i)
    assert list(o.shape) == [
        1,
        resnet_structure[0][0],
        h // 2,
        w // 2,
        d // 2,
    ], "3d output shape is not correct"


def test_fpn_3d_maxpool_structure():
    def adn_fn(s):
        return ActDropNorm(s, norm_fn=torch.nn.BatchNorm3d)

    sd = 3
    i = torch.rand(size=[1, c, h, w, d])
    backbone = ResNetBackbone(
        sd,
        in_channels=1,
        structure=resnet_structure,
        maxpool_structure=maxpool_structure,
        adn_fn=adn_fn,
    )
    a = FeaturePyramidNetworkBackbone(
        backbone,
        sd,
        structure=resnet_structure,
        maxpool_structure=maxpool_structure,
        adn_fn=adn_fn,
    )
    o = a(i)
    assert list(o.shape) == [
        1,
        resnet_structure[0][0],
        h // 2,
        w // 2,
        d // 2,
    ], "3d output shape for fpn with maxpool structure is not correct"
