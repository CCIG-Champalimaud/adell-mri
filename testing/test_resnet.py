import os
import sys

sys.path.append(os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
from itertools import product

import pytest
import torch

from adell_mri.modules.layers.adn_fn import ActDropNormBuilder
from adell_mri.modules.layers.res_net import ResNet

h, w, d, c = 32, 32, 20, 1
res_types = ["resnet", "resnext", "convnext", "convnextv2"]

resnet_args = {
    "backbone_args": {
        "in_channels": 1,
        "structure": [[1, 32, 3, 1], [32, 64, 3, 1], [32, 64, 3, 1]],
        "maxpool_structure": None,
    },
    "projection_head_args": {
        "in_channels": 32,
        "structure": [32, 64, 32],
        "adn_fn": ActDropNormBuilder(norm_fn=torch.nn.BatchNorm1d),
    },
    "prediction_head_args": {
        "in_channels": 32,
        "structure": [32, 64],
        "adn_fn": ActDropNormBuilder(norm_fn=torch.nn.BatchNorm1d),
    },
}


@pytest.mark.parametrize(
    "spatial_dim, be, res_type",
    list(product([2, 3], range(0, 5, 2), res_types)),
)
def test_resnet(spatial_dim, be, res_type):
    resnet_args["backbone_args"]["spatial_dim"] = spatial_dim
    resnet_args["backbone_args"]["batch_ensemble"] = be
    resnet_args["backbone_args"]["res_type"] = res_type
    if spatial_dim == 2:
        resnet_args["backbone_args"]["adn_fn"] = ActDropNormBuilder(
            norm_fn=torch.nn.BatchNorm2d
        )
        input_tensor = torch.ones([2, 1, h, w])
    if spatial_dim == 3:
        resnet_args["backbone_args"]["adn_fn"] = ActDropNormBuilder(
            norm_fn=torch.nn.BatchNorm3d
        )
        input_tensor = torch.ones([2, 1, h, w, d])
    resnet = ResNet(**resnet_args)
    resnet(input_tensor)
