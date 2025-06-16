import sys
import os

sys.path.append(os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
import pytest

import torch
from adell_mri.modules.segmentation.unet import UNet
from adell_mri.modules.layers.res_net import ResNet
from adell_mri.modules.classification.classification import GenericEnsemble
from adell_mri.modules.layers.adn_fn import get_adn_fn, ActDropNormBuilder

resnet_args = {
    "backbone_args": {
        "spatial_dim": 3,
        "in_channels": 1,
        "structure": None,
        "maxpool_structure": None,
        "res_type": "resnet",
        "adn_fn": ActDropNormBuilder(),
    },
    "projection_head_args": {
        "in_channels": 512,
        "structure": [1024, 512, 256],
        "adn_fn": ActDropNormBuilder(norm_fn=torch.nn.BatchNorm1d),
    },
    "prediction_head_args": {
        "in_channels": 256,
        "structure": [512, 256],
        "adn_fn": ActDropNormBuilder(norm_fn=torch.nn.BatchNorm1d),
    },
}

input_shape = [16, 1, 32, 32, 32]
depth = [16, 32, 64]
in_channels = 1
n_classes = 2
n_elements = 3
head_adn_fn = get_adn_fn(1, "batch", act_fn="swish", dropout_param=0.1)
S = [[2, 2, 1], [2, 2, 1], [2, 2, 2], [2, 2, 2]]


@pytest.mark.parametrize(
    "sd,sae", [(2, False), (2, True), (3, False), (3, True)]
)
def test_ensemble_unet_encoders(sd, sae):
    networks = [
        UNet(sd, in_channels=in_channels, depth=depth, encoder_only=True)
        for _ in range(n_elements)
    ]

    ge = GenericEnsemble(
        spatial_dimensions=sd,
        networks=networks,
        n_classes=n_classes,
        n_features=[depth[-1] for _ in range(n_elements)],
        head_structure=[64, 64],
        head_adn_fn=head_adn_fn,
        sae=sae,
    )

    input_tensor = torch.rand(input_shape[: (2 + sd)])
    input_tensor_list = [
        torch.rand(input_shape[: (2 + sd)]) for _ in range(n_elements)
    ]
    out = ge(input_tensor)
    assert list(out.shape) == [input_shape[0], n_classes - 1]
    out = ge(input_tensor_list)
    assert list(out.shape) == [input_shape[0], n_classes - 1]


@pytest.mark.parametrize(
    "sd,sae", [(2, False), (2, True), (3, False), (3, True)]
)
def test_ensemble_resnet(sd, sae):
    resnet_args["backbone_args"]["spatial_dim"] = sd
    resnet_args["backbone_args"]["structure"] = [[d, d, 3, 1] for d in depth]
    resnet_args["backbone_args"]["maxpool_structure"] = [s[:sd] for s in S]
    if sd == 2:
        resnet_args["backbone_args"]["adn_fn"] = ActDropNormBuilder(
            norm_fn=torch.nn.BatchNorm2d
        )
    if sd == 3:
        resnet_args["backbone_args"]["adn_fn"] = ActDropNormBuilder(
            norm_fn=torch.nn.BatchNorm3d
        )
    networks = [ResNet(**resnet_args).backbone for _ in range(n_elements)]

    ge = GenericEnsemble(
        spatial_dimensions=sd,
        networks=networks,
        n_classes=n_classes,
        n_features=[depth[-1] for _ in range(n_elements)],
        head_structure=[64, 64],
        head_adn_fn=head_adn_fn,
        sae=sae,
    )

    input_tensor = torch.rand(input_shape[: (2 + sd)])
    input_tensor_list = [
        torch.rand(input_shape[: (2 + sd)]) for _ in range(n_elements)
    ]
    out = ge(input_tensor)
    assert list(out.shape) == [input_shape[0], n_classes - 1]
    out = ge(input_tensor_list)
    assert list(out.shape) == [input_shape[0], n_classes - 1]
