import os
import sys

sys.path.append(os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
import pytest

import torch
from lib.modules.classification.classification import UNetEncoder
from lib.modules.layers.adn_fn import ActDropNormBuilder

h, w, d, c = 32, 32, 20, 1
bs = 4

depths = [[16, 32, 64], [16, 32, 64, 128]]
spatial_dims = [2, 3]
n_classes = 2

param_list = []
for dim in [2, 3]:
    for D in depths:
        for conv_type in ["regular", "resnet"]:
            for strides in ["regular", "irregular"]:
                param_list.append((D, dim, conv_type, strides))


@pytest.mark.parametrize("D,sd,conv_type,strides", param_list)
def test_unet_encoder(D, sd, conv_type, strides):
    K = [3 for _ in D]
    if strides == "irregular":
        if sd == 2:
            S = [[2, 2], *[[2, 1] for _ in D[1:]]]
        elif sd == 3:
            S = [[2, 2, 2], *[[2, 2, 1] for _ in D[1:]]]
    else:
        S = [2 for _ in D]
    if sd == 2:
        i = torch.rand(size=[bs, c, h, w])
        output_size = [bs, n_classes - 1]
    elif sd == 3:
        i = torch.rand(size=[bs, c, h, w, d])
        output_size = [bs, n_classes - 1]
    a = UNetEncoder(
        spatial_dimensions=sd,
        head_structure=[D[-1] for _ in range(3)],
        head_adn_fn=ActDropNormBuilder(norm_fn=torch.nn.BatchNorm1d),
        depth=D,
        upscale_type="transpose",
        padding=1,
        strides=S,
        kernel_sizes=K,
        conv_type=conv_type,
        link_type="identity",
        n_classes=n_classes,
    )
    o = a(i)
    assert list(o.shape) == output_size
