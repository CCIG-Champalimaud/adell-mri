import sys
import os

sys.path.append(os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
import pytest

import torch
from lib.modules.segmentation.unet import BrUNet

h, w, d, c = 32, 32, 20, 1

depths = [[16, 32, 64], [16, 32, 64, 128]]
spatial_dims = [2, 3]
n_input = 2

param_list = []
for dim in [2, 3]:
    for D in depths:
        for conv_type in ["regular", "resnet"]:
            for strides in ["regular", "irregular"]:
                param_list.append((D, dim, conv_type, strides))


@pytest.mark.parametrize("D,sd,conv_type,strides", param_list)
def test_unet_base(D, sd, conv_type, strides):
    K = [3 for _ in D]
    if strides == "irregular":
        if sd == 2:
            S = [[2, 2], *[[2, 1] for _ in D[1:]]]
        elif sd == 3:
            S = [[2, 2, 2], *[[2, 2, 1] for _ in D[1:]]]
    else:
        S = [2 for _ in D]
    if sd == 2:
        i = [torch.rand(size=[1, c, h, w]) for _ in range(n_input)]
        output_size = [1, 1, h, w]
    elif sd == 3:
        i = [torch.rand(size=[1, c, h, w, d]) for _ in range(n_input)]
        output_size = [1, 1, h, w, d]
    a = BrUNet(
        sd,
        n_input,
        depth=D,
        upscale_type="transpose",
        padding=1,
        strides=S,
        kernel_sizes=K,
        conv_type=conv_type,
        link_type="identity",
    )
    weights = [torch.ones([1]) for _ in range(n_input)]
    o, bb = a(i, weights)
    assert list(o.shape) == output_size


@pytest.mark.parametrize("D,sd,conv_type,strides", param_list)
def unet_base_missing(D, sd, conv_type, strides):
    K = [3 for _ in D]
    if strides == "irregular":
        if sd == 2:
            S = [[2, 2], *[[2, 1] for _ in D[1:]]]
        elif sd == 3:
            S = [[2, 2, 2], *[[2, 2, 1] for _ in D[1:]]]
    else:
        S = [2 for _ in D]
    if sd == 2:
        i = [
            [torch.rand(size=[c, h, w]), None],
            [None, torch.rand(size=[c, h, w])],
        ]
        output_size = [2, 1, h, w]
    elif sd == 3:
        i = [
            [torch.rand(size=[c, h, w, d]), None],
            [None, torch.rand(size=[c, h, w, d])],
        ]
        output_size = [2, 1, h, w, d]
    a = BrUNet(
        sd,
        n_input,
        depth=D,
        upscale_type="transpose",
        padding=1,
        strides=S,
        kernel_sizes=K,
        conv_type=conv_type,
        link_type="identity",
    )
    i, weights = BrUNet.fix_input(i)
    o, bb = a(i, weights)
    assert list(o.shape) == output_size


param_list = []
for dim in [2, 3]:
    for D in depths:
        for conv_type in ["regular", "resnet"]:
            param_list.append((D, dim, conv_type))


@pytest.mark.parametrize("D,sd,conv_type", param_list)
def unet_skip(D, sd, conv_type):
    K = [3 for _ in D]
    S = [2 for _ in D]
    if sd == 2:
        i = [torch.rand(size=[1, c, h, w]) for _ in range(n_input)]
        i_skip = torch.rand(size=[1, c, h, w])
        output_size = [1, 1, h, w]
    elif sd == 3:
        i = [torch.rand(size=[1, c, h, w, d]) for _ in range(n_input)]
        i_skip = torch.rand(size=[1, c, h, w, d])
        output_size = [1, 1, h, w, d]
    a = BrUNet(
        sd,
        n_input,
        depth=D,
        upscale_type="transpose",
        padding=1,
        strides=S,
        kernel_sizes=K,
        skip_conditioning=1,
        link_type="conv",
        conv_type=conv_type,
    )
    weights = [torch.ones([1]) for _ in range(n_input)]
    o, bb = a(i, weights, X_skip_layer=i_skip)
    assert list(o.shape) == output_size


param_list = []
for dim in [2, 3]:
    for D in depths:
        param_list.append((D, dim))


@pytest.mark.parametrize("D,sd", param_list)
def unet_skip_feature(D, sd):
    n_features = 4
    K = [3 for _ in D]
    S = [2 for _ in D]
    if sd == 2:
        i = [torch.rand(size=[2, c, h, w]) for _ in range(n_input)]
        i_skip = torch.rand(size=[2, 1, h, w])
        i_feat = torch.rand(size=[2, n_features])
        output_size = [2, 1, h, w]
    elif sd == 3:
        i = [torch.rand(size=[2, c, h, w, d]) for _ in range(n_input)]
        i_skip = torch.rand(size=[2, 1, h, w, d])
        i_feat = torch.rand(size=[2, n_features])
        output_size = [2, 1, h, w, d]
    a = BrUNet(
        sd,
        n_input,
        depth=D,
        upscale_type="transpose",
        padding=1,
        strides=S,
        kernel_sizes=K,
        feature_conditioning=n_features,
        skip_conditioning=1,
        feature_conditioning_params={
            "mean": torch.zeros_like(i_feat),
            "std": torch.ones_like(i_feat),
        },
        link_type="conv",
    )
    weights = [torch.ones([2]) for _ in range(n_input)]
    o, bb = a(i, weights, X_feature_conditioning=i_feat, X_skip_layer=i_skip)
    assert list(o.shape) == output_size


test_unet_base(depths[0], 2, "regular", "irregular")
