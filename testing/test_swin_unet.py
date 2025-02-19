import os
import sys

sys.path.append(os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
import pytest

import torch
from adell_mri.modules.segmentation.unetr import SWINUNet
from copy import deepcopy

h, w, d, c = 16, 16, 16, 1

depths = [[16, 32], [8, 16, 32]]
spatial_dims = [2, 3]


def get_swin_params():
    return deepcopy(
        {
            "image_size": [h, w, d],
            "patch_size": [2, 2, 2],
            "window_size": [8, 8, 8],
            "shift_sizes": [0, 1],
            "n_heads": 4,
            "dropout_rate": 0.1,
            "embed_method": "linear",
            "embedding_size": [8, 16, 32, 64],
        }
    )


param_list = []
for dim in [2, 3]:
    for D in depths:
        for conv_type in ["regular", "resnet"]:
            for embed_method in ["linear", "convolutional"]:
                for pool in ["regular", "irregular"]:
                    param_list.append((D, dim, conv_type, embed_method, pool))


@pytest.mark.parametrize("D,sd,conv_type,embed_method,pool", param_list)
def test_swin_base(D, sd, conv_type, embed_method, pool):
    K = [3 for _ in D]
    if sd == 2:
        i = torch.rand(size=[1, c, h, w])
        output_size = [1, 1, h, w]
    elif sd == 3:
        i = torch.rand(size=[1, c, h, w, d])
        output_size = [1, 1, h, w, d]

    pooling_structure = [2] * len(D)
    if pool == "irregular":
        pooling_structure[0] = [2, 2, 1][:sd]
    swin_params = get_swin_params()
    swin_params["image_size"] = swin_params["image_size"][:sd]
    swin_params["patch_size"] = swin_params["patch_size"][:sd]
    swin_params["window_size"] = swin_params["window_size"][:sd]
    swin_params["embed_method"] = embed_method
    swin_params["embedding_size"] = swin_params["embedding_size"][: len(D)]

    a = SWINUNet(
        **swin_params,
        spatial_dimensions=sd,
        depth=D,
        upscale_type="transpose",
        padding=1,
        kernel_sizes=K,
        conv_type=conv_type,
        norm_type="layer",
        link_type="identity",
        strides=pooling_structure,
    )
    o, bb = a(i)
    assert list(o.shape) == output_size


param_list = []
for dim in [2, 3]:
    for D in depths:
        for conv_type in ["regular", "resnet"]:
            param_list.append((D, dim, conv_type))


@pytest.mark.parametrize("D,sd,conv_type", param_list)
def test_swin_skip(D, sd, conv_type):
    K = [3 for _ in D]
    if sd == 2:
        i = torch.rand(size=[1, c, h, w])
        i_skip = torch.rand(size=[1, c, h, w])
        output_size = [1, 1, h, w]
    elif sd == 3:
        i = torch.rand(size=[1, c, h, w, d])
        i_skip = torch.rand(size=[1, c, h, w, d])
        output_size = [1, 1, h, w, d]

    swin_params = get_swin_params()
    swin_params["image_size"] = swin_params["image_size"][:sd]
    swin_params["patch_size"] = swin_params["patch_size"][:sd]
    swin_params["window_size"] = swin_params["window_size"][:sd]
    swin_params["embedding_size"] = swin_params["embedding_size"][: len(D)]

    a = SWINUNet(
        **swin_params,
        spatial_dimensions=sd,
        depth=D,
        upscale_type="transpose",
        padding=1,
        kernel_sizes=K,
        skip_conditioning=1,
        link_type="conv",
        conv_type=conv_type,
    )
    o, bb = a(i, X_skip_layer=i_skip)
    assert list(o.shape) == output_size


param_list = []
for dim in [2, 3]:
    for D in depths:
        param_list.append((D, dim))


@pytest.mark.parametrize("D,sd", param_list)
def test_swin_skip_feature(D, sd):
    n_features = 4
    K = [3 for _ in D]
    if sd == 2:
        i = torch.rand(size=[2, c, h, w])
        i_skip = torch.rand(size=[2, 1, h, w])
        i_feat = torch.rand(size=[2, n_features])
        output_size = [2, 1, h, w]
    elif sd == 3:
        i = torch.rand(size=[2, c, h, w, d])
        i_skip = torch.rand(size=[2, 1, h, w, d])
        i_feat = torch.rand(size=[2, n_features])
        output_size = [2, 1, h, w, d]

    swin_params = get_swin_params()
    swin_params["image_size"] = swin_params["image_size"][:sd]
    swin_params["patch_size"] = swin_params["patch_size"][:sd]
    swin_params["window_size"] = swin_params["window_size"][:sd]
    swin_params["embedding_size"] = swin_params["embedding_size"][: len(D)]

    a = SWINUNet(
        **swin_params,
        spatial_dimensions=sd,
        depth=D,
        upscale_type="transpose",
        padding=1,
        kernel_sizes=K,
        feature_conditioning=n_features,
        skip_conditioning=1,
        feature_conditioning_params={
            "mean": torch.zeros_like(i_feat),
            "std": torch.ones_like(i_feat),
        },
        link_type="conv",
    )
    o, bb = a(i, X_feature_conditioning=i_feat, X_skip_layer=i_skip)
    assert list(o.shape) == output_size


# test_swin_base(depths[0],2,"regular","convolutional")
