import sys
import os

sys.path.append(os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
import pytest

import torch
from lib.modules.layers.vit import (
    SWINTransformerBlock,
    SWINTransformerBlockStack,
)
from lib.modules.layers.adn_fn import get_adn_fn

image_size = [32, 32, 16]
patch_size = [4, 4, 2]
window_size = [8, 8, 4]
n_channels = 1
attention_dim = 164
hidden_dim = 128
batch_size = 4
n_heads = 2
adn_fn = get_adn_fn(1, "identity", "gelu", 0.1)

args = []
for s in range(4):
    for image_size in [[32, 32], [32, 32, 16]]:
        for patch_size in [[4, 4], [4, 4, 2]]:
            for window_size in [[8, 8], [8, 8, 4]]:
                for embed_method in ["linear", "convolutional"]:
                    if len(image_size) == len(patch_size) and len(
                        image_size
                    ) == len(window_size):
                        args.append(
                            [
                                s,
                                image_size,
                                patch_size,
                                window_size,
                                embed_method,
                            ]
                        )


@pytest.mark.parametrize(
    "s,image_size,patch_size,window_size,embed_method", args
)
def test_swin(s, image_size, patch_size, window_size, embed_method):
    st = SWINTransformerBlock(
        image_size=image_size,
        patch_size=patch_size,
        window_size=window_size,
        n_channels=n_channels,
        attention_dim=attention_dim,
        hidden_dim=hidden_dim,
        shift_size=s,
        n_heads=n_heads,
        dropout_rate=0.1,
        embed_method="linear",
        mlp_structure=[128, 128],
        adn_fn=torch.nn.Identity,
    )
    out = st(torch.rand(size=[batch_size, n_channels, *image_size]), scale=1)
    assert_dim = [batch_size, n_channels, *image_size]
    assert list(out.shape) == assert_dim


args = []
for scale in range(1, 3):
    for image_size in [[32, 32], [32, 32, 16]]:
        for patch_size in [[4, 4], [4, 4, 2]]:
            for window_size in [[8, 8], [8, 8, 4]]:
                for embed_method in ["linear", "convolutional"]:
                    if len(image_size) == len(patch_size) and len(
                        image_size
                    ) == len(window_size):
                        args.append(
                            [
                                scale,
                                image_size,
                                patch_size,
                                window_size,
                                embed_method,
                            ]
                        )


@pytest.mark.parametrize(
    "scale,image_size,patch_size,window_size,embed_method", args
)
def test_swin_stack(scale, image_size, patch_size, window_size, embed_method):
    st = SWINTransformerBlockStack(
        image_size=image_size,
        patch_size=patch_size,
        window_size=window_size,
        n_channels=n_channels,
        attention_dim=attention_dim,
        hidden_dim=hidden_dim,
        shift_sizes=[0, 1, 2],
        n_heads=n_heads,
        dropout_rate=0.1,
        embed_method=embed_method,
        mlp_structure=[512],
        adn_fn=torch.nn.Identity,
    )
    out = st(
        torch.rand(size=[batch_size, n_channels, *image_size]), scale=scale
    )
    new_n_channels = n_channels * scale ** len(image_size)
    new_image_size = [x // scale for x in image_size]
    assert_dim = [batch_size, new_n_channels, *new_image_size]
    assert list(out.shape) == assert_dim
