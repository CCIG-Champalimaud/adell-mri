import sys
import os

sys.path.append(os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
import pytest

import torch
import numpy as np
from adell_mri.modules.classification import ViTClassifier
from adell_mri.modules.layers.adn_fn import get_adn_fn

image_size = [32, 32, 32]
patch_size = [4, 4, 4]
window_size = [16, 16, 16]
in_channels = 2

attention_dim = 64
hidden_dim = 64
batch_size = 4
n_heads = 4
adn_fn = get_adn_fn(1, "identity", "gelu", 0.1)
n_classes = 4


@pytest.mark.parametrize(
    "embed_method,scale",
    [
        ("linear", 1),
        ("convolutional", 1),
        ("linear", 2),
        ("convolutional", 2),
    ],
)
def test_transformer(embed_method, scale):
    vit = ViTClassifier(
        image_size=image_size,
        patch_size=patch_size,
        in_channels=in_channels,
        number_of_blocks=4,
        attention_dim=attention_dim,
        hidden_dim=hidden_dim,
        n_heads=n_heads,
        dropout_rate=0.1,
        embed_method=embed_method,
        mlp_structure=[64, 64],
        adn_fn=adn_fn,
        n_classes=4,
    )
    im_size = [batch_size] + [in_channels] + image_size
    output_image_size = [batch_size, in_channels * scale ** len(image_size)]
    output_image_size += [x // scale for x in image_size]
    im = torch.rand(im_size)
    out = vit(im)
    assert list(out.shape) == [batch_size, n_classes]


@pytest.mark.parametrize(
    "embed_method,scale",
    [
        ("linear", 1),
        ("convolutional", 1),
        ("linear", 2),
        ("convolutional", 2),
    ],
)
def test_transformer_registers(embed_method, scale):
    vit = ViTClassifier(
        image_size=image_size,
        patch_size=patch_size,
        in_channels=in_channels,
        number_of_blocks=4,
        attention_dim=attention_dim,
        hidden_dim=hidden_dim,
        n_heads=n_heads,
        dropout_rate=0.1,
        embed_method=embed_method,
        mlp_structure=[64, 64],
        adn_fn=adn_fn,
        n_classes=4,
        n_registers=4,
    )
    im_size = [batch_size] + [in_channels] + image_size
    output_image_size = [batch_size, in_channels * scale ** len(image_size)]
    output_image_size += [x // scale for x in image_size]
    im = torch.rand(im_size)
    out = vit(im)
    assert list(out.shape) == [batch_size, n_classes]
