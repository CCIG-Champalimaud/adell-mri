import sys
import os

sys.path.append(os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))

import torch
from adell_mri.modules.self_supervised.ibot import iBOT


def test_ibot():
    # Create a sample input tensor
    batch_size = 2
    channels = 1
    img_size = 224
    patch_size = 16
    attention_dim = 64
    feature_map_dimensions = [
        im // p
        for im, p in zip([img_size, img_size], [patch_size, patch_size])
    ]
    n_tokens = feature_map_dimensions[0] * feature_map_dimensions[1]
    use_class_token = True
    n_patches = 8
    x = torch.randn(batch_size, channels, img_size, img_size)

    # Initialize iBOT model
    model = iBOT(
        backbone_args={
            "image_size": [img_size, img_size],
            "patch_size": [patch_size, patch_size],
            "in_channels": 1,
            "number_of_blocks": 4,
            "attention_dim": attention_dim,
            "embedding_size": attention_dim,
            "use_class_token": use_class_token,
        },
        projection_head_args={"structure": [128]},
        feature_map_dimensions=feature_map_dimensions,
        out_dim=128,
        n_encoder_features=attention_dim,
        min_patch_size=[1, 1],
        max_patch_size=[2, 2],
        n_patches=n_patches,
    )

    # Set model to evaluation mode
    model.eval()

    # Forward pass
    with torch.no_grad():
        reduced_out, out, mask_coords = model.forward_training(x, mask=True)

    # Check output shape
    assert reduced_out.shape == (batch_size, model.out_dim)
    assert out.shape == (batch_size, n_tokens, model.out_dim)

    # Check if output contains valid values
    assert not torch.isnan(reduced_out).any(), "Output contains NaN values"
    assert not torch.isinf(
        reduced_out
    ).any(), "Output contains infinity values"
