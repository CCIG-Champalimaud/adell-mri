"""Reusable lightweight checkpoints for tests."""

import torch

from adell_mri.modules.layers.adn_fn import get_adn_fn
from adell_mri.modules.layers.conv_next import ConvNeXt
from adell_mri.modules.layers.res_net import ResNet
from adell_mri.modules.layers.vit import ViT
from adell_mri.modules.segmentation.unet import UNet
from adell_mri.utils.encoder_loading import build_encoder_manifest


def resnet_checkpoint() -> dict:
    """
    Returns a tiny ResNet self-supervised checkpoint.

    Returns:
        dict: checkpoint containing a state dict and encoder manifest.
    """
    backbone_args = {
        "spatial_dim": 2,
        "in_channels": 1,
        "structure": [[8, 8, 3, 1], [16, 16, 3, 1]],
        "maxpool_structure": [2, 2],
        "adn_fn": get_adn_fn(2, "batch", "relu", 0.0),
    }
    network = ResNet(
        backbone_args=backbone_args,
        projection_head_args={
            "in_channels": 16,
            "structure": [16, 8],
            "adn_fn": torch.nn.Identity,
        },
    )
    manifest = build_encoder_manifest(
        net_type="resnet",
        ssl_method="simclr",
        network_config={
            "backbone_args": backbone_args,
            "norm_fn": "batch",
            "act_fn": "relu",
        },
        spatial_dim=2,
        in_channels=1,
    )
    return {"state_dict": network.state_dict(), "encoder_config": manifest}


def resnet3d_checkpoint() -> dict:
    """
    Returns a tiny 3D ResNet self-supervised checkpoint.

    Returns:
        dict: checkpoint containing a state dict and encoder manifest.
    """
    backbone_args = {
        "spatial_dim": 3,
        "in_channels": 1,
        "structure": [[8, 8, 3, 1], [16, 16, 3, 1]],
        "maxpool_structure": [[2, 2, 2], [2, 2, 2]],
        "adn_fn": get_adn_fn(3, "batch", "relu", 0.0),
    }
    network = ResNet(
        backbone_args=backbone_args,
        projection_head_args={
            "in_channels": 16,
            "structure": [16, 8],
            "adn_fn": torch.nn.Identity,
        },
    )
    manifest = build_encoder_manifest(
        net_type="resnet",
        ssl_method="simclr",
        network_config={
            "backbone_args": backbone_args,
            "norm_fn": "batch",
            "act_fn": "relu",
        },
        spatial_dim=3,
        in_channels=1,
    )
    return {"state_dict": network.state_dict(), "encoder_config": manifest}


def convnext_checkpoint() -> dict:
    """
    Returns a tiny ConvNeXt self-supervised checkpoint.

    Returns:
        dict: checkpoint containing a state dict and encoder manifest.
    """
    backbone_args = {
        "spatial_dim": 2,
        "in_channels": 1,
        "structure": [[8, 8, 3, 1], [16, 16, 3, 1]],
        "maxpool_structure": [2, 2],
        "first_layer_stride": 4,
    }
    network = ConvNeXt(
        backbone_args=backbone_args,
        projection_head_args={
            "in_channels": 16,
            "structure": [16, 8],
            "adn_fn": torch.nn.Identity,
        },
    )
    manifest = build_encoder_manifest(
        net_type="convnext",
        ssl_method="simclr",
        network_config={
            "backbone_args": backbone_args,
            "norm_fn": "batch",
            "act_fn": "relu",
        },
        spatial_dim=2,
        in_channels=1,
    )
    return {"state_dict": network.state_dict(), "encoder_config": manifest}


def vit_checkpoint() -> dict:
    """
    Returns a tiny ViT self-supervised checkpoint.

    Returns:
        dict: checkpoint containing a state dict and encoder manifest.
    """
    backbone_args = {
        "image_size": [32, 32],
        "patch_size": [8, 8],
        "in_channels": 1,
        "number_of_blocks": 4,
        "attention_dim": 16,
        "hidden_dim": 16,
        "embedding_size": 16,
        "n_heads": 4,
        "use_class_token": True,
    }
    network = ViT(**backbone_args)
    manifest = build_encoder_manifest(
        net_type="vit",
        ssl_method="dino",
        network_config={
            "backbone_args": backbone_args,
            "norm_fn": "layer",
            "act_fn": "gelu",
        },
        spatial_dim=2,
        in_channels=1,
    )
    state_dict = {
        f"encoder_.{key}": value for key, value in network.state_dict().items()
    }
    return {"state_dict": state_dict, "encoder_config": manifest}


def mae_checkpoint() -> dict:
    """
    Returns a tiny MAE self-supervised checkpoint.

    Returns:
        dict: checkpoint containing a state dict and encoder manifest.
    """
    from adell_mri.modules.self_supervised.autoencoders import (
        ViTMaskedAutoEncoder,
    )

    encoder_args = {
        "number_of_blocks": 2,
        "hidden_dim": 16,
        "n_heads": 4,
        "mlp_structure": [32],
    }
    decoder_args = {
        "number_of_blocks": 1,
        "hidden_dim": 16,
        "n_heads": 4,
        "mlp_structure": [32],
    }
    network = ViTMaskedAutoEncoder(
        image_size=[32, 32],
        patch_size=[8, 8],
        in_channels=1,
        input_dim_size=16,
        encoder_args=encoder_args,
        decoder_args=decoder_args,
    )
    manifest = build_encoder_manifest(
        net_type="vit",
        ssl_method="mae",
        network_config={
            "image_size": [32, 32],
            "patch_size": [8, 8],
            "in_channels": 1,
            "input_dim_size": 16,
            "encoder_args": encoder_args,
            "decoder_args": decoder_args,
        },
        spatial_dim=2,
        in_channels=1,
    )
    state_dict = {
        f"model.{key}": value for key, value in network.state_dict().items()
    }
    return {"state_dict": state_dict, "encoder_config": manifest}


def unet_checkpoint() -> dict:
    """
    Returns a tiny U-Net encoder self-supervised checkpoint.

    Returns:
        dict: checkpoint containing a state dict and encoder manifest.
    """
    unet_args = {
        "spatial_dimensions": 2,
        "in_channels": 1,
        "depth": [8, 16, 32],
        "kernel_sizes": [3, 3, 3],
        "strides": [2, 2, 2],
        "encoder_only": True,
    }
    network = UNet(**unet_args)
    manifest = build_encoder_manifest(
        net_type="unet_encoder",
        ssl_method="simclr",
        network_config=unet_args,
        spatial_dim=2,
        in_channels=1,
    )
    return {"state_dict": network.state_dict(), "encoder_config": manifest}
