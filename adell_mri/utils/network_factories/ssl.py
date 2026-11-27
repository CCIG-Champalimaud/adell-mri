"""
Self-supervised network factories.
"""

from typing import Any, Callable

import torch
from lightning.pytorch import LightningModule

from adell_mri.modules.layers.adn_fn import get_adn_fn
from adell_mri.modules.self_supervised.pl import (
    DINOPL,
    IJEPA,
    IJEPAPL,
    BarlowTwinsPL,
    ConvNeXt,
    ResNet,
    SelfSLConvNeXtPL,
    SelfSLResNetPL,
    SelfSLUNetPL,
    UNet,
    ViTMaskedAutoEncoderPL,
    iBOTPL,
)
from adell_mri.utils.network_factories.common import compile_if_necessary
from adell_mri.utils.optimizer_factory import OPTIMIZER_EPS_DEFAULT
from adell_mri.utils.python_logging import get_logger
from adell_mri.utils.torch_utils import ExponentialMovingAverage

logger = get_logger(__name__)


def _default_resnet_backbone_args() -> dict[str, Any]:
    """
    Returns the default ResNet backbone arguments for SSL methods.

    Returns:
        dict[str, Any]: default backbone arguments.
    """
    return {
        "spatial_dim": 2,
        "in_channels": 1,
        "structure": [
            (64, 64, 3, 2),
            (128, 128, 3, 2),
            (256, 256, 3, 2),
            (512, 512, 3, 2),
        ],
        "maxpool_structure": [2, 2, 2, 2],
        "adn_fn": torch.nn.Identity,
        "res_type": "resnet",
    }


def _default_resnet_projection_head_args() -> dict[str, Any]:
    """
    Returns the default ResNet projection head arguments for SSL methods.

    Returns:
        dict[str, Any]: default projection head arguments.
    """
    return {
        "in_channels": 512,
        "structure": [512, 128],
        "adn_fn": torch.nn.Identity,
    }


def _default_resnet_prediction_head_args() -> dict[str, Any]:
    """
    Returns the default ResNet prediction head arguments for SSL methods.

    Returns:
        dict[str, Any]: default prediction head arguments.
    """
    return {
        "in_channels": 128,
        "structure": [512, 128],
        "adn_fn": torch.nn.Identity,
    }


def _vit_backbone_defaults() -> dict[str, Any]:
    """
    Returns the default ViT backbone arguments for the SSL methods that
    require a ViT backbone (IJEPA, DINO, iBOT).

    Returns:
        dict[str, Any]: default backbone arguments.
    """
    return {
        "image_size": [224, 224],
        "patch_size": [16, 16],
        "in_channels": 1,
        "number_of_blocks": 4,
        "attention_dim": 96,
        "embedding_size": 96,
        "n_heads": 3,
    }


def _compute_vit_dims(
    backbone_args: dict[str, Any],
) -> tuple[list[int], int]:
    """
    Computes the feature map dimensions and the number of encoder features
    from ViT backbone arguments.

    Args:
        backbone_args (dict[str, Any]): ViT backbone arguments containing
            ``image_size``, ``patch_size`` and either ``embedding_size`` or
            ``attention_dim``.

    Returns:
        tuple[list[int], int]: feature map dimensions (image size divided by
            patch size, element-wise) and the number of encoder features.
    """
    feature_map_dimensions = [
        s // p
        for s, p in zip(
            backbone_args["image_size"], backbone_args["patch_size"]
        )
    ]
    n_encoder_features = backbone_args.get(
        "embedding_size", backbone_args["attention_dim"]
    )
    return feature_map_dimensions, n_encoder_features


SSL_METHODS_WITH_EMA = {"byol", "ijepa", "dino", "ibot"}
SSL_METHODS_WITH_STOP_GRADIENT = {"byol", "ijepa", "dino", "ibot"}


def get_ema(ssl_method: str, n_steps: int) -> ExponentialMovingAverage | None:
    """
    Returns the exponential moving average teacher for a given SSL method.

    Args:
        ssl_method (str): the SSL method.
        n_steps (int): the number of optimization steps.

    Returns:
        ExponentialMovingAverage | None: the EMA module if the method uses
            an EMA teacher, otherwise ``None``.
    """
    if ssl_method not in SSL_METHODS_WITH_EMA:
        return None
    decay = 0.99 if ssl_method == "ijepa" else 0.996
    return ExponentialMovingAverage(
        decay=decay, final_decay=1.0, n_steps=n_steps
    )


def get_stop_gradients(ssl_method: str) -> bool:
    """
    Returns whether gradients should be stopped for a given SSL method.

    Args:
        ssl_method (str): the SSL method.

    Returns:
        bool: ``True`` if gradients should be stopped, ``False`` otherwise.
    """
    return ssl_method in SSL_METHODS_WITH_STOP_GRADIENT


@compile_if_necessary
def get_ssl_network(
    train_loader_call: Callable,
    max_epochs: int,
    max_steps_optim: int,
    warmup_steps: int,
    ssl_method: str,
    ema: torch.nn.Module,
    net_type: str,
    network_config: dict[str, Any],
    stop_gradient: bool,
    optimizer_eps: float = OPTIMIZER_EPS_DEFAULT,
) -> LightningModule:
    """
    Returns a SSL network.

    Args:
        train_loader_call (Callable): the training data loader.
        max_epochs (int): the maximum number of epochs.
        max_steps_optim (int): the maximum number of steps for optimization.
        warmup_steps (int): the number of warmup steps.
        ssl_method (str): the SSL method to use.
        ema (torch.nn.Module): the exponential moving average.
        net_type (str): the type of network to use.
        network_config (dict[str, Any]): the configuration of the network.
        stop_gradient (bool): whether to stop gradient.

    Returns:
        A :class:`LightningModule` SSL network.
    """
    # Common configuration for SSL methods
    common_params = {
        "training_dataloader_call": train_loader_call,
        "n_epochs": max_epochs,
        "n_steps": max_steps_optim,
        "warmup_steps": warmup_steps,
        "ema": ema,
        "batch_size": network_config.get("batch_size", 32),
        "optimizer_eps": optimizer_eps,
    }

    # pass the optimisation hyperparameters through when present in the
    # configuration file (keeps the module-level defaults otherwise)
    for key in ["learning_rate", "weight_decay"]:
        if key in network_config:
            common_params[key] = network_config[key]

    if ssl_method in ["simclr", "byol", "vicreg", "vicregl"]:
        # These methods use the standard ResNet architecture
        config = {
            "backbone_args": network_config.get(
                "backbone_args", _default_resnet_backbone_args()
            ),
            "projection_head_args": network_config.get(
                "projection_head_args",
                _default_resnet_projection_head_args(),
            ),
            "prediction_head_args": (
                None
                if ssl_method == "simclr"
                else network_config.get(
                    "prediction_head_args",
                    _default_resnet_prediction_head_args(),
                )
            ),
            "ssl_method": ssl_method,
            "aug_image_key_1": "augmented_image_1",
            "aug_image_key_2": "augmented_image_2",
            "stop_gradient": stop_gradient,
            "temperature": network_config.get("temperature", 0.1),
            "vic_reg_loss_params": network_config.get(
                "vic_reg_loss_params", {}
            ),
        }
        ssl = SelfSLResNetPL(**{**common_params, **config})

    elif ssl_method == "ijepa":
        # IJEPA specific configuration (ViT backbone + transformer predictor)
        if net_type != "vit":
            raise TypeError(
                "IJEPA only supports net_type='vit', got %s" % net_type
            )
        backbone_args: dict = network_config.get(
            "backbone_args", _vit_backbone_defaults()
        )
        feature_map_dimensions, n_encoder_features = _compute_vit_dims(
            backbone_args
        )
        predictor_head_args: dict = network_config.get(
            "projection_head_args",
            {
                "number_of_blocks": 2,
                "attention_dim": n_encoder_features,
                "hidden_dim": n_encoder_features,
                "n_heads": 3,
            },
        )
        config = {
            "image_key": "image",
            "backbone_args": backbone_args,
            "projection_head_args": predictor_head_args,
            "feature_map_dimensions": feature_map_dimensions,
            "n_encoder_features": n_encoder_features,
            "min_patch_size": network_config.get("min_patch_size", [4, 4]),
            "max_patch_size": network_config.get("max_patch_size", [8, 8]),
            "n_patches": network_config.get("n_patches", 1),
            "n_masked_patches": network_config.get("n_masked_patches", 4),
            "predictor_dim": network_config.get("predictor_dim", None),
            "stop_gradient": stop_gradient,
            "optimizer_eps": optimizer_eps,
        }
        ssl = IJEPAPL(**{**common_params, **config})

    elif ssl_method == "mae":
        # MAE specific configuration
        encoder_args: dict = network_config.get("encoder_args", {})
        decoder_args: dict = network_config.get("decoder_args", {})
        config = {
            "image_key": "image",
            "image_size": encoder_args.get("image_size", (224, 224)),
            "patch_size": encoder_args.get("patch_size", (16, 16)),
            "in_channels": encoder_args.get("in_channels", 1),
            "input_dim_size": encoder_args.get("embed_dim", 96),
            "encoder_args": encoder_args,
            "decoder_args": decoder_args,
            "mask_fraction": network_config.get("mask_fraction", 0.75),
            "optimizer_eps": optimizer_eps,
        }
        del common_params["ema"]
        ssl = ViTMaskedAutoEncoderPL(**{**common_params, **config})

    elif ssl_method == "dino":
        # DINO specific configuration (ViT backbone + MLP projection head)
        if net_type != "vit":
            raise TypeError(
                "DINO only supports net_type='vit', got %s" % net_type
            )
        backbone_args: dict = network_config.get(
            "backbone_args", _vit_backbone_defaults()
        )
        projection_head_args: dict = network_config.get(
            "projection_head_args",
            {"structure": [512, 256, 128]},
        )
        projection_head_args.setdefault(
            "adn_fn", get_adn_fn(1, "identity", "gelu", dropout_param=0.0)
        )
        config = {
            "aug_image_key_1": "augmented_image_1",
            "aug_image_key_2": "augmented_image_2",
            "backbone_args": backbone_args,
            "projection_head_args": projection_head_args,
            "out_dim": network_config.get("out_dim", 65536),
            "temperature": network_config.get("temperature", 0.1),
            "centers_m": network_config.get("centers_m", 0.9),
            "teacher_score_method": network_config.get(
                "teacher_score_method", "center"
            ),
            "teacher_temperature": network_config.get(
                "teacher_temperature", 0.04
            ),
            "teacher_temperature_final": network_config.get(
                "teacher_temperature_final", 0.07
            ),
            "teacher_temperature_warmup_epochs": network_config.get(
                "teacher_temperature_warmup_epochs", 30
            ),
            "freeze_last_layer_epochs": network_config.get(
                "freeze_last_layer_epochs", 1
            ),
            "stop_gradient": stop_gradient,
            "optimizer_eps": optimizer_eps,
        }
        ssl = DINOPL(**{**common_params, **config})

    elif ssl_method == "ibot":
        # iBOT specific configuration (ViT backbone + MLP projection head)
        if net_type != "vit":
            raise TypeError(
                "iBOT only supports net_type='vit', got %s" % net_type
            )
        backbone_args: dict = network_config.get(
            "backbone_args", _vit_backbone_defaults()
        )
        feature_map_dimensions, n_encoder_features = _compute_vit_dims(
            backbone_args
        )
        projection_head_args: dict = network_config.get(
            "projection_head_args",
            {"structure": [512, 256, 128]},
        )
        projection_head_args.setdefault(
            "adn_fn", get_adn_fn(1, "identity", "gelu", dropout_param=0.0)
        )
        config = {
            "aug_image_key_1": "augmented_image_1",
            "aug_image_key_2": "augmented_image_2",
            "backbone_args": backbone_args,
            "projection_head_args": projection_head_args,
            "out_dim": network_config.get("out_dim", 65536),
            "feature_map_dimensions": feature_map_dimensions,
            "n_encoder_features": n_encoder_features,
            "min_patch_size": network_config.get("min_patch_size", [4, 4]),
            "max_patch_size": network_config.get("max_patch_size", [8, 8]),
            "temperature": network_config.get("temperature", 0.1),
            "centers_m": network_config.get("centers_m", 0.9),
            "teacher_score_method": network_config.get(
                "teacher_score_method", "center"
            ),
            "stop_gradient": stop_gradient,
            "optimizer_eps": optimizer_eps,
        }
        ssl = iBOTPL(**{**common_params, **config})

    elif ssl_method == "barlow":
        # barlow twins-specific configuration (resnet backbone, projector
        # head whose output is used directly in the cross-correlation loss)
        backbone_args: dict = network_config.get(
            "backbone_args", _default_resnet_backbone_args()
        )
        projection_head_args: dict = network_config.get(
            "projection_head_args",
            {"in_channels": 512, "structure": [2048, 8192]},
        )
        config = {
            "image_key": "augmented_image_1",
            "augmented_image_key": "augmented_image_2",
            "backbone_args": backbone_args,
            "projection_head_args": projection_head_args,
            "loss_lam": network_config.get("loss_lam", 0.005),
            "optimizer_eps": optimizer_eps,
        }
        ssl = BarlowTwinsPL(
            training_dataloader_call=train_loader_call,
            learning_rate=network_config.get("learning_rate", 0.001),
            weight_decay=network_config.get("weight_decay", 0.005),
            batch_size=network_config.get("batch_size", 32),
            **config,
        )

    else:
        boilerplate = {
            "training_dataloader_call": train_loader_call,
            "aug_image_key_1": "augmented_image_1",
            "aug_image_key_2": "augmented_image_2",
            "box_key_1": "box_1",
            "box_key_2": "box_2",
            "n_epochs": max_epochs,
            "n_steps": max_steps_optim,
            "warmup_steps": warmup_steps,
            "ssl_method": ssl_method,
            "ema": ema,
            "stop_gradient": stop_gradient,
            "temperature": 0.1,
            "optimizer_eps": optimizer_eps,
        }
        if net_type == "unet_encoder":
            ssl = SelfSLUNetPL(**boilerplate, **network_config)
        elif net_type == "convnext":
            network_config["backbone_args"] = {
                k: network_config["backbone_args"][k]
                for k in network_config["backbone_args"]
                if k not in ["res_type"]
            }
            ssl = SelfSLConvNeXtPL(**boilerplate, **network_config)
        else:
            ssl = SelfSLResNetPL(**boilerplate, **network_config)

    return ssl


@compile_if_necessary
def get_ssl_network_no_pl(
    ssl_method: str, net_type: str, network_config: dict[str, Any]
) -> torch.nn.Module:
    """
    Returns a SSL network.

    Args:
        ssl_method (str): the SSL method to use.
        net_type (str): the type of network to use.
        network_config (dict[str, Any]): the configuration of the network.

    Returns:
        A :class:`torch.nn.Module` SSL network.
    """
    if ssl_method == "ijepa":
        ssl = IJEPA(**network_config)

    else:
        if net_type == "unet_encoder":
            ssl = UNet(**network_config)
        elif net_type == "vit":
            from adell_mri.modules.layers.vit import ViT

            ssl = ViT(**network_config)
        elif net_type == "convnext":
            network_config["backbone_args"] = {
                k: network_config["backbone_args"][k]
                for k in network_config["backbone_args"]
                if k not in ["res_type"]
            }
            ssl = ConvNeXt(**network_config)
        else:
            ssl = ResNet(**network_config)

    return ssl
