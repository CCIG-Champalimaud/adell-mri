"""
Utilities for describing and loading self-supervised encoder checkpoints.

Self-supervised checkpoints contain a training wrapper (an encoder plus
projection/prediction heads and, for some methods, an EMA teacher). The
functions in this module extract a reusable encoder from these checkpoints so
that it can be used by the classification and segmentation entrypoints.
"""

from __future__ import annotations

import re
from copy import deepcopy
from typing import Any

import numpy as np
import torch

from adell_mri.modules.activations import activation_factory
from adell_mri.modules.layers.adn_fn import get_adn_fn, norm_fn_dict
from adell_mri.modules.layers.conv_next import ConvNeXt
from adell_mri.modules.layers.res_net import ResNet
from adell_mri.modules.layers.vit import ViT
from adell_mri.modules.segmentation.unet import UNet
from adell_mri.utils.python_logging import get_logger

logger = get_logger(__name__)

MANIFEST_KEY = "encoder_config"
MANIFEST_FORMAT_VERSION = 1

_FAMILIES = {"resnet", "convnext", "vit", "mae", "unet"}

# Keys from the training wrapper that should never be transferred to an
# encoder.
_EXCLUDED_PATTERNS = (
    r"^ema",
    r"^projection_head",
    r"^projection_",
    r"^prediction_head",
    r"^predictor",
    r"^patch_masker",
    r"^last_layer_",
    r"^head_",
    r"^centers",
    r"^loss",
    r"^criterion",
    r"^decoder",
    r"^decoder_pred",
    r"^classification_layer",
    r"^gaussian_process",
)

_UNET_ARG_KEYS = (
    "spatial_dimensions",
    "depth",
    "kernel_sizes",
    "strides",
    "conv_type",
    "link_type",
    "upscale_type",
    "interpolation",
    "norm_type",
    "dropout_type",
    "padding",
    "dropout_param",
    "activation_fn",
    "in_channels",
    "n_classes",
)


def _activation_name(activation: Any) -> Any:
    """
    Converts an activation class into its registered name.

    Args:
        activation (Any): activation class or string.

    Returns:
        Any: the registered activation name or the input value.
    """
    if isinstance(activation, str):
        return activation
    for name, fn in activation_factory.items():
        if fn is activation:
            return name
    return activation


def _activation_from_name(activation: Any) -> Any:
    """
    Converts a registered activation name into a class.

    Args:
        activation (Any): activation name, class or ``None``.

    Returns:
        Any: the activation class if a name was provided, otherwise the
            input value.
    """
    if isinstance(activation, str) and activation in activation_factory:
        return activation_factory[activation]
    return activation


def _norm_name(norm_fn: Any) -> Any:
    """
    Converts a normalization class into its registered name.

    Args:
        norm_fn (Any): normalization class or ``None``.

    Returns:
        Any: the registered normalization name or the input value.
    """
    if norm_fn is None or isinstance(norm_fn, str):
        return norm_fn
    for name, fns in norm_fn_dict.items():
        if any(fn is norm_fn for fn in fns.values()):
            return name
    return norm_fn


def build_encoder_manifest(
    net_type: str,
    ssl_method: str,
    network_config: dict[str, Any],
    spatial_dim: int | None = None,
    in_channels: int | None = None,
) -> dict[str, Any]:
    """
    Builds a checkpoint manifest describing the encoder of an SSL model.

    Args:
        net_type (str): encoder type (``resnet``, ``convnext``, ``unet_encoder``
            or ``vit``).
        ssl_method (str): self-supervised method used to train the encoder.
        network_config (dict[str, Any]): parsed network configuration. This
            should be the configuration used to instantiate the SSL network.
        spatial_dim (int, optional): number of spatial dimensions. Inferred
            from the configuration when absent. Defaults to None.
        in_channels (int, optional): number of input channels. Inferred from
            the configuration when absent. Defaults to None.

    Returns:
        dict[str, Any]: a serialisable manifest. The manifest contains the
            family of the encoder, the arguments needed to rebuild it and the
            prefix used by its parameters in the checkpoint.
    """
    config = deepcopy(network_config)

    if net_type == "unet_encoder":
        return _build_unet_manifest(
            ssl_method, config, spatial_dim, in_channels
        )

    backbone_args = dict(config.get("backbone_args", {}))

    if net_type == "convnext" or ssl_method in {
        "dino",
        "ibot",
        "ijepa",
        "mae",
    }:
        if ssl_method == "mae":
            return _build_mae_manifest(ssl_method, config, in_channels)
        backbone_args.pop("res_type", None)
        return _build_conv_or_vit_manifest(
            net_type,
            ssl_method,
            config,
            backbone_args,
            spatial_dim,
            in_channels,
        )

    return _build_conv_or_vit_manifest(
        net_type, ssl_method, config, backbone_args, spatial_dim, in_channels
    )


def _build_conv_or_vit_manifest(
    net_type: str,
    ssl_method: str,
    config: dict[str, Any],
    backbone_args: dict[str, Any],
    spatial_dim: int | None,
    in_channels: int | None,
) -> dict[str, Any]:
    adn_fn = backbone_args.pop("adn_fn", None)
    norm_fn = config.get("norm_fn")
    act_fn = config.get("act_fn")
    dropout_param = config.get("dropout_param", 0.0)
    if adn_fn is not None:
        norm_fn = norm_fn or _norm_name(getattr(adn_fn, "norm_fn", None))
        act_fn = act_fn or _activation_name(getattr(adn_fn, "act_fn", None))
        dropout_param = getattr(adn_fn, "dropout_param", dropout_param)

    if "patch_size" in backbone_args:
        family = "vit"
        spatial_dim = spatial_dim or len(backbone_args["patch_size"])
        prefix = "encoder_."
    else:
        res_type = backbone_args.get("res_type", "resnet")
        if net_type == "convnext" or res_type in {"convnext", "convnextv2"}:
            family = "convnext"
            backbone_args.pop("res_type", None)
        else:
            family = "resnet"
        spatial_dim = spatial_dim or backbone_args.get("spatial_dim", 2)
        prefix = "backbone."

    if family != "vit" and adn_fn is None:
        if config.get("norm_fn") is None:
            logger.warning(
                "Could not recover the encoder normalization; defaulting to "
                "'batch'. Pass --config_file to the conversion script if the "
                "checkpoint was trained with a different normalization."
            )
        if config.get("act_fn") is None:
            logger.warning(
                "Could not recover the encoder activation; defaulting to "
                "'swish'. Pass --config_file to the conversion script if the "
                "checkpoint was trained with a different activation."
            )

    manifest = {
        "format_version": MANIFEST_FORMAT_VERSION,
        "family": family,
        "net_type": net_type,
        "ssl_method": ssl_method,
        "spatial_dim": spatial_dim,
        "in_channels": backbone_args.get("in_channels", in_channels),
        "backbone_args": backbone_args,
        "norm_fn": norm_fn or "batch",
        "act_fn": act_fn or "swish",
        "dropout_param": dropout_param,
        "state_dict_prefix": prefix,
    }
    if family == "vit":
        manifest["image_size"] = backbone_args.get("image_size")
        manifest["return_at"] = config.get("return_at")
    return manifest


def _build_unet_manifest(
    ssl_method: str,
    config: dict[str, Any],
    spatial_dim: int | None,
    in_channels: int | None,
) -> dict[str, Any]:
    unet_args = {key: config[key] for key in _UNET_ARG_KEYS if key in config}
    if "activation_fn" in unet_args:
        unet_args["activation_fn"] = _activation_name(
            unet_args["activation_fn"]
        )
    unet_args.pop("n_classes", None)
    return {
        "format_version": MANIFEST_FORMAT_VERSION,
        "family": "unet",
        "net_type": "unet_encoder",
        "ssl_method": ssl_method,
        "spatial_dim": spatial_dim or unet_args.get("spatial_dimensions", 3),
        "in_channels": unet_args.get("in_channels", in_channels),
        "unet_args": unet_args,
        "state_dict_prefix": "",
    }


def _build_mae_manifest(
    ssl_method: str,
    config: dict[str, Any],
    in_channels: int | None,
) -> dict[str, Any]:
    encoder_args = dict(config.get("encoder_args", {}))
    return {
        "format_version": MANIFEST_FORMAT_VERSION,
        "family": "mae",
        "net_type": "vit",
        "ssl_method": ssl_method,
        "spatial_dim": len(config.get("patch_size", [16, 16])),
        "in_channels": config.get("in_channels", in_channels),
        "mae_args": {
            "image_size": config.get("image_size"),
            "patch_size": config.get("patch_size"),
            "in_channels": config.get("in_channels", in_channels),
            "input_dim_size": config.get("input_dim_size"),
            "embed_method": config.get("embed_method", "linear"),
            "dropout_rate": encoder_args.get("dropout_rate", 0.0),
            "encoder_args": encoder_args,
        },
        "state_dict_prefix": "model.",
    }


def read_encoder_manifest(
    checkpoint: str | dict[str, Any],
) -> dict[str, Any]:
    """
    Reads the encoder manifest from a checkpoint.

    The function supports checkpoints produced by :func:`build_encoder_manifest`
    as well as older checkpoints, for which the manifest is inferred from the
    stored hyperparameters and the state dict.

    Args:
        checkpoint (str | dict[str, Any]): path to a checkpoint or a loaded
            checkpoint dictionary.

    Returns:
        dict[str, Any]: the encoder manifest.

    Raises:
        ValueError: if the checkpoint does not contain enough information to
            identify the encoder.
    """
    if isinstance(checkpoint, str):
        ckpt = torch.load(checkpoint, map_location="cpu", weights_only=False)
    else:
        ckpt = checkpoint

    if MANIFEST_KEY in ckpt:
        return ckpt[MANIFEST_KEY]

    metadata = _extract_metadata(ckpt)
    if metadata is not None and "encoder_config" in metadata:
        return metadata["encoder_config"]

    state_dict = ckpt.get("state_dict", {})
    family = infer_encoder_family(state_dict)
    hyper_parameters = ckpt.get("hyper_parameters", {})

    if metadata is not None and "network_config" in metadata:
        network_config = metadata["network_config"]
        if family == "unet":
            return _build_unet_manifest(
                metadata.get("ssl_method", "unknown"),
                network_config,
                network_config.get("spatial_dimensions"),
                network_config.get("in_channels"),
            )
        return _build_conv_or_vit_manifest(
            metadata.get("net_type", family),
            metadata.get("ssl_method", "unknown"),
            network_config,
            dict(network_config.get("backbone_args", {})),
            network_config.get("spatial_dim"),
            network_config.get("in_channels"),
        )

    if family == "unet" and "depth" in hyper_parameters:
        return _build_unet_manifest(
            hyper_parameters.get("ssl_method", "unknown"),
            hyper_parameters,
            hyper_parameters.get("spatial_dimensions"),
            hyper_parameters.get("in_channels"),
        )

    backbone_args = hyper_parameters.get("backbone_args")
    if backbone_args is not None:
        net_type = "convnext" if family == "convnext" else "vit"
        if family == "resnet":
            net_type = "resnet"
        return _build_conv_or_vit_manifest(
            net_type,
            hyper_parameters.get("ssl_method", "unknown"),
            hyper_parameters,
            dict(backbone_args),
            None,
            hyper_parameters.get("in_channels"),
        )

    if family == "mae":
        return _build_mae_manifest(
            hyper_parameters.get("ssl_method", "unknown"),
            hyper_parameters,
            hyper_parameters.get("in_channels"),
        )

    raise ValueError(
        "Could not determine the encoder configuration from the checkpoint. "
        "Re-train the self-supervised model with this version of adell-mri or "
        "provide an encoder_config entry."
    )


def _extract_metadata(checkpoint: dict[str, Any]) -> dict[str, Any] | None:
    callbacks = checkpoint.get("callbacks", {})
    if not isinstance(callbacks, dict):
        return None
    for callback in callbacks.values():
        if isinstance(callback, dict) and "metadata" in callback:
            metadata = callback["metadata"]
            if isinstance(metadata, dict):
                return metadata
    return None


def infer_encoder_family(state_dict: dict[str, torch.Tensor]) -> str:
    """
    Infers the encoder family from a state dict.

    Args:
        state_dict (dict[str, torch.Tensor]): model state dict.

    Returns:
        str: one of ``resnet``, ``convnext``, ``vit``, ``mae`` or ``unet``.

    Raises:
        ValueError: if the family cannot be inferred.
    """
    keys = list(state_dict)
    if any(key.startswith("model.proj.") for key in keys):
        return "mae"
    if any(key.startswith("encoder_.") for key in keys):
        return "vit"
    if any(key.startswith("encoding_operations.") for key in keys):
        return "unet"
    if any(key.startswith("backbone.") for key in keys):
        for key, value in state_dict.items():
            if key.endswith("backbone.input_layer.0.weight"):
                kernel_size = value.shape[2:]
                if kernel_size and all(size == 4 for size in kernel_size):
                    return "convnext"
                return "resnet"
        return "resnet"
    raise ValueError(
        "Could not infer the encoder family from the checkpoint state dict"
    )


def instantiate_encoder(manifest: dict[str, Any]) -> torch.nn.Module:
    """
    Instantiates an encoder from a manifest.

    Args:
        manifest (dict[str, Any]): encoder manifest.

    Returns:
        torch.nn.Module: the encoder module.

    Raises:
        ValueError: if the manifest family is not supported.
    """
    family = manifest["family"]
    if family == "resnet":
        backbone_args = _backbone_args_with_adn(manifest)
        return ResNet(
            backbone_args=backbone_args, projection_head_args=None
        ).backbone
    if family == "convnext":
        backbone_args = _backbone_args_with_adn(manifest)
        return ConvNeXt(
            backbone_args=backbone_args, projection_head_args=None
        ).backbone
    if family == "vit":
        return ViT(**manifest["backbone_args"])
    if family == "mae":
        return _instantiate_mae_encoder(manifest)
    if family == "unet":
        unet_args = dict(manifest["unet_args"])
        if "activation_fn" in unet_args:
            unet_args["activation_fn"] = _activation_from_name(
                unet_args["activation_fn"]
            )
        unet_args["encoder_only"] = True
        return UNet(**unet_args)
    raise ValueError(f"Unsupported encoder family: {family}")


def _backbone_args_with_adn(manifest: dict[str, Any]) -> dict[str, Any]:
    backbone_args = dict(manifest["backbone_args"])
    backbone_args["adn_fn"] = get_adn_fn(
        manifest["spatial_dim"],
        manifest.get("norm_fn", "batch"),
        manifest.get("act_fn", "swish"),
        dropout_param=manifest.get("dropout_param", 0.0),
    )
    return backbone_args


def _instantiate_mae_encoder(manifest: dict[str, Any]) -> torch.nn.Module:
    mae_args = manifest["mae_args"]
    encoder_args = mae_args["encoder_args"]
    n_features = int(np.prod(mae_args["patch_size"])) * mae_args["in_channels"]
    vit_args = {
        "image_size": mae_args["image_size"],
        "patch_size": mae_args["patch_size"],
        "in_channels": mae_args["in_channels"],
        "number_of_blocks": encoder_args["number_of_blocks"],
        "attention_dim": n_features,
        "hidden_dim": encoder_args.get("hidden_dim"),
        "embedding_size": None,
        "n_heads": encoder_args["n_heads"],
        "mlp_structure": encoder_args.get("mlp_structure", [128]),
        "dropout_rate": mae_args.get("dropout_rate", 0.0),
        "embed_method": mae_args.get("embed_method", "linear"),
        "use_class_token": False,
        "use_pos_embed": True,
    }
    return ViT(**vit_args)


def extract_encoder_state_dict(
    state_dict: dict[str, torch.Tensor], manifest: dict[str, Any]
) -> dict[str, torch.Tensor]:
    """
    Extracts the encoder weights from a self-supervised state dict.

    Args:
        state_dict (dict[str, torch.Tensor]): checkpoint state dict.
        manifest (dict[str, Any]): encoder manifest.

    Returns:
        dict[str, torch.Tensor]: state dict compatible with the encoder
            returned by :func:`instantiate_encoder`.
    """
    family = manifest["family"]
    prefix = manifest.get("state_dict_prefix", "")

    out = {}
    for key, value in state_dict.items():
        if any(re.search(pattern, key) for pattern in _EXCLUDED_PATTERNS):
            continue

        if family == "mae":
            if key.startswith("model.proj."):
                out["embedding." + key[len("model.proj.") :]] = value
            elif key.startswith("model.encoder."):
                out["tbs." + key[len("model.encoder.") :]] = value
            continue

        if prefix and key.startswith(prefix):
            out[key[len(prefix) :]] = value
        elif (family != "vit" and key.startswith("encoder_.")) or (
            family != "resnet" and key.startswith("backbone.")
        ):
            continue
        else:
            out[key] = value
    return out


def load_ssl_encoder(
    checkpoint: str | dict[str, Any],
    manifest: dict[str, Any] | None = None,
    strict: bool = False,
) -> torch.nn.Module:
    """
    Loads a self-supervised encoder from a checkpoint.

    Args:
        checkpoint (str | dict[str, Any]): path to a checkpoint or a loaded
            checkpoint dictionary.
        manifest (dict[str, Any], optional): encoder manifest. Read from the
            checkpoint when absent. Defaults to None.
        strict (bool, optional): whether to use strict state dict loading.
            Defaults to False.

    Returns:
        torch.nn.Module: the loaded encoder.

    Raises:
        RuntimeError: if the state dict contains keys that do not belong to
            the encoder.
    """
    if isinstance(checkpoint, str):
        ckpt = torch.load(checkpoint, map_location="cpu", weights_only=False)
    else:
        ckpt = checkpoint
    manifest = manifest or read_encoder_manifest(ckpt)
    encoder = instantiate_encoder(manifest)
    state_dict = extract_encoder_state_dict(ckpt["state_dict"], manifest)
    result = encoder.load_state_dict(state_dict, strict=strict)
    if result.unexpected_keys:
        raise RuntimeError(
            "The following keys from the checkpoint do not belong to the "
            f"encoder: {result.unexpected_keys}"
        )
    logger.info(
        "Loaded SSL encoder (%s) with missing keys: %s",
        manifest["family"],
        result.missing_keys,
    )
    return encoder


def load_encoder_into(
    model: torch.nn.Module,
    checkpoint: str | dict[str, Any],
    prefix: str = "",
    freeze: bool = False,
) -> torch.nn.Module:
    """
    Loads an SSL encoder into a target module.

    Args:
        model (torch.nn.Module): target module.
        checkpoint (str | dict[str, Any]): SSL checkpoint.
        prefix (str, optional): prefix prepended to the encoder state dict
            keys before loading. Defaults to "".
        freeze (bool, optional): whether to freeze the encoder parameters.
            Defaults to False.

    Returns:
        torch.nn.Module: the target module (updated in place).
    """
    encoder = load_ssl_encoder(checkpoint)
    state_dict = {
        prefix + key: value for key, value in encoder.state_dict().items()
    }
    result = model.load_state_dict(state_dict, strict=False)
    if result.unexpected_keys:
        raise RuntimeError(
            "The following encoder keys do not belong to the target model: "
            f"{result.unexpected_keys}"
        )
    if freeze:
        for parameter in model.parameters():
            parameter.requires_grad = False
    return model
