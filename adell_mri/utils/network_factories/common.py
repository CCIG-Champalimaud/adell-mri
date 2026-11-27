"""
Shared helpers for the PyTorch Lightning network factories.
"""

import os
from functools import wraps
from typing import Any, Callable

import torch

from adell_mri.utils.encoder_loading import load_ssl_encoder
from adell_mri.utils.python_logging import get_logger

logger = get_logger(__name__)


class _ViTFeatureExtractor(torch.nn.Module):
    """
    Adapts a ViT encoder so that it returns a spatial feature map.
    """

    def __init__(self, vit: torch.nn.Module):
        """
        Args:
            vit (torch.nn.Module): ViT encoder.
        """
        super().__init__()
        self.vit = vit

    def forward(self, X: torch.Tensor) -> torch.Tensor:
        tokens, _ = self.vit(X)
        return self.vit.tokens_to_feature_map(tokens)


def _feature_extractor_from_encoder(
    encoder: torch.nn.Module, family: str
) -> torch.nn.Module:
    """
    Converts an encoder into a module returning a spatial feature map.

    Args:
        encoder (torch.nn.Module): pretrained encoder.
        family (str): encoder family.

    Returns:
        torch.nn.Module: a module returning a spatial feature map.
    """
    if family in {"vit", "mae"}:
        return _ViTFeatureExtractor(encoder)
    return encoder


def _resolve_single_encoder_checkpoint(encoder_checkpoint, context: str):
    """
    Resolves a single encoder checkpoint from a CLI-style list.

    Args:
        encoder_checkpoint: checkpoint path, dict or list.
        context (str): description used in error messages.

    Returns:
        The single checkpoint value.

    Raises:
        ValueError: if more than one checkpoint is provided.
    """
    if isinstance(encoder_checkpoint, (list, tuple)):
        if len(encoder_checkpoint) != 1:
            raise ValueError(
                f"Only one encoder checkpoint can be used for {context}"
            )
        return encoder_checkpoint[0]
    return encoder_checkpoint


def _build_encoder_feature_extractor(
    encoder_checkpoint, manifest: dict[str, Any], keys: list[str]
) -> torch.nn.Module:
    """
    Builds a feature extractor returning a spatial map for a pretrained encoder.

    Args:
        encoder_checkpoint: self-supervised checkpoint.
        manifest (dict[str, Any]): encoder manifest.
        keys (list[str]): input image keys.

    Returns:
        torch.nn.Module: feature extractor.
    """
    pretrained_encoder = load_ssl_encoder(encoder_checkpoint, manifest=manifest)
    feature_extractor = _feature_extractor_from_encoder(
        pretrained_encoder, manifest["family"]
    )
    if manifest["in_channels"] != len(keys):
        spatial_dimensions = manifest.get("spatial_dim", 3)
        conv_op = (
            torch.nn.Conv2d if spatial_dimensions == 2 else torch.nn.Conv3d
        )
        adapter = conv_op(len(keys), manifest["in_channels"], 1)
        feature_extractor = torch.nn.Sequential(adapter, feature_extractor)
    if manifest.get("image_size") is not None:
        feature_extractor.input_shape = (
            len(keys),
            *[int(x) for x in manifest["image_size"]],
        )
    return feature_extractor


ALLOWED_NET_TYPES = {
    "classification": [
        "unet",
        "vit",
        "factorized_vit",
        "cat",
        "ord",
        "vgg",
    ],
    "segmentation": [
        "unet",
        "brunet",
        "unetpp",
        "unetr",
        "monai_unetr",
        "swin",
        "monai_swin",
        "encoder_unet",
    ],
}


def compile_if_necessary(func: Callable) -> Callable:
    """
    Decorator to compile the output of a function if TORCH_COMPILE is set to
    True.

    Args:
        func: a function outputing a :class:`torch.nn.Module`.

    Returns:
        The compiled :class:`torch.nn.Module`.
    """

    @wraps(func)
    def wrapper(*args, **kwargs):
        model = func(*args, **kwargs)
        tc = os.environ.get("TORCH_COMPILE", "False").lower()
        if tc in ["true", "1"]:
            logger.info(f"Compiling model because TORCH_COMPILE={tc}")
            if not isinstance(model, torch._dynamo.eval_frame.OptimizedModule):
                model = torch.compile(model)
        return model

    return wrapper
