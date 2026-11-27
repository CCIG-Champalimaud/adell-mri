"""
Shared helpers for the segmentation entrypoints.
"""

import torch


def if_none_else(x, obj):
    """
    Returns ``obj`` when ``x`` is None, otherwise returns ``x``.

    Args:
        x: value to check.
        obj: fallback value.

    Returns:
        The input value or the fallback.
    """
    if x is None:
        return obj
    return x


def inter_size(a, b) -> int:
    """
    Returns the size of the intersection between two collections.

    Args:
        a: first collection.
        b: second collection.

    Returns:
        int: number of shared elements.
    """
    return len(set.intersection(set(a), set(b)))


def validate_encoder_checkpoint(args, require_checkpoint: bool = False) -> None:
    """
    Validates the encoder checkpoint and segmentation network type.

    Args:
        args: parsed segmentation arguments.
        require_checkpoint (bool, optional): whether an encoder checkpoint is
            mandatory for ``net_type == "encoder_unet"``. Defaults to False.

    Raises:
        ValueError: if the encoder checkpoint is incompatible with the network
            type or is missing when required.
    """
    if args.encoder_checkpoint is not None and args.net_type != "encoder_unet":
        raise ValueError(
            "--encoder_checkpoint can only be used with "
            "--net_type encoder_unet"
        )
    if (
        require_checkpoint
        and args.net_type == "encoder_unet"
        and args.encoder_checkpoint is None
    ):
        raise ValueError(
            "--net_type encoder_unet requires --encoder_checkpoint"
        )


def read_encoder_config_from_checkpoint(checkpoint):
    """
    Reads the encoder manifest from a trained checkpoint.

    Args:
        checkpoint: checkpoint path, list of paths or None.

    Returns:
        dict: encoder manifest.

    Raises:
        ValueError: if no checkpoint is provided or it has no manifest.
    """
    if checkpoint is None:
        raise ValueError(
            "--net_type encoder_unet requires --encoder_checkpoint or a "
            "trained checkpoint containing an encoder_config entry"
        )
    path = (
        checkpoint[0] if isinstance(checkpoint, (list, tuple)) else checkpoint
    )
    ckpt = torch.load(path, map_location="cpu", weights_only=False)
    encoder_config = ckpt.get("encoder_config")
    if encoder_config is None:
        raise ValueError(
            "The checkpoint does not contain an encoder_config entry. "
            "Provide --encoder_checkpoint explicitly."
        )
    return encoder_config
