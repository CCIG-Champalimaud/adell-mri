#!/usr/bin/env python
"""
Adds an ``encoder_config`` manifest to an older self-supervised checkpoint.

The script reads the encoder configuration from the checkpoint metadata,
hyperparameters or state dict, writes it under the ``encoder_config`` key and
saves a new checkpoint containing all original entries.
"""

import argparse
from pathlib import Path

import torch
import yaml

from adell_mri.utils.encoder_loading import (
    MANIFEST_KEY,
    build_encoder_manifest,
    read_encoder_manifest,
)

_FAMILY_TO_NET_TYPE = {
    "resnet": "resnet",
    "convnext": "convnext",
    "vit": "vit",
    "mae": "vit",
    "unet": "unet_encoder",
}


def parse_arguments() -> argparse.Namespace:
    """
    Parses command-line arguments.

    Returns:
        argparse.Namespace: parsed arguments.
    """
    parser = argparse.ArgumentParser(
        description="Adds an encoder_config manifest to an SSL checkpoint."
    )
    parser.add_argument(
        "--checkpoint",
        required=True,
        help="Path to the self-supervised checkpoint to convert.",
    )
    parser.add_argument(
        "--output",
        default=None,
        help="Path to the output checkpoint. Defaults to "
        "<checkpoint>_with_encoder_config.ckpt.",
    )
    parser.add_argument(
        "--config_file",
        default=None,
        help="Optional SSL YAML configuration. Use this when the checkpoint "
        "does not contain enough metadata to recover the normalization or "
        "activation functions.",
    )
    return parser.parse_args()


def main() -> None:
    """
    Converts an SSL checkpoint to the current format.
    """
    args = parse_arguments()
    checkpoint_path = Path(args.checkpoint)
    output_path = (
        Path(args.output)
        if args.output is not None
        else checkpoint_path.with_name(
            f"{checkpoint_path.stem}_with_encoder_config"
            f"{checkpoint_path.suffix}"
        )
    )

    checkpoint = torch.load(
        checkpoint_path, map_location="cpu", weights_only=False
    )
    if MANIFEST_KEY in checkpoint:
        raise ValueError(
            f"{checkpoint_path} already contains an {MANIFEST_KEY} entry"
        )

    inferred_manifest = read_encoder_manifest(checkpoint)
    manifest = inferred_manifest
    if args.config_file is not None:
        with open(args.config_file, "r") as config_file:
            network_config = yaml.safe_load(config_file)
        family = inferred_manifest["family"]
        ssl_method = (
            "mae"
            if family == "mae"
            else inferred_manifest.get("ssl_method", "unknown")
        )
        manifest = build_encoder_manifest(
            net_type=_FAMILY_TO_NET_TYPE[family],
            ssl_method=ssl_method,
            network_config=network_config,
            spatial_dim=inferred_manifest.get("spatial_dim"),
            in_channels=inferred_manifest.get("in_channels"),
        )
        if family in {"vit", "mae"} and manifest.get("image_size") is None:
            manifest["image_size"] = inferred_manifest.get("image_size")

    checkpoint[MANIFEST_KEY] = manifest
    torch.save(checkpoint, output_path)
    print(f"Wrote {output_path} with encoder family '{manifest['family']}'")


if __name__ == "__main__":
    main()
