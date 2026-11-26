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

from adell_mri.utils.encoder_loading import MANIFEST_KEY, read_encoder_manifest


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

    manifest = read_encoder_manifest(checkpoint)
    checkpoint[MANIFEST_KEY] = manifest
    torch.save(checkpoint, output_path)
    print(f"Wrote {output_path} with encoder family '{manifest['family']}'")


if __name__ == "__main__":
    main()
