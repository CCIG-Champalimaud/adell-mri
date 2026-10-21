"""
Utilities to cache autoencoder latents and compute their statistics.
"""

import os
import re
from collections.abc import Callable

import monai
import numpy as np
import torch

from adell_mri.utils.generic_utils import safe_collate


def _sanitize_filename(identifier: str) -> str:
    """
    Replaces characters that are unsafe in file names.

    Args:
        identifier (str): identifier to sanitize.

    Returns:
        str: sanitized identifier.
    """
    return re.sub(r"[^A-Za-z0-9_.-]", "_", str(identifier))


def infer_latent_shape(
    autoencoder: torch.nn.Module,
    spatial_shape: list[int] | tuple[int],
) -> tuple[int, ...]:
    """
    Infers the shape of the latent produced by an autoencoder for a given
    spatial input shape.

    Args:
        autoencoder (torch.nn.Module): autoencoder with a
            ``latent_downsample_factor`` attribute and either ``latent_channels``
            or ``block_out_channels``.
        spatial_shape (list[int] | tuple[int]): spatial shape of the input.

    Returns:
        tuple[int, ...]: latent shape, excluding the batch dimension.
    """
    factor = autoencoder.latent_downsample_factor
    channels = autoencoder.latent_channels
    if channels is None:
        channels = autoencoder.block_out_channels[-1]
    spatial = tuple(int(size) // (2**factor) for size in spatial_shape)
    return (channels, *spatial)


def cache_latents(
    autoencoder: torch.nn.Module,
    data_list: list[dict],
    transforms: Callable | None,
    output_dir: str,
    batch_size: int = 1,
    num_workers: int = 0,
    device: str | torch.device = "cpu",
    overwrite: bool = False,
    identifier_key: str = "identifier",
) -> list[dict]:
    """
    Encodes a dataset with an autoencoder and saves one ``.npy`` file per case.

    The input records are not modified; a new list of records is returned where
    the ``latent`` key points to the cached file. Records must have an
    identifier, which is used as the file name.

    Args:
        autoencoder (torch.nn.Module): autoencoder with an ``encode`` method.
        data_list (list[dict]): list of records with image paths.
        transforms (Callable | None): transforms applied to each record. They
            should be deterministic (i.e. no random augmentation).
        output_dir (str): directory where the latents are saved.
        batch_size (int, optional): batch size. Defaults to 1.
        num_workers (int, optional): number of dataloader workers. Defaults to
            0.
        device (str | torch.device, optional): device used for encoding.
            Defaults to "cpu".
        overwrite (bool, optional): whether existing latent files should be
            overwritten. Defaults to False.
        identifier_key (str, optional): key used to identify each record.
            Defaults to "identifier".

    Returns:
        list[dict]: records with an additional ``latent`` key.

    Raises:
        KeyError: if a record has no identifier.
        ValueError: if the dataloader does not return an ``image`` key.
    """
    os.makedirs(output_dir, exist_ok=True)
    dataset = monai.data.Dataset(data_list, transforms)
    dataloader = torch.utils.data.DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        collate_fn=safe_collate,
    )
    autoencoder = autoencoder.to(device)
    autoencoder.eval()

    output = []
    offset = 0
    with torch.inference_mode():
        for batch in dataloader:
            if "image" not in batch:
                raise ValueError(
                    "transforms must produce an 'image' key to cache latents"
                )
            images = batch["image"].to(device)
            latents = autoencoder.encode(images)
            identifiers = batch.get(identifier_key)
            if identifiers is None:
                identifiers = [
                    data_list[offset + i][identifier_key]
                    for i in range(images.shape[0])
                ]
            if isinstance(identifiers, torch.Tensor):
                identifiers = identifiers.tolist()
            latents = latents.detach().float().cpu().numpy()
            for i in range(latents.shape[0]):
                record = dict(data_list[offset + i])
                identifier = _sanitize_filename(identifiers[i])
                path = os.path.join(output_dir, identifier + ".npy")
                if overwrite is False and os.path.exists(path) is True:
                    pass
                else:
                    np.save(path, latents[i])
                record["latent"] = path
                output.append(record)
            offset += latents.shape[0]
    return output


def compute_latent_statistics(
    latent_paths: list[str], n_channels: int | None = None
) -> dict[str, float | list[float]]:
    """
    Computes the mean and standard deviation of a set of cached latents.

    Args:
        latent_paths (list[str]): paths to ``.npy`` latent files.
        n_channels (int | None, optional): number of latent channels. If None,
            the shape of the first latent is used. Defaults to None.

    Returns:
        dict[str, float | list[float]]: dictionary with the global ``mean``,
            ``std`` and the ``scale`` used to normalise latents before
            diffusion modelling (``1 / std``).

    Raises:
        ValueError: if no latents are provided or the standard deviation is
            zero.
    """
    if len(latent_paths) == 0:
        raise ValueError("At least one latent path is required")
    if n_channels is None:
        n_channels = np.load(latent_paths[0], mmap_mode="r").shape[0]
    total = np.zeros(n_channels, dtype=np.float64)
    total_sq = np.zeros(n_channels, dtype=np.float64)
    count = 0
    for path in latent_paths:
        latent = np.load(path, mmap_mode="r")
        axes = (1, 2, 3) if latent.ndim == 4 else (1, 2)
        latent = latent.astype(np.float64)
        total += latent.sum(axis=axes)
        total_sq += np.square(latent).sum(axis=axes)
        count += int(np.prod([latent.shape[axis] for axis in axes]))
    mean = total / count
    variance = total_sq / count - np.square(mean)
    std = np.sqrt(np.maximum(variance, 0.0))
    if np.all(std == 0.0):
        raise ValueError("Latent standard deviation is zero")
    return {
        "mean": float(mean.mean()),
        "std": float(std.mean()),
        "scale": float(1.0 / std.mean()),
    }
