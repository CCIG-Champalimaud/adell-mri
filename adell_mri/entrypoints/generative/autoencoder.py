"""
Helpers for training autoencoders and connecting them to latent diffusion
models.
"""

import os
from typing import Any

import monai
import torch
from lightning.pytorch import Trainer
from lightning.pytorch.callbacks import RichProgressBar

from adell_mri.modules.gan.pl.ae import AutoEncoderPL, VariationalAutoEncoderPL
from adell_mri.utils.generic_utils import safe_collate
from adell_mri.utils.pl_callbacks import EMACallback, SetSamplerEpoch
from adell_mri.utils.pl_utils import get_ckpt_callback
from adell_mri.utils.samplers import DistributedRandomSampler
from adell_mri.utils.torch_utils import (
    get_global_rank,
    load_checkpoint_to_model,
)

AUTOENCODER_SUFFIX = "_ae"


def resolve_device(device: str) -> str | torch.device:
    """
    Resolves a multi-GPU device string to the current process device.

    Args:
        device (str): device string, e.g. ``cuda:0,1`` or ``cpu``.

    Returns:
        str | torch.device: device for the current process.
    """
    if isinstance(device, str) and "," in device and "cuda" in device:
        local_rank = int(os.environ.get("LOCAL_RANK", get_global_rank()))
        return torch.device("cuda", local_rank)
    return device


def build_autoencoder(parsed_config: dict[str, Any]) -> torch.nn.Module:
    """
    Builds an autoencoder module from a parsed configuration.

    Args:
        parsed_config (dict[str, Any]): output of
            :func:`adell_mri.modules.config_parsing.parse_config_ae`.

    Returns:
        torch.nn.Module: the autoencoder lightning module.
    """
    model_config = dict(parsed_config["model"])
    common_args = {
        "input_image_key": "image",
        "loss_weights": parsed_config["loss_weights"],
        "rec_loss": parsed_config["rec_loss"],
        "perceptual_kwargs": parsed_config["perceptual_kwargs"],
        "discriminator_config": parsed_config["discriminator"],
        "reconstruction_every_n_epochs": parsed_config.get(
            "reconstruction_every_n_epochs", 0
        ),
        "reconstruction_max_batches": parsed_config.get(
            "reconstruction_max_batches", 1
        ),
    }
    common_args.update(parsed_config.get("optimizer", {}))
    module_class = (
        VariationalAutoEncoderPL
        if parsed_config["is_variational"]
        else AutoEncoderPL
    )
    return module_class(**model_config, **common_args)


def get_autoencoder_checkpoint_name(checkpoint_name: str) -> str:
    """
    Returns the checkpoint name used for the autoencoder.

    Args:
        checkpoint_name (str): base checkpoint name.

    Returns:
        str: autoencoder checkpoint name.
    """
    return checkpoint_name + AUTOENCODER_SUFFIX


def resolve_autoencoder_checkpoint_name(
    checkpoint_name: str, ae_checkpoint_name: str | None = None
) -> str:
    """
    Returns the autoencoder checkpoint name.

    Args:
        checkpoint_name (str): base checkpoint name.
        ae_checkpoint_name (str | None, optional): explicit autoencoder
            checkpoint name. Defaults to None, in which case ``_ae`` is
            appended to ``checkpoint_name``.

    Returns:
        str: autoencoder checkpoint name.
    """
    if ae_checkpoint_name is not None:
        return ae_checkpoint_name
    return get_autoencoder_checkpoint_name(checkpoint_name)


def get_autoencoder_checkpoint(
    checkpoint_dir: str, ae_checkpoint_name: str
) -> str:
    """
    Returns the path to the last autoencoder checkpoint.

    Args:
        checkpoint_dir (str): directory where checkpoints are stored.
        ae_checkpoint_name (str): autoencoder checkpoint name, without the
            ``_last.ckpt`` suffix.

    Returns:
        str: path to the last autoencoder checkpoint.
    """
    return os.path.join(checkpoint_dir, ae_checkpoint_name + "_last.ckpt")


def attach_autoencoder(
    network: torch.nn.Module,
    autoencoder: torch.nn.Module,
) -> None:
    """
    Connects a frozen autoencoder to a diffusion model.

    The diffusion model receives images, encodes them with ``autoencoder``
    before the diffusion process and decodes its samples afterwards. The
    autoencoder remains a separate module and is not included in the diffusion
    model state dict.

    Args:
        network (torch.nn.Module): diffusion module with ``input_encoder`` and
            ``output_decoder`` attributes.
        autoencoder (torch.nn.Module): frozen autoencoder.
    """
    parameter = next(autoencoder.parameters())

    def encode_images(images: torch.Tensor) -> torch.Tensor:
        images = images.to(device=parameter.device, dtype=parameter.dtype)
        with torch.no_grad():
            return autoencoder.encode(images)

    def decode_latents(latents: torch.Tensor) -> torch.Tensor:
        latents = latents.to(device=parameter.device, dtype=parameter.dtype)
        with torch.no_grad():
            return autoencoder.decode(latents)

    network.input_encoder = encode_images
    network.output_decoder = decode_latents


def load_autoencoder(
    parsed_config: dict[str, Any],
    checkpoint_path: str,
    device: str | torch.device,
    logger: Any,
    prefer_ema: bool = True,
) -> torch.nn.Module:
    """
    Builds an autoencoder and loads its weights from a checkpoint.

    Args:
        parsed_config (dict[str, Any]): parsed autoencoder configuration.
        checkpoint_path (str): path to the autoencoder checkpoint.
        device (str | torch.device): device where the autoencoder is placed.
        logger (Any): python logger.
        prefer_ema (bool, optional): load the EMA weights when available.
            Defaults to True.

    Returns:
        torch.nn.Module: the autoencoder in evaluation mode.
    """
    autoencoder = build_autoencoder(parsed_config)
    load_checkpoint_to_model(
        autoencoder, checkpoint_path, [], prefer_ema=prefer_ema
    )
    autoencoder = autoencoder.to(device)
    autoencoder.eval()
    logger.info("Loading autoencoder checkpoint %s", checkpoint_path)
    return autoencoder


def get_latent_shape(
    autoencoder: torch.nn.Module,
    spatial_shape: list[int] | tuple[int],
) -> tuple[int, ...]:
    """
    Infers the shape of the latent produced by an autoencoder for a given
    spatial input shape.

    Args:
        autoencoder (torch.nn.Module): autoencoder with a
            ``latent_downsample_factor`` attribute (an int or a per-axis
            tuple) and either ``latent_channels`` or ``block_out_channels``.
        spatial_shape (list[int] | tuple[int]): spatial shape of the input.

    Returns:
        tuple[int, ...]: latent shape, excluding the batch dimension.
    """
    factor = autoencoder.latent_downsample_factor
    if isinstance(factor, int):
        factors = (factor,) * len(spatial_shape)
    else:
        factors = factor
    channels = autoencoder.latent_channels
    if channels is None:
        channels = autoencoder.block_out_channels[-1]
    spatial = tuple(
        int(size) // int(axis_factor)
        for size, axis_factor in zip(spatial_shape, factors)
    )
    return (channels, *spatial)


def make_autoencoder_loader(
    data_list: list[dict],
    transforms: Any,
    batch_size: int,
    n_workers: int,
    seed: int,
    shuffle: bool = True,
    steps_per_epoch: int | None = None,
    n_devices: int = 1,
    rank: int = 0,
) -> monai.data.ThreadDataLoader:
    """
    Builds the dataloader used to train the autoencoder.

    Args:
        data_list (list[dict]): list of records.
        transforms (Any): transforms applied to each record.
        batch_size (int): batch size.
        n_workers (int): number of dataloader workers.
        seed (int): random seed.
        shuffle (bool, optional): whether the dataset is shuffled. Defaults to
            True.
        steps_per_epoch (int | None, optional): number of batches per rank.
            Defaults to None (iterates the full dataset).
        n_devices (int, optional): number of devices. Defaults to 1.
        rank (int, optional): rank of the current process. Defaults to 0.

    Returns:
        monai.data.ThreadDataLoader: dataloader.
    """
    dataset = monai.data.Dataset(data_list, transforms)
    generator = torch.Generator()
    generator.manual_seed(seed)
    kwargs = {
        "batch_size": batch_size,
        "num_workers": n_workers,
        "collate_fn": safe_collate,
        "pin_memory": True,
        "drop_last": shuffle and steps_per_epoch is not None,
        "shuffle": shuffle,
        "generator": generator,
    }
    if steps_per_epoch is not None:
        num_samples = steps_per_epoch * batch_size
        if n_devices > 1:
            kwargs["sampler"] = DistributedRandomSampler(
                dataset,
                num_samples=num_samples * n_devices,
                num_replicas=n_devices,
                rank=rank,
                replacement=(num_samples * n_devices) > len(dataset),
                seed=seed,
            )
        else:
            kwargs["sampler"] = torch.utils.data.RandomSampler(
                dataset,
                replacement=num_samples > len(dataset),
                num_samples=num_samples,
                generator=generator,
            )
        kwargs["shuffle"] = False
    if n_workers > 0:
        kwargs["persistent_workers"] = True
        kwargs["prefetch_factor"] = 8
    return monai.data.ThreadDataLoader(dataset, **kwargs)


def _read_autoencoder_config(checkpoint_path: str) -> dict[str, Any] | None:
    """
    Reads the autoencoder configuration stored in a checkpoint.

    Args:
        checkpoint_path (str): path to a checkpoint.

    Returns:
        dict[str, Any] | None: the stored configuration, if available.
    """
    checkpoint = torch.load(
        checkpoint_path, map_location="cpu", weights_only=False
    )
    callbacks = checkpoint.get("callbacks", {})
    for callback_state in callbacks.values():
        if isinstance(callback_state, dict) and "metadata" in callback_state:
            metadata = callback_state["metadata"]
            if "ae_config" in metadata:
                return metadata["ae_config"]
    return None


def ensure_autoencoder(
    args: Any,
    parsed_config: dict[str, Any],
    train_list: list[dict],
    val_list: list[dict],
    test_list: list[dict],
    transforms: Any,
    eval_transforms: Any,
    checkpoint_dir: str | None,
    checkpoint_name: str,
    accelerator: str,
    devices: Any,
    strategy: Any,
    n_devices: int,
    pl_logger: Any,
    output_file: Any,
    logger: Any,
) -> tuple[torch.nn.Module, str | None, bool]:
    """
    Trains the autoencoder if needed and returns it.

    The autoencoder checkpoint name is always ``{checkpoint_name}_ae``.
    Training is skipped when the last checkpoint already reached
    ``--ae_max_epochs``. The returned flag indicates whether the autoencoder was
    trained in this call.

    Args:
        args (Any): parsed command-line arguments.
        parsed_config (dict[str, Any]): parsed autoencoder configuration.
        train_list (list[dict]): training records.
        val_list (list[dict]): validation records.
        test_list (list[dict]): test records.
        transforms (Any): transforms applied to the training images.
        eval_transforms (Any): deterministic transforms for validation and
            test images.
        checkpoint_dir (str | None): directory where checkpoints are stored.
        checkpoint_name (str): base checkpoint name.
        accelerator (str): lightning accelerator.
        devices (Any): lightning devices.
        strategy (Any): lightning strategy.
        n_devices (int): number of devices.
        pl_logger (Any): lightning logger.
        output_file (Any): file where test metrics are written.
        logger (Any): python logger.

    Returns:
        tuple[torch.nn.Module, str | None, bool]: the autoencoder, the path to
            its last checkpoint (if any) and whether it was trained in this
            call.

    Raises:
        ValueError: if a checkpoint directory or name is missing, or if an
            existing checkpoint was trained with a different architecture.
    """
    if checkpoint_dir is None or checkpoint_name is None:
        raise ValueError(
            "--checkpoint_dir and --checkpoint_name are required to train an "
            "autoencoder"
        )

    ae_checkpoint_name = resolve_autoencoder_checkpoint_name(
        checkpoint_name, args.ae_checkpoint_name
    )
    device = resolve_device(args.dev)
    max_epochs = args.ae_max_epochs or args.max_epochs
    ckpt_callback, ckpt_path, status = get_ckpt_callback(
        checkpoint_dir=checkpoint_dir,
        checkpoint_name=ae_checkpoint_name,
        max_epochs=max_epochs,
        resume_from_last=True,
        val_fold=None,
        monitor="val_loss",
        metadata={"model_type": "ae", "ae_config": parsed_config},
    )
    last_checkpoint = get_autoencoder_checkpoint(
        checkpoint_dir, ae_checkpoint_name
    )
    if status == "finished":
        stored_config = _read_autoencoder_config(last_checkpoint)
        if stored_config is not None and stored_config.get(
            "model"
        ) != parsed_config.get("model"):
            raise ValueError(
                "The existing autoencoder checkpoint was trained with a "
                "different architecture. Remove the existing checkpoint or "
                "use a different --checkpoint_name."
            )
        autoencoder = load_autoencoder(
            parsed_config, last_checkpoint, device, logger
        )
        return autoencoder, last_checkpoint, False

    autoencoder = build_autoencoder(parsed_config)
    callbacks = [RichProgressBar()]
    if ckpt_callback is not None:
        callbacks.append(ckpt_callback)
    if args.ema_decay is not None:
        callbacks.append(
            EMACallback(decay=args.ema_decay, use_ema_weights=True)
        )
    callbacks.append(SetSamplerEpoch())

    accumulate_grad_batches = args.accumulate_grad_batches
    if parsed_config["loss_weights"].get("adversarial", 0.0) > 0.0:
        if accumulate_grad_batches != 1:
            logger.warning(
                "gradient accumulation is not supported for adversarial "
                "autoencoder training; using accumulate_grad_batches=1"
            )
        accumulate_grad_batches = 1

    batch_size = max(1, parsed_config["batch_size"] // n_devices)
    steps_per_epoch = args.steps_per_epoch or max(
        1, len(train_list) // batch_size
    )
    train_loader = make_autoencoder_loader(
        train_list,
        transforms,
        batch_size=batch_size,
        n_workers=args.n_workers,
        seed=args.seed,
        steps_per_epoch=steps_per_epoch,
        n_devices=n_devices,
        rank=get_global_rank(),
    )
    val_loader = make_autoencoder_loader(
        val_list,
        eval_transforms,
        batch_size=batch_size,
        n_workers=args.n_workers,
        seed=args.seed,
        shuffle=False,
    )
    test_loader = make_autoencoder_loader(
        test_list,
        eval_transforms,
        batch_size=batch_size,
        n_workers=args.n_workers,
        seed=args.seed,
        shuffle=False,
    )
    logger.info(
        "Training autoencoder with steps_per_epoch=%s (train batches=%s, "
        "batch_size=%s, devices=%s, samples_per_epoch=%s, train_size=%s)",
        steps_per_epoch,
        len(train_loader),
        batch_size,
        n_devices,
        steps_per_epoch * batch_size * n_devices,
        len(train_list),
    )

    trainer = Trainer(
        accelerator=accelerator,
        devices=devices,
        logger=pl_logger,
        callbacks=callbacks,
        max_epochs=max_epochs,
        enable_checkpointing=ckpt_callback is not None,
        gradient_clip_val=args.gradient_clip_val,
        strategy=strategy,
        accumulate_grad_batches=accumulate_grad_batches,
        check_val_every_n_epoch=args.check_val_every_n_epoch,
        precision=args.precision,
        deterministic="warn",
        limit_train_batches=steps_per_epoch,
        use_distributed_sampler=n_devices == 1,
    )
    trainer.fit(
        autoencoder,
        train_loader,
        val_loader,
        ckpt_path=ckpt_path,
        weights_only=False,
    )

    if os.path.exists(last_checkpoint) is True:
        autoencoder = load_autoencoder(
            parsed_config, last_checkpoint, device, logger
        )
        test_metrics = trainer.test(autoencoder, test_loader)[0]
        for key, value in test_metrics.items():
            if isinstance(value, torch.Tensor):
                value = value.detach().cpu().item()
            output_file.write(f"ae_{key},{value}\n")
    autoencoder = autoencoder.to(device)
    autoencoder.eval()
    return autoencoder, last_checkpoint, True
