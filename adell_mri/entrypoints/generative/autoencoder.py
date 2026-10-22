"""
Helpers for training autoencoders from the generative entrypoint.
"""

import os
from typing import Any

import monai
import torch
from lightning.pytorch import Trainer
from lightning.pytorch.callbacks import RichProgressBar

from adell_mri.modules.gan.pl.ae import AutoEncoderPL, VariationalAutoEncoderPL
from adell_mri.utils.generic_utils import safe_collate
from adell_mri.utils.pl_callbacks import EMACallback
from adell_mri.utils.pl_utils import get_ckpt_callback
from adell_mri.utils.torch_utils import load_checkpoint_to_model


def build_autoencoder(
    parsed_config: dict[str, Any],
) -> tuple[torch.nn.Module, dict[str, Any]]:
    """
    Builds an autoencoder module from a parsed configuration.

    Args:
        parsed_config (dict[str, Any]): output of
            :func:`adell_mri.modules.config_parsing.parse_config_ae`.

    Returns:
        tuple[torch.nn.Module, dict[str, Any]]: the lightning module and the
            model configuration used to build it.
    """
    model_config = dict(parsed_config["model"])
    common_args = {
        "input_image_key": "image",
        "loss_weights": parsed_config["loss_weights"],
        "rec_loss": parsed_config["rec_loss"],
        "perceptual_kwargs": parsed_config["perceptual_kwargs"],
        "discriminator_config": parsed_config["discriminator"],
    }
    common_args.update(parsed_config.get("optimizer", {}))
    module_class = (
        VariationalAutoEncoderPL
        if parsed_config["is_variational"]
        else AutoEncoderPL
    )
    return module_class(**model_config, **common_args), model_config


def get_autoencoder_checkpoint(
    checkpoint_dir: str, checkpoint_name: str
) -> str:
    """
    Returns the path to the last autoencoder checkpoint.

    Args:
        checkpoint_dir (str): directory where checkpoints are stored.
        checkpoint_name (str): base checkpoint name. The autoencoder checkpoint
            appends ``_ae`` to this name.

    Returns:
        str: path to the last autoencoder checkpoint.
    """
    return os.path.join(checkpoint_dir, checkpoint_name + "_ae_last.ckpt")


def get_latent_shape(
    autoencoder: torch.nn.Module,
    spatial_shape: list[int] | tuple[int],
) -> tuple[int, ...]:
    """
    Infers the shape of the latent produced by an autoencoder for a given
    spatial input shape.

    Args:
        autoencoder (torch.nn.Module): autoencoder with a
            ``latent_downsample_factor`` attribute and either
            ``latent_channels`` or ``block_out_channels``.
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


def make_autoencoder_loader(
    data_list: list[dict],
    transforms: Any,
    batch_size: int,
    n_workers: int,
    seed: int,
) -> monai.data.ThreadDataLoader:
    """
    Builds the dataloader used to train the autoencoder.

    Args:
        data_list (list[dict]): list of records.
        transforms (Any): transforms applied to each record.
        batch_size (int): batch size.
        n_workers (int): number of dataloader workers.
        seed (int): random seed.

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
        "drop_last": len(dataset) > batch_size,
        "shuffle": True,
        "generator": generator,
    }
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
    transforms: Any,
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
        transforms (Any): transforms applied to the training images.
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

    max_epochs = args.ae_max_epochs or args.max_epochs
    ckpt_callback, ckpt_path, status = get_ckpt_callback(
        checkpoint_dir=checkpoint_dir,
        checkpoint_name=checkpoint_name + "_ae",
        max_epochs=max_epochs,
        resume_from_last=True,
        val_fold=None,
        monitor="val_loss",
        metadata={"model_type": "ae", "ae_config": parsed_config},
    )
    last_checkpoint = get_autoencoder_checkpoint(
        checkpoint_dir, checkpoint_name
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
        autoencoder, _ = build_autoencoder(parsed_config)
        load_checkpoint_to_model(autoencoder, last_checkpoint, [])
        logger.info("Reusing autoencoder checkpoint %s", last_checkpoint)
        return autoencoder, last_checkpoint, False

    autoencoder, _ = build_autoencoder(parsed_config)
    callbacks = [RichProgressBar()]
    if ckpt_callback is not None:
        callbacks.append(ckpt_callback)
    if args.ema_decay is not None:
        callbacks.append(
            EMACallback(decay=args.ema_decay, use_ema_weights=True)
        )

    accumulate_grad_batches = args.accumulate_grad_batches
    if parsed_config["loss_weights"].get("adversarial", 0.0) > 0.0:
        if accumulate_grad_batches != 1:
            logger.warning(
                "gradient accumulation is not supported for adversarial "
                "autoencoder training; using accumulate_grad_batches=1"
            )
        accumulate_grad_batches = 1

    batch_size = max(1, parsed_config["batch_size"] // n_devices)
    loader = make_autoencoder_loader(
        train_list,
        transforms,
        batch_size=batch_size,
        n_workers=args.n_workers,
        seed=args.seed,
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
    )
    trainer.fit(autoencoder, loader, loader, ckpt_path=ckpt_path)

    if os.path.exists(last_checkpoint) is True:
        load_checkpoint_to_model(autoencoder, last_checkpoint, [])
        test_metrics = trainer.test(
            autoencoder,
            loader,
            ckpt_path=last_checkpoint,
            weights_only=False,
        )[0]
        for key in test_metrics:
            value = test_metrics[key]
            if isinstance(value, torch.Tensor):
                value = value.detach().cpu().item()
            output_file.write(f"{value}\n")
    autoencoder.eval()
    return autoencoder, last_checkpoint, True
