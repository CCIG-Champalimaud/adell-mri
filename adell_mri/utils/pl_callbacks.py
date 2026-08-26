from typing import Any

import lightning.pytorch as pl
import numpy as np
import torch
import torch.nn.functional as F
from lightning.pytorch import LightningModule, Trainer
from lightning.pytorch.callbacks import Callback
from lightning.pytorch.callbacks.model_checkpoint import ModelCheckpoint
from PIL import Image
from torch.nn.utils.parametrizations import spectral_norm

from adell_mri.utils.python_logging import get_logger
from adell_mri.utils.torch_utils import ExponentialMovingAverage

logger = get_logger(__name__)


def coerce_to_uint8(x: np.ndarray):
    x = (x - x.min()) / (x.max() - x.min()) * 255
    return x.astype(np.uint8)


def split_and_cat(x: np.ndarray, split_dim: int, cat_dim: int) -> np.ndarray:
    arrays = np.split(x, x.shape[split_dim], axis=split_dim)
    arrays = np.concatenate([arr.squeeze(split_dim) for arr in arrays], cat_dim)
    return arrays


def log_image(
    trainer: Trainer,
    key: str,
    images: list[torch.Tensor],
    slice_dim: int | None = None,
    n_slices_out: int | None = None,
    caption: list[str] = None,
    rgb: bool = False,
):
    """
    Logs images to the PyTorch Lightning logger.

    This callback takes a batch of images, slices them along the provided
    slice dimension, converts them to uint8, creates PIL images, and logs
    them to the PyTorch Lightning logger.

    Args:
        trainer (Trainer): PyTorch Lightning Trainer instance.
        key (str): key for the logged images.
        images (list[torch.Tensor]): list of images to log as a list of torch
            Tensors.
        slice_dim (int, optional): dimension to slice the images along (if
            images are 3D). Defaults to None but must be specified for 3D
            images.
        n_slices_out (int, optional): number of image slices to output.
            Defaults to None but must be specified for 3D images.
        caption (list[str], optional): Optional list of captions, one for each
            image.
        RGB (bool, optional): Whether 3-channel images should be considered
            RGB. Defaults to False.
    """
    if hasattr(trainer.logger, "log_image") is False:
        return None
    images = images.detach().to("cpu")
    if len(images.shape) == 5:
        n_slices = images.shape[slice_dim]
        slice_idxs = np.linspace(
            0, n_slices, num=n_slices_out + 2, dtype=np.int32
        )[1:-1]
        images = torch.index_select(
            images, slice_dim, torch.as_tensor(slice_idxs)
        )
        images = torch.split(images, 1, dim=slice_dim)
        images = torch.cat(images, -2).squeeze(-1)
    # do some form of acceptable conversion here if images are not BW, RGB or
    # RGBA
    if images.shape[1] not in [1, 3, 4]:
        images = torch.argmax(images, dim=1, keepdim=True)
    images = torch.split(images, 1, 0)
    # squeeze only  if necessary
    images = [x.squeeze(0) if len(x.shape) > 3 else x for x in images]
    images = [x.permute(1, 2, 0).numpy() for x in images]
    images = [
        (
            coerce_to_uint8(split_and_cat(x, -1, 0))
            if rgb is False
            else coerce_to_uint8(x)
        )
        for x in images
    ]
    images = [x.squeeze(-1) if x.shape[-1] == 1 else x for x in images]
    images = [Image.fromarray(x) for x in images]

    step = trainer.global_step
    if caption is not None:
        trainer.logger.log_image(
            key=key, images=images, caption=caption, step=step
        )
    else:
        trainer.logger.log_image(key=key, images=images, step=step)


def reshape_weight_to_matrix(
    weight: torch.Tensor, dim: int = 0
) -> torch.Tensor:
    """
    Reshapes an n-dimensional tensor into a matrix.

    From https://pytorch.org/docs/stable/_modules/torch/nn/utils/spectral_norm.html

    Args:
        weight (torch.Tensor): weight matrix.
        dim (int, optional): dimension corresponing to first matrix dimension.
            Defaults to 0.

    Returns:
        torch.Tensor: reshaped tensor.
    """
    weight_mat = weight
    if dim != 0:
        weight_mat = weight_mat.permute(
            dim, *[d for d in range(weight_mat.dim()) if d != dim]
        )
    height = weight_mat.size(0)
    return weight_mat.reshape(height, -1)


class SpectralNorm(pl.Callback):
    """
    PyTorch Lightning Callback that dynamically applies PyTorch's native,
    autograd-safe spectral normalization to all compatible layers before
    training starts.
    """

    def __init__(
        self,
        name: str = "weight",
        exclude_modules: list[str] | None = None,
        n_power_iterations: int = 1,
        eps: float = 1e-12,
        bake_on_save: bool = False,
    ):
        """
        Args:
            name (str): The name of the parameter to normalize (usually "weight").
            exclude_modules (list[str] | None): List of module names to exclude
                from spectral normalization. Defaults to None (which excludes
                GaussianProcessLayer, LayerNorm and batch normalizations).
            n_power_iterations (int): Number of power iterations to calculate spectral norm.
            eps (float): Epsilon to avoid division by zero.
            bake_on_save (bool): If True, replace the parametrization keys
                (``.original``, ``._u``, ``._v``) in saved checkpoints with the
                baked normalized weight under the plain parameter name. The
                resulting checkpoint can be loaded into a non-parametrized model
                without ``calculate_sn_weights``. Note that such a checkpoint
                cannot be used to resume SN training (the power-iteration
                vectors are discarded). Defaults to False.
        """
        super().__init__()
        self.name = name
        self.n_power_iterations = n_power_iterations
        self.eps = eps
        self.bake_on_save = bake_on_save

        self.exclude_modules = exclude_modules
        if self.exclude_modules is None:
            self.exclude_modules = [
                "GaussianProcessLayer",
                "LayerNorm",
                "BatchNorm1d",
                "BatchNorm2d",
                "BatchNorm3d",
            ]
        self.exclude_modules.extend(
            [
                "_SpectralNorm",
            ]
        )

    def setup(self, trainer, pl_module, stage=None):
        """
        Recursively traverses the model and applies spectral normalization.
        """
        if getattr(pl_module, "_spectral_norm_applied", False):
            return
        if stage == "fit" or stage is None:
            logger.info("Applying spectral normalization")
            self._apply_spectral_norm(pl_module)

    def _apply_spectral_norm(self, model: torch.nn.Module):
        """
        Traverses every module in the network and checks its direct (non-recursive)
        parameters. If a parameter matches the naming rule and has sufficient dimension,
        we register the native spectral norm.
        """
        for module in model.modules():
            module_name = module.__class__.__name__
            if module_name in self.exclude_modules:
                continue
            for param_name, param in list(
                module.named_parameters(recurse=False)
            ):
                if param.requires_grad is False:
                    continue
                if self.name in param_name:
                    if param.ndim >= 2:
                        try:
                            spectral_norm(
                                module,
                                name=param_name,
                                n_power_iterations=self.n_power_iterations,
                                eps=self.eps,
                            )
                        except Exception as e:
                            # Catch-all to gracefully skip any edge-case non-standard modules
                            print(
                                f"Skipping spectral norm on {module.__class__.__name__}.{param_name}: {e}"
                            )
        model._spectral_norm_applied = True

    def on_save_checkpoint(self, trainer, pl_module, checkpoint) -> None:
        """
        Optionally bakes the spectrally-normalized weights into the checkpoint,
        replacing the parametrization keys (``.original``, ``._u``, ``._v``)
        with the plain normalized parameter.
        """
        if not self.bake_on_save:
            return
        sd = checkpoint.get("state_dict")
        if sd is None:
            return
        for module_name, module in pl_module.named_modules():
            if not hasattr(module, "parametrizations"):
                continue
            for param_name in list(module.parametrizations.keys()):
                if module_name:
                    prefix = f"{module_name}.parametrizations.{param_name}"
                else:
                    prefix = f"parametrizations.{param_name}"

                orig_key = f"{prefix}.original"
                if orig_key not in sd:
                    continue
                u_key = f"{prefix}.0._u"
                v_key = f"{prefix}.0._v"
                out_key = (
                    f"{module_name}.{param_name}" if module_name else param_name
                )
                with torch.no_grad():
                    normalized = module.weight.detach().clone()
                sd[out_key] = normalized
                for k in (orig_key, u_key, v_key):
                    sd.pop(k, None)


class LogImage(Callback):
    """
    Logs image outputs from the validation loop to a Lightning logger.

    This callback logs image outputs during validation by slicing
    the batch along the slice dimension, logging a subset of slices,
    and optionally adding captions. It logs after a set number of
    batches have been processed.
    """

    def __init__(
        self,
        image_keys: list[str] = "image",
        caption_keys: list[str] = None,
        output_idxs: list[int] = None,
        n_slices: int = 1,
        slice_dim: int = 4,
        log_frequency: int = 5,
    ):
        """
        Args:
            image_keys (list[str], optional): keys corresponding to images.
                Defaults to "image".
            caption_keys (list[str], optional): keys corresponding to captions.
                Defaults to None.
            output_idxs (list[int], optional): indices corresponding to logged
                images in the output. Defaults to None.
            n_slices (int, optional): number of logged slices per volume.
                Defaults to 1.
            slice_dim (int, optional): dimension for slices. Defaults to 4.
            log_frequency (int, optional): frequency of logging. Defaults to 5.
        """
        self.image_keys = image_keys
        self.caption_keys = caption_keys
        self.output_idxs = output_idxs
        self.n_slices = n_slices
        self.slice_dim = slice_dim
        self.log_frequency = log_frequency

    def on_validation_batch_end(
        self,
        trainer: Trainer,
        pl_module: LightningModule,
        outputs: tuple[Any],
        batch: dict[str, Any],
        batch_idx: int,
        dataloader_idx: int = 0,
    ) -> None:
        if batch_idx % self.log_frequency == 0:
            if self.caption_keys is not None:
                captions = []
                for key in self.caption_keys:
                    for i in range(len(batch[key])):
                        if len(captions) < i + 1:
                            captions.append([])
                        captions[i].append(f"{key}: {batch[key][i]}")
                captions = [";".join(x) for x in captions]
            else:
                captions = None
            if self.image_keys is not None:
                for key in self.image_keys:
                    log_image(
                        trainer,
                        key,
                        batch[key],
                        slice_dim=self.slice_dim,
                        n_slices_out=self.n_slices,
                        caption=captions,
                    )
            if self.output_idxs is not None:
                for idx in self.output_idxs:
                    log_image(
                        trainer,
                        f"output.{idx}",
                        outputs[idx],
                        slice_dim=self.slice_dim,
                        n_slices_out=self.n_slices,
                        caption=captions,
                    )


class LogImageFromDiffusionProcess(Callback):
    """
    Logs images from diffusion models. Expects the lightning module to have a
    `generate_image` function.
    """

    def __init__(
        self,
        size: list[int],
        n_slices: int = 3,
        slice_dim: int = 4,
        n_images: int = 2,
        every_n_epochs: int = 1,
    ):
        """
        Args:
            size (list[int]): size of the generated image.
            output_idxs (list[int], optional): indices corresponding to logged
                images in the output. Defaults to None.
            n_slices (int, optional): number of logged slices per volume.
                Defaults to 1.
            slice_dim (int, optional): dimension for slices. Defaults to 4.
            every_n_epochs (int, optional): frequency of logging. Defaults to
                1.
        """

        self.size = size
        self.n_slices = n_slices
        self.slice_dim = slice_dim
        self.n_images = n_images
        self.every_n_epochs = every_n_epochs

    def on_validation_epoch_end(
        self, trainer: Trainer, pl_module: LightningModule
    ) -> None:
        ep = pl_module.current_epoch
        if ep % self.every_n_epochs == 0 and ep > 0:
            was_training = pl_module.training
            pl_module.eval()
            with torch.inference_mode():
                generate_kwargs = {}
                n_cond_channels = pl_module.in_channels - pl_module.out_channels
                if n_cond_channels > 0:
                    generate_kwargs["concat_condition"] = torch.zeros(
                        [
                            self.n_images,
                            n_cond_channels,
                            *self.size,
                        ],
                        device=pl_module.device,
                    )
                images = pl_module.generate_image(
                    size=self.size, n=self.n_images, **generate_kwargs
                )
            if was_training:
                pl_module.train()
            log_image(
                trainer,
                key="Generated images",
                images=images,
                slice_dim=self.slice_dim,
                n_slices_out=self.n_slices,
            )


class LogImageFromGAN(Callback):
    """
    Logs images from GAN models. Expects the lightning module to have a
    `generate_image` function.
    """

    def __init__(
        self,
        size: list[int],
        n_slices: int = 3,
        slice_dim: int = 4,
        n_images: int = 2,
        every_n_epochs: int = 1,
        generate_kwargs: dict[str, Any] = None,
        conditional: bool = False,
        conditional_key: str | int = None,
        additional_image_keys: list[str | int] = None,
        rgb: bool = False,
    ):
        """
        Args:
            size (list[int]): size of the generated image.
            output_idxs (list[int], optional): indices corresponding to logged
                images in the output. Defaults to None.
            n_slices (int, optional): number of logged slices per volume.
                Defaults to 1.
            slice_dim (int, optional): dimension for slices. Defaults to 4.
            every_n_epochs (int, optional): frequency of logging. Defaults to
                1.
            generate_kwargs (dict[str, Any], optional): keyword arguments for
                generate function. Defaults to None.
            conditional (bool, optional): generates conditionally on a part of
                the batch. Defaults to False.
            conditional_key (str | int, optional): key for the conditional
                generation on the batch. Defaults to None.
            additional_image_keys (list[str | int], optional): keys for
                additional images which should be logged. Defaults to None.
            rgb (bool, optional): considers the images to be RGB. Defaults to
                False.
        """

        self.size = size
        self.n_slices = n_slices
        self.slice_dim = slice_dim
        self.n_images = n_images
        self.every_n_epochs = every_n_epochs
        self.generate_kwargs = generate_kwargs
        self.conditional = conditional
        self.conditional_key = conditional_key
        self.additional_image_keys = additional_image_keys
        self.rgb = rgb

        if isinstance(self.additional_image_keys, (str, int)):
            self.additional_image_keys = [self.additional_image_keys]

        if self.conditional:
            assert (
                self.conditional_key is not None
            ), "conditional_key must be defined for conditional generation"

    def on_train_epoch_start(self, trainer, pl_module):
        self.storage = {}
        if self.conditional_key is not None:
            self.storage[self.conditional_key] = []
        if self.additional_image_keys is not None:
            for key in self.additional_image_keys:
                self.storage[key] = []

    def on_train_batch_start(self, trainer, pl_module, batch, batch_idx):
        if self.conditional:
            self.storage[self.conditional_key].extend(
                batch[self.conditional_key].detach().to("cpu")
            )
        if self.additional_image_keys:
            for key in self.additional_image_keys:
                self.storage[key].extend(batch[key].detach().to("cpu"))

    def on_train_epoch_end(
        self, trainer: Trainer, pl_module: LightningModule
    ) -> None:
        dev = next(pl_module.parameters()).device
        ep = pl_module.current_epoch
        if ep % self.every_n_epochs == 0 and ep > 0:
            idxs = None
            images_to_log = {}
            with torch.inference_mode():
                if self.generate_kwargs is None:
                    kwargs = {}
                else:
                    kwargs = self.generate_kwargs
                if self.conditional:
                    idxs = np.random.choice(
                        len(self.storage[self.conditional_key]),
                        size=self.n_images,
                    )
                    kwargs["input_tensor"] = torch.stack(
                        [
                            self.storage[self.conditional_key][idx]
                            for idx in idxs
                        ]
                    ).to(dev)
                    images_to_log["Input images"] = kwargs["input_tensor"]
                else:
                    kwargs["size"] = [self.n_images, *self.size]
                if self.additional_image_keys:
                    for key in self.additional_image_keys:
                        if idxs is None:
                            idxs = np.random.choice(
                                len(self.storage[key]), size=self.n_images
                            )
                        images_to_log[f"{key} images"] = torch.stack(
                            [self.storage[key][idx] for idx in idxs]
                        )
                gen_images, cl, reg = pl_module.generate(**kwargs)
                captions = []
                for i in range(gen_images.shape[0]):
                    caption = [None, None]
                    if cl is not None:
                        caption[0] = f"Class: {cl[i]}"
                    if reg is not None:
                        caption[1] = f"Reg: {reg[i].item()}"
                    caption = [c for c in caption if c is not None]
                    if len(caption) > 0:
                        caption = ", ".join(caption)
                    else:
                        caption = ""
                    captions.append(caption)
                images_to_log["Generated images"] = gen_images
            for key in images_to_log:
                log_image(
                    trainer,
                    key=key,
                    images=images_to_log[key],
                    slice_dim=self.slice_dim,
                    n_slices_out=self.n_slices,
                    caption=captions,
                    rgb=self.rgb,
                )


class ModelCheckpointWithMetadata(ModelCheckpoint):
    """
    Identifcal to ModelCheckpoint but allows for metadata to be stored.
    """

    def __init__(
        self,
        metadata: dict[str, Any] = None,
        link_best_as_last: bool = True,
        *args,
        **kwargs,
    ):
        """
        Args:
            metadata (dict[str,Any], optional): dictionary containing all the
                relevant metadata. Defaults to None.
            link_best_as_last (bool, optional):instead of writing, links the
                last checkpoint to the best saved one. Defaults to True
                (default Lightning behaviour).
        """
        super().__init__(*args, **kwargs)
        self.metadata = metadata
        self.link_best_as_last = link_best_as_last

    def state_dict(self) -> dict[str, Any]:
        sd = super().state_dict()
        if self.metadata is not None:
            sd["metadata"] = self.metadata
        return sd

    def _save_last_checkpoint(
        self, trainer: "Trainer", monitor_candidates: dict[str, torch.Tensor]
    ) -> None:
        if not self.save_last:
            return

        filepath = self.format_checkpoint_name(
            monitor_candidates, self.CHECKPOINT_NAME_LAST
        )

        if self._enable_version_counter:
            version_cnt = self.STARTING_VERSION
            while (
                self.file_exists(filepath, trainer)
                and filepath != self.last_model_path
            ):
                filepath = self.format_checkpoint_name(
                    monitor_candidates,
                    self.CHECKPOINT_NAME_LAST,
                    ver=version_cnt,
                )
                version_cnt += 1

        # set the last model path before saving because it will be part of the state.
        previous, self.last_model_path = self.last_model_path, filepath
        if (
            self._fs.protocol == "file"
            and self._last_checkpoint_saved
            and self.save_top_k != 0
            and self.link_best_as_last
        ):
            self._link_checkpoint(
                trainer, self._last_checkpoint_saved, filepath
            )
        else:
            self._save_checkpoint(trainer, filepath)
        if previous and self._should_remove_checkpoint(
            trainer, previous, filepath
        ):
            self._remove_checkpoint(trainer, previous)


class EMACallback(Callback):
    """
    Model Exponential Moving Average. Empirically it has been found that using
    the moving average
    of the trained parameters of a deep network is better than using its trained
    parameters directly.

    If `use_ema_weights`, then the ema parameters of the network is set after
    training end.
    """

    def __init__(
        self,
        decay: float = 0.9999,
        final_decay: float | None = None,
        n_steps: int = None,
        use_ema_weights: bool = True,
        update_train_weights: bool = False,
    ):
        self.decay = decay
        self.final_decay = final_decay
        self.n_steps = n_steps
        self.ema = None
        self.use_ema_weights = use_ema_weights
        self.update_train_weights = update_train_weights

    def on_fit_start(self, trainer: Trainer, pl_module: LightningModule):
        """
        Initialise exponential moving average of the model.
        """
        self.ema = ExponentialMovingAverage(
            decay=self.decay,
            final_decay=self.final_decay,
            n_steps=self.n_steps,
        )

    def on_train_batch_end(
        self,
        trainer,
        pl_module,
        outputs,
        batch,
        batch_idx,
    ):
        """
        Updates EMA of the model.
        """
        # Update currently maintained parameters.
        self.ema.update(pl_module)
        if self.update_train_weights:
            self.copy_to(self.ema.parameters(), pl_module.parameters())

    def on_validation_epoch_start(
        self, trainer: Trainer, pl_module: LightningModule
    ):
        """
        Validation is performed using EMA parameters.
        """
        if self.update_train_weights is False:
            self.store(pl_module.parameters())
            self.copy_to(self.ema.parameters(), pl_module.parameters())

    def on_validation_end(self, trainer, pl_module):
        """
        Restores original parameters after validation.
        """
        if self.update_train_weights is False:
            self.restore(pl_module.parameters())

    def on_train_end(self, trainer: Trainer, pl_module: LightningModule):
        """
        Update module weights to EMA version.
        """
        if self.use_ema_weights:
            self.copy_to(self.ema.parameters(), pl_module.parameters())

    def on_save_checkpoint(self, trainer, pl_module, checkpoint):
        """
        Saves EMA weights.
        """
        checkpoint["state_dict_ema"] = self.ema.state_dict()

    def on_load_checkpoint(self, trainer, pl_module, checkpoint):
        """
        Restores EMA weights.
        """
        if self.ema is not None:
            self.ema.load_state_dict(checkpoint["state_dict_ema"])

    def store(self, parameters):
        """
        Saves EMA weights in ``self.collected_params``.
        """
        self.collected_params = [param.clone() for param in parameters]

    def restore(self, parameters):
        """
        Restore the parameters stored with the `store` method.
        Useful to validate the model with EMA parameters without affecting the
        original optimization process.
        """
        for c_param, param in zip(self.collected_params, parameters):
            param.data.copy_(c_param.data)

    def copy_to(self, shadow_parameters, parameters):
        """
        Copy current parameters into given collection of parameters.
        """
        for s_param, param in zip(shadow_parameters, parameters):
            if param.requires_grad:
                param.data.copy_(s_param.data)
