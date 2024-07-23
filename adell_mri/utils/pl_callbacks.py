import numpy as np
import torch
import torch.nn.functional as F
import lightning.pytorch as pl
from typing import Sequence, Any
from copy import deepcopy
from PIL import Image
from lightning.pytorch import LightningModule, Trainer
from lightning.pytorch.callbacks import Callback
from lightning.pytorch.callbacks.model_checkpoint import ModelCheckpoint


def coerce_to_uint8(x: np.ndarray):
    x = (x - x.min()) / (x.max() - x.min()) * 255
    return x.astype(np.uint8)


def split_and_cat(x: np.ndarray, split_dim: int, cat_dim: int) -> np.ndarray:
    arrays = np.split(x, x.shape[split_dim], axis=split_dim)
    arrays = np.concatenate(
        [arr.squeeze(split_dim) for arr in arrays], cat_dim
    )
    return arrays


def log_image(
    trainer: Trainer,
    key: str,
    images: list[torch.Tensor],
    slice_dim: int | None = None,
    n_slices_out: int | None = None,
    caption: list[str] = None,
):
    """Logs images to the PyTorch Lightning logger.

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
        caption (list[str]): Optional list of captions, one for each image.
    """
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
    images = torch.split(images, 1, 0)
    images = [x.squeeze(0).permute(1, 2, 0).numpy() for x in images]
    images = [coerce_to_uint8(split_and_cat(x, -1, 0)) for x in images]
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
    def __init__(self, power_iterations, eps=1e-8, name="weight"):
        """
        Callback that performs spectral normalization before each training
        batch. It uses the same power iteration implementation as specified in
        [1] and is largely based in the PyTorch implementation [2].

        Importantly, this stores u and v as a parameter dict within the
        callback rather than as a part of

        [1] https://arxiv.org/abs/1802.05957
        [2] https://pytorch.org/docs/stable/_modules/torch/nn/utils/spectral_norm.html

        Args:
            power_iterations (_type_): _description_
            eps (_type_, optional): _description_. Defaults to 1e-8.
            name (str, optional): _description_. Defaults to "weight".
        """
        self.power_iterations = power_iterations
        self.eps = eps
        self.name = name

        self.u_dict = torch.nn.Parameterdict({})
        self.v_dict = torch.nn.Parameterdict({})

    def on_train_batch_start(
        self,
        trainer: pl.Trainer,
        pl_module: pl.LightningModule,
        batch: Sequence,
        batch_idx: int,
    ):
        return self(pl_module)

    def __call__(self, module):
        for k, param in module.named_parameters():
            if self.name in k:
                weight = deepcopy(param.data)
                sh = weight.shape
                weight_mat = reshape_weight_to_matrix(weight)
                h, w = weight_mat.size()
                if k not in self.u_dict:
                    u = weight.new_empty(h).normal_(0, 1)
                    v = weight.new_empty(w).normal_(0, 1)
                    self.u_dict[k.replace(".", "_")] = torch.nn.Parameter(u)
                    self.v_dict[k.replace(".", "_")] = torch.nn.Parameter(v)

                u = self.u_dict[k.replace(".", "_")].data
                v = self.v_dict[k.replace(".", "_")].data

                with torch.no_grad():
                    for p in range(self.power_iterations):
                        v = F.normalize(
                            torch.mv(weight_mat.t(), u),
                            dim=0,
                            eps=self.eps,
                            out=v,
                        )
                        u = F.normalize(
                            torch.mv(weight_mat, v), dim=0, eps=self.eps, out=u
                        )

                self.u_dict[k.replace(".", "_")].data = u
                self.v_dict[k.replace(".", "_")].data = v

                sigma = torch.dot(u, torch.mv(weight_mat, v))
                weight = weight / sigma

                param.data = weight.reshape(sh)


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
                        key,
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
            with torch.inference_mode():
                images = pl_module.generate_image(
                    size=self.size, n=self.n_images
                )
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
        """

        self.size = size
        self.n_slices = n_slices
        self.slice_dim = slice_dim
        self.n_images = n_images
        self.every_n_epochs = every_n_epochs
        self.generate_kwargs = generate_kwargs

    def on_train_epoch_end(
        self, trainer: Trainer, pl_module: LightningModule
    ) -> None:
        ep = pl_module.current_epoch
        if ep % self.every_n_epochs == 0 and ep > 0:
            with torch.inference_mode():
                if self.generate_kwargs is None:
                    images = pl_module.generate(
                        size=[self.n_images, *self.size]
                    )
                else:
                    images = pl_module.generate(
                        size=[self.n_images, *self.size],
                        **self.generate_kwargs,
                    )
            log_image(
                trainer,
                key="Generated images",
                images=images,
                slice_dim=self.slice_dim,
                n_slices_out=self.n_slices,
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
