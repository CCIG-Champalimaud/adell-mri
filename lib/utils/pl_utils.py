"""
Functions that return correctly configured callbacks and objects for PyTorch
Lightning. 
"""

import os
import atexit
import numpy as np
import torch
import wandb
from PIL import Image
from lightning import Trainer
from lightning.pytorch import LightningModule, Trainer
from lightning.pytorch.callbacks import Callback
from lightning.pytorch.loggers import WandbLogger
from lightning.pytorch.callbacks.model_checkpoint import ModelCheckpoint

from typing import List, Union, Tuple, Any, Dict


def coerce_to_uint8(x: np.ndarray):
    x = (x - x.min()) / (x.max() - x.min()) * 255
    return x.astype(np.uint8)


def log_image(
    trainer: Trainer,
    key: str,
    images: List[torch.Tensor],
    slice_dim: int,
    n_slices_out: int,
    caption: List[str] = None,
):
    images = images.detach().to("cpu")
    if len(images.shape) == 5:
        n_slices = images.shape[slice_dim]
        slice_idxs = np.arange(0, n_slices, n_slices_out + 2)[1:-1]
        images = torch.index_select(
            images, slice_dim, torch.as_tensor(slice_idxs)
        )
        images = torch.split(images, 1, dim=slice_dim)
        images = torch.cat(images, -2).squeeze(-1)
    images = torch.split(images, 1, 0)
    images = [x.squeeze(0).permute(1, 2, 0).numpy() for x in images]
    images = [coerce_to_uint8(x).squeeze(-1) for x in images]
    images = [Image.fromarray(x) for x in images]

    if caption is not None:
        trainer.logger.log_image(key=key, images=images, caption=caption)
    else:
        trainer.logger.log_image(key=key, images=images)


def allocated_memory_per_gpu() -> Dict[int, int]:
    """
    Returns a dictionary with the allocated memory per GPU.

    Returns:
        Dict: dictionary with GPU ids (0,1,2,...) as keys and the amount of
            allocated memory as values.
    """
    output = {}
    for i in range(torch.cuda.device_count()):
        output[i] = torch.cuda.memory_allocated(i)
    return output


def get_emptiest_gpus(n: int = 1) -> List[int]:
    """
    Gets the ids for the n emptiest GPUs.

    Args:
        n (int, optional): number of ids to retrieve. Defaults to 1.

    Returns:
        int:
    """
    mem = allocated_memory_per_gpu()
    least_mem_gpu = sorted(
        allocated_memory_per_gpu().keys(), key=lambda i: mem[i]
    )[:n]
    return least_mem_gpu


class LogImage(Callback):
    def __init__(
        self,
        image_keys: List[str] = "image",
        caption_keys: List[str] = None,
        output_idxs: List[int] = None,
        n_slices: int = 1,
        slice_dim: int = 4,
        log_frequency: int = 5,
    ):
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
        outputs: Tuple[Any],
        batch: Dict[str, Any],
        batch_idx: int,
        dataloader_idx: int = 0,
    ):
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
        size: List[int],
        n_slices: int = 3,
        slice_dim: int = 4,
        n_images: int = 2,
        every_n_epochs: int = 1,
    ):
        self.size = size
        self.n_slices = n_slices
        self.slice_dim = slice_dim
        self.n_images = n_images
        self.every_n_epochs = every_n_epochs

    def on_validation_epoch_end(
        self, trainer: Trainer, pl_module: LightningModule
    ):
        ep = pl_module.current_epoch
        if ep % self.every_n_epochs == 0 and ep > 0:
            images = pl_module.generate_image(size=self.size, n=self.n_images)
            log_image(
                trainer,
                key="Generated images",
                images=images,
                slice_dim=self.slice_dim,
                n_slices_out=self.slice_dim,
            )


class GPULock:
    def __init__(self, path: str = ".gpu_lock"):
        self.path = path

        self.locked_gpus = []
        self.available_devices = [
            str(i) for i in range(torch.cuda.device_count())
        ]
        atexit.register(self.unlock_all)

    def get_locked_gpus(self):
        if os.path.exists(self.path):
            with open(self.path, "r") as o:
                locked_gpus = [
                    x.strip() for x in o.readlines() if x.strip() != ""
                ]
            return locked_gpus
        else:
            return []

    def lock(self, i: int):
        i = str(i)
        if i not in self.available_devices:
            raise Exception(
                f"GPU {i} not in available devices {self.available_devices}"
            )
        locked_gpus = self.get_locked_gpus()
        if i not in locked_gpus + self.locked_gpus:
            locked_gpus.append(i)
        else:
            raise Exception(f"GPU {i} is already locked")
        with open(self.path, "w") as o:
            o.write("\n".join(locked_gpus))
        self.locked_gpus.append(i)

    def unlock(self, i: int):
        i = str(i)
        locked_gpus = [x for x in self.get_locked_gpus() if x != i]
        with open(self.path, "w") as o:
            o.write("\n".join(locked_gpus))

    def lock_first_available(self):
        locked_gpus = self.get_locked_gpus()
        gpu_to_lock = None
        for i in range(torch.cuda.device_count()):
            i = str(i)
            if i not in locked_gpus:
                gpu_to_lock = i
                break
        if gpu_to_lock is None:
            raise Exception(f"No available GPUs to lock")
        self.lock(gpu_to_lock)
        return gpu_to_lock

    def unlock_all(self):
        print(f"Unlocking GPUs {self.locked_gpus}")
        locked_gpus = self.get_locked_gpus()
        for locked_gpu in locked_gpus:
            self.unlock(locked_gpu)


class ModelCheckpointWithMetadata(ModelCheckpoint):
    """Identifcal to ModelCheckpoint but allows for metadata to be stored."""

    def __init__(
        self,
        metadata: Dict[str, Any] = None,
        link_best_as_last: bool = True,
        *args,
        **kwargs,
    ):
        """
        Args:
            metadata (Dict[str,Any], optional): dictionary containing all the
                relevant metadata. Defaults to None.
            link_best_as_last (bool, optional):instead of writing, links the
                last checkpoint to the best saved one. Defaults to True
                (default Lightning behaviour).
        """
        super().__init__(*args, **kwargs)
        self.metadata = metadata
        self.link_best_as_last = link_best_as_last

    def state_dict(self) -> Dict[str, Any]:
        sd = super().state_dict()
        if self.metadata is not None:
            sd["metadata"] = self.metadata
        return sd

    def _save_last_checkpoint(
        self, trainer: "Trainer", monitor_candidates: Dict[str, torch.Tensor]
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


def delete_checkpoints(trainer: Trainer) -> None:
    """Convenience function to delete checkpoints.

    Args:
        trainer (Trainer): a Lightning Trainer object.
    """

    def delete(path: str, verbose: bool = False) -> None:
        if os.path.exists(path):
            os.remove(path)
        elif verbose is True:
            print(f"Warning: {path} does not exist and has not been deleted")

    if hasattr(trainer, "checkpoint_callbacks"):
        for ckpt_callback in trainer.checkpoint_callbacks:
            if hasattr(ckpt_callback, "best_model_path"):
                if isinstance(ckpt_callback.best_model_path, (list, tuple)):
                    for bmp in ckpt_callback.best_model_path:
                        delete(bmp)
                else:
                    delete(ckpt_callback.best_model_path)
            if hasattr(ckpt_callback, "last_model_path"):
                delete(ckpt_callback.last_model_path)


def get_step_information(
    max_epochs: int,
    steps_per_epoch: int,
    warmup_epochs: int,
    accumulate_grad_batches: int,
    n_devices: int,
    n_images: int,
    batch_size: int,
) -> Tuple[int, int, int, int, int]:
    """Gets step information (maximum number of steps, maximum number of
    optimizer steps, number of warmup steps how often validation checks are run
    and the validation check interval) from the maximum number of epochs, the
    number of steps per epochs, the number of warmup epochs, for how many batches
    gradients are accumulated, how many devices the network is being runned on,
    the total number of images used for training and the batch size.

    This does all the heavy work that is necessary for LR schedulers and for the
    Lightning optimizer.

    If the number of steps per epoch is `None`, then max_steps is -1 as this is
    what is expected by Lightning.

    Args:
        max_epochs (int): maximum number of epochs.
        steps_per_epoch (int): number of steps per epoch.
        warmup_epochs (int): number of warmup steps.
        accumulate_grad_batches (int): number of gradient accumulation steps.
        n_devices (int): number of traing devices.
        n_images (int): number of images (samples) in dataset.
        batch_size (int): batch size.

    Returns:
        max_steps (int) - total number of steps.
        max_step_params (int) - total number of optimizer steps.
        warmup_steps (int) -
        check_val_every_n_epoch (int)
        val_check_interval (int)
    """
    agb = accumulate_grad_batches
    if steps_per_epoch is not None:
        steps_per_epoch = steps_per_epoch
        steps_per_epoch_optim = int(np.ceil(steps_per_epoch / agb))
        max_steps = max_epochs * steps_per_epoch
        max_epochs = -1
        max_steps_optim = max_epochs * steps_per_epoch_optim
        warmup_steps = warmup_epochs * steps_per_epoch_optim
        check_val_every_n_epoch = None
        val_check_interval = 5 * steps_per_epoch
    else:
        bs = batch_size
        steps_per_epoch = n_images // (bs * n_devices)
        steps_per_epoch = int(np.ceil(steps_per_epoch / agb))
        max_steps = -1
        max_steps_optim = max_epochs * steps_per_epoch
        warmup_steps = warmup_epochs * steps_per_epoch
        check_val_every_n_epoch = 5
        val_check_interval = None

    warmup_steps = int(warmup_steps)
    max_steps_optim = int(max_steps_optim)
    return (
        max_steps,
        max_steps_optim,
        warmup_steps,
        check_val_every_n_epoch,
        val_check_interval,
    )


def get_ckpt_callback(
    checkpoint_dir: str,
    checkpoint_name: str,
    max_epochs: int,
    max_steps: int = None,
    resume_from_last: bool = False,
    val_fold: int = None,
    monitor: str = "val_loss",
    n_best_ckpts: int = 1,
    metadata: dict = None,
) -> ModelCheckpoint:
    """Gets a checkpoint callback for PyTorch Lightning. The format for
    for the last and 2 best checkpoints, respectively is:
    1. "{name}_fold{fold}_last.ckpt"
    2. "{name}_fold{fold}_best_{epoch}_{monitor:.3f}.ckpt"

    Args:
        checkpoint_dir (str): directory where checkpoints will be stored.
        checkpoint_name (str): root name for checkpoint.
        max_epochs (int): maximum number of training epochs (used to check if
            training has finished when resume_from_last==True).
        max_steps (int, optional): maximum number of training steps (used to
            check if training has finished when resume_from_last==True).
            Defaults to None.
        resume_from_last (bool, optional): whether training should be resumed in
            case a checkpoint is detected. Defaults to True.
        val_fold (int, optional): ID for the validation fold. Defaults to None.
        monitor (str, optional): metric which should be monitored when defining
            the best checkpoints. Defaults to "val_loss".
        n_best_ckpts (int, optional): number of best performing models to be
            saved. Defaults to 1.
        metadata (dict, optional): metadata to store with checkpoint. Defaults
            to None.

    Returns:
        ModelCheckpoint: PyTorch Lightning checkpoint callback.
    """
    ckpt_path = None
    ckpt_callback = None
    status = None

    if (checkpoint_dir is not None) and (checkpoint_name is not None):
        if val_fold is not None:
            ckpt_name = checkpoint_name + "_fold" + str(val_fold)
            ckpt_last = checkpoint_name + "_fold" + str(val_fold)
        else:
            ckpt_name = checkpoint_name
            ckpt_last = checkpoint_name
        ckpt_name = ckpt_name + "_best_{epoch}_{" + monitor + ":.3f}"
        if "loss" in monitor:
            mode = "min"
        else:
            mode = "max"
        ckpt_callback = ModelCheckpointWithMetadata(
            dirpath=checkpoint_dir,
            filename=ckpt_name,
            monitor=monitor,
            save_last=True,
            save_top_k=n_best_ckpts,
            mode=mode,
            metadata=metadata,
            link_best_as_last=False,
        )

        ckpt_last = ckpt_last + "_last"
        ckpt_callback.CHECKPOINT_NAME_LAST = ckpt_last
        ckpt_last_full = os.path.join(checkpoint_dir, ckpt_last + ".ckpt")
        if os.path.exists(ckpt_last_full) and resume_from_last is True:
            ckpt_path = ckpt_last_full
            if max_steps is not None:
                value = max_steps
                key = "step"
            else:
                value = max_epochs
                key = "epoch"
            ckpt_value = torch.load(ckpt_path)[key]
            if ckpt_value >= (value - 1):
                print("Training has finished for this fold, skipping")
                status = "finished"
            else:
                print(
                    f"Resuming training from checkpoint in {ckpt_path} ({key}={ckpt_value})"
                )
    return ckpt_callback, ckpt_path, status


def get_logger(
    summary_name: str,
    summary_dir: str,
    project_name: str,
    resume: str,
    fold: int = None,
) -> WandbLogger:
    """Defines a Wandb logger for PyTorch Lightning. Each run is configured
    as "{project_name}/{summary_name}_fold{fold}".

    Args:
        summary_name (str): name of the Wandb run.
        summary_dir (str): directory where summaries are stored.
        project_name (str): name of the Wandb project.
        resume (str): how the metric registry in Wandb should be resumed.
            Details in https://docs.wandb.ai/guides/track/advanced/resuming.
        fold (int, optional): ID for the validation fold. Defaults to None.

    Returns:
        WandbLogger: _description_
    """
    if (summary_name is not None) and (project_name is not None):
        wandb.finish()
        wandb_resume = resume
        if wandb_resume == "none":
            wandb_resume = None
        run_name = summary_name.replace(":", "_")
        if fold is not None:
            run_name = run_name + f"_fold{fold}"
        logger = WandbLogger(
            save_dir=summary_dir,
            project=project_name,
            name=run_name,
            version=run_name,
            reinit=True,
            resume=wandb_resume,
            dir=summary_dir,
        )
    else:
        logger = None
    return logger


def get_devices(
    device_str: str, strategy: str = "ddp_find_unused_parameters_true"
) -> Tuple[str, Union[List[int], int], str]:
    """Takes a string with form "{device}:{device_ids}" where device_ids is a
    comma separated list of device IDs (i.e. cuda:0,1).

    Args:
        device_str (str): device string. Can be "cpu" or "cuda" if no
            parallelization is necessary or "cuda:0,1" if training is to be
            distributed across GPUs 0 and 1, for instance.
        strategy (str): parallelization strategy. Defaults to "ddp".

    Returns:
        Tuple[str,Union[List[int],int],str]: a tuple containing the accelerator
            ("cpu" or "gpu") the devices (None or a list of devices as
            specified after the ":" in the device_str) and the parallelization
            strategy ("ddp" if len(devices) > 0, None otherwise)
    """
    strategy_out = "auto"
    if ":" in device_str:
        accelerator = "gpu" if "cuda" in device_str else "cpu"
        try:
            devices = [int(i) for i in device_str.split(":")[-1].split(",")]
        except:
            devices = "auto"
        if len(devices) > 1:
            strategy_out = strategy
    else:
        accelerator = "gpu" if "cuda" in device_str else "cpu"
        devices = 1
    return accelerator, devices, strategy_out
