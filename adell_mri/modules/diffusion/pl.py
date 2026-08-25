"""
Lightning module for the diffusion model. Uses the MONAI ``generative`` package
to do all of the heavy lifting and combines it with flexible condition embedding
capabilities.
"""

from typing import Callable, List

import lightning.pytorch as pl
import numpy as np
import torch
from generative.inferers import DiffusionInferer
from generative.networks.nets import DiffusionModelUNet
from generative.networks.nets.diffusion_model_unet import AttentionBlock

from adell_mri.constants import DEFAULT_SEED
from adell_mri.modules.diffusion.embedder import Embedder
from adell_mri.modules.diffusion.scheduler import DDPMScheduler
from adell_mri.modules.learning_rate import CosineAnnealingWithWarmupLR
from adell_mri.utils.optimizer_factory import OPTIMIZER_EPS_DEFAULT
from adell_mri.utils.torch_utils import get_global_rank, meta_tensors_to_tensors


class DiffusionUNetPL(DiffusionModelUNet, pl.LightningModule):
    def __init__(
        self,
        inferer: Callable = DiffusionInferer,
        scheduler: Callable = DDPMScheduler,
        embedder: Embedder = None,
        image_key: str = "image",
        cat_condition_key: str = "cat_condition",
        num_condition_key: str = "num_condition",
        concat_condition_key: str = None,
        uncondition_proba: float = 0.0,
        n_epochs: int = 100,
        warmup_steps: int = 0,
        start_decay: int = 0,
        training_dataloader_call: Callable = None,
        batch_size: int = 16,
        learning_rate: float = 0.001,
        weight_decay: float = 0.0,
        seed: int = DEFAULT_SEED,
        optimizer_eps: float = OPTIMIZER_EPS_DEFAULT,
        *args,
        **kwargs,
    ):
        super().__init__(*args, **kwargs)

        self.inferer = inferer
        self.scheduler = scheduler
        self.embedder = embedder
        self.image_key = image_key
        self.cat_condition_key = cat_condition_key
        self.num_condition_key = num_condition_key
        self.concat_condition_key = concat_condition_key
        self.uncondition_proba = uncondition_proba
        self.n_epochs = n_epochs
        self.warmup_steps = warmup_steps
        self.start_decay = start_decay
        self.training_dataloader_call = training_dataloader_call
        self.batch_size = batch_size
        self.learning_rate = learning_rate
        self.weight_decay = weight_decay
        self.seed = seed
        self.optimizer_eps = optimizer_eps

        self.g = torch.Generator()
        self.g.manual_seed(self.seed + get_global_rank())
        self._device_generators = {}
        self.rng = np.random.default_rng(self.seed + get_global_rank())
        self.noise_steps = self.scheduler.num_train_timesteps
        self.loss_fn = torch.nn.MSELoss()

        self._prune_unused_parameters()

    def _prune_unused_parameters(self):
        """
        Removes parameters that are never used during training so that
        DDP can be used without ``find_unused_parameters=True``.

        This is all because ``DiffusionModelUNet`` always gets instatiated with
        ``proj_attn``, which is unncessary when training without conditioning.
        """
        if self.with_conditioning:
            pass
        else:
            for module in self.modules():
                if isinstance(module, AttentionBlock):
                    del module.proj_attn

    def calculate_loss(
        self, prediction: torch.Tensor, epsilon: torch.Tensor
    ) -> torch.Tensor:
        """
        Calculates the loss between the predicted noise and the true noise.

        Args:
            prediction (torch.Tensor): The predicted noise.
            epsilon (torch.Tensor): The true noise.

        Returns:
            torch.Tensor: The mean loss.
        """
        loss = self.loss_fn(prediction, epsilon)
        return loss.mean()

    def _device_generator(self, device: torch.device) -> torch.Generator:
        """
        Returns a generator seeded with the instance seed for the given
        device, creating it if necessary. Sampling random tensors directly
        on the target device requires a device-local generator.

        Args:
            device (torch.device): target device.

        Returns:
            torch.Generator: generator for the target device.
        """
        key = (device.type, device.index)
        if key not in self._device_generators:
            generator = torch.Generator(device=device)
            generator.manual_seed(self.seed + get_global_rank())
            self._device_generators[key] = generator
        return self._device_generators[key]

    def randn_like(self, x: torch.Tensor) -> torch.Tensor:
        """
        Generates a tensor of random normal values with the same shape as the
        input tensor `x`.

        Args:
            x (torch.Tensor): The input tensor whose shape will be used to
                generate the random tensor.

        Returns:
            torch.Tensor: A tensor of random normal values with the same shape
                as `x`.
        """
        return torch.randn(
            size=x.shape,
            dtype=x.dtype,
            layout=x.layout,
            device=x.device,
            generator=self._device_generator(x.device),
        )

    def timesteps_like(self, x: torch.Tensor) -> torch.Tensor:
        """
        Generates a tensor of random integer values between 0 and
        `self.noise_steps` with the same batch size as the input tensor `x`.

        Args:
            x (torch.Tensor): The input tensor whose batch size will be used to
                generate the random tensor.

        Returns:
            torch.Tensor: A tensor of random integer values between 0 and
                `self.noise_steps` with the same batch size as `x`.
        """
        return torch.randint(
            0,
            self.noise_steps,
            (x.shape[0],),
            device=x.device,
            generator=self._device_generator(x.device),
        ).long()

    def step(
        self,
        x: torch.Tensor,
        timesteps: torch.Tensor = None,
        context: torch.Tensor = None,
        concat_condition: torch.Tensor = None,
    ) -> torch.Tensor:
        """
        Computes the loss for a single step of the diffusion model.

        Args:
            x (torch.Tensor): The input tensor to the diffusion model.
            timesteps (torch.Tensor, optional): The timesteps for the
                diffusion. If not provided (None), they will be generated
                randomly. Defaults to None.
            context (torch.Tensor, optional): The conditioning context for the
                diffusion model. Defaults to None.
            concat_condition (torch.Tensor, optional): conditioning tensor
                concatenated along the channel dimension of the noisy input.
                Defaults to None.

        Returns:
            torch.Tensor: The mean loss for the current step.

        """
        epsilon = self.randn_like(x)
        if timesteps is None:
            timesteps = self.timesteps_like(x)
        else:
            timesteps = timesteps.long()
        inferer_kwargs = {
            "inputs": x,
            "diffusion_model": self,
            "noise": epsilon,
            "timesteps": timesteps,
            "condition": context,
        }
        if concat_condition is not None:
            inferer_kwargs["concat_condition"] = concat_condition
        epsilon_pred = self.inferer(**inferer_kwargs)
        loss = self.calculate_loss(epsilon_pred, epsilon)
        return loss

    def unpack_batch(
        self, batch: dict[str, torch.Tensor]
    ) -> tuple[torch.Tensor, torch.Tensor | None, torch.Tensor | None]:
        """
        Convenience function to unpack a batch for model training.

        Args:
            batch (dict[str, torch.Tensor]): dictionary containing the correct
                entries for each batch. Should have inputs corresponding to
                ``self.image_key`` and to conditioning keys (i.e.
                ``self.cat_condition_key``, ``self.num_condition_key`` and
                ``self.concat_condition_key``) if conditioning is required.

        Returns:
            tuple[torch.Tensor, torch.Tensor, torch.Tensor]: tensor with input
                image, embedded condition (if provided) and concatenated
                conditioning channels (if provided).
        """
        x = batch[self.image_key]
        concat_condition = None
        if self.concat_condition_key is not None:
            concat_condition = batch[self.concat_condition_key]
        if self.with_conditioning is True:
            uncondition = (
                "all" if self.rng.random() < self.uncondition_proba else None
            )
            if self.cat_condition_key is not None:
                cat_condition = batch[self.cat_condition_key]
            else:
                cat_condition = None
            if self.num_condition_key is not None:
                num_condition = batch[self.num_condition_key]
            else:
                num_condition = None
            # expects three dimensions (batch, seq, embedding size)
            condition = self.embedder(
                cat_condition,
                num_condition,
                uncondition_cat_idx=uncondition,
                uncondition_num_idx=uncondition,
            )

            if len(condition.shape) < 3:
                condition = condition.unsqueeze(1)
        else:
            condition = None
        return x, condition, concat_condition

    def on_before_batch_transfer(
        self, batch: dict, dataloader_idx: int
    ) -> dict:
        """
        Lightning hook to convert MONAI metatensors to tensors.

        Args:
            batch (dict): batch for lightning step.
            dataloader_idx (int): index for the dataloader (not used).

        Returns:
            dict: batch with metatensors converted to tensors.
        """
        return meta_tensors_to_tensors(batch)

    def training_step(self, batch: dict, batch_idx: int) -> torch.Tensor:
        """
        Training step hook for lightning.

        Args:
            batch (dict): batch.
            batch_idx (int): batch index.

        Returns:
            torch.Tensor: loss value.
        """
        x, condition, concat_condition = self.unpack_batch(batch)
        loss = self.step(
            x, context=condition, concat_condition=concat_condition
        )
        self.log("loss", loss, on_step=True, prog_bar=True)
        return loss

    def validation_step(self, batch: dict, batch_idx: int):
        """
        Validation step hook for lightning.

        Args:
            batch (dict): batch.
            batch_idx (int): batch index.

        Returns:
            torch.Tensor: loss value.
        """
        x, condition, concat_condition = self.unpack_batch(batch)
        loss = self.step(
            x, context=condition, concat_condition=concat_condition
        )
        self.log("val_loss", loss, on_epoch=True, prog_bar=True)
        return loss

    def test_step(self, batch: dict, batch_idx: int):
        """
        Test step hook for lightning.

        Args:
            batch (dict): batch.
            batch_idx (int): batch index.

        Returns:
            torch.Tensor: loss value.
        """
        x, condition, concat_condition = self.unpack_batch(batch)
        loss = self.step(
            x, context=condition, concat_condition=concat_condition
        )
        self.log("test_loss", loss)
        return loss

    @property
    def device(self) -> torch.device:
        """
        Convenience function that returns the device where the model parameters
        are hosted.

        Returns:
            torch.device: the torch device.
        """
        return next(self.parameters()).device

    @torch.inference_mode
    def generate_image(
        self,
        size: List[int],
        n: int,
        input_image: torch.Tensor = None,
        skip_steps: int = 0,
        cat_condition: torch.Tensor = None,
        num_condition: torch.Tensor = None,
        uncondition_cat_idx: int | list[int] | None = None,
        uncondition_num_idx: int | list[int] | None = None,
        guidance_strength: float = 1.0,
        concat_condition: torch.Tensor = None,
    ) -> torch.Tensor:
        """
        Generates an image using the learned diffusion model. Can be used for:
            - pure generation (if no input image is provided)
            - for vector-conditional generation (if categorical or numerical
            conditions are provided)
            - for image-conditioned re-generation (i.e. the image goes through
            part of the diffusion process in a way that only partially destroys
            the content and the rest of the process is recapitulated with
            standard DDPM)
            - for image/mask-conditioned generation through channel
            concatenation (if ``concat_condition`` is provided)

        Part of this support also involves using non-conditioned inputs through
        ``uncondition_cat_idx`` and ``uncondition_num_idx``. In theory, this
        should generate images which are not conditioned on anything in
        particular.

        Args:
            size (List[int]): size (shape) of the output image.
            n (int): number of generated images.
            input_image (torch.Tensor, optional): input image for conditional
                generation or for generating images using classifier guidance or
                similar approaches. Defaults to None.
            skip_steps (int, optional): number of steps that should be skipped
                from the backwards diffusion process. Defaults to 0.
            cat_condition (torch.Tensor, optional): categorical condition.
                Defaults to None.
            num_condition (torch.Tensor, optional): numerical condition.
                Defaults to None.
            uncondition_cat_idx (int | list[int] | None, optional): indices
                corresponding to the non-conditioned categorical conditions
                (uses the learned representation for non-conditional
                generation). Defaults to None.
            uncondition_num_idx (int | list[int] | None, optional): indices
                corresponding to the non-conditioned numerical conditions (uses
                the learned representation for non-conditional generation).
                Defaults to None.
            guidance_strength (float, optional): strength of the classifier
                guidance. Defaults to 1.0.
            concat_condition (torch.Tensor, optional): conditioning tensor
                (e.g. concatenation of input images and one-hot encoded masks)
                concatenated along the channel dimension of the sampled image.
                Can be combined with ``cat_condition``/``num_condition`` for
                mixed conditioning. Defaults to None.

        Returns:
            torch.Tensor: generated (or re-generated) sample.
        """
        noise = torch.randn([n, self.out_channels, *size], device=self.device)
        if input_image is None:
            input_image = noise
        else:
            input_image = self.inferer.scheduler.add_noise(
                original_samples=input_image,
                noise=noise[0].to(input_image),
                timesteps=torch.as_tensor(
                    self.scheduler.num_train_timesteps - skip_steps
                ),
            )
        condition = None
        uncondition = None
        if self.embedder is not None:
            condition = self.embedder(
                X_cat=cat_condition,
                X_num=num_condition,
                batch_size=n,
                update_queues=False,
            )
            uncondition_idx = {
                "uncondition_cat_idx": (
                    uncondition_cat_idx
                    if uncondition_cat_idx is not None
                    else "all"
                ),
                "uncondition_num_idx": (
                    uncondition_num_idx
                    if uncondition_num_idx is not None
                    else "all"
                ),
            }
            uncondition = self.embedder(
                X_cat=cat_condition,
                X_num=num_condition,
                batch_size=n,
                update_queues=False,
                **uncondition_idx,
            )
            if len(condition.shape) < 3:
                condition = condition.unsqueeze(1)
            if uncondition is not None and len(uncondition.shape) < 3:
                uncondition = uncondition.unsqueeze(1)
        sample = self.inferer.sample(
            input_noise=input_image,
            diffusion_model=self,
            scheduler=self.scheduler,
            conditioning=condition,
            concat_condition=concat_condition,
            unconditioning=uncondition,
            skip_steps=skip_steps,
            guidance_strength=guidance_strength,
        )

        return sample

    def train_dataloader(self) -> torch.utils.data.DataLoader:
        """
        Lightning hook which returns the training data loader for the model.

        Returns:
            torch.utils.data.DataLoader: The training data loader.
        """
        return self.training_dataloader_call(self.batch_size)

    def configure_optimizers(
        self,
    ) -> dict[
        str,
        torch.optim.Optimizer | torch.optim.lr_scheduler._LRScheduler | str,
    ]:
        """
        Lightning hook for optimizer configuration.

        Returns:
            a dictionary containing the optimizer, the learning rate scheduler
                and the metric which is monitored during training.
        """
        optimizer = torch.optim.AdamW(
            self.parameters(),
            lr=self.learning_rate,
            weight_decay=self.weight_decay,
            eps=self.optimizer_eps,
        )
        lr_schedulers = CosineAnnealingWithWarmupLR(
            optimizer,
            T_max=self.n_epochs,
            start_decay=self.start_decay,
            n_warmup_steps=self.warmup_steps,
            eta_min=0.0,
        )

        return {
            "optimizer": optimizer,
            "lr_scheduler": lr_schedulers,
            "monitor": "val_loss",
        }
