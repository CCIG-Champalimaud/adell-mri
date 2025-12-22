"""
Lightning module for the diffusion model. Uses the MONAI ``generative`` package
to do all of the heavy lifting and combines it with flexible condition embedding
capabilities.
"""

from typing import Callable, List

import lightning.pytorch as pl
import numpy as np
import torch

from adell_mri.modules.classification.pl import meta_tensors_to_tensors
from adell_mri.modules.diffusion.embedder import Embedder
from adell_mri.modules.learning_rate import CosineAnnealingWithWarmupLR
from generative.inferers import DiffusionInferer
from generative.networks.nets import DiffusionModelUNet
from generative.networks.schedulers import DDPMScheduler


class DiffusionUNetPL(DiffusionModelUNet, pl.LightningModule):
    def __init__(
        self,
        inferer: Callable = DiffusionInferer,
        scheduler: Callable = DDPMScheduler,
        embedder: Embedder = None,
        image_key: str = "image",
        cat_condition_key: str = "cat_condition",
        num_condition_key: str = "num_condition",
        uncondition_proba: float = 0.0,
        n_epochs: int = 100,
        warmup_steps: int = 0,
        start_decay: int = 0,
        training_dataloader_call: Callable = None,
        batch_size: int = 16,
        learning_rate: float = 0.001,
        weight_decay: float = 0.0,
        seed: int = 42,
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
        self.uncondition_proba = uncondition_proba
        self.n_epochs = n_epochs
        self.warmup_steps = warmup_steps
        self.start_decay = start_decay
        self.training_dataloader_call = training_dataloader_call
        self.batch_size = batch_size
        self.learning_rate = learning_rate
        self.weight_decay = weight_decay
        self.seed = seed

        self.g = torch.Generator()
        self.g.manual_seed(self.seed)
        self.rng = np.random.default_rng(self.seed)
        self.noise_steps = self.scheduler.num_train_timesteps
        self.loss_fn = torch.nn.MSELoss()

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
        return (
            torch.randn(
                size=x.shape, generator=self.g, dtype=x.dtype, layout=x.layout
            )
            .contiguous()
            .to(x.device)
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
        return (
            torch.randint(0, self.noise_steps, (x.shape[0],), generator=self.g)
            .to(x.device)
            .long()
        )

    def step(
        self,
        x: torch.Tensor,
        timesteps: torch.Tensor = None,
        context: torch.Tensor = None,
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

        Returns:
            torch.Tensor: The mean loss for the current step.

        """
        epsilon = self.randn_like(x)
        if timesteps is None:
            timesteps = self.timesteps_like(x)
        else:
            timesteps = timesteps.long()
        epsilon_pred = self.inferer(
            inputs=x,
            diffusion_model=self,
            noise=epsilon,
            timesteps=timesteps,
            condition=context,
        )
        loss = self.calculate_loss(epsilon_pred, epsilon)
        return loss

    def unpack_batch(
        self, batch: dict[str, torch.Tensor]
    ) -> tuple[torch.Tensor, torch.Tensor | None]:
        """
        Convenience function to unpack a batch for model training.

        Args:
            batch (dict[str, torch.Tensor]): dictionary containing the correct
                entries for each batch. Should have inputs corresponding to
                ``self.image_key`` and to conditioning keys (i.e.
                ``self.cat_condition_key`` and ``self.num_condition_key``) if
                conditioning is required.

        Returns:
            tuple[torch.Tensor, torch.Tensor]: tensor with input image and
                embedded condition (if provided).
        """
        x = batch[self.image_key]
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
        return x, condition

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
        x, condition = self.unpack_batch(batch)
        loss = self.step(x, context=condition)
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
        x, condition = self.unpack_batch(batch)
        loss = self.step(x, context=condition)
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
        x, condition = self.unpack_batch(batch)
        loss = self.step(x, context=condition)
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

        Returns:
            torch.Tensor: generated (or re-generated) sample.
        """
        noise = torch.randn([n, self.in_channels, *size], device=self.device)
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
        if self.embedder is not None:
            condition = self.embedder(
                X_cat=cat_condition,
                X_num=num_condition,
                batch_size=n,
                update_queues=False,
            )
            uncondition = self.embedder(
                X_cat=cat_condition,
                X_num=num_condition,
                batch_size=n,
                update_queues=False,
                uncondition_cat_idx="all",
                uncondition_num_idx="all",
            )
            if len(condition.shape) < 3:
                condition = condition.unsqueeze(1)
            if len(uncondition.shape) < 3:
                uncondition = uncondition.unsqueeze(1)
        else:
            condition, uncondition = None, None
        sample = self.inferer.sample(
            input_noise=input_image,
            diffusion_model=self,
            scheduler=self.scheduler,
            conditioning=condition,
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
        str, torch.optim.Optimizer | torch.optim.lr_scheduler._LRScheduler | str
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
