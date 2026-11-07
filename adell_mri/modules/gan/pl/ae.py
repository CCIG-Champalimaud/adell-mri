"""
Lightning modules for autoencoder training.
"""

from typing import Any

import lightning.pytorch as pl
import torch
import torch.nn.functional as F

from adell_mri.modules.gan.ae import AutoEncoder
from adell_mri.modules.gan.losses import GaussianKLLoss
from adell_mri.modules.gan.vae import VariationalAutoEncoder
from adell_mri.modules.learning_rate import CosineAnnealingWithWarmupLR
from adell_mri.utils.optimizer_factory import OPTIMIZER_EPS_DEFAULT

DEFAULT_LOSS_WEIGHTS = {
    "rec": 1.0,
    "kl": 0.0,
    "perceptual": 0.0,
    "adversarial": 0.0,
}

LOSS_WEIGHT_KEYS = {
    "rec_loss": "rec",
    "kl_loss": "kl",
    "perceptual_loss": "perceptual",
    "adv_loss": "adversarial",
}


class _AutoEncoderLossMixin:
    """
    Shared loss, optimisation and logging logic for autoencoder modules.
    """

    is_variational: bool = False
    adversarial_start_epoch: int = 0
    adversarial_warmup_epochs: int = 0
    discriminator_update_every_n_steps: int = 1

    def init_losses(
        self,
        loss_weights: dict[str, float] | None,
        rec_loss: str,
        discriminator_config: dict[str, Any] | None,
        perceptual_kwargs: dict[str, Any] | None,
        spectrally_normalize_discriminator: bool = False,
        adversarial_criterion: str = "bce",
    ) -> None:
        """
        Sets up the reconstruction, perceptual and adversarial losses.

        Args:
            loss_weights (dict[str, float] | None): weights for the loss
                terms. Valid keys are ``rec``, ``kl``, ``perceptual`` and
                ``adversarial``. Unspecified terms default to zero, except for
                ``rec`` which defaults to 1.0.
            rec_loss (str): reconstruction loss, either ``mse`` or ``l1``.
            discriminator_config (dict[str, Any] | None): keyword arguments
                for :class:`generative.networks.nets.PatchDiscriminator`.
            perceptual_kwargs (dict[str, Any] | None): keyword arguments for
                :class:`generative.losses.PerceptualLoss`.
            spectrally_normalize_discriminator (bool, optional): applies
                spectral normalization to the discriminator convolutions.
                Defaults to False.
            adversarial_criterion (str, optional): adversarial loss criterion,
                either ``bce``, ``least_squares`` or ``hinge``. Defaults to
                "bce".

        Raises:
            ValueError: if an unknown loss weight, an invalid reconstruction
                loss or an unexpected argument is provided.
        """
        loss_weights = loss_weights or {}
        unknown = set(loss_weights) - set(DEFAULT_LOSS_WEIGHTS)
        if len(unknown) > 0:
            raise ValueError(
                f"Unknown loss weights: {sorted(unknown)}. Valid keys are "
                f"{sorted(DEFAULT_LOSS_WEIGHTS)}"
            )
        self.loss_weights = {**DEFAULT_LOSS_WEIGHTS, **loss_weights}
        if rec_loss not in ["mse", "l1"]:
            raise ValueError(
                f"rec_loss must be either 'mse' or 'l1', got '{rec_loss}'"
            )
        self.rec_loss = rec_loss
        self.reconstruction_loss_fn = (
            F.mse_loss if rec_loss == "mse" else F.l1_loss
        )
        self.variational_loss_fn = GaussianKLLoss()

        self.perceptual_loss = None
        if self.loss_weights["perceptual"] > 0.0:
            try:
                from monai.losses import PerceptualLoss
            except ImportError as exc:
                raise ImportError(
                    "PerceptualLoss requires the generative package and lpips"
                ) from exc
            if self.spatial_dims == 2:
                network_type = "squeeze"
            else:
                network_type = "medicalnet_resnet10_23datasets"
            perceptual_config = {
                "spatial_dims": self.spatial_dims,
                "network_type": network_type,
                "fake_3d_ratio": 0.5,
            }
            if perceptual_kwargs is not None:
                perceptual_config.update(perceptual_kwargs)
            self.perceptual_loss = PerceptualLoss(**perceptual_config)

        self.discriminator = None
        self.adversarial_loss = None
        if self.loss_weights["adversarial"] > 0.0:
            try:
                from monai.losses import PatchAdversarialLoss
                from monai.networks.nets import PatchDiscriminator
            except ImportError as exc:
                raise ImportError(
                    "PatchAdversarialLoss requires the generative package"
                ) from exc
            if adversarial_criterion not in ["bce", "hinge", "least_squares"]:
                raise ValueError(
                    "adversarial_criterion must be 'bce', 'least_squares' "
                    f"or 'hinge', got '{adversarial_criterion}'"
                )
            discriminator_config = discriminator_config or {}
            disc_config = {
                "spatial_dims": self.spatial_dims,
                "channels": 64,
                "in_channels": self.in_channels,
                "out_channels": self.out_channels,
                "num_layers_d": 3,
                "kernel_size": 4,
                "norm": "INSTANCE",
                "padding": 1,
            }
            disc_config.update(discriminator_config)
            self.discriminator = PatchDiscriminator(**disc_config)
            if spectrally_normalize_discriminator:
                for module in self.discriminator.modules():
                    if isinstance(module, torch.nn.Conv3d):
                        torch.nn.utils.parametrizations.spectral_norm(module)
            self.adversarial_loss = PatchAdversarialLoss(
                criterion=adversarial_criterion
            )
        self.automatic_optimization = not self.adversarial_on

    def init_adversarial_schedule(
        self,
        start_epoch: int,
        warmup_epochs: int,
        discriminator_update_every_n_steps: int,
    ) -> None:
        """
        Configures the delay, warm-up and update frequency for the adversarial
        loss.

        Args:
            start_epoch (int): epoch at which the adversarial loss starts.
            warmup_epochs (int): number of epochs over which the adversarial
                loss is linearly ramped up after ``start_epoch``.
            discriminator_update_every_n_steps (int): update the discriminator
                once every N training steps.

        Raises:
            ValueError: if any value is negative or if the discriminator update
                frequency is less than one.
        """
        if start_epoch < 0 or warmup_epochs < 0:
            raise ValueError(
                "adversarial_start_epoch and adversarial_warmup_epochs must "
                "be non-negative"
            )
        if discriminator_update_every_n_steps < 1:
            raise ValueError(
                "discriminator_update_every_n_steps must be greater than or "
                "equal to 1"
            )
        self.adversarial_start_epoch = start_epoch
        self.adversarial_warmup_epochs = warmup_epochs
        self.discriminator_update_every_n_steps = (
            discriminator_update_every_n_steps
        )

    @property
    def adversarial_on(self) -> bool:
        """
        Whether adversarial training is enabled.

        Returns:
            bool: True if the adversarial loss weight is positive.
        """
        return self.loss_weights["adversarial"] > 0.0

    @property
    def adversarial_scale(self) -> float:
        """
        Scale applied to the adversarial loss at the current epoch.

        The scale is zero before ``adversarial_start_epoch`` and then ramps
        linearly to one over ``adversarial_warmup_epochs`` epochs.

        Returns:
            float: adversarial loss scale in the ``[0, 1]`` range.
        """
        if self.adversarial_on is False:
            return 0.0
        if self.current_epoch < self.adversarial_start_epoch:
            return 0.0
        if self.adversarial_warmup_epochs <= 0:
            return 1.0
        elapsed = self.current_epoch - self.adversarial_start_epoch
        return min(1.0, elapsed / self.adversarial_warmup_epochs)

    @property
    def discriminator_update_due(self) -> bool:
        """
        Whether the discriminator should be updated on the current step.

        Returns:
            bool: True if the current global step is a discriminator update
                step.
        """
        if self.discriminator_update_every_n_steps <= 1:
            return True
        return self.global_step % self.discriminator_update_every_n_steps == 0

    def unpack_output(
        self, output: torch.Tensor | tuple[torch.Tensor, ...]
    ) -> tuple[torch.Tensor, torch.Tensor | None, torch.Tensor | None]:
        """
        Extracts the reconstruction and posterior parameters from a forward
        pass.

        Args:
            output (torch.Tensor | tuple[torch.Tensor, ...]): output of the
                autoencoder.

        Returns:
            tuple[torch.Tensor, torch.Tensor | None, torch.Tensor | None]:
                reconstruction, posterior mean and posterior log variance.
        """
        if isinstance(output, tuple):
            return output[0], output[1], output[2]
        return output, None, None

    def compute_adversarial_loss(
        self,
        logits: torch.Tensor,
        target_is_real: bool,
        for_discriminator: bool,
    ) -> torch.Tensor:
        """
        Computes the adversarial loss outside autocast.

        ``PatchAdversarialLoss`` with the ``bce`` criterion uses
        ``torch.nn.BCELoss``, which is unsafe under autocast. Running it in
        float32 keeps mixed-precision training stable while preserving the
        gradient path to the discriminator.

        Args:
            logits (torch.Tensor): discriminator output.
            target_is_real (bool): whether the target is real.
            for_discriminator (bool): whether the loss is for the discriminator.

        Returns:
            torch.Tensor: adversarial loss.
        """
        device_type = "cuda" if logits.is_cuda else "cpu"
        with torch.autocast(device_type=device_type, enabled=False):
            return self.adversarial_loss(
                logits.float(),
                target_is_real=target_is_real,
                for_discriminator=for_discriminator,
            )

    def compute_losses(
        self,
        x: torch.Tensor,
        include_adversarial: bool = True,
        include_perceptual: bool = True,
    ) -> tuple[
        dict[str, torch.Tensor], torch.Tensor | tuple[torch.Tensor, ...]
    ]:
        """
        Computes the (unweighted) loss terms and the total weighted loss.

        Args:
            x (torch.Tensor): input image.
            include_adversarial (bool, optional): whether the adversarial
                generator loss should be computed. Defaults to True.
            include_perceptual (bool, optional): whether the perceptual
                generator loss should be computed. This is relevant only because
                the first training iteration might produce a constant vector
                with std=0, which produces nan after normalising for MedicalNet.
                Defaults to True.

        Returns:
            tuple[dict[str, torch.Tensor], torch.Tensor | tuple[torch.Tensor,
                ...]]: dictionary with the unweighted losses and the total
                weighted loss, and the output of the forward pass.
        """
        output = self.forward(x)
        recon, mu, logvar = self.unpack_output(output)
        losses = {
            "rec_loss": self.reconstruction_loss_fn(recon, x),
        }
        if mu is not None and logvar is not None:
            losses["kl_loss"] = self.variational_loss_fn(mu, logvar)
        if self.perceptual_loss is not None and include_perceptual:
            losses["perceptual_loss"] = self.perceptual_loss(recon, x)
        if (
            include_adversarial
            and self.discriminator is not None
            and self.adversarial_loss is not None
        ):
            logits_fake = self.discriminator(recon)[-1]
            losses["adv_loss"] = self.compute_adversarial_loss(
                logits_fake, target_is_real=True, for_discriminator=False
            )
        total = []
        for key, value in losses.items():
            if key not in LOSS_WEIGHT_KEYS:
                continue
            weight = self.loss_weights[LOSS_WEIGHT_KEYS[key]]
            if key == "adv_loss":
                weight = weight * self.adversarial_scale
            total.append(weight * value)
        losses["loss"] = sum(total)
        return losses, output

    def compute_discriminator_losses(
        self, x: torch.Tensor, recon: torch.Tensor
    ) -> dict[str, torch.Tensor]:
        """
        Computes the discriminator loss.

        Args:
            x (torch.Tensor): real image.
            recon (torch.Tensor): reconstructed image (detached).

        Returns:
            dict[str, torch.Tensor]: dictionary with the discriminator loss.
        """
        logits_fake = self.discriminator(recon)[-1]
        loss_fake = self.compute_adversarial_loss(
            logits_fake, target_is_real=False, for_discriminator=True
        )
        logits_real = self.discriminator(x)[-1]
        loss_real = self.compute_adversarial_loss(
            logits_real, target_is_real=True, for_discriminator=True
        )
        loss = 0.5 * (loss_fake + loss_real)
        return {"disc_loss": loss}

    def log_losses(
        self, losses: dict[str, torch.Tensor], prefix: str = ""
    ) -> None:
        """
        Logs a dictionary of losses.

        Args:
            losses (dict[str, torch.Tensor]): losses to log.
            prefix (str, optional): prefix for the metric names. Defaults to
                an empty string.
        """
        for key, value in losses.items():
            if key == "loss":
                continue
            self.log(
                f"{prefix}{key}",
                value,
                on_epoch=True,
                prog_bar=key in ["rec_loss", "loss"],
                on_step=False,
            )

    def maybe_log_reconstruction(
        self,
        images: torch.Tensor,
        reconstruction: torch.Tensor,
        batch_idx: int,
        prefix: str = "val",
    ) -> None:
        """
        Logs input and reconstructed images every N epochs.

        Args:
            images (torch.Tensor): input images.
            reconstruction (torch.Tensor): reconstructed images.
            batch_idx (int): batch index.
            prefix (str, optional): prefix for the logged image keys. Defaults
                to "val".
        """
        if self.reconstruction_every_n_epochs <= 0:
            return
        if (
            self.current_epoch % self.reconstruction_every_n_epochs != 0
            or batch_idx >= self.reconstruction_max_batches
        ):
            return
        trainer = getattr(self, "trainer", None)
        if trainer is None or getattr(trainer, "is_global_zero", True) is False:
            return
        logger = getattr(trainer, "logger", None)
        if logger is None or hasattr(logger, "log_image") is False:
            return

        from adell_mri.utils.pl_callbacks import log_image

        slice_dim = 4 if images.ndim == 5 else None
        n_slices_out = 3 if images.ndim == 5 else None
        log_image(
            trainer,
            f"{prefix}_input_images",
            images,
            slice_dim=slice_dim,
            n_slices_out=n_slices_out,
        )
        log_image(
            trainer,
            f"{prefix}_reconstructed_images",
            reconstruction,
            slice_dim=slice_dim,
            n_slices_out=n_slices_out,
        )

    def training_step(
        self, batch: dict[str, Any], batch_idx: int
    ) -> torch.Tensor:
        """
        Lightning training step.

        Args:
            batch (dict[str, Any]): batch from dataloader.
            batch_idx (int): batch index.

        Returns:
            torch.Tensor: loss value.
        """
        x = batch[self.input_image_key]
        if self.adversarial_on:
            opt_g, opt_d = self.optimizers()
            adversarial_scale = self.adversarial_scale
            losses, output = self.compute_losses(
                x,
                include_adversarial=adversarial_scale > 0.0,
                include_perceptual=self.global_step > 0,
            )
            opt_g.zero_grad()
            self.manual_backward(losses["loss"])
            self.clip_gradients(
                opt_g, gradient_clip_val=1.0, gradient_clip_algorithm="norm"
            )
            opt_g.step()

            if adversarial_scale > 0.0 and self.discriminator_update_due:
                recon, _, _ = self.unpack_output(output)
                disc_losses = self.compute_discriminator_losses(
                    x, recon.detach()
                )
                opt_d.zero_grad()
                self.manual_backward(disc_losses["disc_loss"])
                self.clip_gradients(
                    opt_d,
                    gradient_clip_val=1.0,
                    gradient_clip_algorithm="norm",
                )
                opt_d.step()
                self.log_losses(disc_losses)

            self.log_losses(losses)
            self.log("adversarial_scale", adversarial_scale, on_epoch=True)
            self.log("loss", losses["loss"].detach(), on_epoch=True)
            return losses["loss"].detach()

        losses, _ = self.compute_losses(
            x,
            include_adversarial=False,
            include_perceptual=self.global_step > 0,
        )
        self.log_losses(losses)
        self.log("loss", losses["loss"], on_epoch=True, prog_bar=True)
        return losses["loss"]

    def validation_step(
        self, batch: dict[str, Any], batch_idx: int
    ) -> torch.Tensor:
        """
        Lightning validation step.

        Args:
            batch (dict[str, Any]): batch from dataloader.
            batch_idx (int): batch index.

        Returns:
            torch.Tensor: loss value.
        """
        x = batch[self.input_image_key]
        losses, output = self.compute_losses(x, include_adversarial=False)
        reconstruction, _, _ = self.unpack_output(output)
        self.maybe_log_reconstruction(x, reconstruction, batch_idx, "val")
        self.log_losses(losses, prefix="val_")
        self.log("val_loss", losses["loss"], on_epoch=True, prog_bar=True)
        return losses["loss"]

    def test_step(self, batch: dict[str, Any], batch_idx: int) -> torch.Tensor:
        """
        Lightning test step.

        Args:
            batch (dict[str, Any]): batch from dataloader.
            batch_idx (int): batch index.

        Returns:
            torch.Tensor: loss value.
        """
        x = batch[self.input_image_key]
        losses, output = self.compute_losses(x, include_adversarial=False)
        reconstruction, _, _ = self.unpack_output(output)
        self.maybe_log_reconstruction(x, reconstruction, batch_idx, "test")
        self.log_losses(losses, prefix="test_")
        self.log("test_loss", losses["loss"], on_epoch=True)
        return losses["loss"]

    def init_routine(self) -> None:
        """
        Saves hyperparameters while ignoring the instantiated loss modules.
        """
        self.save_hyperparameters(
            ignore=[
                "discriminator",
                "perceptual_loss",
                "reconstruction_loss_fn",
                "variational_loss_fn",
                "adversarial_loss",
            ]
        )

    def configure_optimizers(self) -> Any:
        """
        Configures the optimiser(s).

        Returns:
            Any: the optimizer(s) and learning rate scheduler(s).
        """
        optimizer_g = torch.optim.AdamW(
            self.generator_parameters(),
            lr=self.learning_rate,
            betas=(self.momentum_beta1, self.momentum_beta2),
            eps=self.optimizer_eps,
        )
        scheduler_g = CosineAnnealingWithWarmupLR(
            optimizer_g,
            T_max=self.n_epochs,
            start_decay=self.start_decay,
            n_warmup_steps=self.warmup_steps,
            eta_min=0.0,
        )
        if not self.adversarial_on:
            return {
                "optimizer": optimizer_g,
                "lr_scheduler": scheduler_g,
                "monitor": "val_loss",
            }
        optimizer_d = torch.optim.AdamW(
            self.discriminator.parameters(),
            lr=self.discriminator_learning_rate,
            betas=(self.momentum_beta1, self.momentum_beta2),
            eps=self.optimizer_eps,
        )
        scheduler_d = CosineAnnealingWithWarmupLR(
            optimizer_d,
            T_max=self.n_epochs,
            start_decay=self.start_decay,
            n_warmup_steps=self.warmup_steps,
            eta_min=0.0,
        )
        return [optimizer_g, optimizer_d], [scheduler_g, scheduler_d]

    def on_train_epoch_end(self) -> None:
        """
        Steps the learning rate schedulers when using manual optimization.
        """
        if self.automatic_optimization:
            return
        schedulers = self.lr_schedulers()
        if schedulers is None:
            return
        if not isinstance(schedulers, (list, tuple)):
            schedulers = [schedulers]
        for scheduler in schedulers:
            scheduler.step()

    def generator_parameters(self):
        """
        Returns the parameters of the autoencoder excluding the discriminator.

        Returns:
            Any: iterable of parameters.
        """
        discriminator_ids = (
            {id(p) for p in self.discriminator.parameters()}
            if self.discriminator is not None
            else set()
        )
        return [p for p in self.parameters() if id(p) not in discriminator_ids]


class AutoEncoderPL(_AutoEncoderLossMixin, AutoEncoder, pl.LightningModule):
    """
    Basic autoencoder lightning module.
    """

    def __init__(
        self,
        input_image_key: str = "input_image",
        learning_rate: float = 0.0002,
        momentum_beta1: float = 0.5,
        momentum_beta2: float = 0.99,
        optimizer_eps: float = OPTIMIZER_EPS_DEFAULT,
        n_epochs: int = 1000,
        warmup_steps: float = 0.0,
        start_decay: float | None = None,
        loss_weights: dict[str, float] | None = None,
        rec_loss: str = "mse",
        discriminator_config: dict[str, Any] | None = None,
        perceptual_kwargs: dict[str, Any] | None = None,
        discriminator_learning_rate: float | None = None,
        reconstruction_every_n_epochs: int = 5,
        reconstruction_max_batches: int = 1,
        adversarial_start_epoch: int = 0,
        adversarial_warmup_epochs: int = 0,
        discriminator_update_every_n_steps: int = 1,
        spectrally_normalize_discriminator: bool = False,
        adversarial_criterion: str = "bce",
        *args,
        **kwargs,
    ):
        """
        Args:
            input_image_key (str, optional): input image key. Defaults to
                "input_image".
            learning_rate (float, optional): learning rate. Defaults to 0.0002.
            momentum_beta1 (float, optional): first beta momentum for Adam
                optimizer. Defaults to 0.5.
            momentum_beta2 (float, optional): second beta momentum for Adam
                optimizer. Defaults to 0.99.
            optimizer_eps (float, optional): epsilon term used by the optimizer
                for numerical stability. Defaults to the AdamW default (1e-8).
            n_epochs (int, optional): total number of training epochs, used by
                the learning rate scheduler. Defaults to 1000.
            warmup_steps (float, optional): learning rate warmup, in epochs or
                as a fraction of ``n_epochs``. Defaults to 0.0.
            start_decay (float | None, optional): epoch or fraction of
                ``n_epochs`` at which decay starts. Defaults to None (right
                after warmup).
            loss_weights (dict[str, float] | None, optional): weights for the
                reconstruction (``rec``), perceptual (``perceptual``) and
                adversarial (``adversarial``) losses. Defaults to None
                (reconstruction only).
            rec_loss (str, optional): reconstruction loss, either ``mse`` or
                ``l1``. Defaults to "mse".
            discriminator_config (dict[str, Any] | None, optional): keyword
                arguments for the patch discriminator. Defaults to None.
            perceptual_kwargs (dict[str, Any] | None, optional): keyword
                arguments for the perceptual loss. Defaults to None.
            discriminator_learning_rate (float | None, optional): learning
                rate for the discriminator. Defaults to None (uses
                ``learning_rate``).
            reconstruction_every_n_epochs (int, optional): logs validation
                reconstructions every N epochs. A value of 0 disables
                reconstruction logging. Defaults to 5.
            reconstruction_max_batches (int, optional): maximum number of
                validation batches logged per reconstruction epoch. Defaults to
                1.
            adversarial_start_epoch (int, optional): epoch at which the
                adversarial loss starts. Defaults to 0 (immediately).
            adversarial_warmup_epochs (int, optional): number of epochs over
                which the adversarial loss is linearly ramped up after
                ``adversarial_start_epoch``. Defaults to 0 (no warm-up).
            discriminator_update_every_n_steps (int, optional): update the
                discriminator once every N training steps. Defaults to 1
                (every step).
            spectrally_normalize_discriminator (bool, optional): applies
                spectral normalization to the weights in the adversarial
                discriminator. Defaults to False.
            adversarial_criterion (str, optional): adversarial loss criterion,
                either ``bce`` or ``hinge``. Defaults to "bce".
            *args: positional arguments passed to :class:`AutoEncoder`.
            **kwargs: keyword arguments passed to :class:`AutoEncoder`.
        """
        super().__init__(*args, **kwargs)
        self.input_image_key = input_image_key
        self.learning_rate = learning_rate
        self.momentum_beta1 = momentum_beta1
        self.momentum_beta2 = momentum_beta2
        self.optimizer_eps = optimizer_eps
        self.n_epochs = n_epochs
        self.warmup_steps = warmup_steps
        self.start_decay = start_decay
        self.discriminator_learning_rate = (
            discriminator_learning_rate
            if discriminator_learning_rate is not None
            else learning_rate
        )
        self.is_variational = False
        self.reconstruction_every_n_epochs = reconstruction_every_n_epochs
        self.reconstruction_max_batches = reconstruction_max_batches
        self.init_adversarial_schedule(
            adversarial_start_epoch,
            adversarial_warmup_epochs,
            discriminator_update_every_n_steps,
        )
        self.init_losses(
            loss_weights,
            rec_loss,
            discriminator_config,
            perceptual_kwargs,
            spectrally_normalize_discriminator,
            adversarial_criterion,
        )
        self.init_routine()


class VariationalAutoEncoderPL(
    _AutoEncoderLossMixin, VariationalAutoEncoder, pl.LightningModule
):
    """
    Lightning module for variational autoencoder training.
    """

    def __init__(
        self,
        input_image_key: str = "input_image",
        learning_rate: float = 0.0002,
        momentum_beta1: float = 0.5,
        momentum_beta2: float = 0.99,
        var_loss_mult: float = 1.0,
        optimizer_eps: float = OPTIMIZER_EPS_DEFAULT,
        n_epochs: int = 1000,
        warmup_steps: float = 0.0,
        start_decay: float | None = None,
        loss_weights: dict[str, float] | None = None,
        rec_loss: str = "mse",
        discriminator_config: dict[str, Any] | None = None,
        perceptual_kwargs: dict[str, Any] | None = None,
        discriminator_learning_rate: float | None = None,
        reconstruction_every_n_epochs: int = 5,
        reconstruction_max_batches: int = 1,
        adversarial_start_epoch: int = 0,
        adversarial_warmup_epochs: int = 0,
        discriminator_update_every_n_steps: int = 1,
        spectrally_normalize_discriminator: bool = False,
        adversarial_criterion: str = "bce",
        *args,
        **kwargs,
    ):
        """
        Args:
            input_image_key (str, optional): input image key. Defaults to
                "input_image".
            learning_rate (float, optional): learning rate. Defaults to 0.0002.
            momentum_beta1 (float, optional): first beta momentum for Adam
                optimizer. Defaults to 0.5.
            momentum_beta2 (float, optional): second beta momentum for Adam
                optimizer. Defaults to 0.99.
            var_loss_mult (float, optional): multiplier for the variational
                loss. Ignored if ``loss_weights`` is provided. Defaults to 1.0.
            optimizer_eps (float, optional): epsilon term used by the optimizer
                for numerical stability. Defaults to the AdamW default (1e-8).
            n_epochs (int, optional): total number of training epochs, used by
                the learning rate scheduler. Defaults to 1000.
            warmup_steps (float, optional): learning rate warmup, in epochs or
                as a fraction of ``n_epochs``. Defaults to 0.0.
            start_decay (float | None, optional): epoch or fraction of
                ``n_epochs`` at which decay starts. Defaults to None (right
                after warmup).
            loss_weights (dict[str, float] | None, optional): weights for the
                reconstruction (``rec``), KL (``kl``), perceptual
                (``perceptual``) and adversarial (``adversarial``) losses.
                Defaults to None (``rec=1.0`` and ``kl=var_loss_mult``).
            rec_loss (str, optional): reconstruction loss, either ``mse`` or
                ``l1``. Defaults to "mse".
            discriminator_config (dict[str, Any] | None, optional): keyword
                arguments for the patch discriminator. Defaults to None.
            perceptual_kwargs (dict[str, Any] | None, optional): keyword
                arguments for the perceptual loss. Defaults to None.
            discriminator_learning_rate (float | None, optional): learning
                rate for the discriminator. Defaults to None (uses
                ``learning_rate``).
            reconstruction_every_n_epochs (int, optional): logs validation
                reconstructions every N epochs. A value of 0 disables
                reconstruction logging. Defaults to 5.
            reconstruction_max_batches (int, optional): maximum number of
                validation batches logged per reconstruction epoch. Defaults to
                1.
            adversarial_start_epoch (int, optional): epoch at which the
                adversarial loss starts. Defaults to 0 (immediately).
            adversarial_warmup_epochs (int, optional): number of epochs over
                which the adversarial loss is linearly ramped up after
                ``adversarial_start_epoch``. Defaults to 0 (no warm-up).
            discriminator_update_every_n_steps (int, optional): update the
                discriminator once every N training steps. Defaults to 1
                (every step).
            spectrally_normalize_discriminator (bool, optional): applies
                spectral normalization to the weights in the adversarial
                discriminator. Defaults to False.
            adversarial_criterion (str, optional): adversarial loss criterion,
                either ``bce`` or ``hinge``. Defaults to "bce".
            *args: positional arguments passed to
                :class:`VariationalAutoEncoder`.
            **kwargs: keyword arguments passed to
                :class:`VariationalAutoEncoder`.
        """
        super().__init__(*args, **kwargs)
        self.input_image_key = input_image_key
        self.learning_rate = learning_rate
        self.momentum_beta1 = momentum_beta1
        self.momentum_beta2 = momentum_beta2
        self.var_loss_mult = var_loss_mult
        self.optimizer_eps = optimizer_eps
        self.n_epochs = n_epochs
        self.warmup_steps = warmup_steps
        self.start_decay = start_decay
        self.discriminator_learning_rate = (
            discriminator_learning_rate
            if discriminator_learning_rate is not None
            else learning_rate
        )
        self.is_variational = True
        self.reconstruction_every_n_epochs = reconstruction_every_n_epochs
        self.reconstruction_max_batches = reconstruction_max_batches
        self.init_adversarial_schedule(
            adversarial_start_epoch,
            adversarial_warmup_epochs,
            discriminator_update_every_n_steps,
        )
        if loss_weights is None:
            loss_weights = {"rec": 1.0, "kl": var_loss_mult}
        self.init_losses(
            loss_weights,
            rec_loss,
            discriminator_config,
            perceptual_kwargs,
            spectrally_normalize_discriminator,
            adversarial_criterion,
        )
        self.init_routine()
