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

    def init_losses(
        self,
        loss_weights: dict[str, float] | None,
        rec_loss: str,
        discriminator_config: dict[str, Any] | None,
        perceptual_kwargs: dict[str, Any] | None,
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
            perceptual_config = {
                "spatial_dims": self.spatial_dims,
                "network_type": "squeeze",
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
            discriminator_config = discriminator_config or {}
            disc_config = {
                "spatial_dims": self.spatial_dims,
                "channels": 64,
                "in_channels": self.in_channels,
                "out_channels": self.out_channels,
                "num_layers_d": 3,
                "kernel_size": 4,
                "norm": "BATCH",
                "padding": 1,
            }
            disc_config.update(discriminator_config)
            self.discriminator = PatchDiscriminator(**disc_config)
            self.adversarial_loss = PatchAdversarialLoss(
                criterion="least_squares", no_activation_leastsq=True
            )
        self.automatic_optimization = not self.adversarial_on

    @property
    def adversarial_on(self) -> bool:
        """
        Whether adversarial training is enabled.

        Returns:
            bool: True if the adversarial loss weight is positive.
        """
        return self.loss_weights["adversarial"] > 0.0

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

    def compute_losses(
        self, x: torch.Tensor, include_adversarial: bool = True
    ) -> tuple[
        dict[str, torch.Tensor], torch.Tensor | tuple[torch.Tensor, ...]
    ]:
        """
        Computes the (unweighted) loss terms and the total weighted loss.

        Args:
            x (torch.Tensor): input image.
            include_adversarial (bool, optional): whether the adversarial
                generator loss should be computed. Defaults to True.

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
        if self.perceptual_loss is not None:
            losses["perceptual_loss"] = self.perceptual_loss(recon, x)
        if (
            include_adversarial
            and self.discriminator is not None
            and self.adversarial_loss is not None
        ):
            logits_fake = self.discriminator(recon)[-1]
            losses["adv_loss"] = self.adversarial_loss(
                logits_fake, target_is_real=True, for_discriminator=False
            )
        total = sum(
            [
                self.loss_weights[LOSS_WEIGHT_KEYS[key]] * losses[key]
                for key in losses
                if key in LOSS_WEIGHT_KEYS
            ]
        )
        losses["loss"] = total
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
        loss_fake = self.adversarial_loss(
            logits_fake, target_is_real=False, for_discriminator=True
        )
        logits_real = self.discriminator(x)[-1]
        loss_real = self.adversarial_loss(
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
            losses, output = self.compute_losses(x)
            opt_g.zero_grad()
            self.manual_backward(losses["loss"])
            self.clip_gradients(
                opt_g, gradient_clip_val=1.0, gradient_clip_algorithm="norm"
            )
            opt_g.step()

            recon, _, _ = self.unpack_output(output)
            disc_losses = self.compute_discriminator_losses(x, recon.detach())
            opt_d.zero_grad()
            self.manual_backward(
                self.loss_weights["adversarial"] * disc_losses["disc_loss"]
            )
            self.clip_gradients(
                opt_d, gradient_clip_val=1.0, gradient_clip_algorithm="norm"
            )
            opt_d.step()

            self.log_losses(losses)
            self.log_losses(disc_losses)
            self.log("loss", losses["loss"].detach(), on_epoch=True)
            return losses["loss"].detach()

        losses, _ = self.compute_losses(x, include_adversarial=False)
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
        losses, _ = self.compute_losses(x, include_adversarial=False)
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
        losses, _ = self.compute_losses(x, include_adversarial=False)
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
            Any: a single optimizer, or a tuple with the generator and
                discriminator optimizers when adversarial training is enabled.
        """
        optimizer_g = torch.optim.AdamW(
            self.generator_parameters(),
            lr=self.learning_rate,
            betas=(self.momentum_beta1, self.momentum_beta2),
            eps=self.optimizer_eps,
        )
        if not self.adversarial_on:
            return optimizer_g
        optimizer_d = torch.optim.AdamW(
            self.discriminator.parameters(),
            lr=self.discriminator_learning_rate,
            betas=(self.momentum_beta1, self.momentum_beta2),
            eps=self.optimizer_eps,
        )
        return optimizer_g, optimizer_d

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
        loss_weights: dict[str, float] | None = None,
        rec_loss: str = "mse",
        discriminator_config: dict[str, Any] | None = None,
        perceptual_kwargs: dict[str, Any] | None = None,
        discriminator_learning_rate: float | None = None,
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
            *args: positional arguments passed to :class:`AutoEncoder`.
            **kwargs: keyword arguments passed to :class:`AutoEncoder`.
        """
        super().__init__(*args, **kwargs)
        self.input_image_key = input_image_key
        self.learning_rate = learning_rate
        self.momentum_beta1 = momentum_beta1
        self.momentum_beta2 = momentum_beta2
        self.optimizer_eps = optimizer_eps
        self.discriminator_learning_rate = (
            discriminator_learning_rate
            if discriminator_learning_rate is not None
            else learning_rate
        )
        self.is_variational = False
        self.init_losses(
            loss_weights, rec_loss, discriminator_config, perceptual_kwargs
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
        loss_weights: dict[str, float] | None = None,
        rec_loss: str = "mse",
        discriminator_config: dict[str, Any] | None = None,
        perceptual_kwargs: dict[str, Any] | None = None,
        discriminator_learning_rate: float | None = None,
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
        self.discriminator_learning_rate = (
            discriminator_learning_rate
            if discriminator_learning_rate is not None
            else learning_rate
        )
        self.is_variational = True
        if loss_weights is None:
            loss_weights = {"rec": 1.0, "kl": var_loss_mult}
        self.init_losses(
            loss_weights, rec_loss, discriminator_config, perceptual_kwargs
        )
        self.init_routine()
