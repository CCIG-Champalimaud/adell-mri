"""
Lightning modules for autoencoder training.
"""

from typing import Any

import lightning.pytorch as pl
import torch

from adell_mri.modules.gan.ae import AutoEncoder
from adell_mri.modules.gan.losses import GaussianKLLoss
from adell_mri.modules.gan.vae import VariationalAutoEncoder


class AutoEncoderPL(AutoEncoder, pl.LightningModule):
    """
    Basic autoencoder lightning module.
    """

    def __init__(
        self,
        input_image_key: str = "input_image",
        learning_rate: float = 0.0002,
        momentum_beta1: float = 0.5,
        momentum_beta2: float = 0.99,
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
        """
        super().__init__(*args, **kwargs)
        self.input_image_key = input_image_key
        self.learning_rate = learning_rate
        self.momentum_beta1 = momentum_beta1
        self.momentum_beta2 = momentum_beta2

        self.loss_fn = torch.nn.MSELoss()
        self.init_routine()

    def init_routine(self):
        """
        Simple initialization routine which saves hyperparameters.
        """
        self.save_hyperparameters(ignore=["loss_fn", "loss_params"])

    def step(self, batch: dict[str, Any]) -> dict[str, torch.Tensor]:
        """
        Basic step calculating the generated input and calculating the loss
        value with ``self.loss_fn``.

        Args:
            batch (dict[str, Any]): batch from dataloader.

        Returns:
            dict[str, torch.Tensor]: loss dictionary (only has "loss").
        """
        x = batch[self.input_image_key]
        output = self.forward(x)
        loss = self.loss_fn(output, x)
        return {"loss": loss}

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
        loss_dict = self.step(batch)
        for k in loss_dict:
            self.log(
                k, loss_dict[k], on_epoch=True, prog_bar=True, on_step=False
            )
        return sum([loss_dict[k] for k in loss_dict])

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
        loss_dict = self.step(batch)
        for k in loss_dict:
            self.log(
                f"val_{k}",
                loss_dict[k],
                on_epoch=True,
                prog_bar=True,
                on_step=False,
            )
        return sum([loss_dict[k] for k in loss_dict])

    def test_step(self, batch: dict[str, Any], batch_idx: int) -> torch.Tensor:
        """
        Lightning test step.

        Args:
            batch (dict[str, Any]): batch from dataloader.
            batch_idx (int): batch index.

        Returns:
            torch.Tensor: loss value.
        """
        loss_dict = self.step(batch)
        for k in loss_dict:
            self.log(
                f"test_{k}",
                loss_dict[k],
                on_epoch=True,
                prog_bar=True,
                on_step=False,
            )
        return sum([loss_dict[k] for k in loss_dict])

    def configure_optimizers(self) -> torch.optim.Optimizer:
        """
        Lightning hook for optimizer configuration.

        Returns:
            torch.optim.Optimizer: Adam optimizer.
        """
        return torch.optim.Adam(
            self.parameters(),
            lr=self.learning_rate,
            betas=(self.momentum_beta1, self.momentum_beta2),
        )


class VariationalAutoEncoderPL(VariationalAutoEncoder, pl.LightningModule):
    """
    Lightning module for variational autoencoder.
    """

    def __init__(
        self,
        input_image_key: str = "input_image",
        learning_rate: float = 0.0002,
        momentum_beta1: float = 0.5,
        momentum_beta2: float = 0.99,
        var_loss_mult: float = 1.0,
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
                loss. Defaults to 1.0.
        """
        super().__init__(*args, **kwargs)
        self.input_image_key = input_image_key
        self.learning_rate = learning_rate
        self.momentum_beta1 = momentum_beta1
        self.momentum_beta2 = momentum_beta2
        self.var_loss_mult = var_loss_mult

        self.loss_fn = torch.nn.MSELoss()
        self.variational_loss_fn = GaussianKLLoss()
        self.init_routine()

    def init_routine(self):
        """
        Simple initialization routine which saves hyperparameters.
        """
        self.save_hyperparameters(ignore=["loss_fn", "loss_params"])

    def step(self, batch: dict[str, Any]) -> dict[str, torch.Tensor]:
        """
        Basic step calculating the generated input and calculating the loss
        value with ``self.loss_fn``.

        Args:
            batch (dict[str, Any]): batch from dataloader.

        Returns:
            dict[str, torch.Tensor]: loss dictionary ("rec_loss" and
                "var_loss").
        """
        x = batch[self.input_image_key]
        output, mu, logvar = self.forward(x)
        var_loss = self.variational_loss_fn(mu, logvar)
        loss = self.loss_fn(output, x)
        return {"rec_loss": loss, "var_loss": self.var_loss_mult * var_loss}

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
        loss_dict = self.step(batch)
        for k in loss_dict:
            self.log(
                k, loss_dict[k], on_epoch=True, prog_bar=True, on_step=False
            )
        return sum([loss_dict[k] for k in loss_dict])

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
        loss_dict = self.step(batch)
        for k in loss_dict:
            self.log(
                f"val_{k}",
                loss_dict[k],
                on_epoch=True,
                prog_bar=True,
                on_step=False,
            )
        return sum([loss_dict[k] for k in loss_dict])

    def test_step(self, batch: dict[str, Any], batch_idx: int) -> torch.Tensor:
        """
        Lightning test step.

        Args:
            batch (dict[str, Any]): batch from dataloader.
            batch_idx (int): batch index.

        Returns:
            torch.Tensor: loss value.
        """
        loss_dict = self.step(batch)
        for k in loss_dict:
            self.log(
                f"test_{k}",
                loss_dict[k],
                on_epoch=True,
                prog_bar=True,
                on_step=False,
            )
        return sum([loss_dict[k] for k in loss_dict])

    def configure_optimizers(self) -> torch.optim.Optimizer:
        """
        Lightning hook for optimizer configuration.

        Returns:
            torch.optim.Optimizer: Adam optimizer.
        """
        return torch.optim.Adam(
            self.parameters(),
            lr=self.learning_rate,
            betas=(self.momentum_beta1, self.momentum_beta2),
        )
