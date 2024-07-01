import torch
import lightning.pytorch as pl
from abc import ABC
from typing import Any
from .losses import AdversarialLoss, GaussianKLLoss
from .gan import GAN
from .ae import AutoEncoder
from .vae import VariationalAutoEncoder


class GANPLABC(pl.LightningModule, ABC):
    def __init__(self):
        super().__init__()

    def init_routine(self):
        if hasattr(self, "generator") is False:
            raise ValueError("A generator must be passed to the constructor.")
        if hasattr(self, "discriminator") is False:
            raise ValueError(
                "A discriminator must be passed to the constructor."
            )
        self.save_hyperparameters(
            ignore=["loss_fn", "loss_params", "generator", "discriminator"]
        )

    def configure_optimizers(self):
        opt_generator = torch.optim.Adam(
            self.generator.parameters(),
            lr=self.learning_rate,
            betas=(self.momentum_beta1, self.momentum_beta2),
        )
        opt_discriminator = torch.optim.Adam(
            self.generator.parameters(),
            lr=self.learning_rate,
            betas=(self.momentum_beta1, self.momentum_beta2),
        )

        return [opt_generator, opt_discriminator], []

    def generate_noise(self, x: torch.Tensor):
        x_sh = list(x.shape)
        x_sh[1] = self.generator.in_channels
        return torch.randn(*x_sh).to(x)

    def step_generator(self, input_tensor: torch.Tensor):
        gen_samples = self.generator(input_tensor)
        gen_pred = self.discriminator(gen_samples)
        if isinstance(gen_pred, tuple):
            gen_pred = gen_pred[0]
        loss = self.adversarial_loss.generator_loss(gen_pred)
        return loss

    def step_discriminator(self, x: torch.Tensor, input_tensor: torch.Tensor):
        gen_samples = self.generator(input_tensor)
        gen_pred = self.discriminator(gen_samples)
        if isinstance(gen_pred, tuple):
            gen_pred = gen_pred[0]
        real_pred = self.discriminator(x)
        if isinstance(real_pred, tuple):
            real_pred = real_pred[0]
        loss = self.adversarial_loss(gen_pred, real_pred)
        return loss


class AutoEncoderPL(AutoEncoder, pl.LightningModule):
    def __init__(
        self,
        input_image_key: str = "input_image",
        additional_features_key: str = None,
        additional_class_target_key: str = None,
        additional_reg_target_key: str = None,
        learning_rate: float = 0.0002,
        momentum_beta1: float = 0.5,
        momentum_beta2: float = 0.99,
        *args,
        **kwargs,
    ):
        super().__init__(*args, **kwargs)
        self.input_image_key = input_image_key
        self.additional_features_key = additional_features_key
        self.additional_class_target_key = additional_class_target_key
        self.additional_reg_target_key = additional_reg_target_key
        self.learning_rate = learning_rate
        self.momentum_beta1 = momentum_beta1
        self.momentum_beta2 = momentum_beta2

        self.loss_fn = torch.nn.MSELoss()
        self.init_routine()

    def init_routine(self):
        self.save_hyperparameters(ignore=["loss_fn", "loss_params"])

    def step(self, batch):
        x = batch[self.input_image_key]
        output = self.forward(x)
        loss = self.loss_fn(output, x)
        return {"loss": loss}

    def training_step(self, batch, batch_idx):
        loss_dict = self.step(batch)
        for k in loss_dict:
            self.log(
                k,
                loss_dict[k],
                on_epoch=True,
                prog_bar=True,
            )
        return sum([loss_dict[k] for k in loss_dict])

    def validation_step(self, batch, batch_idx):
        loss_dict = self.step(batch)
        for k in loss_dict:
            self.log(
                f"val_{k}",
                loss_dict[k],
                on_epoch=True,
                prog_bar=True,
            )
        return sum([loss_dict[k] for k in loss_dict])

    def test_step(self, batch, batch_idx):
        loss_dict = self.step(batch)
        for k in loss_dict:
            self.log(
                f"test_{k}",
                loss_dict[k],
                on_epoch=True,
                prog_bar=True,
            )
        return sum([loss_dict[k] for k in loss_dict])

    def configure_optimizers(self):
        return torch.optim.Adam(
            self.parameters(),
            lr=self.learning_rate,
            betas=(self.momentum_beta1, self.momentum_beta2),
        )


class VariationalAutoEncoderPL(VariationalAutoEncoder, pl.LightningModule):
    def __init__(
        self,
        input_image_key: str = "input_image",
        additional_features_key: str = None,
        additional_class_target_key: str = None,
        additional_reg_target_key: str = None,
        learning_rate: float = 0.0002,
        momentum_beta1: float = 0.5,
        momentum_beta2: float = 0.99,
        var_loss_mult: float = 1.0,
        *args,
        **kwargs,
    ):
        super().__init__(*args, **kwargs)
        self.input_image_key = input_image_key
        self.additional_features_key = additional_features_key
        self.additional_class_target_key = additional_class_target_key
        self.additional_reg_target_key = additional_reg_target_key
        self.learning_rate = learning_rate
        self.momentum_beta1 = momentum_beta1
        self.momentum_beta2 = momentum_beta2
        self.var_loss_mult = var_loss_mult

        self.loss_fn = torch.nn.MSELoss()
        self.variational_loss_fn = GaussianKLLoss()
        self.init_routine()

    def init_routine(self):
        self.save_hyperparameters(ignore=["loss_fn", "loss_params"])

    def step(self, batch):
        x = batch[self.input_image_key]
        output, mu, logvar = self.forward(x)
        var_loss = self.variational_loss_fn(mu, logvar)
        loss = self.loss_fn(output, x)
        return {"rec_loss": loss, "var_loss": self.var_loss_mult * var_loss}

    def training_step(self, batch, batch_idx):
        loss_dict = self.step(batch)
        for k in loss_dict:
            self.log(
                k,
                loss_dict[k],
                on_epoch=True,
                prog_bar=True,
            )
        return sum([loss_dict[k] for k in loss_dict])

    def validation_step(self, batch, batch_idx):
        loss_dict = self.step(batch)
        for k in loss_dict:
            self.log(
                f"val_{k}",
                loss_dict[k],
                on_epoch=True,
                prog_bar=True,
            )
        return sum([loss_dict[k] for k in loss_dict])

    def test_step(self, batch, batch_idx):
        loss_dict = self.step(batch)
        for k in loss_dict:
            self.log(
                f"test_{k}",
                loss_dict[k],
                on_epoch=True,
                prog_bar=True,
            )
        return sum([loss_dict[k] for k in loss_dict])

    def configure_optimizers(self):
        return torch.optim.Adam(
            self.parameters(),
            lr=self.learning_rate,
            betas=(self.momentum_beta1, self.momentum_beta2),
        )


class GANPL(GAN, GANPLABC):
    def __init__(
        self,
        input_image_key: str = "input_image",
        additional_features_key: str = None,
        additional_class_target_key: str = None,
        additional_reg_target_key: str = None,
        learning_rate: float = 0.0002,
        momentum_beta1: float = 0.5,
        momentum_beta2: float = 0.99,
        smoothing: float = 0.0,
        *args,
        **kwargs,
    ):
        super().__init__(*args, **kwargs)
        self.input_image_key = input_image_key
        self.additional_features_key = additional_features_key
        self.additional_class_target_key = additional_class_target_key
        self.additional_reg_target_key = additional_reg_target_key
        self.learning_rate = learning_rate
        self.momentum_beta1 = momentum_beta1
        self.momentum_beta2 = momentum_beta2
        self.smoothing = smoothing

        self.adversarial_loss = AdversarialLoss(smoothing=self.smoothing)
        self.init_routine()

        self.automatic_optimization = False

    def training_step(self, batch: dict[str, Any], batch_idx: int):
        optimizer_g, optimizer_d = self.optimizers()

        x = batch[self.input_image_key]
        noise = self.generate_noise(x)

        self.toggle_optimizer(optimizer_g)
        loss_g = self.step_generator(noise)
        self.manual_backward(loss_g)
        optimizer_g.step()
        optimizer_g.zero_grad()
        self.untoggle_optimizer(optimizer_g)

        self.toggle_optimizer(optimizer_d)
        loss_d = self.step_discriminator(x, noise)
        self.manual_backward(loss_d)
        optimizer_d.step()
        optimizer_d.zero_grad()
        self.untoggle_optimizer(optimizer_d)

        self.log("loss_g", loss_g, on_epoch=True, prog_bar=True)
        self.log("loss_d", loss_d, on_epoch=True, prog_bar=True)

    def validation_step(self, batch: dict[str, Any], batch_idx: int):
        x = batch[self.input_image_key]
        noise = self.generate_noise(x)

        self.log(
            "val_loss_g",
            self.step_generator(noise),
            on_epoch=True,
            prog_bar=True,
        )
        self.log(
            "val_loss_d",
            self.step_discriminator(noise, x),
            on_epoch=True,
            prog_bar=True,
        )

    def test_step(self, batch: dict[str, Any], batch_idx: int):
        x = batch[self.input_image_key]
        noise = self.generate_noise(x)

        self.log(
            "test_loss_g",
            self.step_generator(noise),
            on_epoch=True,
        )
        self.log(
            "test_loss_d",
            self.step_discriminator(noise, x),
            on_epoch=True,
        )
