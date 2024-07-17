import torch
import lightning.pytorch as pl
from abc import ABC
from typing import Any, Mapping
from .losses import (
    AdversarialLoss,
    WGANGPLoss,
    SemiSLAdversarialLoss,
    GaussianKLLoss,
)
from .gan import GAN
from .ae import AutoEncoder
from .vae import VariationalAutoEncoder
from .losses import apply_discriminator
from ..diffusion.embedder import Embedder


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
        self.init_converters()

    def init_converters(self):
        self.embed = False
        if hasattr(self, "classification_target_key"):
            if self.classification_target_key is not None:
                if self.class_target_specification is None:
                    raise ValueError(
                        "A class_target_specification must be passed to the \
                        constructor if an additional_class_target_key is \
                        specified."
                    )
                self.embed = True
        if hasattr(self, "regression_target_key"):
            if self.regression_target_key is not None:
                if self.reg_target_specification is None:
                    raise ValueError(
                        "A reg_target_specification must be passed to the \
                        constructor if an additional_reg_target_key is \
                        specified."
                    )
                self.embed = True
        if self.embed == True:
            self.embedder = Embedder(
                self.class_target_specification,
                self.reg_target_specification,
                embedding_size=self.generator.cross_attention_dim,
            )

    def configure_optimizers(self):
        opt_generator = torch.optim.Adam(
            self.generator.parameters(),
            lr=self.learning_rate,
            betas=(self.momentum_beta1, self.momentum_beta2),
        )
        opt_discriminator = torch.optim.Adam(
            self.discriminator.parameters(),
            lr=self.learning_rate,
            betas=(self.momentum_beta1, self.momentum_beta2),
        )

        schedulers = []
        if all(
            [
                hasattr(self, k)
                for k in ["epochs", "steps_per_epoch", "pct_start"]
            ]
        ):
            if self.epochs is not None and self.steps_per_epoch is not None:
                sch_gen = torch.optim.lr_scheduler.OneCycleLR(
                    opt_generator,
                    max_lr=self.learning_rate,
                    steps_per_epoch=self.steps_per_epoch,
                    epochs=self.epochs,
                    pct_start=self.pct_start,
                )
                sch_dis = torch.optim.lr_scheduler.OneCycleLR(
                    opt_discriminator,
                    max_lr=self.learning_rate,
                    steps_per_epoch=self.steps_per_epoch,
                    epochs=self.epochs,
                    pct_start=self.pct_start,
                )
                schedulers = [
                    {"scheduler": sch_gen, "interval": "step"},
                    {"scheduler": sch_dis, "interval": "step"},
                ]

        return [opt_generator, opt_discriminator], schedulers

    def on_train_batch_start(self, batch: Any, batch_idx: int) -> int | None:
        schs = self.lr_schedulers()
        if schs is not None:
            for i, sch in enumerate(schs):
                sch.step()
                self.log(
                    f"lr_{i}",
                    sch.get_last_lr()[0],
                    on_step=True,
                    on_epoch=False,
                    prog_bar=True,
                )
        return super().on_train_batch_start(batch, batch_idx)

    def generate_noise(self, x: torch.Tensor):
        x_sh = list(x.shape)
        x_sh[1] = self.generator.in_channels
        return torch.randn(*x_sh).to(x)

    def apply_generator(
        self,
        input_tensor: torch.Tensor,
        class_tensor: torch.Tensor | None = None,
        reg_tensor: torch.Tensor | None = None,
    ):
        if self.embed:
            context, converted_X = self.embedder(
                class_tensor, reg_tensor, return_X=True
            )
            return self.generator(input_tensor, context=context), converted_X
        else:
            return self.generator(input_tensor)

    def step_generator(self, input_tensor: torch.Tensor):
        gen_samples = self.apply_generator(input_tensor)
        gen_pred = apply_discriminator(gen_samples, self.discriminator)
        loss = self.adversarial_loss.generator_loss(gen_pred=gen_pred)
        return loss

    def step_discriminator(self, x: torch.Tensor, input_tensor: torch.Tensor):
        gen_samples = self.apply_generator(input_tensor)
        loss = self.adversarial_loss.discriminator_loss(
            gen_samples=gen_samples,
            real_samples=x,
            discriminator=self.discriminator,
        )
        return loss


class AutoEncoderPL(AutoEncoder, pl.LightningModule):
    def __init__(
        self,
        input_image_key: str = "input_image",
        learning_rate: float = 0.0002,
        momentum_beta1: float = 0.5,
        momentum_beta2: float = 0.99,
        *args,
        **kwargs,
    ):
        super().__init__(*args, **kwargs)
        self.input_image_key = input_image_key
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
                k, loss_dict[k], on_epoch=True, prog_bar=True, on_step=False
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
                on_step=False,
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
                on_step=False,
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
        learning_rate: float = 0.0002,
        momentum_beta1: float = 0.5,
        momentum_beta2: float = 0.99,
        var_loss_mult: float = 1.0,
        *args,
        **kwargs,
    ):
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
                k, loss_dict[k], on_epoch=True, prog_bar=True, on_step=False
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
                on_step=False,
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
                on_step=False,
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
        learning_rate: float = 0.0002,
        momentum_beta1: float = 0.9,
        momentum_beta2: float = 0.99,
        n_critic: int = 1,
        n_generator: int = 1,
        lambda_gp: float = 0.0,
        epochs: int = None,
        steps_per_epoch: int = None,
        pct_start: float = 0.1,
        *args,
        **kwargs,
    ):
        super().__init__(*args, **kwargs)
        self.input_image_key = input_image_key
        self.learning_rate = learning_rate
        self.momentum_beta1 = momentum_beta1
        self.momentum_beta2 = momentum_beta2
        self.n_critic = n_critic
        self.n_generator = n_generator
        self.lambda_gp = lambda_gp
        self.epochs = epochs
        self.steps_per_epoch = steps_per_epoch
        self.pct_start = pct_start

        if self.lambda_gp > 0.0:
            self.adversarial_loss = WGANGPLoss(lambda_gp=self.lambda_gp)
        else:
            self.adversarial_loss = AdversarialLoss()
        self.init_routine()

        self.automatic_optimization = False

    def training_step(self, batch: dict[str, Any], batch_idx: int):
        optimizer_g, optimizer_d = self.optimizers()

        x = batch[self.input_image_key]
        noise = self.generate_noise(x)

        # optimize discriminator
        self.toggle_optimizer(optimizer_d)
        loss_d = self.step_discriminator(x, noise)
        self.manual_backward(loss_d)
        optimizer_d.step()
        optimizer_d.zero_grad()
        self.untoggle_optimizer(optimizer_d)
        self.log("loss_d", loss_d, on_epoch=True, prog_bar=True, on_step=False)

        if batch_idx % self.n_critic == 0:
            self.toggle_optimizer(optimizer_g)
            loss_g = self.step_generator(noise)
            self.manual_backward(loss_g)
            optimizer_g.step()
            optimizer_g.zero_grad()
            self.untoggle_optimizer(optimizer_g)
            self.log(
                "loss_g", loss_g, on_epoch=True, prog_bar=True, on_step=False
            )

    def validation_step(self, batch: dict[str, Any], batch_idx: int):
        x = batch[self.input_image_key]
        noise = self.generate_noise(x)

        self.log(
            "val_loss_g",
            self.step_generator(noise),
            on_epoch=True,
            prog_bar=True,
            on_step=False,
        )
        self.log(
            "val_loss_d",
            self.step_discriminator(x, noise),
            on_epoch=True,
            prog_bar=True,
            on_step=False,
        )

    def test_step(self, batch: dict[str, Any], batch_idx: int):
        x = batch[self.input_image_key]
        noise = self.generate_noise(x)

        self.log(
            "test_loss_g",
            self.step_generator(noise),
            on_epoch=True,
            on_step=False,
        )
        self.log(
            "test_loss_d",
            self.step_discriminator(x, noise),
            on_epoch=True,
            on_step=False,
        )


class ClassGANPL(GAN, GANPLABC):
    def __init__(
        self,
        input_image_key: str = "input_image",
        classification_target_key: str = None,
        regression_target_key: str = None,
        class_target_specification: str = None,
        reg_target_specification: str = None,
        learning_rate: float = 0.0002,
        momentum_beta1: float = 0.9,
        momentum_beta2: float = 0.99,
        smoothing: float = 0.0,
        n_critic: int = 1,
        lambda_gp: float = 0.0,
        epochs: int = None,
        steps_per_epoch: int = None,
        pct_start: float = 0.3,
        *args,
        **kwargs,
    ):
        super().__init__(*args, **kwargs)
        self.input_image_key = input_image_key
        self.classification_target_key = classification_target_key
        self.regression_target_key = regression_target_key
        self.class_target_specification = class_target_specification
        self.reg_target_specification = reg_target_specification
        self.learning_rate = learning_rate
        self.momentum_beta1 = momentum_beta1
        self.momentum_beta2 = momentum_beta2
        self.smoothing = smoothing
        self.n_critic = n_critic
        self.lambda_gp = lambda_gp
        self.epochs = epochs
        self.steps_per_epoch = steps_per_epoch
        self.pct_start = pct_start

        if self.lambda_gp > 0.0:
            self.adversarial_loss = WGANGPLoss(lambda_gp=self.lambda_gp)
        else:
            self.adversarial_loss = SemiSLAdversarialLoss()
        self.init_routine()

        self.automatic_optimization = False

    def step_generator(
        self,
        input_tensor: torch.Tensor,
        class_target: torch.Tensor | None = None,
        reg_target: torch.Tensor | None = None,
    ):
        gen_samples, class_target = self.apply_generator(
            input_tensor, class_target, reg_target
        )
        gen_pred = self.discriminator(gen_samples, self.discriminator)
        gen_pred, class_pred, reg_pred = (
            gen_pred[0],
            gen_pred[1] if gen_pred[1] is not None else None,
            gen_pred[2] if gen_pred[1] is not None else None,
        )
        losses = self.adversarial_loss.generator_loss(
            gen_pred=gen_pred,
            class_pred=class_pred,
            reg_pred=reg_pred,
            class_target=class_target,
            reg_target=reg_target,
        )
        return losses

    def step_discriminator(
        self,
        x: torch.Tensor,
        input_tensor: torch.Tensor,
        class_target: torch.Tensor | list[torch.Tensor] | None = None,
        reg_target: torch.Tensor | list[torch.Tensor] | None = None,
    ):
        gen_samples, class_target = self.apply_generator(
            input_tensor, class_target, reg_target
        )
        losses = self.adversarial_loss.discriminator_loss(
            gen_samples=gen_samples,
            real_samples=x,
            class_target=class_target,
            reg_target=reg_target,
            discriminator=self.discriminator,
        )
        return losses

    def get_targets(self, batch: dict[str, Any]):
        class_target = (
            batch[self.classification_target_key]
            if self.classification_target_key is not None
            else None
        )
        reg_target = (
            batch[self.regression_target_key]
            if self.regression_target_key is not None
            else None
        )
        return class_target, reg_target

    def training_step(self, batch: dict[str, Any], batch_idx: int):
        optimizer_g, optimizer_d = self.optimizers()

        x = batch[self.input_image_key]
        class_target, reg_target = self.get_targets(batch)
        noise = self.generate_noise(x)

        # optimize discriminator
        self.toggle_optimizer(optimizer_d)
        losses = self.step_discriminator(
            x, noise, class_target=class_target, reg_target=reg_target
        )
        loss_d = sum([losses[k] for k in losses]) / len(losses)
        self.manual_backward(loss_d)
        optimizer_d.step()
        optimizer_d.zero_grad()
        self.untoggle_optimizer(optimizer_d)
        for k in losses:
            self.log(
                f"loss_{k}_d",
                losses[k],
                on_epoch=True,
                prog_bar=True,
                on_step=False,
            )

        if batch_idx % self.n_critic == 0:
            self.toggle_optimizer(optimizer_g)
            losses = self.step_generator(
                noise, class_target=class_target, reg_target=reg_target
            )
            loss_g = sum([losses[k] for k in losses]) / len(losses)
            self.manual_backward(loss_g)
            optimizer_g.step()
            optimizer_g.zero_grad()
            self.untoggle_optimizer(optimizer_g)
            for k in losses:
                self.log(
                    f"loss_{k}_g",
                    losses[k],
                    on_epoch=True,
                    prog_bar=True,
                    on_step=False,
                )

    def validation_step(self, batch: dict[str, Any], batch_idx: int):
        x = batch[self.input_image_key]
        noise = self.generate_noise(x)
        class_target, reg_target = self.get_targets(batch)

        losses_g = self.step_generator(
            noise, class_target=class_target, reg_target=reg_target
        )
        losses_d = self.step_discriminator(
            x, noise, class_target=class_target, reg_target=reg_target
        )
        losses_log = {
            **{f"val_loss_{k}_g": losses_g[k] for k in losses_g},
            **{f"val_loss_{k}_d": losses_d[k] for k in losses_g},
        }
        self.log_dict(losses_log, on_epoch=True, prog_bar=True, on_step=False)

    def test_step(self, batch: dict[str, Any], batch_idx: int):
        x = batch[self.input_image_key]
        noise = self.generate_noise(x)
        class_target, reg_target = self.get_targets(batch)

        losses_g = self.step_generator(
            noise, class_target=class_target, reg_target=reg_target
        )
        losses_d = self.step_discriminator(
            x, noise, class_target=class_target, reg_target=reg_target
        )
        losses_log = {
            **{f"test_loss_{k}_g": losses_g[k] for k in losses_g},
            **{f"test_loss_{k}_d": losses_d[k] for k in losses_g},
        }
        self.log_dict(losses_log, on_epoch=True, prog_bar=True, on_step=False)

    def generate(
        self, x: torch.Tensor | None, *args, **kwargs
    ) -> torch.Tensor:
        if x is None:
            x = self.generate_noise()
        return self.apply_generator(x, *args, **kwargs)[0]
