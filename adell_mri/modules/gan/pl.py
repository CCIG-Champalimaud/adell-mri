import numpy as np
import torch
import torch.nn.functional as F
import lightning.pytorch as pl
from abc import ABC
from itertools import chain
from typing import Any
from .losses import (
    AdversarialLoss,
    WGANGPLoss,
    SemiSLAdversarialLoss,
    SemiSLWGANGPLoss,
    GaussianKLLoss,
)
from .gan import GAN
from .ae import AutoEncoder
from .vae import VariationalAutoEncoder
from ..diffusion.embedder import Embedder


def cat_not_none(tensors: list[torch.Tensor | None], *args, **kwargs):
    tensors = [t for t in tensors if t is not None]
    if len(tensors) > 0:
        return torch.cat(tensors, *args, **kwargs)
    return None


def patchify(
    x: torch.Tensor,
    patch_size: tuple[int, int] | tuple[int, int, int],
    stride: tuple[int, int] | tuple[int, int, int] | None = None,
    y: torch.Tensor | list[torch.Tensor] | None = None,
) -> tuple[torch.Tensor, torch.Tensor | None]:
    """
    Produces a patched version of the input tensor X and repeats y accordingly.

    Args:
        x (torch.Tensor): input image tensor. Can be 2D or 3D.
        patch_size (tuple[int, int] | tuple[int, int, int]): size of patch.
        stride (tuple[int, int] | tuple[int, int, int] | None, optional):
            stride for the patching. Defaults to None (same as patch_size).
        y (torch.Tensor | list[torch.Tensor] | None, optional): classification
            tensor(s). Defaults to None.

    Returns:
        tuple[torch.Tensor, torch.Tensor | None]: patched X and repeated y if
            y is provided.
    """
    n_dim = len(patch_size)
    dims = [2, 3, 4][: len(patch_size)]
    if stride is None:
        stride = patch_size
    for p, s, d in zip(patch_size, stride, dims):
        x = x.unfold(d, p, s)
    if n_dim == 2:
        x = x.permute(0, *dims, 1, -2, -1)
    elif n_dim == 3:
        x = x.permute(0, *dims, 1, -3, -2, -1)
    n_patches = np.prod(x.shape[1 : n_dim + 1])
    x = x.flatten(end_dim=-n_dim - 2)
    if y is not None:
        if isinstance(y, list):
            y = [
                (
                    torch.repeat_interleave(y_, n_patches, 0)
                    if y_ is not None
                    else None
                )
                for y_ in y
            ]
        else:
            y = torch.repeat_interleave(y, n_patches, 0)
    return x, y


class GANPLABC(pl.LightningModule, ABC):
    def __init__(self):
        super().__init__()

    def optimization_step_and_logging(
        self,
        optimizer: torch.optim.Optimizer,
        step_fn: callable,
        suffix: str,
        *step_fn_args,
        **step_fn_kwargs,
    ):
        self.toggle_optimizer(optimizer)
        losses = step_fn(*step_fn_args, **step_fn_kwargs)
        loss_sum = sum([losses[k] for k in losses]) / len(losses)
        self.manual_backward(loss_sum)
        optimizer.step()
        optimizer.zero_grad()
        self.untoggle_optimizer(optimizer)
        for k in losses:
            self.log(
                f"loss_{k}_{suffix}",
                losses[k],
                on_epoch=True,
                prog_bar=True,
                on_step=False,
            )
        return loss_sum

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
        embeding_parameters = self.embedder.parameters() if self.embed else []
        opt_generator = torch.optim.Adam(
            chain(self.generator.parameters(), embeding_parameters),
            lr=self.learning_rate,
            betas=(self.momentum_beta1, self.momentum_beta2),
        )
        opt_discriminator = torch.optim.Adam(
            chain(self.discriminator.parameters(), embeding_parameters),
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

    def apply_discriminator(
        self,
        x: torch.Tensor,
        y: torch.Tensor | list[torch.Tensor] | None = None,
    ):
        if hasattr(self, "patch_size"):
            if self.patch_size is not None:
                x, y = patchify(
                    x, patch_size=self.patch_size, stride=self.patch_size, y=y
                )
        x = self.discriminator(x)
        if y is None:
            return x
        return x, y

    def prepare_image_data(self, batch: dict[str, Any]):
        input_tensor = None
        real_tensor = batch[self.real_image_key]
        if hasattr(self, "input_image_key"):
            if self.input_image_key is not None:
                input_tensor = batch[self.input_image_key]
        if input_tensor is None:
            input_tensor = self.generate_noise(real_tensor)
        return real_tensor, input_tensor


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
        real_image_key: str = "real_image",
        input_image_key: str = None,
        learning_rate: float = 0.0002,
        momentum_beta1: float = 0.9,
        momentum_beta2: float = 0.99,
        n_critic: int = 1,
        n_generator: int = 1,
        lambda_gp: float = 0.0,
        lambda_feature_matching: float = 0.0,
        patch_size: tuple[int, int] | tuple[int, int, int] = None,
        epochs: int = None,
        steps_per_epoch: int = None,
        pct_start: float = 0.1,
        *args,
        **kwargs,
    ):
        super().__init__(*args, **kwargs)
        self.real_image_key = real_image_key
        self.input_image_key = input_image_key
        self.learning_rate = learning_rate
        self.momentum_beta1 = momentum_beta1
        self.momentum_beta2 = momentum_beta2
        self.n_critic = n_critic
        self.n_generator = n_generator
        self.lambda_gp = lambda_gp
        self.lambda_feature_matching = lambda_feature_matching
        self.patch_size = patch_size
        self.epochs = epochs
        self.steps_per_epoch = steps_per_epoch
        self.pct_start = pct_start

        if self.lambda_gp > 0.0:
            self.adversarial_loss = WGANGPLoss(lambda_gp=self.lambda_gp)
        else:
            self.adversarial_loss = AdversarialLoss()
        self.init_routine()

        self.automatic_optimization = False

    def step_generator(
        self,
        x: torch.Tensor,
        input_tensor: torch.Tensor,
        x_condition: torch.Tensor | None = None,
    ):
        gen_samples = self.apply_generator(input_tensor)
        gen_samples = cat_not_none([gen_samples, x_condition], 1)
        x = cat_not_none([x, x_condition], 1)

        gen_pred, _, _, gen_feat = self.apply_discriminator(gen_samples)
        losses = self.adversarial_loss.generator_loss(gen_pred=gen_pred)
        if self.lambda_feature_matching > 0.0:
            _, _, _, real_feat = self.apply_discriminator(x)
            losses["feature_matching"] = (
                F.mse_loss(gen_feat.mean(0), real_feat.mean(0))
                * self.lambda_feature_matching
            )
        return losses

    def step_discriminator(
        self,
        x: torch.Tensor,
        input_tensor: torch.Tensor,
        x_condition: torch.Tensor | None = None,
    ):
        gen_samples = self.apply_generator(input_tensor)
        gen_samples = cat_not_none([gen_samples, x_condition], 1)
        x = cat_not_none([x, x_condition], 1)

        real_pred, _, _, _ = self.apply_discriminator(x)
        gen_pred, _, _, _ = self.apply_discriminator(gen_samples)
        losses = self.adversarial_loss.discriminator_loss(
            gen_samples=gen_samples,
            real_samples=x,
            real_pred=real_pred,
            gen_pred=gen_pred,
            discriminator=self.discriminator,
        )
        return losses

    def training_step(self, batch: dict[str, Any], batch_idx: int):
        optimizer_g, optimizer_d = self.optimizers()
        x, input_tensor = self.prepare_image_data(batch)

        # optimize discriminator
        self.optimization_step_and_logging(
            optimizer=optimizer_d,
            step_fn=self.step_discriminator,
            suffix="d",
            x=x,
            input_tensor=input_tensor,
        )

        # optimize generator
        if batch_idx % self.n_critic == 0:
            self.optimization_step_and_logging(
                optimizer=optimizer_g,
                step_fn=self.step_generator,
                suffix="g",
                x=x,
                input_tensor=input_tensor,
            )

    def validation_step(self, batch: dict[str, Any], batch_idx: int):
        x, input_tensor = self.prepare_image_data(batch)

        self.log(
            "val_loss_g",
            self.step_generator(x=x, input_tensor=input_tensor),
            on_epoch=True,
            prog_bar=True,
            on_step=False,
        )
        self.log(
            "val_loss_d",
            self.step_discriminator(x=x, input_tensor=input_tensor),
            on_epoch=True,
            prog_bar=True,
            on_step=False,
        )

    def test_step(self, batch: dict[str, Any], batch_idx: int):
        x, input_tensor = self.prepare_image_data(batch)

        self.log(
            "test_loss_g",
            self.step_generator(x=x, input_tensor=input_tensor),
            on_epoch=True,
            on_step=False,
        )
        self.log(
            "test_loss_d",
            self.step_discriminator(x=x, input_tensor=input_tensor),
            on_epoch=True,
            on_step=False,
        )


class ClassGANPL(GAN, GANPLABC):
    def __init__(
        self,
        real_image_key: str = "real_image",
        input_image_key: str = None,
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
        lambda_feature_matching: float = 0.0,
        patch_size: tuple[int, int] | tuple[int, int, int] = None,
        epochs: int = None,
        steps_per_epoch: int = None,
        pct_start: float = 0.3,
        *args,
        **kwargs,
    ):
        super().__init__(*args, **kwargs)
        self.real_image_key = real_image_key
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
        self.lambda_feature_matching = lambda_feature_matching
        self.patch_size = patch_size
        self.epochs = epochs
        self.steps_per_epoch = steps_per_epoch
        self.pct_start = pct_start

        if self.lambda_gp > 0.0:
            self.adversarial_loss = SemiSLWGANGPLoss(lambda_gp=self.lambda_gp)
        else:
            self.adversarial_loss = SemiSLAdversarialLoss()
        self.init_routine()

        self.automatic_optimization = False

    def step_generator(
        self,
        x: torch.Tensor,
        input_tensor: torch.Tensor,
        class_target: torch.Tensor | None = None,
        reg_target: torch.Tensor | None = None,
        x_condition: torch.Tensor | None = None,
    ):
        gen_samples = self.apply_generator(
            input_tensor, class_target, reg_target
        )
        if self.embed:
            gen_samples, class_target = gen_samples
        gen_samples = cat_not_none([gen_samples, x_condition], 1)
        x = cat_not_none([x, x_condition], 1)

        gen_pred, (class_target, reg_target) = self.apply_discriminator(
            gen_samples, [class_target, reg_target]
        )
        gen_pred, class_pred, reg_pred, gen_feat = gen_pred

        losses = self.adversarial_loss.generator_loss(
            gen_pred=gen_pred,
            class_pred=class_pred,
            reg_pred=reg_pred,
            class_target=class_target,
            reg_target=reg_target,
        )
        if self.lambda_feature_matching > 0.0:
            _, _, _, real_feat = self.apply_discriminator(x)
            losses["feature_matching"] = (
                F.mse_loss(gen_feat.mean(0), real_feat.mean(0))
                * self.lambda_feature_matching
            )
        return losses

    def step_discriminator(
        self,
        x: torch.Tensor,
        input_tensor: torch.Tensor,
        class_target: torch.Tensor | list[torch.Tensor] | None = None,
        reg_target: torch.Tensor | list[torch.Tensor] | None = None,
        x_condition: torch.Tensor | None = None,
    ):
        gen_samples = self.apply_generator(
            input_tensor, class_target, reg_target
        )
        if self.embed:
            gen_samples, class_target = gen_samples
        gen_samples = cat_not_none([gen_samples, x_condition], 1)
        x = cat_not_none([x, x_condition], 1)

        gen_pred = self.apply_discriminator(gen_samples)
        real_pred, (class_target, reg_target) = self.apply_discriminator(
            x, [class_target, reg_target]
        )

        gen_pred, gen_class_pred, gen_reg_pred, _ = gen_pred
        real_pred, real_class_pred, real_reg_pred, _ = real_pred

        losses = self.adversarial_loss.discriminator_loss(
            gen_samples=gen_samples,
            real_samples=x,
            class_target=class_target,
            reg_target=reg_target,
            gen_pred=gen_pred,
            gen_class_pred=gen_class_pred,
            gen_reg_pred=gen_reg_pred,
            real_pred=real_pred,
            real_class_pred=real_class_pred,
            real_reg_pred=real_reg_pred,
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

        x, input_tensor = self.prepare_image_data(batch)
        class_target, reg_target = self.get_targets(batch)

        # optimize discriminator
        self.optimization_step_and_logging(
            optimizer=optimizer_d,
            step_fn=self.step_discriminator,
            suffix="d",
            x=x,
            input_tensor=input_tensor,
            class_target=class_target,
            reg_target=reg_target,
        )

        # optimize generator
        if batch_idx % self.n_critic == 0:
            self.optimization_step_and_logging(
                optimizer=optimizer_g,
                step_fn=self.step_generator,
                suffix="g",
                x=x,
                input_tensor=input_tensor,
                class_target=class_target,
                reg_target=reg_target,
            )

    def validation_step(self, batch: dict[str, Any], batch_idx: int):
        x, input_tensor = self.prepare_image_data(batch)
        class_target, reg_target = self.get_targets(batch)

        losses_g = self.step_generator(
            x=x,
            input_tensor=input_tensor,
            class_target=class_target,
            reg_target=reg_target,
        )
        losses_d = self.step_discriminator(
            x=x,
            input_tensor=input_tensor,
            class_target=class_target,
            reg_target=reg_target,
        )
        losses_log = {
            **{f"val_loss_{k}_g": losses_g[k] for k in losses_g},
            **{f"val_loss_{k}_d": losses_d[k] for k in losses_g},
        }
        self.log_dict(losses_log, on_epoch=True, prog_bar=True, on_step=False)

    def test_step(self, batch: dict[str, Any], batch_idx: int):
        x, input_tensor = self.prepare_image_data(batch)
        class_target, reg_target = self.get_targets(batch)

        losses_g = self.step_generator(
            x=x,
            input_tensor=input_tensor,
            class_target=class_target,
            reg_target=reg_target,
        )
        losses_d = self.step_discriminator(
            x=x,
            input_tensor=input_tensor,
            class_target=class_target,
            reg_target=reg_target,
        )
        losses_log = {
            **{f"test_loss_{k}_g": losses_g[k] for k in losses_g},
            **{f"test_loss_{k}_d": losses_d[k] for k in losses_g},
        }
        self.log_dict(losses_log, on_epoch=True, prog_bar=True, on_step=False)

    def generate(
        self,
        x: torch.Tensor | None = None,
        size: list[int] | None = None,
        *args,
        **kwargs,
    ) -> torch.Tensor:
        x = self.generate_noise(x=x, size=size)
        return self.apply_generator(x, *args, **kwargs)[0]
