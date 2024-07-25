import numpy as np
import torch
import torch.nn.functional as F
import lightning.pytorch as pl
from abc import ABC
from itertools import chain
from typing import Any
from ..losses import SemiSLAdversarialLoss, SemiSLWGANGPLoss
from ..gan import GAN
from ...diffusion.embedder import Embedder


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
            chain(self.discriminator.parameters()),
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

    def feature_matching_loss(
        self, x_1: torch.Tensor, x_2: torch.Tensor
    ) -> torch.Tensor:
        if hasattr(self, "lambda_feature_matching"):
            lfm = self.lambda_feature_matching
            return F.mse_loss(x_1.mean(0), x_2.mean(0)) * lfm
        else:
            raise ValueError("lambda_feature_matching should be defined")

    def identity_loss(
        self, x_1: torch.Tensor, x_2: torch.Tensor
    ) -> torch.Tensor:
        if hasattr(self, "lambda_identity"):
            li = self.lambda_identity
            return F.mse_loss(x_1, x_2) * li
        else:
            raise ValueError("lambda_identity should be defined")

    def prepare_image_data(self, batch: dict[str, Any]):
        input_tensor = None
        real_tensor = batch[self.real_image_key]
        if hasattr(self, "input_image_key"):
            if self.input_image_key is not None:
                input_tensor = batch[self.input_image_key]
        if input_tensor is None:
            input_tensor = self.generate_noise(real_tensor)
        return real_tensor, input_tensor


class GANPL(GAN, GANPLABC):
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
        lambda_identity: float = 0.0,
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
        self.lambda_identity = lambda_identity
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
        real_samples: torch.Tensor,
        input_tensor: torch.Tensor,
        class_target: torch.Tensor | None = None,
        reg_target: torch.Tensor | None = None,
    ):
        gen_samples = self.apply_generator(
            input_tensor, class_target, reg_target
        )
        if self.embed:
            gen_samples, class_target = gen_samples
        if self.input_image_key:
            x_condition = input_tensor
        gen_samples = cat_not_none([gen_samples, x_condition], 1)
        if self.lambda_feature_matching > 0.0 or self.lambda_identity > 0.0:
            real_samples = cat_not_none([real_samples, x_condition], 1)

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
            _, _, _, real_feat = self.apply_discriminator(real_samples)
            losses["feature_matching"] = self.feature_matching_loss(
                gen_feat, real_feat
            )
        if self.lambda_identity > 0.0:
            losses["identity"] = self.identity_loss(gen_samples, real_samples)
        return losses

    def step_discriminator(
        self,
        real_samples: torch.Tensor,
        input_tensor: torch.Tensor,
        class_target: torch.Tensor | list[torch.Tensor] | None = None,
        reg_target: torch.Tensor | list[torch.Tensor] | None = None,
    ):
        gen_samples = self.apply_generator(
            input_tensor, class_target, reg_target
        )
        if self.embed:
            gen_samples, class_target = gen_samples
        if self.input_image_key:
            x_condition = input_tensor
        gen_samples = cat_not_none([gen_samples, x_condition], 1)
        real_samples = cat_not_none([real_samples, x_condition], 1)

        gen_pred = self.apply_discriminator(gen_samples)
        real_pred, (class_target, reg_target) = self.apply_discriminator(
            real_samples, [class_target, reg_target]
        )

        gen_pred, gen_class_pred, gen_reg_pred, _ = gen_pred
        real_pred, real_class_pred, real_reg_pred, _ = real_pred

        losses = self.adversarial_loss.discriminator_loss(
            gen_samples=gen_samples,
            real_samples=real_samples,
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

        real_samples, input_tensor = self.prepare_image_data(batch)
        class_target, reg_target = self.get_targets(batch)

        # optimize discriminator
        self.optimization_step_and_logging(
            optimizer=optimizer_d,
            step_fn=self.step_discriminator,
            suffix="d",
            real_samples=real_samples,
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
                real_samples=real_samples,
                input_tensor=input_tensor,
                class_target=class_target,
                reg_target=reg_target,
            )

    def validation_step(self, batch: dict[str, Any], batch_idx: int):
        real_samples, input_tensor = self.prepare_image_data(batch)
        class_target, reg_target = self.get_targets(batch)

        losses_g = self.step_generator(
            real_samples=real_samples,
            input_tensor=input_tensor,
            class_target=class_target,
            reg_target=reg_target,
        )
        losses_d = self.step_discriminator(
            real_samples=real_samples,
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
        real_samples, input_tensor = self.prepare_image_data(batch)
        class_target, reg_target = self.get_targets(batch)

        losses_g = self.step_generator(
            real_samples=real_samples,
            input_tensor=input_tensor,
            class_target=class_target,
            reg_target=reg_target,
        )
        losses_d = self.step_discriminator(
            real_samples=real_samples,
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
        input_tensor: torch.Tensor | None = None,
        *args,
        **kwargs,
    ) -> torch.Tensor:
        if input_tensor is None:
            input_tensor = self.generate_noise(x=x, size=size)
        image = self.apply_generator(input_tensor, *args, **kwargs)
        if isinstance(image, tuple):
            image = image[0]
        return image
