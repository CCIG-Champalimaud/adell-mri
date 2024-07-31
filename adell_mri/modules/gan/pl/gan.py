import numpy as np
import torch
import torch.nn.functional as F
import lightning.pytorch as pl
from itertools import chain
from typing import Any
from ..losses import SemiSLAdversarialLoss, SemiSLWGANGPLoss
from ..discriminator import Discriminator
from ..generator import Generator
from ...diffusion.embedder import Embedder


def cat_not_none(
    tensors: list[torch.Tensor | None], *args, **kwargs
) -> torch.Tensor | None:
    """
    Concatenates entries which are not None.

    Args:
        tensors (list[torch.Tensor  |  None]): list of tensors or None.

    Returns:
        torch.Tensor | None: concatenated tensor of non-None elements in
            tensors or None if no non-None elements in tensors.
    """
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


class GANPL(pl.LightningModule):
    """
    Generative adversarial network with support for:
        - image-to-image conditional generation [1]
        - class and regression target-based conditional generation (with
            corresponding semi-supervised learning objectives) [2]
        - cycle-consistency losses [3]
        - identity matching loss [1]
        - feature matching loss [4]
        - label smoothing [4]
        - patching for discriminator [1,4]
        - Wasserstein GANs with gradient penalty [5,6]

    [1] https://arxiv.org/abs/1611.07004
    [2] https://arxiv.org/abs/1610.09585
    [3] https://arxiv.org/abs/1703.10593
    [4] https://arxiv.org/abs/1606.03498
    [5] https://arxiv.org/abs/1701.07875
    [6] https://arxiv.org/abs/1704.00028
    """

    def __init__(
        self,
        generator: Generator,
        discriminator: Discriminator,
        generator_cycle: Generator | None = None,
        discriminator_cycle: Discriminator | None = None,
        real_image_key: str = "real_image",
        input_image_key: str = None,
        classification_target_key: str = None,
        regression_target_key: str = None,
        class_target_specification: list[int, list[Any]] = None,
        reg_target_specification: int = None,
        learning_rate: float = 0.0002,
        momentum_beta1: float = 0.9,
        momentum_beta2: float = 0.99,
        smoothing: float = 0.0,
        n_critic: int = 1,
        lambda_gp: float = 0.0,
        lambda_feature_matching: float = 0.0,
        lambda_identity: float = 0.0,
        cycle_consistency: bool = False,
        patch_size: tuple[int, int] | tuple[int, int, int] | None = None,
        epochs: int | None = None,
        steps_per_epoch: int | None = None,
        pct_start: float = 0.3,
        *args,
        **kwargs,
    ):
        """
        Args:
            generator (Generator): generator.
            discriminator (Discriminator): discriminator.
            generator_cycle (Generator | None, optional): generator for cycle
                consistency. Should have the same input/output signature as
                ``generator``. Defaults to None.
            discriminator_cycle (Discriminator | None, optional): discriminator
                for cycle consistency. Should have the same input/output
                signature as ``discriminator``. Defaults to None.
            real_image_key (str, optional): key corresponding to the real
                images in the batch. Defaults to "real_image".
            input_image_key (str, optional): key corresponding to the input
                key for conditional image generation. Defaults to None.
            classification_target_key (str, optional): classification target
                key for conditional and semi-SL generation. Defaults to None.
            regression_target_key (str, optional): regression target key for
                conditional and semi-SL generation. Defaults to None.
            class_target_specification (list[int, list[Any]], optional):
                classification target specification. Should be a list of
                i) lists with categories or ii) integers. If a list of lists is
                specified, it will setup automatic conversion between the
                specified categories and integers; if a list of integers is
                specified, assumes each integer corresponds to the number of
                classes. Each element inthe outter list corresponds to a class.
                Defaults to None.
            reg_target_specification (int, optional): number of variables to
                be used for regression in conditional generation
                generation/semi-SL training. Defaults to None.
            learning_rate (float, optional): learning rate. Defaults to 0.0002.
            momentum_beta1 (float, optional): first momentum beta. Defaults to
                0.9.
            momentum_beta2 (float, optional): second momentum beta. Defaults to
                0.99.
            smoothing (float, optional): label smoothing for discriminator
                training. Defaults to 0.0.
            n_critic (int, optional): number of critic optimization iterations
                for each generator optimization iteration. Defaults to 1.
            lambda_gp (float, optional): weight for gradient penalisation for
                Wasserstein GAN with gradient penalty. Defaults to 0.0.
            lambda_feature_matching (float, optional): weight for feature
                matching loss. Defaults to 0.0 (no feature matching).
            lambda_identity (float, optional): weight for identity loss (for
                image-to-image conditional generation). Defaults to 0.0.
            cycle_consistency (bool, optional): triggers cycle-consistent GAN.
                Requires specifying ``generator_cycle`` and
                ``discriminator_cycle``. Defaults to False.
            patch_size (tuple[int, int] | tuple[int, int, int], optional):
                patch size for patchGAN discriminator loss. Defaults to None.
            epochs (int | None , optional): number of epochs. Specifying this
                is required to trigger OneCycle learning rate schedule.
                Defaults to None.
            steps_per_epoch (int | None , optional): number of steps per epoch.
                Specifying this is required to trigger OneCycle learning rate
                schedule. Defaults to None.
            pct_start (float, optional): fraction of steps for warm-up.
                Defaults to 0.3.
        """
        super().__init__(*args, **kwargs)
        self.generator = generator
        self.discriminator = discriminator
        self.generator_cycle = generator_cycle
        self.discriminator_cycle = discriminator_cycle
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
        self.cycle_consistency = cycle_consistency
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

    # utilities

    @property
    def device(self) -> torch.device:
        return next(self.generator.parameters()).device

    # initialization
    def init_routine(self):
        if hasattr(self, "generator") is False:
            raise ValueError("A generator must be passed to constructor.")
        if hasattr(self, "discriminator") is False:
            raise ValueError("A discriminator must be passed to constructor.")
        if self.cycle_consistency is True:
            if hasattr(self, "generator_cycle") is False:
                raise ValueError(
                    "generator_cycle must be passed to constructor if \
                        cycle_consistency is True."
                )
            if hasattr(self, "discriminator_cycle") is False:
                raise ValueError(
                    "discriminator_cycle must be passed to constructor if \
                        cycle_consistency is True."
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
        if self.embed:
            self.embedder = Embedder(
                self.class_target_specification,
                self.reg_target_specification,
                embedding_size=self.generator.cross_attention_dim,
            )

    # optimizations and ad-hoc steps
    def optimization_step_and_logging(
        self,
        optimizer: torch.optim.Optimizer,
        step_fn: callable,
        suffix: str,
        additional_losses: dict[str, torch.Tensor] | None = None,
        *step_fn_args,
        **step_fn_kwargs,
    ):
        self.toggle_optimizer(optimizer)
        output = step_fn(*step_fn_args, **step_fn_kwargs)
        if isinstance(output, (tuple, list)):
            losses = output[0]
        else:
            losses = output
        if additional_losses is not None:
            losses = {**losses, **additional_losses}
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
                sync_dist=True,
            )
        return output

    def step_generator(
        self,
        real_samples: torch.Tensor,
        input_tensor: torch.Tensor,
        gen_samples: torch.Tensor | None = None,
        generator: torch.nn.Module | None = None,
        discriminator: torch.nn.Module | None = None,
        class_target: torch.Tensor | None = None,
        reg_target: torch.Tensor | None = None,
    ):
        if generator is None:
            generator = self.generator
        if discriminator is None:
            discriminator = self.discriminator

        if gen_samples is None:
            gen_samples = self.apply_generator(
                input_tensor, generator, class_target, reg_target
            )
        if self.embed:
            gen_samples, class_target = gen_samples
        if self.input_image_key:
            x_condition = input_tensor
        gen_samples = cat_not_none([gen_samples, x_condition], 1)
        if self.lambda_feature_matching > 0.0 or self.lambda_identity > 0.0:
            real_samples = cat_not_none([real_samples, x_condition], 1)

        gen_pred, (class_target, reg_target) = self.apply_discriminator(
            gen_samples, discriminator, [class_target, reg_target]
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
            _, _, _, real_feat = self.apply_discriminator(
                real_samples, discriminator
            )
            losses["feature_matching"] = self.feature_matching_loss(
                gen_feat, real_feat
            )
        if self.lambda_identity > 0.0:
            losses["identity"] = self.identity_loss(gen_samples, real_samples)
        return losses, gen_samples

    def step_discriminator(
        self,
        real_samples: torch.Tensor,
        input_tensor: torch.Tensor,
        generator: torch.nn.Module | None = None,
        discriminator: torch.nn.Module | None = None,
        class_target: torch.Tensor | list[torch.Tensor] | None = None,
        reg_target: torch.Tensor | list[torch.Tensor] | None = None,
    ):
        if generator is None:
            generator = self.generator
        if discriminator is None:
            discriminator = self.discriminator

        gen_samples = self.apply_generator(
            input_tensor, generator, class_target, reg_target
        )
        if self.embed:
            gen_samples, class_target = gen_samples
        if self.input_image_key:
            x_condition = input_tensor
        gen_samples = cat_not_none([gen_samples, x_condition], 1)
        real_samples = cat_not_none([real_samples, x_condition], 1)

        gen_pred = self.apply_discriminator(gen_samples, discriminator)
        real_pred, (class_target, reg_target) = self.apply_discriminator(
            real_samples, discriminator, [class_target, reg_target]
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
            discriminator=discriminator,
        )
        return losses

    def step_cycle(
        self,
        gen_samples_a: torch.Tensor,
        gen_samples_b: torch.Tensor,
        generator_a_to_b: torch.nn.Module,
        generator_b_to_a: torch.nn.Module,
        class_target: torch.Tensor | None,
        reg_target: torch.Tensor | None,
    ):
        recon_samples_b = self.apply_generator(
            gen_samples_a,
            generator=generator_a_to_b,
            class_tensor=class_target,
            reg_target=reg_target,
            return_converted_x=False,
        )
        recon_samples_a = self.apply_generator(
            gen_samples_b,
            generator=generator_b_to_a,
            class_tensor=class_target,
            reg_target=reg_target,
            return_converted_x=False,
        )
        losses = {
            "cyc_b_to_a": self.identity_loss(recon_samples_a, gen_samples_a),
            "cyc_a_to_b": self.identity_loss(recon_samples_b, gen_samples_b),
        }
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

    def regular_optimization(
        self,
        real_samples: torch.Tensor,
        input_tensor: torch.Tensor,
        class_target: torch.Tensor | None,
        reg_target: torch.Tensor | None,
        batch_idx: int,
        extra_suffix: str | None = None,
    ):
        opt_g, opt_d = self.optimizers()
        # optimize discriminator
        self.optimization_step_and_logging(
            optimizer=opt_d,
            step_fn=self.step_discriminator,
            suffix="d" if extra_suffix is None else f"d_{extra_suffix}",
            generator=self.generator,
            discriminator=self.discriminator,
            real_samples=real_samples,
            input_tensor=input_tensor,
            class_target=class_target,
            reg_target=reg_target,
        )

        # optimize generator
        if batch_idx % self.n_critic == 0:
            self.optimization_step_and_logging(
                optimizer=opt_g,
                step_fn=self.step_generator,
                suffix="g" if extra_suffix is None else f"g_{extra_suffix}",
                generator=self.generator,
                discriminator=self.discriminator,
                real_samples=real_samples,
                input_tensor=input_tensor,
                class_target=class_target,
                reg_target=reg_target,
            )

    def cycle_consistency_optimization(
        self,
        real_samples: torch.Tensor,
        input_tensor: torch.Tensor,
        class_target: torch.Tensor | None,
        reg_target: torch.Tensor | None,
        batch_idx: int,
    ):
        opt_g, opt_d, opt_g_cyc, opt_d_cyc = self.optimizers()
        # optimize discriminator
        self.optimization_step_and_logging(
            optimizer=opt_d,
            step_fn=self.step_discriminator,
            suffix="d",
            generator=self.generator,
            discriminator=self.discriminator,
            real_samples=real_samples,
            input_tensor=input_tensor,
            class_target=class_target,
            reg_target=reg_target,
        )

        # optimize discriminator for cycle
        self.optimization_step_and_logging(
            optimizer=opt_d_cyc,
            step_fn=self.step_discriminator,
            suffix="cycle_d",
            generator=self.generator_cycle,
            discriminator=self.discriminator_cycle,
            real_samples=input_tensor,
            input_tensor=real_samples,
            class_target=class_target,
            reg_target=reg_target,
        )

        # optimize generator
        if batch_idx % self.n_critic == 0:
            # generate samples and register gradients
            self.toggle_optimizer(opt_g)
            self.toggle_optimizer(opt_g_cyc)
            gen_samples = self.apply_generator(
                input_tensor, self.generator, class_target, reg_target
            )
            gen_samples_cycle = self.apply_generator(
                real_samples, self.generator_cycle, class_target, reg_target
            )

            # calculate cycle-consistency loss to include in both backward steps
            cycle_losses = self.step_cycle(
                gen_samples_a=gen_samples_cycle,
                gen_samples_b=gen_samples,
                generator_a_to_b=self.generator,
                generator_b_to_a=self.generator_cycle,
                class_target=class_target,
                reg_target=reg_target,
            )

            # optimize generator with CL loss
            self.optimization_step_and_logging(
                optimizer=opt_g,
                step_fn=self.step_generator,
                suffix="g",
                additional_losses=cycle_losses,
                gen_samples=gen_samples,
                generator=self.generator,
                discriminator=self.discriminator,
                real_samples=real_samples,
                input_tensor=input_tensor,
                class_target=class_target,
                reg_target=reg_target,
            )

            # optimize generator for cycle with CL loss
            self.optimization_step_and_logging(
                optimizer=opt_g_cyc,
                step_fn=self.step_generator,
                suffix="cycle_g",
                additional_losses=cycle_losses,
                gen_samples=gen_samples_cycle,
                generator=self.generator_cycle,
                discriminator=self.discriminator_cycle,
                real_samples=input_tensor,
                input_tensor=real_samples,
                class_target=class_target,
                reg_target=reg_target,
            )

    # loss definitions

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

    # generation, data processing, functional bits

    def generate_noise(
        self, x: torch.Tensor | None = None, size: list[int] | None = None
    ) -> torch.Tensor:
        if x is None:
            x_sh = size
        else:
            x_sh = list(x.shape)
        x_sh[1] = self.generator.in_channels
        return torch.randn(*x_sh).to(self.device)

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
        image = self.apply_generator(
            input_tensor, self.generator, *args, **kwargs
        )
        if isinstance(image, tuple):
            image = image[0]
        return image

    def apply_generator(
        self,
        input_tensor: torch.Tensor,
        generator: torch.nn.Module,
        class_target: torch.Tensor | None = None,
        reg_target: torch.Tensor | None = None,
        return_converted_x: bool = True,
    ):
        if self.embed:
            context, converted_X = self.embedder(
                class_target, reg_target, return_X=True
            )
            if return_converted_x is True:
                return generator(input_tensor, context=context), converted_X
            else:
                return generator(input_tensor, context=context)
        else:
            return generator(input_tensor)

    def apply_discriminator(
        self,
        x: torch.Tensor,
        discriminator: torch.nn.Module,
        y: torch.Tensor | list[torch.Tensor] | None = None,
    ):
        if hasattr(self, "patch_size"):
            if self.patch_size is not None:
                x, y = patchify(
                    x, patch_size=self.patch_size, stride=self.patch_size, y=y
                )
        x = discriminator(x)
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

    # lightning-specific stuff

    def training_step(self, batch: dict[str, Any], batch_idx: int):
        real_samples, input_tensor = self.prepare_image_data(batch)
        class_target, reg_target = self.get_targets(batch)

        if self.cycle_consistency is False:
            self.regular_optimization(
                real_samples=real_samples,
                input_tensor=input_tensor,
                class_target=class_target,
                reg_target=reg_target,
                batch_idx=batch_idx,
            )
        else:
            self.cycle_consistency_optimization(
                real_samples=input_tensor,
                input_tensor=real_samples,
                class_target=class_target,
                reg_target=reg_target,
                batch_idx=batch_idx,
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

    def configure_optimizers(self):
        emb_pars = self.embedder.parameters() if self.embed else []
        opt_gen = torch.optim.Adam(
            chain(self.generator.parameters(), emb_pars),
            lr=self.learning_rate,
            betas=(self.momentum_beta1, self.momentum_beta2),
        )
        opt_disc = torch.optim.Adam(
            chain(self.discriminator.parameters()),
            lr=self.learning_rate,
            betas=(self.momentum_beta1, self.momentum_beta2),
        )
        optimizers = [opt_gen, opt_disc]
        if self.cycle_consistency is True:
            opt_gen_cycle = torch.optim.Adam(
                chain(self.generator_cycle.parameters(), emb_pars),
                lr=self.learning_rate,
                betas=(self.momentum_beta1, self.momentum_beta2),
            )
            opt_disc_cycle = torch.optim.Adam(
                chain(self.discriminator_cycle.parameters()),
                lr=self.learning_rate,
                betas=(self.momentum_beta1, self.momentum_beta2),
            )
            optimizers.extend([opt_gen_cycle, opt_disc_cycle])

        schedulers = []
        if all(
            [
                hasattr(self, k)
                for k in ["epochs", "steps_per_epoch", "pct_start"]
            ]
        ):
            if self.epochs is not None and self.steps_per_epoch is not None:
                for opt in optimizers:
                    sch = torch.optim.lr_scheduler.OneCycleLR(
                        opt,
                        max_lr=self.learning_rate,
                        steps_per_epoch=self.steps_per_epoch,
                        epochs=self.epochs,
                        pct_start=self.pct_start,
                    )
                    schedulers.append({"scheduler": sch, "interval": "step"})

        return optimizers, schedulers

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
                    sync_dist=True,
                )
        return super().on_train_batch_start(batch, batch_idx)

    # forward for the sake of having a forward
    def forward(
        self,
        X: torch.Tensor,
        class_target: torch.Tensor | None = None,
        reg_target: torch.Tensor | None = None,
    ):
        return self.apply_generator(
            X, self.generator, class_target, reg_target
        )
