"""
Lightning modules for generative adversarial network training.
"""

from copy import deepcopy
from itertools import chain
from typing import Any, Callable

import lightning.pytorch as pl
import numpy as np
import torch
import torch.nn.functional as F
import torch.optim

from ...diffusion.embedder import Embedder
from ..discriminator import Discriminator
from ..generator import Generator
from ..losses import (SemiSLAdversarialLoss, SemiSLRelativisticGANLoss,
                      SemiSLWGANGPLoss)


def mean(x: list[torch.Tensor]) -> torch.Tensor:
    """
    Convenience function to calculate mean of tensor list.

    Args:
        x (list[torch.Tensors]): list of tensors.

    Returns:
        torch.Tensor: average of x.
    """
    if len(x) == 0:
        return 0.0
    return sum(x) / len(x)


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


def l2norm(x: torch.Tensor, axis=None):
    """
    L2 normalization of a tensor.

    Args:
        x (torch.Tensor): input tensor.
        axis (int, optional): axis to normalize over. Defaults to None.
    """
    return x / torch.sqrt(torch.sum(x**2, axis=axis, keepdim=True))


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
        numerical_moments: tuple[list[float], list[float]] | None = None,
        learning_rate: float = 0.0002,
        momentum_beta1: float = 0.9,
        momentum_beta2: float = 0.99,
        smoothing: float = 0.0,
        n_critic: int = 1,
        lambda_gp: float = 0.0,
        lambda_feature_matching: float = 0.0,
        lambda_feature_map_matching: float = 0.0,
        lambda_identity: float = 0.0,
        cycle_consistency: bool = False,
        cycle_symmetry: bool = False,
        batch_size: int = 1,
        patch_size: tuple[int, int] | tuple[int, int, int] | None = None,
        epochs: int | None = None,
        steps_per_epoch: int | None = None,
        pct_start: float = 0.3,
        training_dataloader_call: Callable = None,
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
            numerical_moments (tuple[list[float], list[float]] | None,
                optional): list of means and standard deviations for numerical
                normalisation of the regression targets. Defaults to None (no
                normalisation).
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
            lambda_feature_map_matching (float, optional): weight for feature
                map matching loss (identical to feature matching but works at
                the feature map level rather than at the feature vector level;
                may be useful if there is an expected spatial structure).
                Defaults to 0.0 (no feature map matching)
            lambda_identity (float, optional): weight for identity loss (for
                image-to-image conditional generation). Defaults to 0.0.
            cycle_consistency (bool, optional): triggers cycle-consistent GAN.
                Requires specifying ``generator_cycle`` and
                ``discriminator_cycle``. Defaults to False.
            cycle_symmetry (bool, optional): triggers cycle symmetry in cycle
                consistency. Defaults to False.
            batch_size (int, optional): batch size. Defaults to 1.
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
            training_dataloader_call (Callable, optional): call for the
                training dataloader. Defaults to None.
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
        self.numerical_moments = numerical_moments
        self.learning_rate = learning_rate
        self.momentum_beta1 = momentum_beta1
        self.momentum_beta2 = momentum_beta2
        self.smoothing = smoothing
        self.n_critic = n_critic
        self.lambda_gp = lambda_gp
        self.lambda_feature_matching = lambda_feature_matching
        self.lambda_feature_map_matching = lambda_feature_map_matching
        self.lambda_identity = lambda_identity
        self.cycle_consistency = cycle_consistency
        self.cycle_symmetry = cycle_symmetry
        self.batch_size = batch_size
        self.patch_size = patch_size
        self.epochs = epochs
        self.steps_per_epoch = steps_per_epoch
        self.pct_start = pct_start
        self.training_dataloader_call = training_dataloader_call

        if self.lambda_gp > 0.0:
            self.adversarial_loss = SemiSLWGANGPLoss(lambda_gp=self.lambda_gp)
        else:
            self.adversarial_loss = SemiSLAdversarialLoss()
        self.init_routine()

        self.automatic_optimization = False

    # utilities

    @property
    def device(self) -> torch.device:
        """
        Easily accessible device argument for Module.

        Returns:
            torch.device: the device where the first parameter is alocated.
        """
        return next(self.generator.parameters()).device

    # initialization
    def init_routine(self):
        """
        A simple initialization routine:
            * Checks if generators and discriminators are specified
            * Checks whether ``generator_cycle`` and ``discriminator_cycle`` are
                specified when ``cycle_consistency`` is True
            * Saves hyperparameters
            * Initializes converters
        """
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
        """
        Initializes converters (embedders) if ``classification_target_key`` or
        ``regression_target_key`` are specified.

        Raises:
            ValueError: if ``classification_target_key`` has been specified but
                ``class_target_specification`` was not specified.
            ValueError: if ``regression_target_key`` has been specified but
                ``reg_target_specification`` was not specified.
        """
        self.embed = False
        if hasattr(self, "classification_target_key"):
            if self.classification_target_key is not None:
                if self.class_target_specification is None:
                    raise ValueError(
                        "A class_target_specification must be passed to the \
                        constructor if an classification_target_key is \
                        specified."
                    )
                self.embed = True
        if hasattr(self, "regression_target_key"):
            if self.regression_target_key is not None:
                if self.reg_target_specification is None:
                    raise ValueError(
                        "A reg_target_specification must be passed to the \
                        constructor if an regression_target_key is \
                        specified."
                    )
                self.embed = True
        if self.embed:
            if self.generator.cross_attention_dim is None:
                raise ValueError(
                    "cross_attention_dim must be specified in the \
                        generator if an additional_class_target_key or \
                        additional_reg_target_key is specified."
                )
            self.embedder = Embedder(
                self.class_target_specification,
                self.reg_target_specification,
                embedding_size=self.generator.cross_attention_dim,
                numerical_moments=self.numerical_moments,
            )

    # optimizations and ad-hoc steps
    def optimization_step_and_logging(
        self,
        optimizers: torch.optim.Optimizer | list[torch.optim.Optimizer],
        step_fns: dict[str, Callable | list | dict],
    ):
        """
        Generic optimization function supporting multiple optimizers and "step"
        functions. Each step function should return a dictionary with losses.

        To accomodate for steps requiring gradient accumulation across diferent
        steps, the ``step`` method is called for all optimizers before the
        ``zero_grad`` method is called. This allows for the same gradients to
        be used for different gradients.

        Args:
            optimizers (torch.optim.Optimizer | list[torch.optim.Optimizer]):
                an optimizer or a list of optimizers which are toggled before
                running any step and untoggled at the end of all steps.
            step_fns (dict[str, Callable  |  list  |  dict]): a dictionary with
                one obligatory key ("step_fn", which should be a callable) and
                two optional keys ("args" and "kwargs" for arguments and keyword
                arguments for ``step_fn``)
        """
        self.toggle_optimizers(optimizers)
        all_losses: dict[str, torch.Tensor] = {}
        for step_key in step_fns:
            step_fn = step_fns[step_key]["step_fn"]
            step_fn_args = step_fns[step_key].get("args", [])
            step_fn_kwargs = step_fns[step_key].get("kwargs", {})
            output = step_fn(*step_fn_args, **step_fn_kwargs)
            if isinstance(output, (tuple, list)):
                losses = output[0]
            else:
                losses = output
            for k in losses:
                all_losses[f"{step_key}_{k}"] = losses[k]
            for k in all_losses:
                self.log(
                    k,
                    all_losses[k].detach().mean(),
                    on_epoch=True,
                    prog_bar=True,
                    on_step=False,
                    sync_dist=True,
                    batch_size=self.batch_size,
                )
        reduced_losses = mean(
            [all_losses[k] for k in all_losses if len(all_losses[k].shape) == 0]
        )
        unreduced_losses = mean(
            [all_losses[k] for k in all_losses if len(all_losses[k].shape) > 0]
        )
        loss_sum = reduced_losses + unreduced_losses.mean()
        self.manual_backward(loss_sum)
        self.step_zero_grad_optimizers(optimizers)
        self.untoggle_optimizers(optimizers)

    def step_generator(
        self,
        real_samples: torch.Tensor,
        input_tensor: torch.Tensor,
        gen_samples: torch.Tensor | None = None,
        generator: torch.nn.Module | None = None,
        discriminator: torch.nn.Module | None = None,
        class_target: torch.Tensor | None = None,
        reg_target: torch.Tensor | None = None,
    ) -> tuple[dict[str, torch.Tensor], torch.Tensor]:
        """
        Generic generator optimization step.

        Args:
            real_samples (torch.Tensor): real samples.
            input_tensor (torch.Tensor): input tensor (the generator will be
                applied to this if ``gen_samples`` is None).
            gen_samples (torch.Tensor | None, optional): generated samples.
                Defaults to None.
            generator (torch.nn.Module | None, optional): generator to generate
                samples using ``input_tensor`` if ``gen_samples`` is not
                specified. Defaults to None (uses ``self.generator``).
            discriminator (torch.nn.Module | None, optional): discriminator.
                Defaults to None (uses ``self.discriminator``).
            class_target (torch.Tensor | None, optional): classification target.
                Defaults to None.
            reg_target (torch.Tensor | None, optional): regression target.
                Defaults to None.

        Returns:
            dict[str, torch.Tensor]: dictionary with loss functions.
            torch.Tensor: generated samples (``gen_samples`` if specified).
        """
        real_feat = None
        if generator is None:
            generator = self.generator
        if discriminator is None:
            discriminator = self.discriminator

        if gen_samples is None:
            gen_samples = self.apply_generator(
                input_tensor, generator, class_target, reg_target
            )
        if self.embed:
            gen_samples, class_target, reg_target = gen_samples
        if self.input_image_key:
            x_condition = input_tensor
        else:
            x_condition = None
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
        if self.lambda_feature_map_matching > 0.0:
            if real_feat is None:
                _, _, _, real_feat = self.apply_discriminator(
                    real_samples, discriminator
                )
            losses["feature_map_matching"] = self.feature_map_matching_loss(
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
    ) -> dict[str, torch.Tensor]:
        """
        Generic discriminator step.

        Args:
            real_samples (torch.Tensor): real samples.
            input_tensor (torch.Tensor): input tensor (the generator will be
                applied to this if ``gen_samples`` is None).
            generator (torch.nn.Module | None, optional): generator to generate
                samples using ``input_tensor``. Defaults to None (uses
                ``self.generator``).
            discriminator (torch.nn.Module | None, optional): discriminator.
                Defaults to None (uses ``self.discriminator``).
            class_target (torch.Tensor | None, optional): classification target.
                Defaults to None.
            reg_target (torch.Tensor | None, optional): regression target.
                Defaults to None.

        Returns:
            dict[str, torch.Tensor]: dictionary with loss functions.
            torch.Tensor: generated samples (``gen_samples`` if specified).
        """
        if generator is None:
            generator = self.generator
        if discriminator is None:
            discriminator = self.discriminator

        gen_samples = self.apply_generator(
            input_tensor, generator, class_target, reg_target
        )
        if self.embed:
            gen_samples, class_target, reg_target = gen_samples
        if self.input_image_key:
            x_condition = input_tensor
        else:
            x_condition = None
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
        input_samples_a: torch.Tensor,
        input_samples_b: torch.Tensor,
        generator_a_to_b: torch.nn.Module,
        generator_b_to_a: torch.nn.Module,
        class_target: torch.Tensor | None,
        reg_target: torch.Tensor | None,
    ) -> dict[str, torch.Tensor]:
        """
        Cycle consistency step.

        Args:
            input_samples_a (torch.Tensor): input/target samples.
            input_samples_b (torch.Tensor): target/input samples.
            generator_a_to_b (torch.nn.Module): generator that converts
                ``input_samples_a`` to the condition of ``input_samples_b``.
            generator_b_to_a (torch.nn.Module): generator that converts
                ``input_samples_b`` to the condition of ``input_samples_a``.
            class_target (torch.Tensor | None, optional): classification target.
                Defaults to None.
            reg_target (torch.Tensor | None, optional): regression target.
                Defaults to None.

        Returns:
            dict[str, torch.Tensor]: dictionary with loss functions.
        """
        boilerplate = dict(
            class_target=class_target,
            reg_target=reg_target,
            return_converted_x=False,
            symmetry=self.cycle_symmetry,
        )
        gen_samples_a = self.apply_generator(
            input_samples_b, generator=generator_b_to_a, **boilerplate
        )
        gen_samples_b = self.apply_generator(
            input_samples_a, generator=generator_a_to_b, **boilerplate
        )
        recon_samples_b = self.apply_generator(
            gen_samples_a, generator=generator_a_to_b, **boilerplate
        )
        recon_samples_a = self.apply_generator(
            gen_samples_b, generator=generator_b_to_a, **boilerplate
        )
        losses = {
            "cyc_b_to_a": self.identity_loss(recon_samples_a, gen_samples_a),
            "cyc_a_to_b": self.identity_loss(recon_samples_b, gen_samples_b),
        }
        return losses

    def regular_optimization(
        self,
        real_samples: torch.Tensor,
        input_tensor: torch.Tensor,
        class_target: torch.Tensor | None,
        reg_target: torch.Tensor | None,
        batch_idx: int,
    ):
        """
        Manual optimization step for GAN/CAN.

        Args:
            real_samples (torch.Tensor): real samples.
            input_tensor (torch.Tensor): input tensor (the generator will be
                applied to this if ``gen_samples`` is None).
            class_target (torch.Tensor | None, optional): classification target.
                Defaults to None.
            reg_target (torch.Tensor | None, optional): regression target.
                Defaults to None.
            batch_idx (int): used to trigger optimization step if
                ``self.n_critic`` > 1.
        """
        kwargs = {
            "generator": self.generator,
            "discriminator": self.discriminator,
            "real_samples": real_samples,
            "input_tensor": input_tensor,
            "class_target": class_target,
            "reg_target": reg_target,
        }
        opt_g, opt_d = self.optimizers()
        # optimize discriminator
        self.optimization_step_and_logging(
            optimizers=opt_d,
            step_fns={
                "d": {"step_fn": self.step_discriminator, "kwargs": kwargs}
            },
        )

        # optimize generator
        if batch_idx % self.n_critic == 0:
            self.optimization_step_and_logging(
                optimizers=opt_g,
                step_fns={
                    "g": {"step_fn": self.step_generator, "kwargs": kwargs}
                },
            )

    def cycle_consistency_optimization(
        self,
        real_samples: torch.Tensor,
        input_tensor: torch.Tensor,
        class_target: torch.Tensor | None,
        reg_target: torch.Tensor | None,
        batch_idx: int,
    ):
        """
        Manual optimization step for CycleGAN.

        Args:
            real_samples (torch.Tensor): real samples.
            input_tensor (torch.Tensor): input tensor (the generator will be
                applied to this if ``gen_samples`` is None).
            class_target (torch.Tensor | None, optional): classification target.
                Defaults to None.
            reg_target (torch.Tensor | None, optional): regression target.
                Defaults to None.
            batch_idx (int): used to trigger optimization step if
                ``self.n_critic`` > 1.
        """
        targets = {"class_target": class_target, "reg_target": reg_target}
        opt_g, opt_d, opt_g_cyc, opt_d_cyc = self.optimizers()
        # optimize discriminator
        self.optimization_step_and_logging(
            optimizers=opt_d,
            step_fns={
                "d": {
                    "step_fn": self.step_discriminator,
                    "kwargs": {
                        "generator": self.generator,
                        "discriminator": self.discriminator,
                        "real_samples": real_samples,
                        "input_tensor": input_tensor,
                        **targets,
                    },
                },
            },
        )

        # optimize discriminator for cycle
        self.optimization_step_and_logging(
            optimizers=opt_d_cyc,
            step_fns={
                "cycle_d": {
                    "step_fn": self.step_discriminator,
                    "kwargs": {
                        "generator": self.generator_cycle,
                        "discriminator": self.discriminator_cycle,
                        "real_samples": input_tensor,
                        "input_tensor": real_samples,
                        **targets,
                    },
                },
            },
        )

        # optimize generator
        if batch_idx % self.n_critic == 0:
            self.optimization_step_and_logging(
                optimizers=[opt_g, opt_g_cyc],
                step_fns={
                    "cycle_consistency": {
                        "step_fn": self.step_cycle,
                        "kwargs": {
                            "input_samples_a": input_tensor,
                            "input_samples_b": real_samples,
                            "generator_a_to_b": self.generator,
                            "generator_b_to_a": self.generator_cycle,
                            **targets,
                        },
                    },
                    "g": {
                        "step_fn": self.step_generator,
                        "kwargs": {
                            "generator": self.generator,
                            "discriminator": self.discriminator,
                            "real_samples": real_samples,
                            "input_tensor": input_tensor,
                            **targets,
                        },
                    },
                    "g_cycle": {
                        "step_fn": self.step_generator,
                        "kwargs": {
                            "generator": self.generator_cycle,
                            "discriminator": self.discriminator_cycle,
                            "real_samples": input_tensor,
                            "input_tensor": real_samples,
                            **targets,
                        },
                    },
                },
            )

    # loss definitions

    def feature_matching_loss(
        self,
        x_1: torch.Tensor,
        x_2: torch.Tensor,
    ) -> torch.Tensor:
        """
        Feature matching loss (an MSE between ``x_1`` and ``x_2`` after pooling
        if necessary).

        Args:
            x_1 (torch.Tensor): tensor.
            x_2 (torch.Tensor): tensor.

        Returns:
            torch.Tensor: feature matching loss weighted by
                ``self.lambda_feature_matching``.
        """

        def pool_if_necessary(x: torch.Tensor) -> torch.Tensor:
            if len(x.shape) > 2:
                x = x.flatten(start_dim=2).mean(-1)
            return x

        return torch.multiply(
            F.mse_loss(
                pool_if_necessary(x_1).mean(0), pool_if_necessary(x_2).mean(0)
            ),
            self.lambda_feature_matching,
        )

    def feature_map_matching_loss(
        self,
        x_1: torch.Tensor,
        x_2: torch.Tensor,
    ) -> torch.Tensor:
        """
        Feature matching loss (an MSE between ``x_1`` and ``x_2``).

        Args:
            x_1 (torch.Tensor): tensor.
            x_2 (torch.Tensor): tensor.

        Returns:
            torch.Tensor: feature matching loss weighted by
                ``self.lambda_feature_map_matching``.
        """

        return torch.multiply(
            F.mse_loss(x_1.mean(0), x_2.mean(0)),
            self.lambda_feature_map_matching,
        )

    def identity_loss(
        self, x_1: torch.Tensor, x_2: torch.Tensor
    ) -> torch.Tensor:
        """
        MSE between ``x_1`` and ``x_2``.

        Args:
            x_1 (torch.Tensor): tensor.
            x_2 (torch.Tensor): tensor.

        Returns:
            torch.Tensor: identity loss weighted by ``self.lambda_identity``.
        """
        return F.mse_loss(x_1, x_2) * self.lambda_identity

    # generation, data processing, functional bits

    def generate_noise(
        self, x: torch.Tensor | None = None, size: list[int] | None = None
    ) -> torch.Tensor:
        """
        Generates noise with the shape of ``x`` after replacing the channel
        dimension with ``self.generator.in_channels``. If ``x`` is not
        specified, ``size`` can be specified (the shape of the input tensor
        including batch size).

        Args:
            x (torch.Tensor | None, optional): tensor. Defaults to None.
            size (list[int] | None, optional): size. Defaults to None.

        Returns:
            torch.Tensor: noise sampled from a random normal distribution.
        """
        if x is None:
            x_sh = size
        else:
            x_sh = list(x.shape)
        x_sh[1] = self.generator.in_channels
        return torch.randn(*x_sh).to(self.device)

    @torch.no_grad
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

        if self.embed:
            n = input_tensor.shape[0]
            if "class_target" not in kwargs:
                kwargs["class_target"] = self.embedder.get_random_cat(n)
            if "reg_target" not in kwargs:
                kwargs["reg_target"] = self.embedder.get_random_num(n)
        image = self.apply_generator(
            input_tensor,
            self.generator,
            return_converted_x=False,
            *args,
            **kwargs,
        )
        if "class_target" not in kwargs:
            kwargs["class_target"] = None
        if "reg_target" not in kwargs:
            kwargs["reg_target"] = None
        return image, kwargs["class_target"], kwargs["reg_target"]

    def apply_generator(
        self,
        input_tensor: torch.Tensor,
        generator: torch.nn.Module,
        class_target: list[Any] | list[list[Any]] | None = None,
        reg_target: torch.Tensor | None = None,
        return_converted_x: bool = True,
        symmetric: bool = False,
    ) -> torch.Tensor | tuple[torch.Tensor, list[list[int]]]:
        """
        Symmetric mode assumes that targets can be symmetrised at the instance
        level. In other words, they have to have an even number of entries.
        Here, what is meant by "symmetric" is somewhat misleading - for a given
        target x with ``n`` elements at the instance level, the symmetric
        conversion returns ``concatenate(x[:n // 2], x[n // 2:])``.

        Args:
            input_tensor (torch.Tensor): input tensor.
            generator (torch.nn.Module): generator.
            class_target (list[Any] | list[list[Any]] | None, optional):
                classification target. Defaults to None.
            reg_target (torch.Tensor | None, optional): regression target.
                Defaults to None.
            return_converted_x (bool, optional): also returns the classification
                target converted to a list of lists of one-hot indices. Defaults
                to True.
            symmetric (bool, optional): symmetrizes the classification and
                regression targets (details above). Defaults to False.

        Raises:
            ValueError: if symmetric is True and either reg_target or
                class_target have an odd number of elements.
            ValueError: if symmetric is True and the first element of
                class_target is not list or tuple.

        Returns:
            torch.Tensor | tuple[torch.Tensor, list[list[int]]]: output tensor
                if no targets are provided or if return_converted_x is False,
                output tensor and converted classification target otherwise.
        """

        def make_reg_target_symmetric(x: torch.Tensor) -> torch.Tensor:
            x = x.clone()
            size = x.shape[1]
            if size % 2 != 0:
                raise ValueError(
                    "target must have even number of elements for symmetric"
                )
            return torch.cat([x[:, size // 2 :], x[:, : size // 2]], 1)

        def make_class_target_symmetric(
            x: list[Any] | list[list[Any]],
        ) -> list[list[Any]]:
            x = deepcopy(x)
            example = x[0]
            if isinstance(example, (list, tuple)) is False:
                raise ValueError(
                    "class_target must be list of lists/tuples for symmetric"
                )
            size = len(example[0])
            half = size // 2
            if size % 2 != 0:
                raise ValueError(
                    "target must have even number of elements for symmetric"
                )
            return [[*y[:half], *y[half:]] for y in x]

        if self.embed:
            if symmetric is True:
                if reg_target is not None:
                    reg_target = make_reg_target_symmetric(reg_target)
                if class_target is not None:
                    class_target = make_class_target_symmetric(class_target)
            context, converted_class_X, converted_reg_X = self.embedder(
                class_target,
                reg_target,
                batch_size=input_tensor.shape[0],
                return_X=True,
            )
            if return_converted_x is True:
                return (
                    generator(input_tensor, context=context),
                    converted_class_X,
                    converted_reg_X,
                )
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
        out = discriminator(x)
        if y is None:
            return out
        return out, y

    def prepare_image_data(
        self, batch: dict[str, Any]
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Retrieves input and target image tensors from input batch.

        Args:
            batch (dict[str, Any]): batch from dataloader.

        Returns:
            torch.Tensor: target tensor.
            torch.Tensor: input tensor.
        """
        input_tensor = None
        real_tensor = batch[self.real_image_key]
        if hasattr(self, "input_image_key"):
            if self.input_image_key is not None:
                input_tensor = batch[self.input_image_key]
        if input_tensor is None:
            input_tensor = self.generate_noise(real_tensor)
        return real_tensor, input_tensor

    def get_targets(
        self, batch: dict[str, Any]
    ) -> tuple[list[list[Any]] | list[Any] | None, torch.Tensor | None]:
        """
        Convenience function to retrieve targets.

        Args:
            batch (dict[str, Any]): batch from dataloader.

        Returns:
            list[list[Any]] | list[Any] | None: classification target if
                ``self.classification_target_key`` is not None, else None.
            torch.Tensor | None: regression target if
                ``self.regression_target_key`` is not None, else None.
        """
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

    # lightning-specific stuff

    def train_dataloader(self) -> torch.utils.data.DataLoader:
        return self.training_dataloader_call(self.batch_size)

    def set_require_grad(
        self, optimizer: torch.optim.Optimizer, requires_grad: bool
    ):
        """
        Sets ``parameter.require_grad`` to ``requires_grad`` in an optimizer for
        all groups.

        Args:
            optimizer (torch.optim.Optimizer): torch optimizer.
            requires_grad (bool): target gradient requirement.
        """
        for group in optimizer.param_groups:
            for param in group["params"]:
                param.requires_grad_(requires_grad)

    def toggle_optimizers(
        self, optimizers: list[torch.optim.Optimizer] | torch.optim.Optimizer
    ):
        """
        Toggles an optimizer or list of optimizers (sets their parameters'
        gradient requirements to True).

        Args:
            optimizers (list[torch.optim.Optimizer] | torch.optim.Optimizer): an
                optimizer or a list of optimizers.
        """
        if isinstance(optimizers, list):
            for optimizer in optimizers:
                self.set_require_grad(optimizer, True)
        else:
            self.set_require_grad(optimizers, True)

    def untoggle_optimizers(
        self, optimizers: list[torch.optim.Optimizer] | torch.optim.Optimizer
    ):
        """
        Untoggles an optimizer or list of optimizers (sets their parameters'
        gradient requirements to True).

        Args:
            optimizers (list[torch.optim.Optimizer] | torch.optim.Optimizer): an
                optimizer or a list of optimizers.
        """
        if isinstance(optimizers, list):
            for optimizer in optimizers:
                self.set_require_grad(optimizer, False)
        else:
            self.set_require_grad(optimizers, False)

    def step_zero_grad_optimizers(
        self, optimizers: list[torch.optim.Optimizer] | torch.optim.Optimizer
    ):
        """
        Takes a step and sets gradients to zero for an optimizer or list of
        optimizers.

        Args:
            optimizers (list[torch.optim.Optimizer] | torch.optim.Optimizer): an
                optimizer or a list of optimizers.
        """
        if isinstance(optimizers, list):
            for optimizer in optimizers:
                optimizer.step()
            for optimizer in optimizers:
                optimizer.zero_grad()
        else:
            optimizers.step()
            optimizers.zero_grad()

    def training_step(self, batch: dict[str, Any], batch_idx: int):
        """
        Lightning manual training step.

        Args:
            batch (dict[str, Any]): batch dictionary.
            batch_idx (int): batch index.
        """
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
                real_samples=real_samples,
                input_tensor=input_tensor,
                class_target=class_target,
                reg_target=reg_target,
                batch_idx=batch_idx,
            )

    def validation_step(self, batch: dict[str, Any], batch_idx: int):
        """
        Lightning manual validation step.

        Args:
            batch (dict[str, Any]): batch dictionary.
            batch_idx (int): batch index.
        """
        real_samples, input_tensor = self.prepare_image_data(batch)
        class_target, reg_target = self.get_targets(batch)

        losses_g = self.step_generator(
            real_samples=real_samples,
            input_tensor=input_tensor,
            class_target=class_target,
            reg_target=reg_target,
        )[0]
        losses_d = self.step_discriminator(
            real_samples=real_samples,
            input_tensor=input_tensor,
            class_target=class_target,
            reg_target=reg_target,
        )
        losses_log = {
            **{f"val_loss_{k}_g": losses_g[k] for k in losses_g},
            **{f"val_loss_{k}_d": losses_d[k] for k in losses_d},
        }
        losses_log["val_loss"] = sum(losses_log.values()) / len(losses_log)
        self.log_dict(losses_log, on_epoch=True, prog_bar=True, on_step=False)

    def test_step(self, batch: dict[str, Any], batch_idx: int):
        """
        Lightning manual test step.

        Args:
            batch (dict[str, Any]): batch dictionary.
            batch_idx (int): batch index.
        """
        real_samples, input_tensor = self.prepare_image_data(batch)
        class_target, reg_target = self.get_targets(batch)

        losses_g = self.step_generator(
            real_samples=real_samples,
            input_tensor=input_tensor,
            class_target=class_target,
            reg_target=reg_target,
        )[0]
        losses_d = self.step_discriminator(
            real_samples=real_samples,
            input_tensor=input_tensor,
            class_target=class_target,
            reg_target=reg_target,
        )
        losses_log = {
            **{f"test_loss_{k}_g": losses_g[k] for k in losses_g},
            **{f"test_loss_{k}_d": losses_d[k] for k in losses_d},
        }
        losses_log["test_loss"] = sum(losses_log.values()) / len(losses_log)
        self.log_dict(losses_log, on_epoch=True, prog_bar=True, on_step=False)

    def configure_optimizers(
        self,
    ) -> tuple[list[torch.optim.Optimizer], list[dict[str, Any]]]:
        """
        Optimizer configuration for lightning.

        Returns:
            list[torch.optim.Optimizer]: list of torch optimizers.
            list[dict[str, Any]]: list of scheduler information.
        """
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
        """
        Lightning hook for on train batch start for learning rate logging and
        optimizer untoggling.

        Args:
            batch (Any): batch from dataloader.
            batch_idx (int): batch index.
        """
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
        self.untoggle_optimizers(self.optimizers())
        return super().on_train_batch_start(batch, batch_idx)

    def forward(
        self,
        input_tensor: torch.Tensor,
        class_target: list[Any] | list[list[Any]] | None = None,
        reg_target: torch.Tensor | None = None,
        return_converted_x: bool = True,
        symmetric: bool = False,
    ) -> torch.Tensor | tuple[torch.Tensor, list[list[int]]]:
        """
        Foward wrapping ``self.apply_generator`` with core generator.

        Args:
            input_tensor (torch.Tensor): input tensor.
            class_target (list[Any] | list[list[Any]] | None, optional):
                classification target. Defaults to None.
            reg_target (torch.Tensor | None, optional): regression target.
                Defaults to None.
            return_converted_x (bool, optional): also returns the classification
                target converted to a list of lists of one-hot indices. Defaults
                to True.
            symmetric (bool, optional): symmetrizes the classification and
                regression targets (details above). Defaults to False.

        Returns:
            torch.Tensor | tuple[torch.Tensor, list[list[int]]]: output tensor
                if no targets are provided or if return_converted_x is False,
                output tensor and converted classification target otherwise.
        """
        return self.apply_generator(
            input_tensor=input_tensor,
            generator=self.generator,
            class_target=class_target,
            reg_target=reg_target,
            return_converted_x=return_converted_x,
            symmetric=symmetric,
        )


class RelativisticGANPL(GANPL):
    """
    Relativistic GAN PL module.
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.adversarial_loss = SemiSLRelativisticGANLoss(self.lambda_gp)
        self.init_routine()

        self.automatic_optimization = False

    def step_generator(
        self,
        real_samples: torch.Tensor,
        input_tensor: torch.Tensor,
        gen_samples: torch.Tensor | None = None,
        generator: torch.nn.Module | None = None,
        discriminator: torch.nn.Module | None = None,
        class_target: torch.Tensor | None = None,
        reg_target: torch.Tensor | None = None,
    ) -> tuple[dict[str, torch.Tensor], torch.Tensor]:
        """
        Generic generator optimization step.

        Args:
            real_samples (torch.Tensor): real samples.
            input_tensor (torch.Tensor): input tensor (the generator will be
                applied to this if ``gen_samples`` is None).
            gen_samples (torch.Tensor | None, optional): generated samples.
                Defaults to None.
            generator (torch.nn.Module | None, optional): generator to generate
                samples using ``input_tensor`` if ``gen_samples`` is not
                specified. Defaults to None (uses ``self.generator``).
            discriminator (torch.nn.Module | None, optional): discriminator.
                Defaults to None (uses ``self.discriminator``).
            class_target (torch.Tensor | None, optional): classification target.
                Defaults to None.
            reg_target (torch.Tensor | None, optional): regression target.
                Defaults to None.

        Returns:
            dict[str, torch.Tensor]: dictionary with loss functions.
            torch.Tensor: generated samples (``gen_samples`` if specified).
        """
        real_feat = None
        if generator is None:
            generator = self.generator
        if discriminator is None:
            discriminator = self.discriminator

        if gen_samples is None:
            gen_samples = self.apply_generator(
                input_tensor, generator, class_target, reg_target
            )
        if self.embed:
            gen_samples, class_target, reg_target = gen_samples
        if self.input_image_key:
            x_condition = input_tensor
        else:
            x_condition = None
        gen_samples = cat_not_none([gen_samples, x_condition], 1)
        real_samples = cat_not_none([real_samples, x_condition], 1)

        gen_pred, (class_target, reg_target) = self.apply_discriminator(
            gen_samples, discriminator, [class_target, reg_target]
        )
        real_pred, _, _, real_feat = self.apply_discriminator(
            real_samples, discriminator
        )
        gen_pred, class_pred, reg_pred, gen_feat = gen_pred

        losses = self.adversarial_loss.generator_loss(
            gen_pred=gen_pred,
            real_pred=real_pred,
            class_pred=class_pred,
            reg_pred=reg_pred,
            class_target=class_target,
            reg_target=reg_target,
        )
        if self.lambda_feature_matching > 0.0:
            losses["feature_matching"] = self.feature_matching_loss(
                gen_feat, real_feat
            )
        if self.lambda_feature_map_matching > 0.0:
            losses["feature_map_matching"] = self.feature_map_matching_loss(
                gen_feat, real_feat
            )
        if self.lambda_identity > 0.0:
            losses["identity"] = self.identity_loss(gen_samples, real_samples)
        return losses, gen_samples
