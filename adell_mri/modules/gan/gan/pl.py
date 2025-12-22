"""
Lightning modules for StyleGAN training. Includes:

- ProGAN
"""

import logging
from typing import Any, Callable

import lightning.pytorch as pl
import torch
import torch.nn.functional as F

from adell_mri.modules.gan.gan.loss_functions import compute_gradient_penalty_r1
from adell_mri.utils.logging import make_grid

logger = logging.getLogger("GAN")
logger.setLevel(logging.INFO)
logger.propagate = False
ch = logging.StreamHandler()
ch.setFormatter(
    logging.Formatter(
        fmt="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )
)
logger.addHandler(ch)


def compute_minibatch_diversity(images: torch.Tensor) -> torch.Tensor:
    """
    Compute minibatch diversity as the mean of the pixel wise standard
    deviation.

    Args:
        images: (torch.Tensor): images.

    Returns:
        (torch.Tensor): minibatch diversity.
    """
    return images.std(0).mean()


def compute_drift(predictions: torch.Tensor) -> torch.Tensor:
    return predictions.square().mean()


class ProGANPL(pl.LightningModule):
    def __init__(
        self,
        generator: torch.nn.Module,
        discriminator: torch.nn.Module,
        steps_per_epoch: int,
        learning_rate: float = 0.001,
        momentum_beta1: float = 0.0,
        momentum_beta2: float = 0.99,
        epochs: int | None = None,
        pct_start: float | None = None,
        transition_epochs: int = 1,
        epochs_per_level: int = 5,
        gradient_penalty_lambda: float = 0.0,
        gradient_penalty_every: int = 1,
        discriminator_step_every: int = 1,
        minibatch_diversity_lambda: float = 0.0,
        drift_lambda: float = 0.001,
    ):
        super().__init__()
        self.generator = generator
        self.discriminator = discriminator
        self.learning_rate = learning_rate
        self.momentum_beta1 = momentum_beta1
        self.momentum_beta2 = momentum_beta2
        self.epochs = epochs
        self.steps_per_epoch = steps_per_epoch
        self.pct_start = pct_start
        self.transition_epochs = transition_epochs
        self.epochs_per_level = epochs_per_level
        self.gradient_penalty_lambda = gradient_penalty_lambda
        self.gradient_penalty_every = gradient_penalty_every
        self.discriminator_step_every = discriminator_step_every
        self.minibatch_diversity_lambda = minibatch_diversity_lambda
        self.drift_lambda = drift_lambda

        assert self.generator.n_levels == self.discriminator.n_levels
        self.n_levels = self.generator.n_levels
        self.transition_steps = self.transition_epochs * self.steps_per_epoch
        self.transition_steps_counter = 0

        self.scaling_gp = 1
        self.current_level = None
        self.current_alpha = None
        self.current_prog_level = None
        self.growing = True

        self.automatic_optimization = False

        self.calculate_level_schedule()

    def calculate_level_schedule(self):
        self.level_schedule = []
        self.prog_level_schedule = []
        counter = 0
        level = self.n_levels
        self.last_growing_epoch = None
        epl = self.epochs_per_level
        te = self.transition_epochs
        for _ in range(self.epochs):
            if all([counter >= epl, counter < epl + te, level > 0]):
                prog_level = level - 1
            else:
                prog_level = None
            if (counter == epl + te) and (level > 0):
                counter = 0
                level -= 1
            self.level_schedule.append(level)
            self.prog_level_schedule.append(prog_level)
            if level == 0 and self.last_growing_epoch is None:
                self.last_growing_epoch = len(self.level_schedule) - 1
            counter += 1

    def configure_optimizers(
        self,
    ) -> tuple[list[torch.optim.Optimizer], list[dict[str, Any]]]:
        logger.info("Setting generator optimizer")
        generator_optimizer = torch.optim.Adam(
            self.generator.parameters(),
            lr=self.learning_rate,
            betas=(self.momentum_beta1, self.momentum_beta2),
        )
        logger.info("Setting discriminator optimizer")
        discriminator_optimizer = torch.optim.Adam(
            self.discriminator.parameters(),
            lr=self.learning_rate,
            betas=(self.momentum_beta1, self.momentum_beta2),
        )
        optimizers = [generator_optimizer, discriminator_optimizer]

        schedulers = []

        if self.pct_start and self.epochs and self.steps_per_epoch:
            logger.info("Setting one cycle LR schedule")
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

    def step_with_optimizer(
        self, fn: Callable, optimizer: torch.optim.Optimizer, *args, **kwargs
    ):
        self.toggle_optimizer(optimizer)
        out = fn(*args, **kwargs)
        self.manual_backward(out)
        optimizer.step()
        optimizer.zero_grad()
        self.untoggle_optimizer(optimizer)
        return out

    def generator_step(
        self,
        z: torch.Tensor,
        alpha: float,
        level: int,
        prog_level: int,
        logging_key: str,
    ):
        fake = self.generator(
            z, alpha=alpha, level=level, prog_level=prog_level
        )
        fake_score = self.discriminator(
            fake, level=level, prog_level=prog_level, alpha=alpha
        ).squeeze(1)
        g_loss = F.softplus(-fake_score).mean()
        self.log(logging_key, g_loss, prog_bar=True)
        if self.minibatch_diversity_lambda > 0:
            minibatch_diversity = compute_minibatch_diversity(fake)
            self.log("minibatch_diversity", minibatch_diversity, prog_bar=True)
            g_loss += -self.minibatch_diversity_lambda * minibatch_diversity
        self.log("level", level, prog_bar=True)
        self.log("alpha", alpha)
        if prog_level:
            self.log("prog_level", prog_level, prog_bar=True)
        self.log("image_height", fake.shape[2], prog_bar=True)
        self.log("image_width", fake.shape[3], prog_bar=True)
        return g_loss

    def discriminator_step(
        self,
        z: torch.Tensor,
        real_images: torch.Tensor,
        alpha: float,
        level: int,
        prog_level: int,
        logging_key: str,
    ):
        fake_images = self.generator(
            z, alpha=alpha, level=level, prog_level=prog_level
        ).detach()
        real_images = F.interpolate(
            real_images,
            size=fake_images.shape[2:],
            mode="bilinear",
        )
        b = real_images.shape[0]
        all_images = torch.cat([fake_images, real_images], 0)
        scores = self.discriminator(
            all_images,
            level=level,
            prog_level=prog_level,
            alpha=alpha,
            split_minibatch_std=True,
        ).squeeze(1)
        d_loss = F.softplus(scores[:b]).mean() + F.softplus(-scores[b:]).mean()
        self.log(logging_key, d_loss, prog_bar=True)
        if (
            self.gradient_penalty_lambda > 0
            and self.global_step % self.gradient_penalty_every == 0
        ):
            gp = compute_gradient_penalty_r1(
                discriminator=self.discriminator,
                real_samples=real_images,
                level=level,
                prog_level=prog_level,
                alpha=alpha,
            )
            d_loss += self.gradient_penalty_lambda * gp * self.scaling_gp
            self.log("gradient_penalty", gp, prog_bar=True)
        if self.drift_lambda > 0:
            drift = compute_drift(scores[b:])
            d_loss += self.drift_lambda * drift
            self.log("drift", drift, prog_bar=True)
        return d_loss

    def training_step(self, batch, batch_idx) -> torch.Tensor:
        real_images = batch["image"]
        batch_size = real_images.shape[0]
        real_images = real_images.to(self.device)
        curr_epoch = self.current_epoch
        curr_level = self.level_schedule[curr_epoch]
        curr_prog_level = self.prog_level_schedule[curr_epoch]
        trans_epochs = self.transition_epochs
        epl = self.epochs_per_level
        epoch_at_level = curr_epoch % epl
        if curr_prog_level is None:
            alpha = 1.0
            self.transition_steps_counter = 0
        elif epoch_at_level < trans_epochs:
            self.transition_steps_counter += 1
            alpha = 1 - self.transition_steps_counter / self.transition_steps
        else:
            self.transition_steps_counter = 0
            alpha = 1.0

        self.current_level = curr_level
        self.current_alpha = alpha
        self.current_prog_level = curr_prog_level

        optimizer_g, optimizer_d = self.optimizers()

        z = torch.randn(
            batch_size, self.generator.input_channels, 1, 1, device=self.device
        )

        # generator step
        self.step_with_optimizer(
            self.generator_step,
            optimizer_g,
            z=z,
            alpha=alpha,
            level=curr_level,
            prog_level=curr_prog_level,
            logging_key="train_generator_loss",
        )

        # discriminator step
        if batch_idx % self.discriminator_step_every == 0:
            self.step_with_optimizer(
                self.discriminator_step,
                optimizer_d,
                z=z,
                real_images=real_images,
                alpha=alpha,
                level=curr_level,
                prog_level=curr_prog_level,
                logging_key="train_discriminator_loss",
            )
            self.scaling_gp = 1
        else:
            self.scaling_gp += 1

    def validation_step(self, batch, batch_idx) -> None:
        pass

    def on_train_epoch_end(self) -> None:
        if hasattr(self.logger, "log_image") is False:
            return
        z = torch.randn(
            16, self.generator.input_channels, 1, 1, device=self.device
        )
        fake_images = self.generator(
            z,
            level=self.current_level,
            prog_level=self.current_prog_level,
            alpha=self.current_alpha,
        )
        image_grid = make_grid(fake_images)
        self.logger.log_image(
            "validation/generated_images", [image_grid], self.current_epoch
        )
