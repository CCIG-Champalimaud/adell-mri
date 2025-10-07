"""
Lightning modules for StyleGAN training. Includes:

- ProGAN
"""

import torch
import lightning.pytorch as pl
import torch.nn.functional as F
from typing import Any, Callable
from adell_mri.modules.gan.gan.loss_functions import compute_gradient_penalty_r1


class ProGANPL(pl.LightningModule):
    def __init__(
        self,
        generator: torch.nn.Module,
        discriminator: torch.nn.Module,
        learning_rate: float = 0.001,
        momentum_beta1: float = 0.0,
        momentum_beta2: float = 0.99,
        epochs: int | None = None,
        steps_per_epoch: int | None = None,
        pct_start: float = 0.3,
        transition_epochs: int = 1,
        epochs_per_level: int = 5,
        gradient_penalty_lambda: float = 0.0,
        gradient_penalty_every: int = 1,
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

        assert self.generator.n_levels == self.discriminator.n_levels
        self.n_levels = self.generator.n_levels
        self.transition_steps = self.transition_epochs * self.steps_per_epoch

        self.automatic_optimization = False

        self.calculate_level_schedule()

    def calculate_level_schedule(self):
        self.level_schedule = []
        counter = 0
        level = self.n_levels
        for _ in range(self.epochs):
            self.level_schedule.append(level)
            if (counter == self.epochs_per_level) and (level > 0):
                counter = 0
                level -= 1
            counter += 1

    def configure_optimizers(
        self,
    ) -> tuple[list[torch.optim.Optimizer], list[dict[str, Any]]]:
        generator_optimizer = torch.optim.Adam(
            self.generator.parameters(),
            lr=self.learning_rate,
            betas=(self.momentum_beta1, self.momentum_beta2),
        )
        discriminator_optimizer = torch.optim.Adam(
            self.discriminator.parameters(),
            lr=self.learning_rate,
            betas=(self.momentum_beta1, self.momentum_beta2),
        )
        optimizers = [generator_optimizer, discriminator_optimizer]

        schedulers = []

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

    def step_with_optimizer(
        self, fn: Callable, optimizer: torch.optim.Optimizer, *args, **kwargs
    ):
        self.toggle_optimizer(optimizer)
        out = fn(*args, **kwargs)
        self.manual_backward()
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
        )
        target_scores = torch.ones(z.shape[0])
        g_loss = F.binary_cross_entropy_with_logits(fake_score, target_scores)
        self.log(logging_key, g_loss)
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
        )
        all_images = torch.cat(real_images, fake_images)
        scores = self.discriminator(
            all_images, level=level, prog_level=prog_level, alpha=alpha
        )
        target_scores = torch.cat(
            [
                torch.ones(fake_images.shape[0], device=fake_images.device),
                torch.zeros(fake_images.shape[0], device=fake_images.device),
            ]
        )
        d_loss = F.binary_cross_entropy_with_logits(scores, target_scores)
        self.log(logging_key, d_loss)
        if (
            self.gradient_penalty_lambda > 0
            and self.global_step % self.gradient_penalty_every == 0
        ):
            gp = compute_gradient_penalty_r1(
                discriminator=self.discriminator, real_samples=real_images
            )
            d_loss += self.gradient_penalty_lambda * gp
            self.log("gradient_penalty", gp)
        return d_loss

    def training_step(self, batch, batch_idx) -> torch.Tensor:
        batch_size = len(batch)
        real_images = batch["image"]
        current_level = self.level_schedule[self.current_epoch]
        if self.current_epoch % self.epochs_per_level < self.transition_epochs:
            alpha = self.global_step % self.transition_steps
            alpha /= self.transition_steps
        else:
            alpha = 1.0
        if alpha < 1.0:
            level = current_level - 1
            prog_level = current_level
        else:
            level = current_level
            prog_level = None

        optimizer_g, optimizer_d = self.optimizers()

        z = torch.randn(batch_size, self.generator.input_channels, 1, 1)

        # generator step
        self.step_with_optimizer(
            self.generator_step,
            optimizer_g,
            z=z,
            alpha=alpha,
            level=level,
            prog_level=prog_level,
            logging_key="train_generator_loss",
        )

        # discriminator step
        self.step_with_optimizer(
            self.discriminator_step,
            optimizer_d,
            z=z,
            real_images=real_images,
            alpha=alpha,
            level=level,
            prog_level=prog_level,
            logging_key="train_discriminator_loss",
        )
