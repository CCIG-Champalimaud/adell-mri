import torch
import lightning.pytorch as pl
from abc import ABC
from typing import Any
from .losses import AdversarialLoss
from .gan import GAN


class GANPLABC(ABC, pl.LightningModule):
    def __init__(self):
        pass

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

    # optimizer_idx = 0
    def step_generator(self, input_tensor: torch.Tensor):
        gen_samples = self.generator(input_tensor)
        gen_pred = self.discriminator(gen_samples)
        loss = self.adversarial_loss.generator_loss(gen_pred)
        return loss

    # optimizer_idx = 1
    def step_discriminator(self, x: torch.Tensor, input_tensor: torch.Tensor):
        gen_samples = self.generator(input_tensor)
        gen_pred = self.discriminator(gen_samples)
        real_pred = self.discriminator(x)
        loss = self.adversarial_loss(gen_pred, real_pred)
        return loss


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

        self.adversarial_loss = AdversarialLoss()
        self.init_routine()

    def training_step_standard(
        self, batch: dict[str, Any], batch_idx: int, optimizer_idx: int
    ):
        x = batch[self.input_image_key]
        noise = self.generate_noise(x)

        if optimizer_idx == 0:
            return self.step_generator(noise)
        elif optimizer_idx == 1:
            return self.step_discriminator(x, noise)
        else:
            raise ValueError(f"Invalid optimizer index {optimizer_idx}")

    def validation_step_standard(
        self, batch: dict[str, Any], batch_idx: int, optimizer_idx: int
    ):
        x = batch[self.input_image_key]
        noise = self.generate_noise(x)

        if optimizer_idx == 0:
            return self.step_generator(noise, prefix="val_")
        elif optimizer_idx == 1:
            return self.step_discriminator(x, noise, prefix="val_")
        else:
            raise ValueError(f"Invalid optimizer index {optimizer_idx}")

    def test_step_standard(
        self, batch: dict[str, Any], batch_idx: int, optimizer_idx: int
    ):
        x = batch[self.input_image_key]
        noise = self.generate_noise(x)

        if optimizer_idx == 0:
            return self.step_generator(noise, prefix="val_")
        elif optimizer_idx == 1:
            return self.step_discriminator(x, noise, prefix="val_")
        else:
            raise ValueError(f"Invalid optimizer index {optimizer_idx}")
