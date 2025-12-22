from typing import Any

import torch
import torch.nn.functional as F

from adell_mri.modules.gan.losses import (
    apply_discriminator,
    compute_gradient_penalty_r1,
)


class StyleGANLoss(torch.nn.Module):
    """
    StyleGAN loss with R1 regularization.

    Both loss functions are calculated using the logits produced by the
    discriminator and are mean(sigmoid(gen_pred)) - mean(sigmoid(real_pred))
    for the discriminator loss and -mean(sigmoid(gen_pred)) for the
    generator.
    """

    def __init__(self, lambda_gp: float = 10.0):
        """
        Args:
            lambda_gp (float, optional): constant used to multiply the
                gradient penalty term. Defaults to 10.0.
        """
        super().__init__()
        self.lambda_gp = lambda_gp

    def generator_loss(self, gen_pred: torch.Tensor) -> dict[str, Any]:
        loss = F.softplus(-gen_pred).mean()
        return {"adversarial": loss}

    def discriminator_loss(
        self,
        gen_samples: torch.Tensor,
        real_samples: torch.Tensor,
        discriminator: torch.nn.Module,
        gen_pred: torch.Tensor | None = None,
        real_pred: torch.Tensor | None = None,
    ) -> torch.Tensor:
        loss = F.sigmoid(gen_pred).mean() - F.sigmoid(real_pred).mean()
        return {
            "adversarial": loss,
            "gradient_penalty": self.lambda_gp
            * compute_gradient_penalty_r1(
                gen_samples=gen_samples,
                real_samples=real_samples,
                discriminator=discriminator,
            ),
        }

    def forward(
        self,
        gen_samples: torch.Tensor,
        real_samples: torch.Tensor,
        discriminator: torch.nn.Module,
        *args,
        **kwargs,
    ) -> torch.Tensor:
        gen_pred = apply_discriminator(
            gen_samples, discriminator, *args, **kwargs
        )
        real_pred = apply_discriminator(
            real_samples, discriminator, *args, **kwargs
        )
        return (
            self.discriminator_loss(
                gen_pred=gen_pred,
                real_pred=real_pred,
                gen_samples=gen_samples,
                real_samples=real_samples,
                discriminator=discriminator,
            ).add(self.generator_loss(pred=gen_pred))
            / 2
        )
