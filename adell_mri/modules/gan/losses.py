import torch
import torch.nn.functional as F

from typing import Callable


def apply_loss(
    pred: torch.Tensor | list[torch.Tensor],
    target: torch.Tensor | list[torch.Tensor],
    loss_fn: Callable,
    *args,
    **kwargs,
) -> torch.Tensor:
    if isinstance(pred, (tuple, list)):
        pred = torch.stack(pred, axis=-1)
    if isinstance(target, (tuple, list)):
        target = torch.stack(target, axis=-1)
    return loss_fn(pred, target, *args, **kwargs)


def apply_discriminator(
    x: torch.Tensor,
    discriminator: torch.nn.Module,
    *args,
    **kwargs,
) -> torch.Tensor:
    x = discriminator(x, *args, **kwargs)
    if isinstance(x, (tuple, list)):
        x = x[0]
    return x


def compute_gradient_penalty(
    gen_samples: torch.Tensor,
    real_samples: torch.Tensor,
    discriminator: torch.nn.Module,
) -> torch.Tensor:
    """
    Calculates the gradient penalty loss for WGAN GP.

    Adapted from: https://github.com/nocotan/pytorch-lightning-gans/blob/master/models/wgan_gp.py
    """
    epsilon_sh = [1 for _ in real_samples.shape]
    epsilon_sh[0] = real_samples.shape[0]
    epsilon = torch.rand(*epsilon_sh).to(real_samples.device)

    interpolates = (
        epsilon * real_samples + ((1 - epsilon) * gen_samples)
    ).requires_grad_(True)
    d_interpolates = apply_discriminator(interpolates, discriminator)[0]
    fake = torch.ones_like(d_interpolates)
    gradients = torch.autograd.grad(
        outputs=d_interpolates,
        inputs=interpolates,
        grad_outputs=fake,
        create_graph=True,
        retain_graph=True,
        only_inputs=True,
    )[0]
    gradients = gradients.view(gradients.size(0), -1).to(real_samples.device)
    gradient_penalty = ((gradients.norm(2, dim=1) - 1).square()).mean()
    return gradient_penalty


class AdversarialLoss(torch.nn.Module):
    def __init__(self, eps: float = 1e-8, smoothing: float = 0.1):
        super().__init__()
        self.eps = eps
        self.smoothing = smoothing

    def ones_like_smooth(self, x: torch.Tensor) -> torch.Tensor:
        return torch.ones_like(x) * (1 - self.smoothing)

    def generator_loss(self, gen_pred: torch.Tensor) -> torch.Tensor:
        return F.binary_cross_entropy(gen_pred, torch.ones_like(gen_pred))

    def discriminator_loss(
        self,
        gen_samples: torch.Tensor,
        real_samples: torch.Tensor,
        discriminator: torch.nn.Module,
        gen_pred: torch.Tensor = None,
        real_pred: torch.Tensor = None,
        *args,
        **kwargs,
    ) -> torch.Tensor:
        if gen_pred is None:
            gen_pred = apply_discriminator(
                gen_samples, discriminator, *args, **kwargs
            )
        if real_pred is None:
            real_pred = apply_discriminator(
                real_samples, discriminator, *args, **kwargs
            )
        return F.binary_cross_entropy(
            torch.cat([real_pred, gen_pred]),
            torch.cat(
                [self.ones_like_smooth(real_pred), torch.zeros_like(gen_pred)]
            ),
        )

    def forward(
        self,
        gen_samples: torch.Tensor,
        real_samples: torch.Tensor,
        discriminator: torch.nn.Module,
    ) -> torch.Tensor:
        return (
            self.generator_loss(
                apply_discriminator(gen_samples, discriminator)
            ).add(
                self.discriminator_loss(
                    apply_discriminator(real_samples, discriminator)
                )
            )
            / 2.0
        )


class WGANGPLoss(torch.nn.Module):
    def __init__(self, lambda_gp: float = 10.0):
        super().__init__()
        self.lambda_gp = lambda_gp

    def generator_loss(self, gen_pred: torch.Tensor):
        return -gen_pred.mean()

    def discriminator_loss(
        self,
        gen_samples: torch.Tensor,
        real_samples: torch.Tensor,
        discriminator: torch.nn.Module,
        gen_pred: torch.Tensor = None,
        real_pred: torch.Tensor = None,
        *args,
        **kwargs,
    ) -> torch.Tensor:
        if gen_pred is None:
            gen_pred = apply_discriminator(
                gen_samples, discriminator, *args, **kwargs
            )
        if real_pred is None:
            real_pred = apply_discriminator(
                real_samples, discriminator, *args, **kwargs
            )
        return torch.add(
            gen_pred.mean() - real_pred.mean(),
            self.lambda_gp
            * compute_gradient_penalty(
                gen_samples=gen_samples,
                real_samples=real_samples,
                discriminator=discriminator,
            ),
        )

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


class SemiSLAdversarialLoss(torch.nn.Module):
    """
    Extends the standard adversarial loss to support classification and
    regression targets in a semi-supervised fashion.
    """

    def __init__(self, eps: float = 1e-8, smoothing: float = 0.1):
        super().__init__()
        self.eps = eps
        self.smoothing = smoothing

    def ones_like_smooth(self, x: torch.Tensor) -> torch.Tensor:
        return torch.ones_like(x) * (1 - self.smoothing)

    def apply_loss(
        self,
        pred: torch.Tensor | list[torch.Tensor],
        target: torch.Tensor | list[torch.Tensor],
        loss_fn: Callable,
    ) -> torch.Tensor:
        if isinstance(pred, (tuple, list)):
            return sum([loss_fn(p, t) for p, t in zip(pred, target)])
        else:
            return loss_fn(pred, target)

    def cat_if_none(self, tensors: list[torch.Tensor | None], *args, **kwargs):
        if all([t is not None for t in tensors]):
            tensors = [
                torch.stack(t, axis=-1) if isinstance(t, list) else t
                for t in tensors
            ]
            return torch.cat(tensors, *args, **kwargs)
        return None

    def pred(
        self, x: torch.Tensor, discriminator: torch.nn.Module, *args, **kwargs
    ) -> torch.Tensor:
        pred = discriminator(x, *args, **kwargs)
        pred, class_pred, reg_pred = (
            pred[0],
            pred[1] if pred[1] is not None else None,
            pred[2] if pred[1] is not None else None,
        )
        return pred, class_pred, reg_pred

    def generator_loss(
        self,
        gen_pred: torch.Tensor,
        class_pred: torch.Tensor | None = None,
        class_target: torch.Tensor | None = None,
        reg_pred: torch.Tensor | None = None,
        reg_target: torch.Tensor | None = None,
    ) -> torch.Tensor:
        losses = {}
        losses["adversarial"] = F.binary_cross_entropy(
            gen_pred, torch.ones_like(gen_pred)
        )
        if (class_pred is not None) and (class_target is not None):
            losses["class"] = apply_loss(
                class_pred, class_target, F.cross_entropy
            )

        if (reg_pred is not None) and (reg_target is not None):
            losses["reg"] = apply_loss(reg_pred, reg_target, F.mse_loss)
        return losses

    def discriminator_loss(
        self,
        gen_samples: torch.Tensor,
        real_samples: torch.Tensor,
        discriminator: torch.nn.Module,
        gen_pred: torch.Tensor = None,
        real_pred: torch.Tensor = None,
        class_pred: torch.Tensor | list[torch.Tensor] | None = None,
        class_target: torch.Tensor | list[torch.Tensor] | None = None,
        reg_pred: torch.Tensor | list[torch.Tensor] | None = None,
        reg_target: torch.Tensor | list[torch.Tensor] | None = None,
        *args,
        **kwargs,
    ) -> torch.Tensor:
        if gen_pred is None:
            gen_pred, gen_class_pred, gen_reg_pred = self.pred(
                gen_samples, discriminator, *args, **kwargs
            )
        if real_pred is None:
            real_pred, real_class_pred, real_reg_pred = self.pred(
                real_samples, discriminator, *args, **kwargs
            )

        class_pred = self.cat_if_none([gen_class_pred, real_class_pred])
        class_target = self.cat_if_none([class_target, class_target])
        reg_pred = self.cat_if_none([gen_reg_pred, real_reg_pred])
        reg_target = self.cat_if_none([class_target, class_target])

        losses = {}
        losses["adversarial"] = F.binary_cross_entropy(
            torch.cat([real_pred, gen_pred]),
            torch.cat(
                [
                    self.ones_like_smooth(real_pred),
                    torch.zeros_like(gen_pred),
                ]
            ),
        )
        if (class_pred is not None) and (class_target is not None):
            losses["class"] = apply_loss(
                class_pred, class_target, F.cross_entropy
            )
        if (reg_pred is not None) and (reg_target is not None):
            losses["reg"] = apply_loss(reg_pred, reg_target, F.mse_loss)
        return losses

    def forward(
        self,
        gen_samples: torch.Tensor,
        real_samples: torch.Tensor,
        discriminator: torch.nn.Module,
        class_pred: torch.Tensor | list[torch.Tensor] | None = None,
        class_target: torch.Tensor | list[torch.Tensor] | None = None,
        reg_pred: torch.Tensor | list[torch.Tensor] | None = None,
        reg_target: torch.Tensor | list[torch.Tensor] | None = None,
    ) -> torch.Tensor:
        return torch.add(
            self.discriminator_loss(
                gen_samples=gen_samples,
                real_samples=real_samples,
                discriminator=discriminator,
                class_pred=class_pred,
                class_target=class_target,
                reg_pred=reg_pred,
                reg_target=reg_target,
            ),
            self.generator_loss(
                gen_pred=apply_discriminator(
                    gen_samples, discriminator=discriminator
                ),
                class_pred=class_pred,
                class_target=class_target,
                reg_pred=reg_pred,
                reg_target=reg_target,
            ),
        )


class GaussianKLLoss(torch.nn.Module):
    def __init__(self, eps: float = 1e-8):
        super().__init__()
        self.eps = eps

    def forward(self, mu: torch.Tensor, logvar: torch.Tensor):
        # from https://arxiv.org/abs/1312.6114
        kld = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp(), dim=1)
        return kld.mean()
