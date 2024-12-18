from typing import Callable

import torch
import torch.nn.functional as F


def cat_if_none(tensors: list[torch.Tensor | None], *args, **kwargs):
    if all([t is not None for t in tensors]):
        tensors = [
            torch.stack(t, axis=-1) if isinstance(t, list) else t
            for t in tensors
        ]
        return torch.cat(tensors, *args, **kwargs)
    return None


def apply_loss(
    pred: torch.Tensor | list[torch.Tensor],
    target: torch.Tensor | list[torch.Tensor],
    loss_fn: Callable,
    *args,
    **kwargs,
) -> torch.Tensor:
    """
    Generic function to apply a function loss_fn over a set of pred and target
    inputs after stacking them on the last dimension.

    Args:
        pred (torch.Tensor | list[torch.Tensor]): prediction tensor.
        target (torch.Tensor | list[torch.Tensor]): target tensor.
        loss_fn (Callable): loss function.

    Returns:
        torch.Tensor: loss value.
    """
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
    """
    Generic function to apply discriminator and return only the first element.

    Args:
        x (torch.Tensor): input tensor.
        discriminator (torch.nn.Module): discriminator module.

    Returns:
        torch.Tensor: first element of discriminator output.
    """
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
    Calculates the gradient penalty loss for WGAN GP. This is calculated as
    (1 - norm(gradients, 2)) ** 2.

    Adapted from: https://github.com/nocotan/pytorch-lightning-gans/blob/master/models/wgan_gp.py

    Args:
        gen_samples (torch.Tensor): generated samples.
        real_samples (torch.Tensor): real samples.
        discriminator (torch.nn.Module): discriminator module.

    Returns:
        torch.Tensor: the value for the gradient penalty.
    """
    if discriminator.training is False:
        return 0.0
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


def compute_gradient_penalty_r3gan(
    samples: torch.Tensor,
    preds: torch.Tensor | None = None,
    discriminator: torch.nn.Module | None = None,
) -> torch.Tensor:
    """
    Calculates the gradient penalty loss for the R3GAN:

    $$
        \frac{\gamma}{2} \times E[|| \nabla_x D(x) ||^2]
    $$

    Source: https://openreview.net/pdf?id=VpIH3Wn9eK

    Args:
        samples (torch.Tensor): samples (real or fake).
        preds (torch.Tensor): predictions for discriminator.
        discriminator (torch.nn.Module): discriminator module.

    Returns:
        torch.Tensor: the value for the gradient penalty.
    """
    if discriminator.training is False:
        return 0.0

    if preds is None and discriminator is not None:
        preds = apply_discriminator(samples, discriminator)[0]
    elif preds is None and discriminator is None:
        raise ValueError("Either preds or discriminator must be provided.")
    gradients = torch.autograd.grad(
        outputs=preds,
        inputs=samples,
        create_graph=True,
        retain_graph=True,
        only_inputs=True,
    )[0]
    gradients = gradients.view(gradients.size(0), -1).to(samples.device)
    gradient_penalty = (gradients.square()).mean()
    return gradient_penalty


class AdversarialLoss(torch.nn.Module):
    """
    Standard adversarial loss.

    Both discriminator and generator losses are parametrized as binary cross
    entropy functions. For the discriminator, 0 is fake and 1 is real, for the
    generator 1 is fake.
    """

    def __init__(self, smoothing: float = 0.0):
        """
        Args:
            smoothing (float, optional): discriminator smoothing. Defaults to
                0.1.
        """
        super().__init__()
        self.smoothing = smoothing

    def ones_like_smooth(self, x: torch.Tensor) -> torch.Tensor:
        return torch.ones_like(x) * (1 - self.smoothing)

    def generator_loss(self, gen_pred: torch.Tensor) -> torch.Tensor:
        loss = F.binary_cross_entropy_with_logits(
            gen_pred, torch.ones_like(gen_pred)
        )
        return {"adversarial": loss}

    def discriminator_loss(
        self,
        gen_pred: torch.Tensor,
        real_pred: torch.Tensor,
        # these are kept for compatibility
        gen_samples: torch.Tensor | None = None,
        real_samples: torch.Tensor | None = None,
        discriminator: torch.nn.Module | None = None,
    ) -> torch.Tensor:
        loss = F.binary_cross_entropy_with_logits(
            torch.cat([real_pred, gen_pred]),
            torch.cat(
                [self.ones_like_smooth(real_pred), torch.zeros_like(gen_pred)]
            ),
        )
        return {"adversarial": loss}

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
    """
    Wasserstein GAN loss with gradient penalty.

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

    def generator_loss(self, gen_pred: torch.Tensor):
        loss = -F.sigmoid(gen_pred).mean()
        return {"adversarial": loss}

    def discriminator_loss(
        self,
        gen_samples: torch.Tensor,
        real_samples: torch.Tensor,
        discriminator: torch.nn.Module,
        gen_pred: torch.Tensor | None = None,
        real_pred: torch.Tensor | None = None,
    ) -> torch.Tensor:
        loss = torch.add(
            F.sigmoid(gen_pred).mean() - F.sigmoid(real_pred).mean(),
            self.lambda_gp
            * compute_gradient_penalty(
                gen_samples=gen_samples,
                real_samples=real_samples,
                discriminator=discriminator,
            ),
        )
        return {"adversarial": loss}

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


class RelativisticGANLoss(torch.nn.Module):
    """
    Relativistic GAN loss from [1].

    [1] https://arxiv.org/pdf/1807.00734
    """

    def __init__(self, lambda_gp: float = 1.0):
        super().__init__()
        self.lambda_gp = lambda_gp

    def generator_loss(
        self, gen_pred: torch.Tensor, real_pred: torch.Tensor
    ) -> torch.Tensor:
        return {"adversarial": torch.mean(-(gen_pred - real_pred))}

    def discriminator_loss(
        self,
        gen_pred: torch.Tensor,
        real_pred: torch.Tensor,
        gen_samples: torch.Tensor | None = None,
        real_samples: torch.Tensor | None = None,
        # this is kept for compatibility
        discriminator: torch.nn.Module | None = None,
    ) -> torch.Tensor:
        r1_gp = compute_gradient_penalty_r3gan(real_samples, real_pred)
        r2_gp = compute_gradient_penalty_r3gan(gen_samples, gen_pred)
        return {
            "adversarial": torch.add(
                torch.mean(-(real_pred - gen_pred)),
                self.lambda_gp / 2 * torch.add(r1_gp, r2_gp),
            )
        }

    def forward(
        self,
        gen_samples: torch.Tensor,
        real_samples: torch.Tensor,
        discriminator: torch.nn.Module,
    ) -> torch.Tensor:
        real_pred = apply_discriminator(real_samples, discriminator)
        gen_pred = apply_discriminator(gen_samples, discriminator)
        return torch.add(
            self.generator_loss(gen_pred=gen_pred, real_pred=real_pred),
            self.discriminator_loss(gen_pred=gen_pred, real_pred=real_pred),
        )


class SemiSLAdversarialLoss(torch.nn.Module):
    """
    Extends the standard adversarial loss to support classification and
    regression targets in a semi-supervised fashion.
    """

    def __init__(self, smoothing: float = 0.0):
        """
        Args:
            smoothing (float, optional): discriminator smoothing. Defaults to
                0.1.
        """

        super().__init__()
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
        # keep gen_samples here for compatibility
        gen_samples: torch.Tensor | None = None,
    ) -> torch.Tensor:
        losses = {}
        losses["adversarial"] = F.binary_cross_entropy_with_logits(
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
        gen_pred: torch.Tensor,
        real_pred: torch.Tensor,
        gen_class_pred: torch.Tensor | list[torch.Tensor] | None = None,
        real_class_pred: torch.Tensor | list[torch.Tensor] | None = None,
        class_target: torch.Tensor | list[torch.Tensor] | None = None,
        gen_reg_pred: torch.Tensor | list[torch.Tensor] | None = None,
        real_reg_pred: torch.Tensor | list[torch.Tensor] | None = None,
        reg_target: torch.Tensor | list[torch.Tensor] | None = None,
        # keep these here for compatibility
        gen_samples: torch.Tensor | None = None,
        real_samples: torch.Tensor | None = None,
        discriminator: torch.nn.Module | None = None,
    ) -> torch.Tensor:
        class_pred = cat_if_none([gen_class_pred, real_class_pred])
        class_target = cat_if_none([class_target, class_target])
        reg_pred = cat_if_none([gen_reg_pred, real_reg_pred])
        reg_target = cat_if_none([class_target, class_target])

        losses = {}
        losses["adversarial"] = F.binary_cross_entropy_with_logits(
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


class SemiSLWGANGPLoss(torch.nn.Module):
    """
    Extends the standard WGAN-GP loss to support classification and regression
    targets in a semi-supervised fashion.
    """

    def __init__(
        self,
        lambda_gp: float = 10.0,
    ):
        """
        Args:
            lambda_gp (float, optional): constant used to multiply the
                gradient penalty term. Defaults to 10.0.
        """

        super().__init__()
        self.lambda_gp = lambda_gp

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
        losses["adversarial"] = -F.sigmoid(gen_pred).mean()
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
        gen_pred: torch.Tensor | None = None,
        real_pred: torch.Tensor | None = None,
        gen_class_pred: torch.Tensor | list[torch.Tensor] | None = None,
        real_class_pred: torch.Tensor | list[torch.Tensor] | None = None,
        class_target: torch.Tensor | list[torch.Tensor] | None = None,
        gen_reg_pred: torch.Tensor | list[torch.Tensor] | None = None,
        real_reg_pred: torch.Tensor | list[torch.Tensor] | None = None,
        reg_target: torch.Tensor | list[torch.Tensor] | None = None,
    ) -> torch.Tensor:
        class_pred = cat_if_none([gen_class_pred, real_class_pred])
        class_target = cat_if_none([class_target, class_target])
        reg_pred = cat_if_none([gen_reg_pred, real_reg_pred])
        reg_target = cat_if_none([reg_target, reg_target])

        losses = {}
        losses["adversarial"] = torch.add(
            F.sigmoid(gen_pred).mean() - F.sigmoid(real_pred).mean(),
            self.lambda_gp
            * compute_gradient_penalty(
                gen_samples=gen_samples,
                real_samples=real_samples,
                discriminator=discriminator,
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
