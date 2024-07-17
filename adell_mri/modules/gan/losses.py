import torch
import torch.nn.functional as F


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
    d_interpolates = apply_discriminator(interpolates, discriminator)
    fake = (
        torch.Tensor(real_samples.shape[0], 1)
        .fill_(1.0)
        .to(real_samples.device)
    )
    gradients = torch.autograd.grad(
        outputs=d_interpolates,
        inputs=interpolates,
        grad_outputs=fake,
        create_graph=True,
        retain_graph=True,
        only_inputs=True,
    )[0]
    gradients = gradients.view(gradients.size(0), -1).to(real_samples.device)
    gradient_penalty = ((gradients.norm(2, dim=1) - 1) ** 2).mean()
    return gradient_penalty


class AdversarialLoss(torch.nn.Module):
    def __init__(self, eps: float = 1e-8, smoothing: float = 0.1):
        super().__init__()
        self.eps = eps
        self.smoothing = smoothing

    def ones_like_smooth(self, x: torch.Tensor) -> torch.Tensor:
        return torch.ones_like(x) * (1 - self.smoothing)

    def generator_loss(self, gen_pred: torch.Tensor = None):
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
    ) -> torch.Tensor:
        if gen_pred is None:
            gen_pred = apply_discriminator(gen_samples, discriminator)
        if real_pred is None:
            real_pred = apply_discriminator(real_samples, discriminator)
        return torch.add(
            gen_pred - real_pred,
            self.lambda_gp
            * compute_gradient_penalty(
                gen_samples, real_samples, discriminator
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


class GaussianKLLoss(torch.nn.Module):
    def __init__(self, eps: float = 1e-8):
        super().__init__()
        self.eps = eps

    def forward(self, mu: torch.Tensor, logvar: torch.Tensor):
        # from https://arxiv.org/abs/1312.6114
        kld = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp(), dim=1)
        return kld.mean()
