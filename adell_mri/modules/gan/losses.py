import torch
import torch.nn.functional as F


class AdversarialLoss(torch.nn.Module):
    def __init__(self, eps: float = 1e-8, smoothing: float = 0.1):
        super().__init__()
        self.eps = eps
        self.smoothing = smoothing

    def ones_like_smooth(self, x: torch.Tensor) -> torch.Tensor:
        return torch.ones_like(x) * (1 - self.smoothing)

    def generator_loss(self, pred: torch.Tensor):
        return F.binary_cross_entropy(
            torch.clamp(pred, min=self.eps), torch.zeros_like(pred)
        )

    def discriminator_loss(self, pred: torch.Tensor):
        return F.binary_cross_entropy(
            torch.clamp(1 - pred, min=self.eps), self.ones_like_smooth(pred)
        )

    def forward(self, gen_pred: torch.Tensor, real_pred: torch.Tensor):
        return (
            self.generator_loss(gen_pred).add(
                self.discriminator_loss(real_pred)
            )
            / 2.0
        )


class GaussianKLLoss(torch.nn.Module):
    def __init__(self, eps: float = 1e-8):
        super().__init__()
        self.eps = eps

    def forward(self, mu: torch.Tensor, logvar: torch.Tensor):
        # from https://arxiv.org/abs/1312.6114
        kld = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp(), dim=1)
        return kld.mean()
