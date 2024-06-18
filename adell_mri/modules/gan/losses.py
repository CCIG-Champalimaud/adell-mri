import torch


class AdversarialLoss(torch.nn.Module):
    def __init__(self, eps: float = 1e-8):
        super().__init__()
        self.eps = eps

    def generator_loss(self, pred: torch.Tensor):
        return -torch.log(torch.clamp(pred, min=self.eps)).mean()

    def discriminator_loss(self, pred: torch.Tensor):
        return -torch.log(torch.clamp(1 - pred, min=self.eps)).mean()

    def forward(self, gen_pred: torch.Tensor, real_pred: torch.Tensor):
        return torch.add(
            self.generator_loss(gen_pred),
            self.discriminator_loss(real_pred),
        )


class GaussianKLLoss(torch.nn.Module):
    def __init__(self, eps: float = 1e-8):
        super().__init__()
        self.eps = eps

    def forward(mu: torch.Tensor, logvar: torch.Tensor):
        # from: https://arxiv.org/abs/1312.6114
        KLD = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp(), dim=1)
        return KLD.mean()
