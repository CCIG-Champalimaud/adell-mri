import torch


class AdversarialLoss(torch.nn.Module):
    def __init__(self, eps: float = 1e-8):
        self.eps = eps

    def generator_loss(self, pred: torch.Tensor):
        return -torch.log(torch.clip(pred, min=self.eps)).mean()

    def discriminator_loss(self, pred: torch.Tensor):
        return -torch.log(torch.clip(1 - pred, min=self.eps)).mean()

    def forward(self, gen_pred: torch.Tensor, real_pred: torch.Tensor):
        return torch.add(
            self.generator_loss(gen_pred),
            self.discriminator_loss(real_pred),
        )
