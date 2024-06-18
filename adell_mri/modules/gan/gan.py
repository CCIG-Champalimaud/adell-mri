import torch
from .discriminator import Discriminator
from .generator import Generator


class GAN(torch.nn.Module):
    def __init__(
        self,
        generator: Generator,
        discriminator: Discriminator,
    ):
        super().__init__()
        self.generator = generator
        self.discriminator = discriminator

    def forward(self, x: torch.Tensor, *args, **kwargs) -> torch.Tensor:
        return self.generator(x, *args, **kwargs)
