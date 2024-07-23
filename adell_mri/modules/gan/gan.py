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

    @property
    def device(self) -> torch.device:
        return next(self.generator.parameters()).device

    def forward(self, x: torch.Tensor, *args, **kwargs) -> torch.Tensor:
        return self.generator(x, *args, **kwargs)

    def generate_noise(
        self, x: torch.Tensor | None = None, size: list[int] | None = None
    ) -> torch.Tensor:
        if x is None:
            x_sh = size
        else:
            x_sh = list(x.shape)
        x_sh[1] = self.generator.in_channels
        return torch.randn(*x_sh).to(self.device)

    def generate(
        self,
        x: torch.Tensor | None = None,
        size: list[int] | None = None,
        *args,
        **kwargs,
    ) -> torch.Tensor:
        x = self.generate_noise(x, size)
        return self.generator(x, *args, **kwargs)
