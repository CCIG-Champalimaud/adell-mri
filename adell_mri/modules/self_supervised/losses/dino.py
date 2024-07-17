import torch
import torch.nn.functional as F


class DinoLoss(torch.nn.Module):
    def __init__(self, temperatures: float | tuple[float, float]):
        super().__init__()
        self.temperatures = temperatures

        if isinstance(self.temperatures, float):
            self.temperatures = [self.temperatures, self.temperatures]

        self.t1 = self.temperatures[0]
        self.t2 = self.temperatures[1]

    def forward(self, a: torch.Tensor, b: torch.Tensor, C: torch.Tensor):
        return (
            torch.sum(
                torch.softmax((b - C) / self.t2, dim=-1)
                * torch.log_softmax(a - self.t1, dim=-1),
                dim=-1,
            )
            .negative()
            .mean()
        )
