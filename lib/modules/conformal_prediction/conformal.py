import torch
import torch.nn.functional as F
from math import ceil


class AdaptivePredictionSets(torch.nn.Module):
    def __init__(self, alpha: float):
        super().__init__()
        self.alpha = alpha

        self.y = []
        self.pred = []

    def update(self, y: torch.Tensor, pred: torch.Tensor):
        self.y.append(y)
        self.pred.append(pred)

    def calculate(self):
        y = torch.concatenate(self.y, 0)
        pred = torch.concatenate(self.pred, 0)
        n = y.shape[0]
        pi = pred.argsort(1, descending=True)
        srt = torch.take_along_dim(pred, pi, axis=1).cumsum(axis=1)
        scores = torch.take_along_dim(srt, pi.argsort(axis=1), axis=1)[
            range(n), y
        ]
        qhat = torch.quantile(
            scores,
            ceil((n + 1) * (1 - self.alpha)) / n,
            interpolation="higher",
        )
        self.qhat = torch.nn.Parameter(qhat, requires_grad=False)

    def reset(self):
        self.y = []
        self.pred = []

    def forward(self, pred: torch.Tensor, logits: bool = False):
        if logits is True:
            pred = F.softmax(pred, -1)
        pi = pred.argsort(1, descending=True)
        srt = torch.take_along_dim(pred, pi, axis=1).cumsum(axis=1)
        pred_sets = torch.take_along_dim(
            srt <= self.qhat, pi.argsort(axis=1), axis=1
        )
        pred_sets[range(pred.shape[0]), torch.argmax(pred, 1)] = True
        return torch.concatenate([pred_sets.float(), pred], 1)
