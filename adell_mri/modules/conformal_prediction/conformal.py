from math import ceil

import torch
import torch.nn.functional as F


class AdaptivePredictionSets(torch.nn.Module):
    """
    Implements adaptive prediction sets for conformal prediction [1].

    Computes a conformal quantile (``qhat``) from calibration data such that
    the resulting prediction sets cover the true class with probability at
    least ``1 - alpha``. Prediction sets are built from the cumulative
    softmax probabilities of ranked classes.

    [1] Angelopoulos & Bates, "Gentle Introduction to Conformal Prediction
        and Distribution-Free Uncertainty Quantification" (arXiv:2107.07511).
    """

    def __init__(self, alpha: float):
        """
        Args:
            alpha (float): miscoverage level, i.e. the target error rate.
                Must be in (0, 1).
        """
        super().__init__()
        self.alpha = alpha

        self.y = []
        self.pred = []

    def update(self, y: torch.Tensor, pred: torch.Tensor):
        """
        Adds a batch of calibration labels and predictions.

        Args:
            y (torch.Tensor): integer class labels.
            pred (torch.Tensor): class probabilities or logits.
        """
        self.y.append(y)
        self.pred.append(pred)

    def calculate(self):
        """
        Computes the conformal quantile (``self.qhat``) from the accumulated
        calibration data.
        """
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
        """
        Clears the accumulated calibration data.
        """
        self.y = []
        self.pred = []

    def forward(self, pred: torch.Tensor, logits: bool = False):
        """
        Computes the prediction sets for a batch of predictions.

        Args:
            pred (torch.Tensor): class probabilities or logits.
            logits (bool, optional): whether ``pred`` contains logits that
                must be converted to probabilities. Defaults to False.

        Returns:
            torch.Tensor: concatenation of the binary prediction set
                indicators and the original probabilities along the last
                dimension.
        """
        if logits is True:
            pred = F.softmax(pred, -1)
        pi = pred.argsort(1, descending=True)
        srt = torch.take_along_dim(pred, pi, axis=1).cumsum(axis=1)
        pred_sets = torch.take_along_dim(
            srt <= self.qhat, pi.argsort(axis=1), axis=1
        )
        pred_sets[range(pred.shape[0]), torch.argmax(pred, 1)] = True
        return torch.concatenate([pred_sets.float(), pred], 1)
