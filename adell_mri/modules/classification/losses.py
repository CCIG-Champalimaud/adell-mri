"""
Implements loss functions for classification tasks.
"""

import torch
import torch.nn.functional as F


def ordinal_sigmoidal_loss(
    pred: torch.Tensor,
    target: torch.Tensor,
    n_classes: int,
    weight: torch.Tensor | None = None,
):
    """
    Computes the ordinal sigmoidal loss between predictions and targets.

    Args:
        pred (torch.Tensor): The model's predicted logits for each class.
        target (torch.Tensor): The ground truth class labels.
        n_classes (int): The number of classes in the classification task.
        weight (torch.Tensor, optional): A tensor of weights for each class. Defaults to None.

    Returns:
        torch.Tensor: The computed ordinal sigmoidal loss.
    """

    def label_to_ordinal(
        label: torch.Tensor, n_classes: int, ignore_0: bool = True
    ):
        """
        Converts class labels to ordinal encoding.

        Args:
            label (torch.Tensor): Input class labels.
            n_classes (int): Number of classes.
            ignore_0 (bool, optional): Whether to ignore class 0 in ordinal encoding. Defaults to True.

        Returns:
            torch.Tensor: Ordinal encoded labels.
        """
        one_hot = F.one_hot(label, n_classes)
        one_hot = one_hot.unsqueeze(1).swapaxes(1, -1).squeeze(-1)
        one_hot = torch.clamp(one_hot, max=1)
        one_hot_cumsum = torch.cumsum(one_hot, axis=1) - one_hot
        output = torch.ones_like(one_hot_cumsum, device=one_hot_cumsum.device)
        output = output - one_hot_cumsum
        if ignore_0 is True:
            output = output[:, 1:]
        return output

    target_ordinal = label_to_ordinal(target, n_classes)
    log_sigmoid = F.logsigmoid(pred)
    term1 = log_sigmoid * target_ordinal
    term2 = (log_sigmoid - pred) * (1 - target_ordinal)
    loss = -(term1 + term2).flatten(start_dim=1).sum(1)
    if weight is not None:
        weight = torch.as_tensor(weight).type_as(pred)
        weight_sample = weight[target]
        loss = loss * weight_sample

    return loss


class OrdinalSigmoidalLoss(torch.nn.Module):
    """
    Module implementation of the ordinal sigmoidal loss.

    This class provides a modular interface for computing the ordinal sigmoidal loss,
    which is useful for training neural networks with ordinal classification tasks.

    Args:
        weight (torch.Tensor): A tensor of weights for each class.
        n_classes (int): The number of classes in the classification task.
    """

    def __init__(self, n_classes: int, weight: torch.Tensor | None = None):
        """
        Initialize the OrdinalSigmoidalLoss module.

        Args:
            weight (torch.Tensor): A tensor of weights for each class.
            n_classes (int): The number of classes in the classification task.
        """
        super().__init__()
        self.n_classes = n_classes
        if weight is not None:
            self.weight = torch.as_tensor(weight)
        else:
            self.weight = None

    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        """
        Compute the ordinal sigmoidal loss.

        Args:
            pred (torch.Tensor): The model's predicted logits for each class.
            target (torch.Tensor): The ground truth class labels.

        Returns:
            torch.Tensor: The computed ordinal sigmoidal loss.
        """
        return ordinal_sigmoidal_loss(pred, target, self.n_classes, self.weight)
