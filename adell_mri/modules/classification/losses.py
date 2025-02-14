"""
Implements loss functions for classification tasks.
"""

import torch
import torch.nn.functional as F


def ordinal_sigmoidal_loss(
    pred: torch.Tensor,
    target: torch.Tensor,
    n_classes: int,
    weight: torch.Tensor = None,
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

    def label_to_ordinal(label, n_classes, ignore_0=True):
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

    weight = torch.as_tensor(weight).type_as(pred)

    target_ordinal = label_to_ordinal(target, n_classes)
    loss = F.binary_cross_entropy_with_logits(
        pred, target_ordinal.float(), reduction="none"
    )
    loss = loss.flatten(start_dim=1).sum(1)
    if weight is not None:
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

    def __init__(self, weight: torch.Tensor, n_classes: int):
        """
        Initialize the OrdinalSigmoidalLoss module.

        Args:
            weight (torch.Tensor): A tensor of weights for each class.
            n_classes (int): The number of classes in the classification task.
        """
        self.n_classes = n_classes
        self.weight = torch.as_tensor(weight)

    def __call__(self, pred, target):
        """
        Compute the ordinal sigmoidal loss.

        Args:
            pred (torch.Tensor): The model's predicted logits for each class.
            target (torch.Tensor): The ground truth class labels.

        Returns:
            torch.Tensor: The computed ordinal sigmoidal loss.
        """
        return ordinal_sigmoidal_loss(pred, target, self.n_classes, self.weight)
