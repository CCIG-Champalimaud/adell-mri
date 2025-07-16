"""
Contains methods that are to be applied to whole batches.
"""

from typing import Tuple

import numpy as np
import torch


def label_smoothing(y: torch.Tensor, smooth_factor: float) -> torch.Tensor:
    """
    Smooths labels using a smoothing factor smoot_factor. Works only for
    binary labels.

    Args:
        y (torch.Tensor): binary classification tensor.
        smooth_factor (torch.Tensor): amount of label smoothing.

    Returns:
        torch.Tensor: smoothened binary classification tensor.
    """
    return torch.where(y < 0.5, y + smooth_factor, y - smooth_factor)


def mixup(
    x: torch.Tensor,
    y: torch.Tensor,
    mixup_alpha: float,
    g: np.random.Generator = None,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
Applies mixup to a set of inputs and labels. The mixup factor M is
    defined as M ~ Beta(mixup_alpha,mixup_alpha).

    Args:
        x (torch.Tensor): set of inputs.
        y (torch.Tensor): set of labels.
        mixup_alpha (float): alpha/beta parametrising the beta distribution
            from which the mixup factor is sampled.
        g (np.random.Generator, optional): numpy random number generator.
            Defaults to None.

    Returns:
        Tuple[torch.Tensor,torch.Tensor]: set of mixed up inputs and labels.
    """
    batch_size = y.shape[0]
    if g is None:
        g = np.random.default_rng()
    mixup_factor = torch.as_tensor(
        g.beta(mixup_alpha, mixup_alpha, batch_size),
        dtype=x.dtype,
        device=x.device,
    )
    mixup_factor_x = mixup_factor.reshape(
        [-1] + [1 for _ in range(1, len(x.shape))]
    )
    mixup_perm = g.permutation(batch_size)
    x = x * mixup_factor_x + x[mixup_perm] * (1.0 - mixup_factor_x)
    y = y * mixup_factor + y[mixup_perm] * (1.0 - mixup_factor)
    return x, y


def partial_mixup(
    x: torch.Tensor,
    y: torch.Tensor,
    mixup_alpha: float,
    mixup_fraction: float = 0.5,
    g: np.random.Generator = None,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
Applies mixup to a fraction of the set of inputs and labels. The inputs
    undergoing mixup are randomly selected according to a Bernoulli
    distribution with p=mixup_fraction. The mixup  factor M is defined as
    M ~ Beta(mixup_alpha,mixup_alpha).

    Args:
        x (torch.Tensor): set of inputs.
        y (torch.Tensor): set of labels.
        mixup_alpha (float): alpha/beta parametrising the beta distribution
            from which the mixup factor is sampled.
        mixup_fraction (float): expected fraction of mixed up samples. Defaults
            to 0.5.
        g (np.random.Generator, optional): numpy random number generator.
            Defaults to None.

    Returns:
        Tuple[torch.Tensor,torch.Tensor]: set of mixed up inputs and labels.
    """
    batch_size = y.shape[0]
    if g is None:
        g = np.random.default_rng()
    mxu_i = g.binomial(1, mixup_fraction, batch_size).astype(bool)
    mixup_factor = torch.as_tensor(
        g.beta(mixup_alpha, mixup_alpha, mxu_i.sum()),
        dtype=x.dtype,
        device=x.device,
    )
    mixup_factor_x = mixup_factor.reshape(
        [-1] + [1 for _ in range(1, len(x.shape))]
    )
    mixup_perm = g.permutation(batch_size)
    x[mxu_i] = torch.add(
        x[mxu_i] * mixup_factor_x, x[mixup_perm][mxu_i] * (1 - mixup_factor_x)
    )
    y[mxu_i] = torch.add(
        y[mxu_i] * mixup_factor, y[mixup_perm][mxu_i] * (1 - mixup_factor)
    )
    return x, y


class BatchPreprocessing:
    """
    Orchestrates batch preprocessing operations such as label smoothing
    and (partial) mixup.
    """

    def __init__(
        self,
        label_smoothing: float = None,
        mixup_alpha: float = None,
        partial_mixup: float = None,
        seed: int = 42,
    ):
        """
        Args:
            label_smoothing (float, optional): amount of label smoothign.
                Defaults to None.
            mixup_alpha (float, optional): alpha/beta parametrising the beta
                distribution from which the mixup factor is sampled. Defaults
                to None (no mixup).
            partial_mixup (float, optional): expected fraction of mixed up
                samples. Defaults to None (no partial mixup).
            seed (int, optional): random seed. Defaults to 42.
        """
        self.label_smoothing = label_smoothing
        self.mixup_alpha = mixup_alpha
        self.partial_mixup = partial_mixup
        self.seed = seed

        if self.mixup_alpha is not None:
            self.g = np.random.default_rng(seed)

    def __call__(
        self, X: torch.Tensor, y: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Applies label smoothing and (partial) mixup given the options specified
        in the constructor.

        Args:
            X (torch.Tensor): set of inputs.
            y (torch.Tensor): set of labels.

        Returns:
            Tuple[torch.Tensor,torch.Tensor]: set of transformed inputs and
                labels.
        """
        if self.label_smoothing is not None:
            y = label_smoothing(y, self.label_smoothing)
        if self.mixup_alpha is not None:
            initial_y_dtype = y.dtype
            y = y.float()
            if self.partial_mixup is not None:
                X, y = partial_mixup(
                    X, y, self.mixup_alpha, self.partial_mixup, self.g
                )
            else:
                X, y = mixup(X, y, self.mixup_alpha, self.g)
            y = y.to(initial_y_dtype)
        return X, y
