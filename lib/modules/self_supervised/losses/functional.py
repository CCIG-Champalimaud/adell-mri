import torch
import torch.nn.functional as F

from typing import Tuple


def cos_sim(x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
    """Calculates the cosine similarity between x and y (wraps the functional
    function for simplicity).

    Args:
        x (torch.Tensor): tensor
        y (torch.Tensor): tensor, must be of same shape to x

    Returns:
        torch.Tensor: cosine similarity between x and y
    """
    return F.cosine_similarity(x[:, None, :], y[None, :, :], dim=-1)


def cos_dist(x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
    """Calculates the cosine distance between x and y.

    Args:
        x (torch.Tensor): tensor
        y (torch.Tensor): tensor, must be of same shape to x

    Returns:
        torch.Tensor: cosine distance between x and y
    """
    return 1 - cos_sim(x, y)


def unravel_index(
    indices: torch.LongTensor, shape: Tuple[int, ...]
) -> torch.LongTensor:
    """Converts flat indices into unraveled coordinates in a target shape.

    This is a `torch` implementation of `numpy.unravel_index`.

    Args:
        indices: A tensor of indices, (*, N).
        shape: The targeted shape, (D,).

    Returns:
        unravel coordinates, (*, N, D).
    """
    # from https://github.com/pytorch/pytorch/issues/35674#issuecomment-739492875

    shape = torch.tensor(shape)
    indices = indices % shape.prod()  # prevent out-of-bounds indices

    coord = torch.zeros(indices.size() + shape.size(), dtype=int)

    for i, dim in enumerate(reversed(shape)):
        coord[..., i] = indices % dim
        indices = torch.div(indices, dim, rounding_mode="floor")

    return coord.flip(-1)


def standardize(x: torch.Tensor, d: int = 0) -> torch.Tensor:
    """Standardizes x (subtracts mean and divides by std) according to
    dimension d.

    Args:
        x (torch.Tensor): tensor
        d (int, optional): dimension along which x will be standardized.
            Defaults to 0

    Returns:
        torch.Tensor: standardized x along dimension d
    """
    return torch.divide(
        x - torch.mean(x, d, keepdim=True), torch.std(x, d, keepdim=True)
    )


def pearson_corr(x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
    """Calculates Pearson correlation between x and y

    Args:
        x (torch.Tensor): tensor
        y (torch.Tensor): tensor

    Returns:
        torch.Tensor: Pearson correlation between x and y
    """
    x, y = x.flatten(start_dim=1), y.flatten(start_dim=1)
    x, y = standardize(x), standardize(y)
    x, y = x.unsqueeze(1), y.unsqueeze(0)
    n = torch.sum(x * y, axis=-1)
    d = torch.multiply(torch.norm(x, 2, -1), torch.norm(y, 2, -1))
    return n / d


def barlow_twins_loss(
    x: torch.Tensor, y: torch.Tensor, l: float = 0.02
) -> torch.Tensor:
    """Calculates the Barlow twins loss between x and y. This loss is composed
    of two terms: the invariance term, which maximises the Pearson correlation
    with views belonging to the same image (invariance term) and minimises the
    correlation between different images (reduction term) to promote greater
    feature diversity.

    Args:
        x (torch.Tensor): tensor
        y (torch.Tensor): tensor
        l (float, optional): term that scales the reduction term. Defaults to
            0.02.

    Returns:
        torch.Tensor: Barlow twins loss
    """
    diag_idx = torch.arange(0, x.shape)
    n = x.shape[0]
    C = pearson_corr(x, y)
    inv_term = torch.diagonal(1 - C)[diag_idx, diag_idx]
    red_term = torch.square(C)
    red_term[diag_idx, diag_idx] = 0
    loss = torch.add(
        inv_term.sum() / x.shape[0], red_term.sum() / (n * (n - 1)) * l
    )
    return loss


def simsiam_loss(x1: torch.Tensor, x2: torch.Tensor) -> torch.Tensor:
    """Loss for the SimSiam protocol.

    Args:
        x1 (torch.Tensor): tensor
        x2 (torch.Tensor): tensor

    Returns:
        torch.Tensor: SimSiam loss
    """
    cos_sim = F.cosine_similarity(x1, x2)
    return -cos_sim.mean()


def byol_loss(x1: torch.Tensor, x2: torch.Tensor) -> torch.Tensor:
    """Loss for the BYOL (bootstrap your own latent) protocol.

    Args:
        x1 (torch.Tensor): tensor
        x2 (torch.Tensor): tensor

    Returns:
        torch.Tensor: BYOL loss
    """
    return 2 * simsiam_loss(x1, x2) + 2
