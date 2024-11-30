from itertools import combinations
from typing import Tuple

import torch


def complete_iou_loss(
    a: torch.Tensor,
    b: torch.Tensor,
    a_center: torch.Tensor = None,
    b_center: torch.Tensor = None,
    ndim: int = 3,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Calculates the complete IoU loss as proposed by Zheng et al. [1] for
    any given number of spatial dimensions.
    Combines three components - the IoU loss, the minimization of the distance
    between centers and the minimization of the difference in aspect ratios.
    To calculate the aspect ratios for n-dimensions, the aspect ratio between
    each dimension pair are averaged.

    [1] https://arxiv.org/abs/1911.08287

    Args:
        a (torch.Tensor): set of n bounding boxes.
        b (torch.Tensor): set of n bounding boxes.
        a_center (torch.Tensor, optional): n bounding box centers for a.
            Defaults to None (computes from bounding boxes).
        b_center (torch.Tensor, optional): n bounding box centers for b.
            Defaults to None (computes from bounding boxes).
        ndim (int, optional): Number of spatial dimensions. Defaults to 3.

    Returns:
        iou: the IoU loss
        cpd_component: the distance between centers component of the loss
        ar_component: the aspect ratio component of the loss
    """
    a_tl, b_tl = a[:, :ndim], b[:, :ndim]
    a_br, b_br = a[:, ndim:], b[:, ndim:]
    inter_tl = torch.maximum(a_tl, b_tl)
    inter_br = torch.minimum(a_br, b_br)
    a_size = a_br - a_tl + 1
    b_size = b_br - b_tl + 1
    inter_size = inter_br - inter_tl + 1
    if a_center is None:
        a_center = (a_tl + a_br) / 2
    if b_center is None:
        b_center = (b_tl + b_br) / 2
    diag_tl = torch.minimum(a_tl, b_tl)
    diag_br = torch.maximum(a_br, b_br)
    # calculate IoU
    inter_area = torch.prod(inter_size, axis=-1)
    union_area = torch.subtract(
        torch.prod(a_size, axis=-1) + torch.prod(b_size, axis=-1), inter_area
    )
    iou = torch.where(
        union_area > 0.0, inter_area / union_area, torch.zeros_like(union_area)
    )
    # distance between centers and between corners of external bounding box
    # for distance IoU loss
    center_distance = torch.square(a_center - b_center).sum(-1)
    bb_distance = torch.square(diag_br - diag_tl).sum(-1)
    cpd_component = center_distance / bb_distance
    # aspect ratio component
    pis = torch.pi**2
    ar_list = []
    for i, j in combinations(range(ndim), 2):
        ar_list.append(
            4
            / pis
            * (
                torch.subtract(
                    torch.arctan(a_size[:, i] / a_size[:, j]),
                    torch.arctan(b_size[:, i] / b_size[:, j]),
                )
            )
            ** 2
        )
    v = sum(ar_list) / len(ar_list)
    alpha = v / ((1 - iou) + v)
    ar_component = v * alpha

    return iou.float(), cpd_component, ar_component
