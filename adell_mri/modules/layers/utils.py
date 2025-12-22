from math import floor
from typing import List

import numpy as np
import torch


def split_int_into_n(i: int, n: int) -> List[int]:
    """
    Divides an integer i into n slots, where i is distributed as equally
    as possible across each slot.

    Args:
        i (int): number to be divided.
        n (int): number of slots.

    Returns:
        List[int]: list of numbers which sum to 1.
    """
    r = i % n
    o = [floor(i / n) for _ in range(n)]
    idx = 0
    while r > 0:
        o[idx] += 1
        r -= 1
        idx += 1
    return o


def crop_to_size(X: torch.Tensor, output_size: list) -> torch.Tensor:
    """
    Crops a tensor to the size given by list. Assumes the first two
        dimensions are the batch and channel dimensions.

        Args:
            X (torch.Tensor): torch Tensor to be cropped
            output_size (list): list with the output dimensions. Should be
            smaller or identical to the current dimensions and the list length
            should be len(X.shape)

        Returns:
            torch.Tensor: a resized torch Tensor
    """
    sh = list(X.shape)[2:]
    diff = [i - j for i, j in zip(sh, output_size)]
    a = [x // 2 for x in diff]
    r = [i - j for i, j in zip(diff, a)]
    b = [i - j for i, j in zip(sh, r)]
    for i, (x, y) in enumerate(zip(a, b)):
        idx = torch.LongTensor(np.r_[x:y]).to(X.device)
        X = torch.index_select(X, i + 2, idx)
    return X


def unsqueeze_to_target(x: torch.Tensor, target: torch.Tensor, dim=-1):
    cur, tar = len(x.shape), len(target.shape)
    if cur < tar:
        for _ in range(tar - cur):
            x = x.unsqueeze(dim)
    return x


class SequentialWithArgs(torch.nn.Sequential):
    """
    Modified Sequential module. The difference is that the forward takes
    arguments.
    """

    def __init__(self, *args: torch.nn.Module):
        """
        Args:
            modules (torch.nn.Module): module
        """
        super(torch.nn.Sequential).__init__(*args)

    def forward(self, X: torch.Tensor, *args, **kwargs) -> torch.Tensor:
        for module in self:
            X = module(X, *args, **kwargs)
        return X
