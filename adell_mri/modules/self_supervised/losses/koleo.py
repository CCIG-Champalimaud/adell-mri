"""
Implementation of the KoLeo loss. Some bits and pieces are based on the KoLeo 
regularizer implemented for the DINOv2 code [1].

[1] https://github.com/facebookresearch/dinov2/blob/e1277af2ba9496fbadf7aec6eba56e8d882d1e35/dinov2/loss/koleo_loss.py
"""

import torch
import torch.nn.functional as F


class KoLeoLoss(torch.nn.Module):
    """
    KoLeo regularizer as implemented in [1]. Formally equivalent to:

        L = - 1/n * sum(log(p)),

    where p is the distance to the nearest neighbour. Optionally, the input
    can be normalised.

    [1] https://arxiv.org/abs/1806.03198
    """

    def __init__(self, epsilon: float = 1e-8, normalize: bool = True):
        """
        Args:
            epsilon (float, optional): small constant to add to logarithm.
                Defaults to 1e-8.
            normalize (bool, optional): whether the input should be
                L2-normalised before computing the KoLeo loss. Defaults to
                True.
        """
        super().__init__()

        self.epsilon = epsilon
        self.normalize = normalize

    def forward(self, X: torch.Tensor):
        # assume that X is (B, F)
        if self.normalize:
            X = F.normalize(X, eps=self.epsilon, p=2, dim=-1)
        n = X.shape[0]
        dists = torch.abs(X[None, :, :] - X[:, None, :])
        dists.view(-1)[:: (n + 1)].fill_(torch.inf)
        return -torch.log(dists.min(1).values + self.epsilon).sum() / n
