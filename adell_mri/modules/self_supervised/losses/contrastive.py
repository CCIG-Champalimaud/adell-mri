from copy import deepcopy

import torch
import torch.nn.functional as F

from adell_mri.modules.self_supervised.losses.functional import cos_dist


class KLDivergence(torch.nn.Module):
    """
    Implementation of the KL divergence method suggested in [1]. Allow for both
    local and global KL divergence calculation.

    [1] https://dl.acm.org/doi/10.1007/978-3-031-34048-2_49
    """

    def __init__(self, mode: str = "global"):
        """
        Args:
            mode (str, optional): whether the input feature maps should be
            reduced to their global average ("global") or simply flattened to
            calculate local differences ("local"). Defaults to "global".
        """
        super().__init__()
        self.mode = mode

        assert mode in [
            "global",
            "local",
        ], f"mode {mode} not supported. available options: 'global', 'local'"

    def average_pooling(self, X: torch.Tensor):
        if len(X.shape) > 2:
            return X.flatten(start_dim=2).mean(-1)
        return X

    def forward(
        self, X_1: torch.Tensor, X_2: torch.Tensor, anchors: torch.Tensor
    ):
        if self.mode == "global":
            X_1 = self.average_pooling(X_1)
            X_2 = self.average_pooling(X_2)
            anchors = self.average_pooling(anchors)
        elif self.mode == "local":
            X_1 = X_1.flatten(start_dim=2)
            X_2 = X_2.flatten(start_dim=2)
            anchors = anchors.flatten(start_dim=2)

        p_1 = F.softmax(F.cosine_similarity(X_1[:, None], anchors), dim=1)
        p_2 = F.softmax(F.cosine_similarity(X_2[None, :], anchors), dim=1)
        kl_div = torch.sum(p_1 * (torch.log(p_1) - torch.log(p_2)))
        return kl_div


class ContrastiveDistanceLoss(torch.nn.Module):
    """
    Implements contrastive losses for self-supervised representation learning.

    It supports both pairwise and triplet losses, as well as euclidean
    and cosine distances. The loss encourages positive pairs to have small
    distances and negative pairs to have large distances in the
    embedding space.

    At initialization, parameters like margin, distance metric, etc. can be
    configured. The forward pass takes in two batches of embeddings X1 and X2,
    and a binary target y indicating whether pairs are from the same class.
    """

    def __init__(
        self,
        dist_p=2,
        random_sample=False,
        margin=1,
        dev="cpu",
        loss_type="pairwise",
        dist_type="euclidean",
    ):
        """
        Args:
            dist_p (int, optional): p-norm for distance. Defaults to 2.
            random_sample (bool, optional): whether a elements are randomly
                sampled. Defaults to False.
            margin (int, optional): margin for triplet loss. Defaults to 1.
            dev (str, optional): device. Defaults to "cpu".
            loss_type (str, optional): type of loss (between "pairwise" and
                "triplet"). Defaults to "pairwise".
            dist_type (str, optional): type of distance (between "euclidean"
                and "cosine"). Defaults to "euclidean".

        Raises:
            Exception: if `loss_options` is not in `["pairwise", "triplet"]`
            Exception: if `dist_options` is not in `["euclidean", "cosine"]`
        """
        super().__init__()
        self.dist_p = dist_p
        self.random_sample = random_sample
        self.margin = margin
        self.dev = dev
        self.loss_type = loss_type
        self.dist_type = dist_type

        self.loss_options = ["pairwise", "triplet"]
        self.dist_options = ["euclidean", "cosine"]
        self.torch_margin = torch.as_tensor(
            [self.margin], dtype=torch.float32, device=self.dev
        )

        if self.loss_type not in self.loss_options:
            raise Exception(
                "Loss `{}` not in `{}`".format(
                    self.loss_type, self.loss_options
                )
            )

        if self.dist_type not in self.dist_options:
            raise Exception(
                "dist_type `{}` not in `{}`".format(
                    self.loss_type, self.dist_options
                )
            )

    def dist(self, x: torch.Tensor, y: torch.Tensor):
        if self.dist_type == "euclidean":
            return torch.cdist(x, y, self.dist_p)
        elif self.dist_type == "cosine":
            return cos_dist(x, y)

    def pairwise_distance(self, X1, X2, is_same):
        X1 = X1.flatten(start_dim=1)
        X2 = X2.flatten(start_dim=1)
        dist = self.dist(X1, X2)
        dist = torch.add(
            is_same * dist,
            (1 - is_same.float())
            * torch.maximum(torch.zeros_like(dist), self.torch_margin - dist),
        )
        if self.random_sample is True:
            # randomly samples one entry for each element
            n = dist.shape[0]
            x_idx = torch.arange(0, n, 1, dtype=torch.int32)
            y_idx = torch.randint(0, n, size=[n])
            dist = dist[x_idx, y_idx]
        else:
            dist = dist.sum(-1) / (dist.shape[-1] - 1)
        return dist

    def triplet_distance(self, X1, X2, is_same):
        X1 = X1.flatten(start_dim=1)
        X2 = X2.flatten(start_dim=1)
        dist = self.dist(X1, X2)
        # retrieve negative examples with the lowest distance to
        # each anchor
        hard_negatives = (
            torch.where(is_same, torch.ones_like(dist) * torch.inf, dist)
            .min(1)
            .values
        )
        # retrieve positive examples with the highest distance to
        # each anchor
        hard_positives = (
            torch.where(
                torch.logical_not(is_same),
                -torch.ones_like(dist) * torch.inf,
                dist,
            )
            .max(1)
            .values
        )
        # calculates loss given both hard negatives and positives
        triplet_loss = torch.maximum(
            torch.zeros_like(hard_negatives),
            self.margin + hard_positives - hard_negatives,
        )
        return triplet_loss

    def forward(
        self, X1: torch.Tensor, X2: torch.Tensor = None, y: torch.Tensor = None
    ):
        if isinstance(X1, list):
            X1, X2 = X1
        if X2 is None:
            X2 = deepcopy(X1)
        if y is None:
            y = torch.ones([X1.shape[0]])
        y1, y2 = y.unsqueeze(0), y.unsqueeze(1)
        is_same = y1 == y2
        if self.loss_type == "pairwise":
            loss = self.pairwise_distance(X1, X2, is_same)
        elif self.loss_type == "triplet":
            loss = self.triplet_distance(X1, X2, is_same)
        return loss.mean()
