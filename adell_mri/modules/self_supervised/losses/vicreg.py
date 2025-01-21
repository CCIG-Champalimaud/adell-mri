"""
Variance-Invariance-Covariance (VIC) regularisation loss implementation.
"""

from math import sqrt
from typing import Tuple

import torch
import torch.nn.functional as F

from .functional import unravel_index


def off_diagonal(x: torch.Tensor) -> torch.Tensor:
    """
    Extracts the elements outside of the diagonal of a given square tensor.

    Args:
        x (torch.Tensor): square tensor.

    Returns:
        torch.Tensor: tensor containing off-diagonal elements.
    """
    # from https://github.com/facebookresearch/vicreg/blob/a73f567660ae507b0667c68f685945ae6e2f62c3/main_vicreg.py
    n, m = x.shape
    assert n == m
    return x.flatten()[:-1].view(n - 1, n + 1)[:, 1:].flatten()


class VICRegLoss(torch.nn.Module):
    def __init__(
        self,
        min_var: float = 1.0,
        eps: float = 1e-4,
        lam: float = 25.0,
        mu: float = 25.0,
        nu: float = 0.1,
    ):
        """
        Implementation of the VICReg loss from [1].

        [1] https://arxiv.org/abs/2105.04906

        Args:
            min_var (float, optional): minimum variance of the features.
                Defaults to 1..
            eps (float, optional): epsilon term to avoid errors due to floating
                point imprecisions. Defaults to 1e-4.
            lam (float, optional): invariance term.
            mu (float, optional): variance term.
            nu (float, optional): covariance term.
        """
        super().__init__()
        self.min_var = min_var
        self.eps = eps
        self.lam = lam
        self.mu = mu
        self.nu = nu

    def variance_loss(self, X: torch.Tensor) -> torch.Tensor:
        """
        Calculates the VICReg variance loss (a Hinge loss for the variance
        which keeps it above `self.min_var`)

        Args:
            X (torch.Tensor): input tensor

        Returns:
            torch.Tensor: variance loss
        """
        reg_std = torch.sqrt(torch.var(X, 0) + self.eps)
        return F.relu(self.min_var - reg_std).mean()

    def covariance_loss(self, X: torch.Tensor) -> torch.Tensor:
        """
        Calculates the covariance loss for VICReg (minimises the L2 norm of
        the off diagonal elements belonging to the covariance matrix of the
        features).

        Args:
            X (torch.Tensor): input tensor

        Returns:
            torch.Tensor: covariance loss.
        """
        X_mean = X.mean(0)
        X_centred = X - X_mean
        cov = (X_centred.T @ X_centred) / (X.shape[0] - 1)
        norm_cov = off_diagonal(cov) / sqrt(X.shape[1])
        return torch.sum(norm_cov.pow_(2))

    def invariance_loss(self, X1: torch.Tensor, X2: torch.Tensor) -> torch.Tensor:
        """
        Calculates the invariance loss for VICReg (minimises the MSE
        between the features calculated from two views of the same image).

        Args:
            X1 (torch.Tensor): input tensor from view 1
            X2 (torch.Tensor): input tensor from view 2

        Returns:
            torch.Tensor: invariance loss
        """
        n = X1.numel()
        mse = torch.sum((X1 - X2) ** 2 / n)
        return mse

    def vicreg_loss(
        self, X1: torch.Tensor, X2: torch.Tensor, adj: float = 1.0
    ) -> torch.Tensor:
        """
        Wrapper for the three components of the VICReg loss.

        Args:
            X1 (torch.Tensor): input tensor from view 1
            X2 (torch.Tensor): input tensor from view 2
            adj (float, optional): adjustment to the covariance loss (helpful
                for local VICReg losses. Defaults to 1.0.

        Returns:
            var_loss (torch.Tensor) variance loss
            cov_loss (torch.Tensor) covariance loss
            inv_loss (torch.Tensor) invariance loss
        """
        var_loss = torch.add(self.variance_loss(X1) / 2, self.variance_loss(X2) / 2)
        cov_loss = torch.add(
            self.covariance_loss(X1) / adj / 2,
            self.covariance_loss(X2) / adj / 2,
        )
        inv_loss = self.invariance_loss(X1, X2)
        return var_loss, cov_loss, inv_loss

    def flatten_if_necessary(self, x):
        if len(x.shape) > 2:
            return x.flatten(start_dim=2).mean(-1)
        return x

    def forward(
        self, X1: torch.Tensor, X2: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Forward method for VICReg loss.

        Args:
            X1 (torch.Tensor): (B,C,H,W,(D)) tensor corresponding to the first
                transform.
            X2 (torch.Tensor): (B,C,H,W,(D)) tensor corresponding to the second
                transform.

        Returns:
            inv_loss (torch.Tensor) weighted invariance loss
            cov_loss (torch.Tensor) weighted covariance loss
            var_loss (torch.Tensor) weighted variance loss
        """

        flat_max_X1 = self.flatten_if_necessary(X1)
        flat_max_X2 = self.flatten_if_necessary(X2)
        var_loss, cov_loss, inv_loss = self.vicreg_loss(flat_max_X1, flat_max_X2)
        return self.lam * inv_loss, self.mu * var_loss, self.nu * cov_loss


class VICRegLocalLoss(VICRegLoss):
    def __init__(
        self,
        min_var: float = 1.0,
        eps: float = 1e-4,
        lam: float = 25.0,
        mu: float = 25.0,
        nu: float = 0.1,
        gamma: int = 10,
    ):
        """
        Local VICRegL loss from [2]. This is, in essence, a version of
        VICReg which leads to better downstream solutions for segmentation
        tasks and other tasks requiring pixel- or superpixel-level inference.
        Default values are according to the paper.

        [2] https://arxiv.org/pdf/2210.01571v1.pdf

        Args:
            min_var (float, optional): minimum variance of the features.
                Defaults to 1..
            eps (float, optional): epsilon term to avoid errors due to floating
                point imprecisions. Defaults to 1e-4.
            lam (float, optional): invariance term.
            mu (float, optional): variance term.
            nu (float, optional): covariance term.
            gamma (int, optional): the local loss term is calculated only for
                the top-gamma feature matches between input images. Defaults
                to 10.
        """
        super().__init__()
        self.min_var = min_var
        self.eps = eps
        self.lam = lam
        self.mu = mu
        self.nu = nu
        self.gamma = gamma

        self.alpha = 0.9
        self.zeros = None
        self.sparse_coords_1 = None
        self.sparse_coords_2 = None

    def transform_coords(self, coords: torch.Tensor, box: torch.Tensor) -> torch.Tensor:
        """
        Takes a set of coords and addapts them to a new coordinate space
        defined by box (0,0 becomes the top left corner of the bounding box).

        Args:
            coords (torch.Tensor): pixel coordinates (x,y)
            box (torch.Tensor): coordinates for a given bounding box
                (x1,y1,x2,y2)

        Returns:
            torch.Tensor: transformed coordinates
        """
        ndim = box.shape[-1] // 2
        a, b = (
            torch.unsqueeze(box[:, :ndim], 1),
            torch.unsqueeze(box[:, ndim:], 1),
        )
        size = b - a
        return coords.unsqueeze(0) * size + a

    def local_loss(self, X1: torch.Tensor, X2: torch.Tensor, all_dists: torch.Tensor):
        g = self.gamma
        b = X1.shape[0]
        _, idxs = torch.topk(all_dists.flatten(start_dim=1), g, 1)
        idxs = [unravel_index(x, all_dists[0].shape) for x in idxs]
        indexes = torch.cat([torch.ones(g) * i for i in range(b)]).long()
        indexes_1 = torch.cat(
            [self.sparse_coords_1[idxs[i][:, 0]].long() for i in range(b)]
        )
        indexes_1 = tuple(
            [indexes, *[indexes_1[:, i] for i in range(indexes_1.shape[1])]]
        )
        indexes_2 = torch.cat(
            [self.sparse_coords_2[idxs[i][:, 0]].long() for i in range(b)]
        )
        indexes_2 = tuple(
            [indexes, *[indexes_2[:, i] for i in range(indexes_2.shape[1])]]
        )
        features_1 = X1.unsqueeze(-1).swapaxes(1, -1).squeeze(1)[indexes_1]
        features_2 = X2.unsqueeze(-1).swapaxes(1, -1).squeeze(1)[indexes_2]
        vrl = sum([x / g for x in self.vicreg_loss(features_1, features_2, g)])
        return vrl

    def location_local_loss(
        self,
        X1: torch.Tensor,
        X2: torch.Tensor,
        box_X1: torch.Tensor,
        box_X2: torch.Tensor,
    ) -> torch.Tensor:
        """
        Given two views of the same image X1 and X2 and their bounding box
        coordinates in the original image space (box_X1 and box_X2), this loss
        function minimises the distance between nearby pixels in both images.
        It does not calculate it for *all* pixels but only for the
        top-self.gamma pixels.

        Args:
            X1 (torch.Tensor): input tensor from view 1
            X2 (torch.Tensor): input tensor from view 2
            box_X1 (torch.Tensor): box containing X1 view
            box_X2 (torch.Tensor): box containing X2 view

        Returns:
            torch.Tensor: local loss for location
        """
        assert X1.shape[0] == X2.shape[0], "X1 and X2 need to have the same batch size"
        X1.shape[0]
        coords_X1 = self.transform_coords(self.sparse_coords_1, box_X1)
        coords_X2 = self.transform_coords(self.sparse_coords_2, box_X2)
        all_dists = torch.cdist(coords_X1, coords_X2, p=2)
        return self.local_loss(X1, X2, all_dists)

    def feature_local_loss(self, X1: torch.Tensor, X2: torch.Tensor):
        """
        Given two views of the same image X1 and X2, this loss
        function minimises the distance between the top-self.gamma closest
        pixels in feature space.

        Args:
            X1 (torch.Tensor): input tensor from view 1
            X2 (torch.Tensor): input tensor from view 2

        Returns:
            torch.Tensor: local loss for features
        """
        assert X1.shape[0] == X2.shape[0], "X1 and X2 need to have the same batch size"
        flat_X1 = X1.flatten(start_dim=2).swapaxes(1, 2)
        flat_X2 = X2.flatten(start_dim=2).swapaxes(1, 2)
        all_dists = torch.cdist(flat_X1, flat_X2, p=2)
        return self.local_loss(X1, X2, all_dists)

    def get_sparse_coords(self, X):
        return (
            torch.stack(
                [
                    x.flatten()
                    for x in torch.meshgrid(
                        *[torch.arange(0, i) for i in X.shape[2:]],
                        indexing="ij",
                    )
                ],
                axis=1,
            )
            .float()
            .to(X.device)
        )

    def forward(
        self,
        X1: torch.Tensor,
        X2: torch.Tensor,
        box_X1: torch.Tensor,
        box_X2: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Forward method for local VICReg loss.

        Args:
            X1 (torch.Tensor): (B,C,H,W,(D)) tensor corresponding to the first
                transform.
            X2 (torch.Tensor): (B,C,H,W,(D)) tensor corresponding to the second
                transform.
            box_X1 (torch.Tensor): coordinates for X1 in the original image
            box_X2 (torch.Tensor): coordinates for X2 in the original image

        Returns:
            var_loss (torch.Tensor)
            cov_loss (torch.Tensor)
            inv_loss (torch.Tensor)
            local_loss (torch.Tensor)
        """

        flat_max_X1 = X1.flatten(start_dim=2).mean(-1)
        flat_max_X2 = X2.flatten(start_dim=2).mean(-1)

        # these steps calculating sparse coordinates and storing them as a class
        # variable assume that image shape remains the same. if this is not the
        # case, then these variables are redefined

        if self.sparse_coords_1 is None:
            self.sparse_coords_1 = self.get_sparse_coords(X1)
            self.shape_1 = X1.shape
        else:
            if self.shape_1 != X1.shape:
                self.sparse_coords_1 = self.get_sparse_coords(X1)
                self.shape_1 = X1.shape

        if self.sparse_coords_2 is None:
            self.sparse_coords_2 = self.get_sparse_coords(X2)
            self.shape_2 = X2.shape
        else:
            if self.shape_2 != X2.shape:
                self.sparse_coords_2 = self.get_sparse_coords(X2)
                self.shape_2 = X2.shape

        var_loss, cov_loss, inv_loss = self.vicreg_loss(flat_max_X1, flat_max_X2)

        # location and feature local losses are non-symmetric so this
        # symmetrises them
        short_range_local_loss = (
            torch.add(
                self.location_local_loss(X1, X2, box_X1, box_X2) * (1 - self.alpha),
                self.location_local_loss(X2, X1, box_X2, box_X1) * (1 - self.alpha),
            )
            / 2
        )
        long_range_local_loss = (
            torch.add(
                self.feature_local_loss(X1, X2) * (1 - self.alpha),
                self.feature_local_loss(X2, X1) * (1 - self.alpha),
            )
            / 2
        )

        local_loss = short_range_local_loss + long_range_local_loss
        return (
            self.lam * inv_loss * self.alpha,
            self.mu * var_loss * self.alpha,
            self.nu * cov_loss * self.alpha,
            local_loss,
        )
