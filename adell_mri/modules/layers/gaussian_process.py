"""
Implements Gaussian process layer.
"""

import math
from typing import Tuple

import torch
from torch.distributions.multivariate_normal import MultivariateNormal

from adell_mri.modules.layers.utils import unsqueeze_to_target


class GaussianProcessLayer(torch.nn.Module):
    """
    Gaussian process layer as implemented in [1]. The model is first trained
    and only self.weights is updated. `self.W` and `self.b`
    are randomly sampled parameters for random Fourier features (this is generally
    used to map the input feature space to a lower dimension space, making the GP
    more efficient) and are not updated.

    If `normalize_input=True`, the input is passed through a learned layer
    normalization before the random Fourier feature transform, as in the
    original SNGP reference implementation.

    The inverse covariance can be updated at the end of the training cheaply
    by running `self.update_inv_cov` over the several batches that comprise a
    training epoch.

    The covariance can then then be calculated (and stored in `self.cov`) by
    running `self.get_cov`. This enables sampling from the inferred distribution
    using `self.rsample(X,n)`, where X is the input tensor and n is the number
    of samples.

    [1] https://arxiv.org/pdf/2006.10108.pdf
    """

    def __init__(
        self,
        in_channels: int,
        n_rff: int,
        n_classes: int,
        n_outputs: int = None,
        m: float = 0.999,
        ordinal: bool = False,
        normalize_input: bool = True,
    ):
        """
        Args:
            in_channels (int): input channels.
            n_rff (int): number of random Fourier features.
            n_classes (int): number of classes.
            n_outputs (int, optional): output dimensionality. Defaults to
                n_rff (backward compatible with old out_channels-only API).
            m (float, optional): momentum for updating inv covariance matrix.
                Defaults to 0.999.
            ordinal (bool, optional): whether this layer is used for ordinal
                classification. Defaults to False.
            normalize_input (bool, optional): whether to apply layer
                normalization to the input before the random Fourier feature
                transform. Defaults to True.
        """
        super().__init__()
        self.in_channels = in_channels
        self.n_rff = n_rff
        self.n_outputs = n_outputs if n_outputs is not None else n_rff
        self.out_channels = self.n_outputs
        self.m = m
        self.ordinal = ordinal
        self.n_classes = n_classes
        self.normalize_input = normalize_input

        self.initialize_params()

    def initialize_params(self):
        """
        Initialises the parameters for the GaussianProcessLayer.
        """
        i = self.in_channels
        r = self.n_rff
        o = self.n_outputs
        W = torch.normal(
            torch.zeros([1, r, i]), torch.ones([1, r, i]) * math.sqrt(2.0 / i)
        )
        self.register_buffer(
            "scaling_term", torch.sqrt(torch.as_tensor(2.0 / r))
        )
        self.register_buffer("W", W.float())
        self.register_buffer("b", torch.rand([1, r]).float() * (2 * torch.pi))
        self.register_buffer("inv_conv", torch.eye(r, r).unsqueeze(0).float())
        if self.normalize_input:
            self.input_norm = torch.nn.LayerNorm(i)
        else:
            self.input_norm = None
        self.weights = torch.nn.Parameter(torch.empty(o, r))
        torch.nn.init.kaiming_uniform_(self.weights, a=math.sqrt(5))

    @torch.no_grad()
    def update_inv_cov(
        self,
        X: torch.Tensor,
        probabilities: torch.Tensor,
        use_momentum: bool = False,
    ):
        """
        Updates the inverse covariance using the output features from a feature
        extractor (i.e. the input features for this GaussianProcessLayer) and
        the output probabilities.

        Args:
            X (torch.Tensor): output features from a feature extractor.
            probabilities (torch.Tensor): probability vector. Should be
                size (b) if `self.n_classes == 2` and `self.ordinal == False`,
                size (b, c) if `self.n_classes > 2` and `self.ordinal == False`,
                size (b, c-1) if `self.n_classes > 2` and `self.ordinal == True`.
            use_momentum (bool, optional): whether the update should be
                momentum-based. Defaults to False.
        """
        phi = self.calculate_phi(X)
        self.update_inv_cov_from_phi(phi, probabilities, use_momentum)

    @torch.no_grad()
    def update_inv_cov_from_phi(
        self,
        phi: torch.Tensor,
        probabilities: torch.Tensor,
        use_momentum: bool = False,
    ):
        """
        Updates the inverted covariance matrix using phi (as calculated using
        `self.calculate_phi`) and the curvature as calculated using the input
        probabilities.

        Args:
            phi (torch.Tensor): low rank matrix used to calculate the kernel
                matrix.
            probabilities (torch.Tensor): probability vector. Should be
                size (b) if `self.n_classes == 2` and `self.ordinal == False`,
                size (b, c) if `self.n_classes > 2` and `self.ordinal == False`,
                size (b, c-1) if `self.n_classes > 2` and `self.ordinal == True`.
            use_momentum (bool, optional): whether the update should be
                momentum-based. Defaults to False.
        """
        probabilities = probabilities.to(dtype=phi.dtype)
        if self.ordinal:
            curvature = (probabilities * (1 - probabilities)).sum(-1)
        elif self.n_classes == 2:
            curvature = probabilities * (1 - probabilities)
        else:
            curvature = (probabilities * (1 - probabilities)).sum(-1)
        phi = phi.flatten(end_dim=-2)
        curvature = curvature.flatten()
        update_term = torch.einsum("n,ni,nj->ij", curvature, phi, phi)
        update_term = update_term.unsqueeze(0)

        if use_momentum:
            self.inv_conv.mul_(self.m).add_(update_term, alpha=1 - self.m)
        else:
            self.inv_conv.add_(update_term)

    @torch.no_grad()
    def reset_inv_cov(self):
        """
        Resets the inverse covariance (i.e. sets it to the identity matrix).
        """
        self.inv_conv.copy_(
            torch.eye(
                self.n_rff,
                dtype=self.inv_conv.dtype,
                device=self.inv_conv.device,
            ).unsqueeze(0)
        )
        if hasattr(self, "cov"):
            del self.cov

    def get_cov(self):
        """
        Calculates the covariance matrix by inverting the inverted covariance
        matrix. If the computation fails due to the matrix being singular,
        this calculates the pseudo-inverse.
        """
        try:
            # small jitter for numerical stability
            jitter = 1e-6 * torch.eye(
                self.inv_conv.shape[-1],
                dtype=self.inv_conv.dtype,
                device=self.inv_conv.device,
            )
            inv_conv_stable = self.inv_conv + jitter
            self.cov = torch.linalg.inv(inv_conv_stable)
        except torch.linalg.LinAlgError:
            # pseudo-inverse if matrix is singular
            self.cov = torch.linalg.pinv(self.inv_conv)

    def calculate_phi(self, X: torch.Tensor):
        """
        Calculates phi (low rank matrix used to calculate the kernel matrix).

        Args:
            X (torch.Tensor): input tensor.

        Returns:
            phi
        """
        if self.input_norm is not None:
            X = self.input_norm(X.swapaxes(1, -1)).swapaxes(1, -1)
        X = X.swapaxes(1, -1)
        X = X.unsqueeze(-1)
        W = unsqueeze_to_target(self.W, X, 1)
        mm = torch.matmul(-W, X).squeeze(-1)
        return self.scaling_term * torch.cos(mm + self.b)

    def forward_with_phi(
        self, X: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        phi = self.calculate_phi(X)
        output = phi @ self.weights.t()
        if len(output.shape) > 2:
            output = output.swapaxes(1, -1)
        return output, phi

    def forward(self, X: torch.Tensor):
        """
        Uses phi to calculate the mean (phi @ self.weights.t()).

        Args:
            X (torch.Tensor): input tensor.

        Returns:
            mean of the Gaussian process
        """
        output, _ = self.forward_with_phi(X)
        return output

    def get_parameters(
        self, X: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Returns the mean and covariance for a set of input samples.

        Args:
            X (torch.Tensor): input tensor.

        Raises:
            Exception: if self.get_cov() has not been called.

        Returns:
            Tuple[torch.Tensor,torch.Tensor]: mean and covariance for Gaussian
                process.
        """
        if hasattr(self, "cov") is False:
            raise Exception(
                "self.get_cov() must be called before getting parameters"
            )
        phi = self.calculate_phi(X)
        mean = phi @ self.weights.t()
        cov_feature_product = self.cov @ phi.unsqueeze(-1)
        var = (phi.unsqueeze(-2) @ cov_feature_product).squeeze(-1).squeeze(-1)
        cov = torch.diag_embed(var.unsqueeze(-1).expand(-1, self.n_outputs))
        return mean, cov

    def rsample(self, X: torch.Tensor, n_samples: int) -> torch.Tensor:
        """
        Samples from fitted Gaussian process conditioned on input samples X.

        Args:
            X (torch.Tensor): input tensor with shape [b,in_channels].
            n_samples (int): number of samples.

        Returns:
            torch.Tensor: tensor with shape [n_samples,b,out_channels].
        """
        mean, cov = self.get_parameters(X)
        try:
            mvn = MultivariateNormal(mean, cov)
            return mvn.rsample([n_samples])
        except ValueError as e:
            print(
                f"Warning: Covariance matrix not positive semidefinite, using diagonal: {e}"
            )
            diag_cov = torch.diag_embed(torch.diagonal(cov, dim1=-2, dim2=-1))
            mvn = MultivariateNormal(mean, diag_cov)
            return mvn.rsample([n_samples])
