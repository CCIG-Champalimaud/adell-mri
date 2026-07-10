"""
Implements Gaussian process layer.
"""

from typing import Tuple

import torch
from torch.distributions.multivariate_normal import MultivariateNormal

from adell_mri.modules.layers.utils import unsqueeze_to_target


class GaussianProcessLayer(torch.nn.Module):
    """
    Gaussian process layer as implemented in [1]. The model is first trained
    and only `self.weights` is updated. `self.W` and `self.b` are randomly sampled
    parameters for random Fourier features (this is generally used to map the
    input feature space to a lower dimension space, making the GP more
    efficient) and are not updated.

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
        n_outputs: int = None,
        m: float = 0.999,
        ordinal: bool = False,
        n_classes: int = None,
    ):
        """
        Args:
            in_channels (int): input channels.
            n_rff (int): number of random Fourier features.
            n_outputs (int, optional): output dimensionality. Defaults to
                n_rff (backward compatible with old out_channels-only API).
            m (float, optional): momentum for updating inv covariance matrix.
                Defaults to 0.999.
            ordinal (bool, optional): whether this layer is used for ordinal
                classification. Defaults to False.
            n_classes (int, optional): number of classes (required when
                ordinal=True). Defaults to None.
        """
        super().__init__()
        self.in_channels = in_channels
        self.n_rff = n_rff
        self.n_outputs = n_outputs if n_outputs is not None else n_rff
        self.out_channels = self.n_outputs
        self.m = m
        self.ordinal = ordinal
        self.n_classes = n_classes

        self.initialize_params()

    def initialize_params(self):
        i = self.in_channels
        r = self.n_rff
        o = self.n_outputs
        self.scaling_term = torch.sqrt(torch.as_tensor(2.0 / i))
        self.W = torch.nn.Parameter(
            torch.normal(torch.zeros([1, r, i]), torch.ones([1, r, i])).float(),
            requires_grad=False,
        )
        self.b = torch.nn.Parameter(
            torch.rand([1, r]).float() * torch.pi, requires_grad=False
        )
        self.weights = torch.nn.Parameter(
            torch.normal(torch.zeros([o, r]), torch.ones([o, r])).float(),
            requires_grad=True,
        )
        self.inv_conv = torch.nn.Parameter(
            torch.eye(r, r).unsqueeze(0).float(), requires_grad=False
        )

    def update_inv_cov(self, X: torch.Tensor, y: torch.Tensor):
        phi = self.calculate_phi(X)
        phi, phi_t = phi.unsqueeze(-2), phi.unsqueeze(-1)
        K = torch.matmul(phi_t, phi)
        if len(K.shape) > 3:
            K = K.flatten(start_dim=1, end_dim=-3)
            K = K.mean(1)

        if self.ordinal:
            # ordinal: scalar latent with cumulative-link likelihood.
            # essentially treats each probability as an individual binary probability.
            with torch.no_grad():
                f_mean = self.forward(X)
                p_cum = torch.sigmoid(f_mean)
                variance = p_cum * (1 - p_cum)
                variance = variance.unsqueeze(-1).expand(
                    -1, K.shape[-1], K.shape[-1]
                )
            update_term = variance * K
        elif y.dim() == 1:
            # binary
            y_float = y.float().unsqueeze(-1)
            variance = y_float * (1 - y_float)
            variance = variance.unsqueeze(-1).expand(
                -1, K.shape[-1], K.shape[-1]
            )
            update_term = variance * K
        else:
            # multi-class
            y_onehot = y.float()
            y_expanded = y_onehot.unsqueeze(-1)
            y_expanded_t = y_onehot.unsqueeze(-2)
            variance = y_expanded * (
                torch.eye(y_onehot.shape[-1], device=y_onehot.device)
                - y_expanded_t
            )
            update_term = torch.matmul(
                torch.matmul(variance, K), variance.transpose(-2, -1)
            )

        self.inv_conv.data = torch.add(
            self.inv_conv * self.m, (1 - self.m) * update_term.sum(0)
        )

    def get_cov(self):
        try:
            # small jitter for numerical stability
            jitter = 1e-6 * torch.eye(
                self.inv_conv.shape[-1], device=self.inv_conv.device
            )
            inv_conv_stable = self.inv_conv + jitter
            self.cov = torch.linalg.inv(inv_conv_stable)
        except torch.linalg.LinAlgError:
            # pseudo-inverse if matrix is singular
            self.cov = torch.linalg.pinv(self.inv_conv)

    def calculate_phi(self, X: torch.Tensor):
        """
        Calculates phi (low rank matrix used to calculate the kernel matrix)

        Args:
            X (torch.Tensor): input tensor.

        Returns:
            phi
        """
        X = X.swapaxes(1, -1)
        X = X.unsqueeze(-1)
        W = unsqueeze_to_target(self.W, X, 1)
        mm = torch.matmul(-W, X).squeeze(-1)
        return self.scaling_term * torch.cos(mm + self.b)

    def forward(self, X: torch.Tensor):
        """
        Uses phi to calculate the mean (phi * self.weights)

        Args:
            X (torch.Tensor): input tensor.

        Returns:
            mean of the Gaussian process
        """
        phi = self.calculate_phi(X)
        output = phi @ self.weights.T
        if len(output.shape) > 2:
            output = output.swapaxes(1, -1)
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
        mean = phi @ self.weights.T
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
