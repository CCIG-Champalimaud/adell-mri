from typing import Tuple

import torch
import torch.nn.functional as F


class UOut(torch.nn.Module):
    """
    Form of dropout suggested in [1]. Rather than dropping out
    specific channels, each channel X is modified such that
    $X' = X + rX$, where $x \sim U(-\beta,\beta)$. This guarantees
    a much smaller variance shift and allows for a dropout-like
    activation layer to be combined with batch-normalization without
    performance drops (more info on this in [1]). This operation is
    performed on the first dimension after the batch dimension (assumed
    to be the channel dimension).

    [1] https://ieeexplore.ieee.org/document/8953671
    """

    def __init__(self, beta: float = 0.0) -> torch.nn.Module:
        """
        Args:
            beta (float, optional): beta parameter for the uniform
            distribution from which $r$ will be sampled for reference, the
            original authors use a value of 0.1. Defaults to 0.

        Returns:
            torch.nn.Module: a Torch Module
        """
        super().__init__()
        self.beta = beta

    def forward(self, X: torch.Tensor) -> torch.Tensor:
        """Forward pass for this Module.

        Args:
            X (torch.Tensor): Tensor

        Returns:
            torch.Tensor: Tensor
        """
        if self.training is True:
            sh = list(X.shape)
            for i in range(2, len(sh)):
                sh[i] = 1
            r = torch.rand(sh).to(X.device)
            r = r * self.beta * 2 - self.beta
            X = X + X * r
            return X
        else:
            return X


class LayerNorm(torch.nn.Module):
    # from: https://github.com/facebookresearch/VICRegL/blob/main/convnext.py
    r"""LayerNorm that supports two data formats: channels_last (default) or
    channels_first. The ordering of the dimensions in the inputs. channels_last
    corresponds to inputs with shape (batch_size, height, width, channels) while
    channels_first corresponds to inputs with shape (batch_size, channels, height,
    width).
    """

    def __init__(self, normalized_shape, eps=1e-6, data_format="channels_last"):
        super().__init__()
        self.weight = torch.nn.Parameter(torch.ones([1, normalized_shape]))
        self.bias = torch.nn.Parameter(torch.zeros([1, normalized_shape]))
        self.eps = eps
        self.data_format = data_format
        if self.data_format not in ["channels_last", "channels_first"]:
            raise NotImplementedError
        self.normalized_shape = (normalized_shape,)

    def forward(self, x):
        if self.data_format == "channels_last":
            return F.layer_norm(
                x, self.normalized_shape, self.weight, self.bias, self.eps
            )
        elif self.data_format == "channels_first":
            n = len(x.shape)
            u = x.mean(1, keepdim=True)
            s = (x - u).pow(2).mean(1, keepdim=True)
            x = (x - u) / torch.sqrt(s + self.eps)
            sh = [1, self.normalized_shape[0], *[1 for _ in range(2, n)]]
            x = self.weight.reshape(sh) * x + self.bias.reshape(sh)
            return x


class LayerNormChannelsFirst(torch.nn.Module):
    # adapted from: https://github.com/facebookresearch/VICRegL/blob/main/convnext.py
    r"""LayerNorm that supports channels_first."""

    def __init__(self, normalized_shape, eps=1e-6):
        super().__init__()
        self.weight = torch.nn.Parameter(torch.ones(normalized_shape))
        self.bias = torch.nn.Parameter(torch.zeros(normalized_shape))
        self.eps = eps
        self.normalized_shape = (normalized_shape,)

    def forward(self, x):
        u = x.mean(1, keepdim=True)
        s = (x - u).pow(2).mean(1, keepdim=True)
        x = (x - u) / torch.sqrt(s + self.eps)
        if x.ndim == 3:
            x = x.multiply(self.weight[:, None]).add(self.bias[:, None])
        if x.ndim == 4:
            x = x.multiply(self.weight[:, None, None]).add(
                self.bias[:, None, None]
            )
        if x.ndim == 5:
            x = x.multiply(self.weight[:, None, None, None]).add(
                self.bias[:, None, None, None]
            )
        return x


class L2NormalizationLayer(torch.nn.Module):
    """
    L2 Normalization layer.
    """

    def __init__(self, in_channels: int | None = None, dim=1, eps=1e-12):
        """
        Args:
            in_channels (int, optional): number of input channels. Not used.
            dim (int, optional): dimension along which to normalize. Defaults
                to 1.
            eps (float, optional): small value to avoid division by zero.
                Defaults to 1e-12.
        """
        super().__init__()
        self.in_channels = in_channels
        self.dim = dim
        self.eps = eps

    def forward(self, x):
        """
        Args:
            x (torch.Tensor): input tensor.

        Returns:
            torch.Tensor: normalized tensor.
        """
        return F.normalize(x, p=2, dim=self.dim, eps=self.eps)


class GRN(torch.nn.Module):
    """
    Global response normalization suggested in the ConvNeXtV2 paper [1]. For a
    given input X:

        1. The p=2 norm is calculated along the provided dimensions (`gx`)
        2. gx is normalised by `nx = gx/mean(gx)`
        3. The input is normalised as `X_adj = (X*nx) + X`
        4. The input is adjusted with two learnable parameters gamma and beta
            such that `output = X_adj * gamma + beta`

    [1] https://arxiv.org/pdf/2301.00808.pdf
    """

    def __init__(self, n_channels: int, reduce_dims: Tuple[int] = (1, 2)):
        """
        Args:
            n_channels (int): number of input channels.
            reduce_dims (Tuple[int]): performs the global reduction along these
                dimensions.
        """
        super().__init__()
        self.n_channels = n_channels
        self.reduce_dims = reduce_dims

    def init_parameters(self):
        """
        Initialise gamma and beta.
        """
        sh = [1 for _ in range(self.spatial_dimensions + 1)]
        sh.append(self.n_channels)
        self.gamma = torch.nn.Parameter(torch.zeros(sh))
        self.beta = torch.nn.Parameter(torch.zeros(sh))

    def forward(self, X: torch.Tensor) -> torch.Tensor:
        gx = torch.norm(X, p=2, dim=self.dims, keepdim=True)
        nx = gx / (gx.mean(dim=-1, keepdim=True) + 1e-6)
        return self.gamma * (X * nx) + self.beta + X


class ChannelDropout(torch.nn.Module):
    def __init__(self, dropout_prob: float, channel_axis: int = 1):
        """Drops out random channels rather than random cells in the Tensor.

        Args:
            dropout_prob (float): probability of dropout.
            channel_axis (int, optional): channel corresponding to the.
                Defaults to 1.
        """
        super().__init__()
        self.dropout_prob = dropout_prob
        self.channel_axis = channel_axis

    def forward(self, X: torch.Tensor) -> torch.Tensor:
        if self.dropout_prob > 0 and self.training is True:
            sh = X.shape
            n_batches = sh[0]
            n_channels = sh[self.channel_axis]
            dropout = torch.rand([n_batches, n_channels]) > self.dropout_prob
            new_shape = []
            for idx in range(len(sh)):
                if idx == 0:
                    new_shape.append(n_batches)
                elif idx == self.channel_axis:
                    new_shape.append(n_channels)
                else:
                    new_shape.append(1)
            dropout = dropout.reshape(*new_shape)
            dropout = dropout.float().to(X.device)
            X = X * dropout
        return X
