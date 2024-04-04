import numpy as np
import torch
import einops
from .linear_blocks import MultiHeadSelfAttention


class SpatialSqueezeAndExcite2d(torch.nn.Module):
    def __init__(self, input_channels: int):
        """Spatial squeeze and excite layer [1] for 2d inputs. Basically a
        modular attention mechanism.

        [1] https://arxiv.org/abs/1803.02579

        Args:
            input_channels (int): number of input channels.
        """
        super().__init__()
        self.input_channels = input_channels

        self.init_layers()

    def init_layers(self):
        self.op = torch.nn.Sequential(
            torch.nn.Conv2d(self.input_channels, 1, kernel_size=1),
            torch.nn.Sigmoid(),
        )

    def forward(self, X: torch.Tensor) -> torch.Tensor:
        spatial_squeeze = self.op(X)
        X = X * spatial_squeeze
        return X


class SpatialSqueezeAndExcite3d(torch.nn.Module):
    def __init__(self, input_channels: int):
        """Spatial squeeze and excite layer [1] for 3d inputs. Basically a
        modular attention mechanism.

        [1] https://arxiv.org/abs/1803.02579

        Args:
            input_channels (int): number of input channels.
        """
        super().__init__()
        self.input_channels = input_channels

        self.init_layers()

    def init_layers(self):
        self.op = torch.nn.Sequential(
            torch.nn.Conv3d(self.input_channels, 1, kernel_size=1),
            torch.nn.Sigmoid(),
        )

    def forward(self, X: torch.Tensor) -> torch.Tensor:
        spatial_squeeze = self.op(X)
        X = X * spatial_squeeze
        return X


class ChannelSqueezeAndExcite(torch.nn.Module):
    def __init__(self, input_channels: int):
        """Channel squeeze and excite. A self-attention mechanism at the
        channel level.

        Args:
            input_channels (int): number of input channels.
        """
        super().__init__()
        self.input_channels = input_channels

        self.init_layers()

    def init_layers(self):
        n_chan = self.input_channels
        self.op = torch.nn.Sequential(
            torch.nn.Linear(n_chan, n_chan),
            torch.nn.ReLU(),
            torch.nn.Linear(n_chan, n_chan),
            torch.nn.Sigmoid(),
        )

    def forward(self, X: torch.Tensor) -> torch.Tensor:
        n = X.dim()
        channel_average = torch.flatten(X, start_dim=2).mean(-1)
        channel_squeeze = self.op(channel_average)
        channel_squeeze = channel_squeeze.reshape(
            *channel_squeeze.shape, *[1 for _ in range(n - 2)]
        )
        X = X * channel_squeeze
        return X


class ConcurrentSqueezeAndExcite2d(torch.nn.Module):
    def __init__(self, input_channels: int):
        """Concurrent squeeze and excite for 2d inputs. Combines channel and
        spatial squeeze and excite by adding the output of both.

        Args:
            input_channels (int): number of input channels.
        """
        super().__init__()
        self.input_channels = input_channels

        self.init_layers()

    def init_layers(self):
        self.spatial = SpatialSqueezeAndExcite2d(self.input_channels)
        self.channel = ChannelSqueezeAndExcite(self.input_channels)

    def forward(self, X):
        s = self.spatial(X)
        c = self.channel(X)
        output = s + c
        return output


class ConcurrentSqueezeAndExcite3d(torch.nn.Module):
    def __init__(self, input_channels: int):
        """Concurrent squeeze and excite for 3d inputs. Combines channel and
        spatial squeeze and excite by adding the output of both.

        Args:
            input_channels (int): number of input channels.
        """
        super().__init__()
        self.input_channels = input_channels

        self.init_layers()

    def init_layers(self):
        self.spatial = SpatialSqueezeAndExcite3d(self.input_channels)
        self.channel = ChannelSqueezeAndExcite(self.input_channels)

    def forward(self, X):
        s = self.spatial(X)
        c = self.channel(X)
        output = s + c
        return output


class SelfAttentionBlock(torch.nn.Module):
    """
    Simple application of a self-attention operation to an image/volume.
    First flattens the input, reshapes it such that each token corresponds
    to a pixel/voxel and applies self-attention to that operation.
    """

    def __init__(
        self,
        ndim: int,
        input_dim: int,
        attention_dim: int,
        patch_size: tuple[int, int] | tuple[int, int, int] = (16, 16, 8),
    ):
        """
        Args:
            ndim (int): number of input dimensions (2 or 3).
            input_dim (int): input dimension (number of channels).
            attention_dim (int): dimension of the attention operation.
            patch_size (tuple[int, int] | tuple[int, int, int], optional):
                patch size for embedding.
        """
        super().__init__()
        self.ndim = ndim
        self.input_dim = input_dim
        self.attention_dim = attention_dim
        self.patch_size = patch_size
        self.input_dim_att = np.prod(patch_size[:ndim]) * input_dim

        self.attention_op = MultiHeadSelfAttention(
            input_dim=self.input_dim_att,
            attention_dim=attention_dim,
            hidden_dim=attention_dim,
            output_dim=self.input_dim_att,
        )

    def get_kwargs_for_rearrange(self, shape: list[int]):
        if self.ndim == 2:
            return {
                "x": self.patch_size[0],
                "y": self.patch_size[1],
                "c": self.input_dim,
                "h": shape[2] // self.patch_size[0],
                "w": shape[3] // self.patch_size[1],
            }
        if self.ndim == 3:
            return {
                "x": self.patch_size[0],
                "y": self.patch_size[1],
                "z": self.patch_size[2],
                "c": self.input_dim,
                "h": shape[2] // self.patch_size[0],
                "w": shape[3] // self.patch_size[1],
                "d": shape[4] // self.patch_size[2],
            }

    def embed(self, X: torch.Tensor) -> torch.Tensor:
        sh = X.shape
        kwargs = self.get_kwargs_for_rearrange(sh)
        if self.ndim == 2:
            return einops.rearrange(
                X, "n c (h x) (w y) -> n (h w) (x y c)", **kwargs
            )
        if self.ndim == 3:
            return einops.rearrange(
                X, "n c (h x) (w y) (d z) -> n (h w d) (x y z c)", **kwargs
            )

    def unembed(self, X: torch.Tensor, sh: list[int]) -> torch.Tensor:
        kwargs = self.get_kwargs_for_rearrange(sh)
        if self.ndim == 2:
            return einops.rearrange(
                X, "n (h w) (x y c) -> n c (h x) (w y)", **kwargs
            )
        if self.ndim == 3:
            return einops.rearrange(
                X, "n (h w d) (x y z c) -> n c (h x) (w y) (d z)", **kwargs
            )

    def forward(self, X: torch.Tensor) -> torch.Tensor:
        X = self.unembed(self.attention_op(self.embed(X)), X.shape)
        return X
