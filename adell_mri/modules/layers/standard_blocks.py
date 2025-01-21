from typing import List

import torch
import torch.nn.functional as F

from ...custom_types import TensorList
from .batch_ensemble import BatchEnsembleWrapper


class GlobalPooling(torch.nn.Module):
    def __init__(self, mode: str = "max"):
        """Wrapper for average and maximum pooling

        Args:
            mode (str, optional): pooling mode. Can be one of "average" or
            "max". Defaults to "max".
        """
        super().__init__()
        self.mode = mode

        self.get_op()

    def get_op(self):
        if self.mode == "average":
            self.op = torch.mean
        elif self.mode == "max":
            self.op = torch.max
        else:
            raise "mode must be one of [average,max]"

    def forward(self, X):
        if len(X.shape) > 2:
            X = self.op(X.flatten(start_dim=2), -1)
            if self.mode == "max":
                X = X.values
        return X


class DepthWiseSeparableConvolution2d(torch.nn.Module):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int = 3,
        padding: int = 1,
        adn_fn: torch.nn.Module = torch.nn.Identity,
    ):
        """Depthwise separable convolution [1] for 2d inputs. Very lightweight
        version of a standard convolutional operation with a relatively small
        drop in performance.

        [1] https://arxiv.org/abs/1902.00927v2

        Args:
            in_channels (int): number of input channels
            out_channels (int): number of output channels
            kernel_size (int, optional): kernel size. Defaults to 3.
            padding (int, optional): amount of padding. Defaults to 1.
            adn_fn (torch.nn.Module, optional): ADN function applied after
                convolutions. Defaults to torch.nn.Identity.
        """
        super().__init__()
        self.input_channels = in_channels
        self.output_channels = out_channels
        self.kernel_size = kernel_size
        self.padding = padding
        self.adn_fn = adn_fn

        self.init_layers()

    def init_layers(self):
        self.depthwise_op = torch.nn.Conv2d(
            self.input_channels,
            self.input_channels,
            kernel_size=self.kernel_size,
            padding=self.paddign,
            groups=self.input_channels,
        )
        self.pointwise_op = torch.nn.Conv2d(
            self.input_channels, self.output_channels, kernel_size=1
        )
        self.act_op = self.adn_fn(self.output_channels)

    def forward(self, X: torch.Tensor) -> torch.Tensor:
        X = self.depthwise_op(X)
        X = self.pointwise_op(X)
        X = self.act_op(X)
        return X


class DepthWiseSeparableConvolution3d(torch.nn.Module):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int = 3,
        padding: int = 1,
        adn_fn: torch.nn.Module = torch.nn.Identity,
    ):
        """Depthwise separable convolution [1] for 3d inputs. Very lightweight
        version of a standard convolutional operation with a relatively small
        drop in performance.

        [1] https://arxiv.org/abs/1902.00927v2

        Args:
            in_channels (int): number of input channels
            out_channels (int): number of output channels
            kernel_size (int, optional): kernel size. Defaults to 3.
            padding (int, optional): amount of padding. Defaults to 1.
            adn_fn (torch.nn.Module, optional): ADN function applied after
                convolutions. Defaults to torch.nn.Identity.
        """
        super().__init__()
        self.input_channels = in_channels
        self.output_channels = out_channels
        self.kernel_size = kernel_size
        self.padding = padding
        self.adn_fn = adn_fn

        self.init_layers()

    def init_layers(self):
        self.depthwise_op = torch.nn.Conv3d(
            self.input_channels,
            self.input_channels,
            kernel_size=self.kernel_size,
            padding=self.padding,
            groups=self.input_channels,
        )
        self.pointwise_op = torch.nn.Conv3d(
            self.input_channels, self.output_channels, kernel_size=1
        )
        self.act_op = self.adn_fn(self.output_channels)

    def forward(self, X: torch.Tensor) -> torch.Tensor:
        X = self.depthwise_op(X)
        X = self.pointwise_op(X)
        X = self.act_op(X)
        return X


class ConvolutionalBlock2d(torch.nn.Module):
    def __init__(
        self,
        in_channels: List[int],
        out_channels: List[int],
        kernel_size: List[int],
        adn_fn: torch.nn.Module = torch.nn.Identity,
        adn_args: dict = {},
        stride: int = 1,
        padding: str = "valid",
    ):
        """Assembles a set of blocks containing convolutions followed by
        adn_fn operations. Used to quickly build convolutional neural
        networks.

        Args:
            in_channels (List[int]): list of input channels for convolutions.
            out_channels (List[int]): list of output channels for convolutions.
            kernel_size (List[int]): list of kernel sizes.
            adn_fn (torch.nn.Module, optional): module applied after
            convolutions. Defaults to torch.nn.Identity.
            adn_args (dict, optional): args for the module applied after
            convolutions. Defaults to {}.
            stride (int, optional): stride for the convolutions. Defaults to 1.
            padding (str, optional): padding for the convolutions. Defaults to
            "valid".
        """

        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding
        self.adn_fn = adn_fn
        self.adn_args = adn_args

        self.initialize_layers()

    def initialize_layers(self):
        """Initialize the layers for this Module."""
        self.mod_list = torch.nn.ModuleList()
        for i, o, k in zip(self.in_channels, self.out_channels, self.kernel_size):
            op = torch.nn.Sequential(
                torch.nn.Conv2d(i, o, k, stride=self.stride, padding=self.padding),
                self.adn_fn(o, **self.adn_args),
            )
            self.mod_list.append(op)
        self.op = torch.nn.Sequential(*self.mod_list)

    def forward(self, X: torch.Tensor) -> torch.Tensor:
        """Forward pass for this Module.

        Args:
            X (torch.Tensor)

        Returns:
            torch.Tensor
        """
        return self.op(X)


class ConvolutionalBlock3d(torch.nn.Module):
    def __init__(
        self,
        in_channels: List[int],
        out_channels: List[int],
        kernel_size: List[int],
        adn_fn: torch.nn.Module = torch.nn.Identity,
        adn_args: dict = {},
        stride: int = 1,
        padding: str = "valid",
    ):
        """Assembles a set of blocks containing convolutions followed by
        adn_fn operations. Used to quickly build convolutional neural
        networks.

        Args:
            in_channels (List[int]): list of input channels for convolutions.
            out_channels (List[int]): list of output channels for convolutions.
            kernel_size (List[int]): list of kernel sizes.
            adn_fn (torch.nn.Module, optional): module applied after
            convolutions. Defaults to torch.nn.Identity.
            adn_args (dict, optional): args for the module applied after
            convolutions. Defaults to {}.
            stride (int, optional): stride for the convolutions. Defaults to 1.
            padding (str, optional): padding for the convolutions. Defaults to
            "valid".
        """

        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding
        self.adn_fn = adn_fn
        self.adn_args = adn_args

        self.initialize_layers()

    def initialize_layers(self):
        """Initialize the layers for this Module."""
        self.mod_list = torch.nn.ModuleList()
        for i, o, k in zip(self.in_channels, self.out_channels, self.kernel_size):
            op = torch.nn.Sequential(
                torch.nn.Conv3d(i, o, k, stride=self.stride, padding=self.padding),
                self.adn_fn(o, **self.adn_args),
            )
            self.mod_list.append(op)
        self.op = torch.nn.Sequential(*self.mod_list)

    def forward(self, X: torch.Tensor) -> torch.Tensor:
        """Forward pass for this Module.

        Args:
            X (torch.Tensor)

        Returns:
            torch.Tensor
        """
        return self.op(X)


class DenseBlock(torch.nn.Module):
    def __init__(
        self,
        spatial_dim: int,
        structure: List[int],
        kernel_size: int,
        adn_fn: torch.nn.Module = torch.nn.PReLU,
        structure_skip: List[int] = None,
        return_all: bool = False,
    ):
        """Implementation of dense block with the possibility of including
        skip connections for architectures such as U-Net++.

        Args:
            spatial_dim (int): dimensionality of the input (2D or 3D).
            structure (List[int]): structure of the convolutions in the dense
                block.
            kernel_size (int): kernel size for convolutions.
            adn_fn (torch.nn.Module, optional): function to be applied after
                each convolution. Defaults to torch.nn.PReLU.
            structure_skip (List[int], optional): structure of the additional
                skip inputs. Skip inputs are optionally fed into the forward
                method and are appended to the outputs of each layer. For this
                reason, len(structure_skip) == len(structure) - 1. Defaults to
                None (normal dense block).
            return_all (bool, optional): Whether all outputs (intermediate
                convolutions) from this layer should be returned. Defaults to
                False.
        """
        super().__init__()
        self.spatial_dim = spatial_dim
        self.structure = structure
        self.kernel_size = kernel_size
        self.adn_fn = adn_fn
        self.structure_skip = structure_skip
        self.return_all = return_all

        if self.structure_skip is None or len(self.structure_skip) == 0:
            self.structure_skip = [0 for _ in range(len(self.structure) - 1)]
        self.init_layers()

    def init_layers(self):
        if self.spatial_dim == 2:
            self.conv_op = torch.nn.Conv2d
        elif self.spatial_dim == 3:
            self.conv_op = torch.nn.Conv3d
        self.ops = torch.nn.ModuleList([])
        self.upscale_ops = torch.nn.ModuleList([])
        prev_d = self.structure[0]
        d = self.structure[1]
        k = self.kernel_size
        self.ops.append(
            torch.nn.Sequential(
                self.conv_op(prev_d, d, k, padding="same"), self.adn_fn(d)
            )
        )
        for i in range(1, len(self.structure) - 1):
            prev_d = sum(self.structure[: (i + 1)]) + self.structure_skip[i - 1]
            d = self.structure[i + 1]
            self.ops.append(
                torch.nn.Sequential(
                    self.conv_op(prev_d, d, k, padding="same"), self.adn_fn(d)
                )
            )

    def forward(self, X: torch.Tensor, X_skip: TensorList = None):
        """
        Args:
            X (torch.Tensor): input tensor.
            X_skip (TensorList, optional): list of tensors with an identical
                size as X (except for the channel dimension). The length of
                this list should be len(self.structure). More details in
                __init__. Defaults to None.

        Returns:
            torch.Tensor or TensorList
        """
        outputs = [X]
        out = X
        for i in range(len(self.ops)):
            if X_skip is not None and i > 0:
                xs = X_skip[i - 1]
                xs = [F.interpolate(xs, out.shape[2:])]
            else:
                xs = []
            out = torch.cat([out, *outputs[:-1], *xs], 1)
            out = self.ops[i](out)
            outputs.append(out)
        if self.return_all is True:
            return outputs
        else:
            return outputs[-1]


class VGGConvolution3d(torch.nn.Module):
    """
    Implementation of simple vgg-style convolutional blocks.
    """

    def __init__(self, input_channels: int, first_depth: int, batch_ensemble: int = 0):
        """
        Args:
            input_channels (List[int]): list of input channels for convolutions.
            first_depth (int): number of output channels for the first convolution.
            batch_ensemble (int, optional): number of batch ensemble modules.
                Defautls to 0.
        """

        super().__init__()
        self.input_channels = input_channels
        self.first_depth = first_depth
        self.batch_ensemble = batch_ensemble

        self.c1 = torch.nn.Conv3d(input_channels, first_depth, 3, padding=1)
        if self.batch_ensemble > 0:
            self.c1_batch_ensemble = BatchEnsembleWrapper(
                None,
                n=self.batch_ensemble,
                in_channels=input_channels,
                out_channels=first_depth,
            )
        self.act1 = torch.nn.GELU()
        self.n1 = torch.nn.BatchNorm3d(first_depth)
        self.c2 = torch.nn.Conv3d(first_depth, first_depth * 2, 3, padding=1)
        self.act2 = torch.nn.GELU()
        self.n2 = torch.nn.BatchNorm3d(first_depth * 2)
        self.m = torch.nn.MaxPool3d(2, 2)

    def forward(self, x: torch.Tensor, batch_idx: int = None) -> torch.Tensor:
        """
        Args:
            X (torch.Tensor): input tensor.
            batch_idx (int, optional): batch index for the batch ensemble
                operation (performed only if batch_ensemble > 0).
                Defaults to None (random batch index).
        Returns:
            torch.Tensor or TensorList
        """
        if self.batch_ensemble > 0:
            x = self.c1_batch_ensemble(x, batch_idx, self.c1)
        else:
            x = self.c1(x)
        x = self.act1(x)
        x = self.n1(x)
        x = self.n2(self.act2(self.c2(x)))
        return self.m(x)


class VGGDeconvolution3d(torch.nn.Module):
    """
    Implementation of simple vgg-style deconvolutional blocks.
    """

    def __init__(
        self,
        input_channels: int,
        first_depth: int,
        batch_ensemble: int = 0,
        last: bool = False,
        last_channels: int = 1,
    ):
        """
        Args:
            input_channels (List[int]): list of input channels for convolutions.
            first_depth (int): number of output channels for the first convolution.
            batch_ensemble (int, optional): number of batch ensemble modules.
                Defautls to 0.
        """

        super().__init__()
        self.input_channels = input_channels
        self.first_depth = first_depth
        self.batch_ensemble = batch_ensemble

        self.c1 = torch.nn.Conv3d(input_channels, first_depth, 3, padding=1)
        if self.batch_ensemble > 0:
            self.c1_batch_ensemble = BatchEnsembleWrapper(
                None,
                n=self.batch_ensemble,
                in_channels=input_channels,
                out_channels=first_depth,
            )
        self.act1 = torch.nn.GELU()
        self.n1 = torch.nn.BatchNorm3d(first_depth)
        if last:
            self.c2 = torch.nn.Conv3d(first_depth, last_channels, 3, padding=1)
            self.act2 = torch.nn.GELU()
            self.n2 = torch.nn.BatchNorm3d(last_channels)
        else:
            self.c2 = torch.nn.Conv3d(first_depth, first_depth, 3, padding=1)
            self.act2 = torch.nn.GELU()
            self.n2 = torch.nn.BatchNorm3d(first_depth)
        self.up = torch.nn.Upsample(scale_factor=2, mode="nearest")

    def forward(self, x: torch.Tensor, batch_idx: int = None) -> torch.Tensor:
        """
        Args:
            X (torch.Tensor): input tensor.
            batch_idx (int, optional): batch index for the batch ensemble
                operation (performed only if batch_ensemble > 0).
                Defaults to None (random batch index).
        Returns:
            torch.Tensor or TensorList
        """
        if self.batch_ensemble > 0:
            x = self.c1_batch_ensemble(x, batch_idx, self.c1)
        else:
            x = self.c1(x)
        x = self.act1(x)
        x = self.n1(x)
        x = self.n2(self.act2(self.c2(x)))
        return self.up(x)
