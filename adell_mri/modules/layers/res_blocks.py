import numpy as np
import torch

from ...custom_types import ModuleList
from .regularization import GRN
from .utils import crop_to_size, split_int_into_n


class ResidualBlock2d(torch.nn.Module):
    """
    Default residual block in 2 dimensions. If `out_channels`
    is different from `in_channels` then a convolution is applied to
    the skip connection to match the number of `out_channels`.
    """

    def __init__(
        self,
        in_channels: int,
        kernel_size: int,
        inter_channels: int = None,
        out_channels: int = None,
        adn_fn: torch.nn.Module = torch.nn.Identity,
        skip_activation: bool = None,
    ):
        """
        Args:
            in_channels (int): number of input channels.
            kernel_size (int): kernel size.
            inter_channels (int): number of intermediary channels. Defaults
                to None.
            out_channels (int): number of output channels. Defaults to None.
            adn_fn (torch.nn.Module, optional): the activation-dropout-normalization
                module used. Defaults to torch.nn.Identity.
            skip_activation (bool, optional): skips final activation during forward
                pass. Defaults to None (False).
        """
        super().__init__()
        self.in_channels = in_channels
        self.kernel_size = kernel_size
        if inter_channels is not None:
            self.inter_channels = inter_channels
        else:
            self.inter_channels = self.in_channels
        if out_channels is not None:
            self.out_channels = out_channels
        else:
            self.out_channels = self.in_channels
        self.adn_fn = adn_fn
        self.skip_activation = skip_activation

        self.init_layers()

    def init_layers(self):
        if self.inter_channels is not None:
            self.op = torch.nn.Sequential(
                torch.nn.Conv2d(self.in_channels, self.inter_channels, 1),
                self.adn_fn(self.inter_channels),
                torch.nn.Conv2d(
                    self.inter_channels,
                    self.inter_channels,
                    self.kernel_size,
                    padding="same",
                ),
                self.adn_fn(self.inter_channels),
                torch.nn.Conv2d(self.inter_channels, self.in_channels, 1),
            )
        else:
            self.op = torch.nn.Sequential(
                torch.nn.Conv2d(
                    self.in_channels,
                    self.in_channels,
                    self.kernel_size,
                    padding="same",
                ),
                self.adn_fn(self.in_channels),
                torch.nn.Conv2d(
                    self.in_channels,
                    self.in_channels,
                    self.kernel_size,
                    padding="same",
                ),
            )

        # convolve residual connection to match possible difference in
        # output channels
        if self.in_channels != self.out_channels:
            self.final_op = torch.nn.Conv2d(
                self.in_channels, self.out_channels, 1
            )
        else:
            self.final_op = torch.nn.Identity()

        self.adn_op = self.adn_fn(self.out_channels)

    def forward(self, X: torch.Tensor, skip_activation: bool = None):
        out = self.final_op(self.op(X) + X)
        skip_activation = (
            skip_activation
            if skip_activation is not None
            else self.skip_activation
        )
        if skip_activation is not True:
            out = self.adn_op(out)
        return out


class ResidualBlock3d(torch.nn.Module):
    """
    Default residual block in 3 dimensions. If `out_channels`
    is different from `in_channels` then a convolution is applied to
    the skip connection to match the number of `out_channels`.
    """

    def __init__(
        self,
        in_channels: int,
        kernel_size: int,
        inter_channels: int = None,
        out_channels: int = None,
        adn_fn: torch.nn.Module = torch.nn.Identity,
        skip_activation: bool = None,
    ):
        """
        Args:
            in_channels (int): number of input channels.
            kernel_size (int): kernel size.
            inter_channels (int): number of intermediary channels. Defaults
                to None.
            out_channels (int): number of output channels. Defaults to None.
            adn_fn (torch.nn.Module, optional): the activation-dropout-normalization
                module used. Defaults to torch.nn.Identity.
            skip_activation (bool, optional): skips final activation during forward
                pass. Defaults to None (False).
        """
        super().__init__()
        self.in_channels = in_channels
        self.kernel_size = kernel_size
        if inter_channels is not None:
            self.inter_channels = inter_channels
        else:
            self.inter_channels = self.in_channels
        if out_channels is not None:
            self.out_channels = out_channels
        else:
            self.out_channels = self.in_channels
        self.adn_fn = adn_fn
        self.skip_activation = skip_activation

        self.init_layers()

    def init_layers(self):
        if self.inter_channels is not None:
            self.op = torch.nn.Sequential(
                torch.nn.Conv3d(self.in_channels, self.inter_channels, 1),
                self.adn_fn(self.inter_channels),
                torch.nn.Conv3d(
                    self.inter_channels,
                    self.inter_channels,
                    self.kernel_size,
                    padding="same",
                ),
                self.adn_fn(self.inter_channels),
                torch.nn.Conv3d(self.inter_channels, self.in_channels, 1),
            )
        else:
            self.op = torch.nn.Sequential(
                torch.nn.Conv3d(
                    self.in_channels,
                    self.in_channels,
                    self.kernel_size,
                    padding="same",
                ),
                self.adn_fn(self.in_channels),
                torch.nn.Conv3d(
                    self.in_channels,
                    self.in_channels,
                    self.kernel_size,
                    padding="same",
                ),
            )

        # convolve residual connection to match possible difference in
        # output channels
        if self.in_channels != self.out_channels:
            self.final_op = torch.nn.Conv3d(
                self.in_channels, self.out_channels, 1
            )
        else:
            self.final_op = torch.nn.Identity()

        self.adn_op = self.adn_fn(self.out_channels)

    def forward(self, X: torch.Tensor, skip_activation: bool = None):
        out = self.final_op(self.op(X) + X)
        skip_activation = (
            skip_activation
            if skip_activation is not None
            else self.skip_activation
        )
        if skip_activation is not True:
            out = self.adn_op(out)
        return out


class ParallelOperationsAndSum(torch.nn.Module):
    def __init__(
        self, operation_list: ModuleList, crop_to_smallest: bool = False
    ) -> torch.nn.Module:
        """Module that uses a set of other modules on the original input
        and sums the output of this set of other modules as the output.

        Args:
            operation_list (ModuleList): list of PyTorch Modules (i.e.
                [torch.nn.Conv2d(16,32,3),torch.nn.Conv2d(16,32,5)])
            crop_to_smallest (bool, optional): whether the output should be
                cropped to the size of the smallest operation output

        Returns:
            torch.nn.Module: a PyTorch Module
        """
        super().__init__()
        self.operation_list = operation_list
        self.crop_to_smallest = crop_to_smallest

    def forward(self, X: torch.Tensor) -> torch.Tensor:
        """Forward pass for this Module.

        Args:
            X (torch.Tensor): 5D tensor

        Returns:
            torch.Tensor: 5D Tensor
        """
        outputs = []
        for operation in self.operation_list:
            outputs.append(operation(X))
        if self.crop_to_smallest is True:
            sh = []
            for output in outputs:
                sh.append(list(output.shape))
            crop_sizes = np.array(sh).min(axis=0)[2:]
            for i in range(len(outputs)):
                outputs[i] = crop_to_size(outputs[i], crop_sizes)
        output = outputs[0] + outputs[1]
        if len(outputs) > 2:
            for o in outputs[2:]:
                output = output + o
        return output


class ResNeXtBlock2d(torch.nn.Module):
    """
    Default ResNeXt block in 2 dimensions. If `out_channels`
    is different from `in_channels` then a convolution is applied to
    the skip connection to match the number of `out_channels`.
    """

    def __init__(
        self,
        in_channels: int,
        kernel_size: int,
        inter_channels: int = None,
        out_channels: int = None,
        adn_fn: torch.nn.Module = torch.nn.Identity,
        n_splits: int = 16,
        skip_activation: bool = None,
    ):
        """
        Args:
            in_channels (int): number of input channels.
            inter_channels (int): number of intermediary channels. Defaults
                to None.
            out_channels (int): number of output channels. Defaults to None.
            kernel_size (int): kernel size.
            adn_fn (torch.nn.Module, optional): the activation-dropout-normalization
                module used. Defaults to torch.nn.Identity.
            n_splits (int, optional): number of branches in intermediate step
                of the ResNeXt module. Defaults to 32.
            skip_activation (bool, optional): skips final activation during forward
                pass. Defaults to None (False).
        """
        super().__init__()
        self.in_channels = in_channels
        self.kernel_size = kernel_size
        if inter_channels is not None:
            self.inter_channels = inter_channels
        else:
            self.inter_channels = self.in_channels
        if out_channels is not None:
            self.out_channels = out_channels
        else:
            self.out_channels = self.in_channels
        self.adn_fn = adn_fn
        self.n_splits = n_splits

        self.init_layers()

    def init_layers(self):
        if self.inter_channels is None:
            self.inter_channels = self.output_channels
        self.n_channels_splits = split_int_into_n(
            self.inter_channels, n=self.n_splits
        )
        self.ops = torch.nn.ModuleList([])
        for n_channels in self.n_channels_splits:
            op = torch.nn.Sequential(
                torch.nn.Conv2d(self.in_channels, n_channels, 1),
                self.adn_fn(n_channels),
                torch.nn.Conv2d(
                    n_channels, n_channels, self.kernel_size, padding="same"
                ),
                self.adn_fn(n_channels),
                torch.nn.Conv2d(n_channels, self.out_channels, 1),
            )
            self.ops.append(op)

        self.op = ParallelOperationsAndSum(self.ops)

        # convolve residual connection to match possible difference in
        # output channels
        if self.in_channels != self.out_channels:
            self.skip_op = torch.nn.Conv3d(
                self.in_channels, self.out_channels, 1
            )
        else:
            self.skip_op = torch.nn.Identity()

        self.final_op = self.adn_fn(self.out_channels)

    def forward(self, X: torch.Tensor, skip_activation: bool = None):
        skip_activation = (
            skip_activation
            if skip_activation is not None
            else self.skip_activation
        )
        out = self.op(X) + self.skip_op(X)
        if skip_activation is not True:
            out = self.final_op(out)
        return out


class ResNeXtBlock3d(torch.nn.Module):
    """
    Default ResNeXt block in 3 dimensions. If `out_channels`
    is different from `in_channels` then a convolution is applied to
    the skip connection to match the number of `out_channels`.
    """

    def __init__(
        self,
        in_channels: int,
        kernel_size: int,
        inter_channels: int = None,
        out_channels: int = None,
        adn_fn: torch.nn.Module = torch.nn.Identity,
        n_splits: int = 32,
        skip_activation: bool = None,
    ):
        """
        Args:
            in_channels (int): number of input channels.
            inter_channels (int): number of intermediary channels. Defaults
                to None.
            out_channels (int): number of output channels. Defaults to None.
            kernel_size (int): kernel size.
            adn_fn (torch.nn.Module, optional): the activation-dropout-normalization
                module used. Defaults to torch.nn.Identity.
            n_splits (int, optional): number of branches in intermediate step
                of the ResNeXt module. Defaults to 32.
            skip_activation (bool, optional): skips final activation during forward
                pass. Defaults to None (False).
        """
        super().__init__()
        self.in_channels = in_channels
        self.kernel_size = kernel_size
        if inter_channels is not None:
            self.inter_channels = inter_channels
        else:
            self.inter_channels = self.in_channels
        if out_channels is not None:
            self.out_channels = out_channels
        else:
            self.out_channels = self.in_channels
        self.adn_fn = adn_fn
        self.n_splits = n_splits
        self.skip_activation = skip_activation

        self.init_layers()

    def init_layers(self):
        if self.inter_channels is None:
            self.inter_channels = self.output_channels
        self.n_channels_splits = split_int_into_n(
            self.inter_channels, n=self.n_splits
        )
        self.ops = torch.nn.ModuleList([])
        for n_channels in self.n_channels_splits:
            op = torch.nn.Sequential(
                torch.nn.Conv3d(self.in_channels, n_channels, 1),
                self.adn_fn(n_channels),
                torch.nn.Conv3d(
                    n_channels, n_channels, self.kernel_size, padding="same"
                ),
                self.adn_fn(n_channels),
                torch.nn.Conv3d(n_channels, self.out_channels, 1),
            )
            self.ops.append(op)

        self.op = ParallelOperationsAndSum(self.ops)

        # convolve residual connection to match possible difference in
        # output channels
        if self.in_channels != self.out_channels:
            self.skip_op = torch.nn.Conv3d(
                self.in_channels, self.out_channels, 1
            )
        else:
            self.skip_op = torch.nn.Identity()

        self.final_op = self.adn_fn(self.out_channels)

    def forward(self, X: torch.Tensor, skip_activation: bool = None):
        skip_activation = (
            skip_activation
            if skip_activation is not None
            else self.skip_activation
        )
        out = self.op(X) + self.skip_op(X)
        if skip_activation is not True:
            out = self.final_op(out)
        return out


class ConvNeXtBlock2d(torch.nn.Module):
    """Two-dimensional ConvNeXt Block. Adapted from [1].

    [1] https://github.com/facebookresearch/ConvNeXt
    """

    def __init__(
        self,
        in_channels: int,
        kernel_size: int,
        inter_channels: int = None,
        out_channels: int = None,
        adn_fn: torch.nn.Module = torch.nn.Identity,
        layer_scale_init_value: float = 1e-6,
        skip_activation: bool = None,
    ):
        """
        Args:
            in_channels (int): number of input channels.
            kernel_size (int): kernel size.
            inter_channels (int): number of intermediary channels. Defaults
                to None.
            out_channels (int): number of output channels. Defaults to None.
            adn_fn (torch.nn.Module, optional): for compability purposes.
            layer_scale_init_value (float, optional): init value for gamma (
                scales non-residual term). Defaults to 1e-6.
            skip_activation (bool, optional): skips final activation during forward
                pass (redundant; for consistency). Defaults to None (False).
        """
        super().__init__()
        self.in_channels = in_channels
        self.kernel_size = kernel_size
        self.inter_channels = inter_channels
        self.out_channels = out_channels
        self.adn_fn = adn_fn
        self.layer_scale_init_value = layer_scale_init_value
        self.skip_activation = skip_activation

        self.dwconv = torch.nn.Conv2d(
            in_channels,
            in_channels,
            kernel_size=kernel_size,
            padding="same",
            groups=in_channels,
        )
        self.norm = torch.nn.LayerNorm(in_channels, eps=1e-6)
        self.pwconv1 = torch.nn.Linear(in_channels, inter_channels)
        self.act = torch.nn.GELU()
        self.pwconv2 = torch.nn.Linear(inter_channels, in_channels)
        self.gamma = (
            torch.nn.Parameter(
                layer_scale_init_value * torch.ones((in_channels)),
                requires_grad=True,
            )
            if layer_scale_init_value > 0
            else None
        )
        if out_channels != in_channels:
            self.out_layer = torch.nn.Sequential(
                torch.nn.Conv2d(
                    in_channels, out_channels, kernel_size=1, padding="same"
                ),
                torch.nn.GELU(),
            )
        else:
            self.out_layer = None

    def forward(self, x, mask=None):
        input = x
        x = self.dwconv(x)
        x = x.permute(0, 2, 3, 1)  # (N, C, H, W) -> (N, H, W, C)
        x = self.norm(x)
        x = self.pwconv1(x)
        x = self.act(x)
        x = self.pwconv2(x)
        if self.gamma is not None:
            x = self.gamma * x
        x = x.permute(0, 3, 1, 2)  # (N, H, W, C) -> (N, C, H, W)

        x = input + x
        if self.out_layer is not None:
            x = self.out_layer(x)

        return x


class ConvNeXtBlock3d(torch.nn.Module):
    """Three-dimensional ConvNeXt Block. Adapted from [1].

    [1] https://github.com/facebookresearch/ConvNeXt
    """

    def __init__(
        self,
        in_channels: int,
        kernel_size: int,
        inter_channels: int,
        out_channels: int,
        adn_fn: torch.nn.Module = torch.nn.Identity,
        layer_scale_init_value: float = 1e-6,
        skip_activation: bool = None,
    ):
        """
        Args:
            in_channels (int): number of input channels.
            kernel_size (int): kernel size.
            inter_channels (int): number of intermediary channels. Defaults
                to None.
            out_channels (int): number of output channels. Defaults to None.
            adn_fn (torch.nn.Module, optional): used only when the output
                channels are different. Defaults to torch.nn.Identity.
            layer_scale_init_value (float, optional): init value for gamma (
                scales non-residual term). Defaults to 1e-6.
            skip_activation (bool, optional): skips final activation during forward
                pass (redundant; for consistency). Defaults to None (False).
        """
        super().__init__()
        self.in_channels = in_channels
        self.kernel_size = kernel_size
        self.inter_channels = inter_channels
        self.out_channels = out_channels
        self.adn_fn = adn_fn
        self.layer_scale_init_value = layer_scale_init_value
        self.skip_activation = skip_activation

        self.dwconv = torch.nn.Conv3d(
            in_channels,
            in_channels,
            kernel_size=kernel_size,
            padding="same",
            groups=in_channels,
        )
        self.norm = torch.nn.LayerNorm(in_channels, eps=1e-6)
        self.pwconv1 = torch.nn.Linear(in_channels, inter_channels)
        self.act = torch.nn.GELU()
        self.pwconv2 = torch.nn.Linear(inter_channels, in_channels)
        self.gamma = (
            torch.nn.Parameter(
                layer_scale_init_value * torch.ones((in_channels)),
                requires_grad=True,
            )
            if layer_scale_init_value > 0
            else None
        )
        if out_channels != in_channels:
            self.out_layer = torch.nn.Sequential(
                torch.nn.Conv3d(
                    in_channels,
                    out_channels,
                    kernel_size=kernel_size,
                    padding="same",
                ),
                torch.nn.GELU(),
            )
        else:
            self.out_layer = None

    def forward(self, x):
        input = x
        x = self.dwconv(x)
        x = x.permute(0, 2, 3, 4, 1)  # (N, C, H, W) -> (N, H, W, C)
        x = self.norm(x)
        x = self.pwconv1(x)
        x = self.act(x)
        x = self.pwconv2(x)
        if self.gamma is not None:
            x = self.gamma * x
        x = x.permute(0, 4, 1, 2, 3)  # (N, H, W, C) -> (N, C, H, W)

        x = input + x
        if self.out_layer is not None:
            x = self.out_layer(x)
        return x


# TODO: implement SparseConvNeXtBlockVTwo2d and SparseConvNeXtBlockVTwo3d


class ConvNeXtBlockVTwo2d(torch.nn.Module):
    """Two-dimensional ConvNeXtV2 Block. Adapted from [1] and [2].

    [1] https://github.com/facebookresearch/ConvNeXt
    [2] https://arxiv.org/pdf/2301.00808.pdf
    """

    def __init__(
        self,
        in_channels: int,
        kernel_size: int,
        inter_channels: int = None,
        out_channels: int = None,
        adn_fn: torch.nn.Module = torch.nn.Identity,
    ):
        """
        Args:
            in_channels (int): number of input channels.
            kernel_size (int): kernel size.
            inter_channels (int): number of intermediary channels. Defaults
                to None.
            out_channels (int): number of output channels. Defaults to None.
            adn_fn (torch.nn.Module, optional): for compability purposes.
            layer_scale_init_value (float, optional): init value for gamma (
                scales non-residual term). Defaults to 1e-6.
        """
        super().__init__()
        self.in_channels = in_channels
        self.kernel_size = kernel_size
        self.inter_channels = inter_channels
        self.out_channels = out_channels
        self.adn_fn = adn_fn

        self.dwconv = torch.nn.Conv2d(
            in_channels,
            in_channels,
            kernel_size=kernel_size,
            padding="same",
            groups=in_channels,
        )
        self.norm = torch.nn.LayerNorm(in_channels, eps=1e-6)
        self.pwconv1 = torch.nn.Linear(in_channels, inter_channels)
        self.act = torch.nn.GELU()
        self.grn = GRN(n_channels=inter_channels, reduce_dims=(1, 2))
        self.pwconv2 = torch.nn.Linear(inter_channels, in_channels)
        if out_channels != in_channels:
            self.out_layer = torch.nn.Sequential(
                torch.nn.Conv2d(
                    in_channels, out_channels, kernel_size=1, padding="same"
                ),
                torch.nn.GELU(),
            )
        else:
            self.out_layer = None

    def forward(self, x, mask=None):
        input = x
        if mask is not None:
            x = self.dwconv(x * mask) * mask
        else:
            x = self.dwconv(x)
        x = x.permute(0, 2, 3, 1)  # (N, C, H, W) -> (N, H, W, C)
        x = self.norm(x)
        x = self.pwconv1(x)
        x = self.act(x)
        x = self.grn(x)
        x = self.pwconv2(x)
        x = x.permute(0, 3, 1, 2)  # (N, H, W, C) -> (N, C, H, W)

        x = input + x
        if self.out_layer is not None:
            x = self.out_layer(x)

        return x


class ConvNeXtBlockVTwo3d(torch.nn.Module):
    """Three-dimensional ConvNeXtV2 Block. Adapted from [1] and [2].

    [1] https://github.com/facebookresearch/ConvNeXt
    [2] https://arxiv.org/pdf/2301.00808.pdf
    """

    def __init__(
        self,
        in_channels: int,
        kernel_size: int,
        inter_channels: int,
        out_channels: int,
        adn_fn: torch.nn.Module = torch.nn.Identity,
    ):
        """
        Args:
            in_channels (int): number of input channels.
            kernel_size (int): kernel size.
            inter_channels (int): number of intermediary channels. Defaults
                to None.
            out_channels (int): number of output channels. Defaults to None.
            adn_fn (torch.nn.Module, optional): used only when the output
                channels are different. Defaults to torch.nn.Identity.
            layer_scale_init_value (float, optional): init value for gamma (
                scales non-residual term). Defaults to 1e-6.
        """
        super().__init__()
        self.in_channels = in_channels
        self.kernel_size = kernel_size
        self.inter_channels = inter_channels
        self.out_channels = out_channels
        self.adn_fn = adn_fn

        self.dwconv = torch.nn.Conv3d(
            in_channels,
            in_channels,
            kernel_size=kernel_size,
            padding="same",
            groups=in_channels,
        )
        self.norm = torch.nn.LayerNorm(in_channels, eps=1e-6)
        self.pwconv1 = torch.nn.Linear(in_channels, inter_channels)
        self.act = torch.nn.GELU()
        self.grn = GRN(n_channels=inter_channels, reduce_dims=(1, 2, 3))
        self.pwconv2 = torch.nn.Linear(inter_channels, in_channels)
        if out_channels != in_channels:
            self.out_layer = torch.nn.Sequential(
                torch.nn.Conv3d(
                    in_channels,
                    out_channels,
                    kernel_size=kernel_size,
                    padding="same",
                ),
                torch.nn.GELU(),
            )
        else:
            self.out_layer = None

    def forward(self, x, mask=None):
        input = x
        if mask is not None:
            x = self.dwconv(x * mask) * mask
        else:
            x = self.dwconv(x)
        x = x.permute(0, 2, 3, 4, 1)  # (N, C, H, W, D) -> (N, H, W, D, C)
        x = self.norm(x)
        x = self.pwconv1(x)
        x = self.act(x)
        x = self.grn(x)
        x = self.pwconv2(x)
        x = x.permute(0, 4, 1, 2, 3)  # (N, H, W, D, C) -> (N, C, H, W, D)

        x = input + x
        if self.out_layer is not None:
            x = self.out_layer(x)
        return x
