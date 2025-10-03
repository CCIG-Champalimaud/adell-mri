"""
Implements StyleGAN architectures. It also implements ProGAN as most StyleGAN
variants are based on ProGAN.
"""

import torch
import torch.nn.functional as F
from typing import Any
from functools import partial
from adell_mri.modules.activations import get_activation
from adell_mri.modules.layers.linear_blocks import MLP
from adell_mri.modules.layers.adn_fn import get_adn_fn


class ProgressiveGANBlock(torch.nn.Module):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_sizes: int | tuple[int, int] = (3, 3),
        activation: (
            str | tuple[str, dict[str, Any]] | torch.nn.Module
        ) = "leaky_relu",
        upsample: str | None = None,
        upsample_factor: int = 2,
        downsample: str | None = None,
        downsample_factor: int = 2,
    ):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_sizes = kernel_sizes
        self.activation = activation
        self.upsample = upsample
        self.upsample_factor = upsample_factor
        self.downsample = downsample
        self.downsample_factor = downsample_factor

        if isinstance(self.kernel_sizes, int):
            self.kernel_sizes = (self.kernel_sizes, self.kernel_sizes)
        if isinstance(self.activation, str):
            self.activation = get_activation(self.activation)
        elif isinstance(self.activation, tuple):
            self.activation = get_activation(
                self.activation[0], **self.activation[1]
            )

        conv1_in_channels = self.in_channels
        if self.upsample is not None:
            if self.upsample == "nearest":
                self.upsample_op = torch.nn.Upsample(
                    scale_factor=upsample_factor, mode="nearest"
                )
            elif self.upsample == "conv":
                self.upsample_op = self.transpose_conv_op(
                    in_channels=self.in_channels,
                    out_channels=self.out_channels,
                    kernel_size=upsample_factor,
                    stride=upsample_factor,
                )
                conv1_in_channels = self.out_channels
            else:
                raise ValueError(f"Unknown upsample mode: {self.upsample}")

        if self.downsample:
            if self.downsample == "nearest":
                self.downsample_op = self.pool_op(
                    scale_factor=downsample_factor, mode="nearest"
                )
            elif self.downsample == "conv":
                self.downsample_op = self.conv_op(
                    in_channels=self.out_channels,
                    out_channels=self.out_channels,
                    kernel_size=downsample_factor,
                    stride=downsample_factor,
                )
            else:
                raise ValueError(f"Unknown downsample mode: {self.downsample}")

        self.conv1 = self.conv_op(
            in_channels=conv1_in_channels,
            out_channels=self.out_channels,
            kernel_size=self.kernel_sizes[0],
            padding=self.kernel_sizes[0] // 2,
            padding_mode="reflect",
        )
        self.normalization1 = self.norm_op(self.out_channels)
        self.conv2 = self.conv_op(
            in_channels=self.out_channels,
            out_channels=self.out_channels,
            kernel_size=self.kernel_sizes[1],
            padding=self.kernel_sizes[0] // 2,
            padding_mode="reflect",
        )
        self.normalization2 = self.norm_op(self.out_channels)

    def forward(
        self, x: torch.Tensor, skip_last_activation: bool = False
    ) -> torch.Tensor:
        if self.upsample:
            x = self.upsample_op(x)
        x = self.conv1(x)
        x = self.normalization1(x)
        x = self.activation(x)
        x = self.conv2(x)
        x = self.normalization2(x)
        if not skip_last_activation:
            x = self.activation(x)
        if self.downsample:
            x = self.downsample_op(x)
        return x

    @property
    def conv_op(self):
        raise NotImplementedError("Subclasses must implement this property.")

    @property
    def norm_op(self):
        raise NotImplementedError("Subclasses must implement this property.")

    @property
    def transpose_conv_op(self):
        raise NotImplementedError("Subclasses must implement this property.")

    @property
    def pool_op(self):
        raise NotImplementedError("Subclasses must implement this property.")


class ProgressiveGANBlock2d(ProgressiveGANBlock):
    conv_op = torch.nn.Conv2d
    norm_op = partial(torch.nn.InstanceNorm2d, affine=True)
    transpose_conv_op = torch.nn.ConvTranspose2d
    pool_op = torch.nn.MaxPool2d


class ProgressiveGANBlock3d(ProgressiveGANBlock):
    conv_op = torch.nn.Conv3d
    norm_op = partial(torch.nn.InstanceNorm3d, affine=True)
    transpose_conv_op = torch.nn.ConvTranspose3d
    pool_op = torch.nn.MaxPool3d


class ProgressiveGenerator(torch.nn.Module):
    def __init__(
        self,
        n_dim: int,
        input_channels: int,
        output_channels: int,
        depths: list[int],
    ):
        super().__init__()
        self.input_channels = input_channels
        self.output_channels = output_channels
        self.depths = depths
        self.n_dim = n_dim

        self.blocks = torch.nn.ModuleList()

        in_channels = input_channels
        for i, depth in enumerate(depths):
            block = {
                "in_channels": in_channels,
                "out_channels": depth,
                "kernel_sizes": 3 if i > 0 else (4, 3),
                "activation": ("leaky_relu", {"negative_slope": 0.2}),
            }
            if i == 0:
                block["upsample"] = "nearest"
                block["upsample_factor"] = 4
            else:
                block["upsample"] = "conv"
                block["upsample_factor"] = 2

            self.blocks.append(self.block(**block))
            in_channels = depth

        self.output_block = self.block(
            in_channels=depths[-1],
            out_channels=output_channels,
            kernel_sizes=3,
            activation=("leaky_relu", {"negative_slope": 0.2}),
        )

    def block(
        self, *args, **kwargs
    ) -> ProgressiveGANBlock2d | ProgressiveGANBlock3d:
        if self.n_dim == 2:
            return ProgressiveGANBlock2d(*args, **kwargs)
        else:
            return ProgressiveGANBlock3d(*args, **kwargs)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        for block in self.blocks:
            x = block(x)
        x = self.output_block(x)
        return x


class StyleGAN(torch.nn.Module):
    """
    Similar to ProgressiveGAN but must have:

    - Style MLP network
    - Generator must have style input
    - Noise addition
    """


class ProgressiveDiscriminator(torch.nn.Module):
    def __init__(
        self,
        n_dim: int,
        input_channels: int,
        output_channels: int,
        depths: list[int],
        mlp_dropout: float = 0.0,
    ):
        super().__init__()

        self.n_dim = n_dim
        self.input_channels = input_channels
        self.output_channels = output_channels
        self.depths = depths

        # we instantiate all blocks to avoid initialising/allocating modules
        # half-way through training
        self.blocks = torch.nn.ModuleList()
        self.input_blocks = torch.nn.ModuleList()

        for i in range(len(depths) - 1):
            is_last = i == len(depths) - 1
            input_block = self.block(
                in_channels=input_channels,
                out_channels=depths[i],
                kernel_sizes=1,
                activation=("leaky_relu", {"negative_slope": 0.2}),
                downsample=None,
            )
            intermediate_block = self.block(
                in_channels=depths[i],
                out_channels=depths[i + 1],
                kernel_sizes=3,
                activation=("leaky_relu", {"negative_slope": 0.2}),
                downsample="conv",
                downsample_factor=4 if is_last else 2,
            )
            self.blocks.append(intermediate_block)
            self.input_blocks.append(input_block)

        self.classification_module = MLP(
            input_dim=depths[-1],
            output_dim=output_channels,
            structure=[depths[-1] // 2, depths[-1] // 4],
            adn_fn=get_adn_fn(
                1,
                norm_fn="batch",
                act_fn="leaky_relu",
                dropout_param=mlp_dropout,
            ),
        )

    def block(
        self, *args, **kwargs
    ) -> ProgressiveGANBlock2d | ProgressiveGANBlock3d:
        if self.n_dim == 2:
            return ProgressiveGANBlock2d(*args, **kwargs)
        else:
            return ProgressiveGANBlock3d(*args, **kwargs)

    def forward(
        self,
        x: torch.Tensor,
        alpha: float = 1.0,
        level: int = 0,
        progressive_level: int | None = None,
    ) -> torch.Tensor:
        if alpha < 0 or alpha > 1:
            raise ValueError("Alpha must be between 0 and 1.")
        if (progressive_level is not None) and (alpha < 1.0):
            if progressive_level + 1 != level:
                raise ValueError(
                    "progressive_level must be one less than level."
                )
            progressive_x = self.input_blocks[progressive_level](x)
            progressive_x = self.blocks[progressive_level](progressive_x)
            x = F.interpolate(x, scale_factor=0.5, mode="nearest")
            x = self.input_blocks[level](x)
            x = alpha * x + (1 - alpha) * progressive_x
            x = self.blocks[level](x)
        else:
            x = self.input_blocks[level](x)
            x = self.blocks[level](x)
        for block in self.blocks[level + 1 :]:
            x = block(x)
        x = x.flatten(start_dim=2)
        if x.shape[-1] > 1:
            x = x.amax(dim=-1)
        x = self.classification_module(x)
        return x
