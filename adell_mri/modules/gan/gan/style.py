"""
Implements StyleGAN architectures. It also implements ProGAN as most StyleGAN
variants are based on ProGAN.
"""

import torch
from typing import Any
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

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.upsample:
            x = self.upsample_op(x)
        x = self.conv1(x)
        x = self.normalization1(x)
        x = self.activation(x)
        x = self.conv2(x)
        x = self.normalization2(x)
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
    norm_op = torch.nn.InstanceNorm2d
    transpose_conv_op = torch.nn.ConvTranspose2d
    pool_op = torch.nn.MaxPool2d


class ProgressiveGANBlock3d(ProgressiveGANBlock):
    conv_op = torch.nn.Conv3d
    norm_op = torch.nn.InstanceNorm3d
    transpose_conv_op = torch.nn.ConvTranspose3d
    pool_op = torch.nn.MaxPool3d


class ProgressiveGAN(torch.nn.Module):
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

            self.blocks.append(
                ProgressiveGANBlock2d(**block)
                if n_dim == 2
                else ProgressiveGANBlock3d(**block)
            )

            in_channels = depth

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        for block in self.blocks:
            x = block(x)
        return x


class Discriminator(torch.nn.Module):
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

        self.blocks = torch.nn.ModuleList()

        in_channels = input_channels
        for i, depth in enumerate(depths):
            is_last = i == len(depths) - 1
            block = {
                "in_channels": in_channels,
                "out_channels": depth,
                "kernel_sizes": 3,
                "activation": ("leaky_relu", {"negative_slope": 0.2}),
                "downsample": "conv",
                "downsample_factor": 4 if is_last else 2,
            }
            self.blocks.append(
                ProgressiveGANBlock2d(**block)
                if n_dim == 2
                else ProgressiveGANBlock3d(**block)
            )

            in_channels = depth

        self.classification_module = MLP(
            input_dim=in_channels,
            output_dim=output_channels,
            structure=[in_channels // 2, in_channels // 4],
            adn_fn=get_adn_fn(
                1,
                norm_fn="batch",
                act_fn="leaky_relu",
                dropout_param=mlp_dropout,
            ),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        for block in self.blocks:
            x = block(x)
        x = x.flatten(start_dim=2)
        if x.shape[-1] > 1:
            x = x.amax(dim=-1)
        x = self.classification_module(x)
        return x
