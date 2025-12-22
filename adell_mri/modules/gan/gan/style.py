"""
Implements StyleGAN architectures. It also implements ProGAN as most StyleGAN
variants are based on ProGAN.
"""

from functools import partial
from math import prod, sqrt
from typing import Any

import torch
import torch.nn.functional as F
from torch.nn.utils import parametrize

from adell_mri.modules.activations import get_activation
from adell_mri.modules.layers.adn_fn import get_adn_fn
from adell_mri.modules.layers.linear_blocks import MLP
from adell_mri.modules.layers.regularization import LRN


class EqualizedLR(torch.nn.Module):
    """
    Parametrization that scales the weight by 1/sqrt(fan_in) at runtime.

    This uses torch.nn.utils.parametrize, so gradients flow to the underlying
    parameter seamlessly and no in-place assignment to module.weight is needed.
    """

    def __init__(self, scale: float) -> None:
        super().__init__()
        self.register_buffer("scale", torch.tensor(float(scale)))

    def forward(self, w: torch.Tensor) -> torch.Tensor:
        return w * self.scale


def compute_fan_in(module: torch.nn.Module) -> int:
    if isinstance(
        module,
        (
            torch.nn.Conv2d,
            torch.nn.Conv3d,
            torch.nn.ConvTranspose2d,
            torch.nn.ConvTranspose3d,
        ),
    ):
        # weight shape: (out_channels, in_channels, *kernel)
        # fan_in is product of all dims except dim 0
        return int(prod(module.weight.shape[1:]))
    elif isinstance(module, torch.nn.Linear):
        return int(module.in_features)
    else:
        raise ValueError("Unsupported module type for equalized LR")


def apply_equalized_learning_rate(conv: torch.nn.Module):
    torch.nn.init.normal_(conv.weight, 0, 1)
    fan_in = compute_fan_in(conv)
    parametrize.register_parametrization(
        conv, "weight", EqualizedLR(1.0 / sqrt(fan_in))
    )


def attach_minibatch_std(x: torch.Tensor, eps: float = 1e-8) -> torch.Tensor:
    if x.shape[0] > 1:
        mean_over_batch = x.mean(dim=0, keepdim=True)
        var = torch.square(x - mean_over_batch).mean(dim=0, keepdim=True)
        std = torch.sqrt(var + eps).mean()
        std = torch.ones_like(x[:, :1]) * std
    else:
        std = torch.zeros(x.shape[0], 1, *x.shape[2:])
    x = torch.cat([x, std], dim=1)
    return x


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
        minibatch_std: bool = False,
        equalized_learning_rate: bool = False,
        noise_injection: bool = False,
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
        self.minibatch_std = minibatch_std
        self.noise_injection = noise_injection

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
            in_channels=conv1_in_channels + self.minibatch_std,
            out_channels=self.out_channels,
            kernel_size=self.kernel_sizes[0],
            padding="same",
        )
        self.normalization1 = self.norm_op(self.out_channels)
        self.conv2 = self.conv_op(
            in_channels=self.out_channels,
            out_channels=self.out_channels,
            kernel_size=self.kernel_sizes[1],
            padding="same",
        )
        self.normalization2 = self.norm_op(self.out_channels)

        if equalized_learning_rate:
            apply_equalized_learning_rate(self.conv1)
            apply_equalized_learning_rate(self.conv2)

            if self.downsample == "conv":
                apply_equalized_learning_rate(self.downsample_op)

            if self.upsample == "conv":
                apply_equalized_learning_rate(self.upsample_op)

        if self.noise_injection:
            self.noise_weight = torch.nn.Parameter(
                torch.zeros(1, conv1_in_channels)
            )

    def forward(
        self,
        x: torch.Tensor,
        skip_last_activation: bool = False,
        split_minibatch_std: bool = False,
    ) -> torch.Tensor:
        if self.upsample:
            x = self.upsample_op(x)
        if self.noise_injection:
            noise = torch.randn(x.shape[0], 1, *x.shape[2:]).to(x.device)
            self.noise_weight.data = self.noise_weight.data.reshape(
                1, -1, *[1 for _ in x.shape[2:]]
            )
            x = x + self.noise_weight * noise
        if self.minibatch_std:
            if split_minibatch_std:
                half_b = x.shape[0] // 2
                x = torch.cat(
                    [
                        attach_minibatch_std(x[:half_b], 1e-8),
                        attach_minibatch_std(x[half_b:], 1e-8),
                    ],
                    dim=0,
                )
            else:
                x = attach_minibatch_std(x, 1e-8)
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
    norm_op = torch.nn.Identity
    transpose_conv_op = torch.nn.ConvTranspose2d
    pool_op = torch.nn.MaxPool2d


class ProgressiveGANBlock3d(ProgressiveGANBlock):
    conv_op = torch.nn.Conv3d
    norm_op = torch.nn.Identity
    transpose_conv_op = torch.nn.ConvTranspose3d
    pool_op = torch.nn.MaxPool3d


class ProgressiveGeneratorBlock2d(ProgressiveGANBlock):
    conv_op = torch.nn.Conv2d
    norm_op = partial(torch.nn.InstanceNorm2d, affine=True)
    transpose_conv_op = torch.nn.ConvTranspose2d
    pool_op = torch.nn.MaxPool2d


class ProgressiveGeneratorBlock3d(ProgressiveGANBlock):
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
        equalized_learning_rate: bool = False,
        noise_injection: bool = False,
    ):
        super().__init__()
        self.input_channels = input_channels
        self.output_channels = output_channels
        self.depths = depths
        self.n_dim = n_dim
        self.equalized_learning_rate = equalized_learning_rate
        self.noise_injection = noise_injection

        self.n_levels = len(depths) - 1

        self.blocks = torch.nn.ModuleList()
        self.output_blocks = torch.nn.ModuleList()

        in_channels = input_channels
        for i, depth in enumerate(depths):
            block_kwargs = {
                "in_channels": in_channels,
                "out_channels": depth,
                "kernel_sizes": 3 if i > 0 else (4, 3),
                "activation": ("leaky_relu", {"negative_slope": 0.2}),
                "noise_injection": noise_injection if i > 0 else False,
            }
            if i == 0:
                block_kwargs["upsample"] = "nearest"
                block_kwargs["upsample_factor"] = 4
            else:
                block_kwargs["upsample"] = "conv"
                block_kwargs["upsample_factor"] = 2

            self.blocks.append(self.block(**block_kwargs))
            in_channels = depth

            output_block = self.block(
                in_channels=depth,
                out_channels=output_channels,
                kernel_sizes=1,
                activation="tanh",
                upsample=None,
            )
            self.output_blocks.append(output_block)

        self.lrn = LRN(in_channels=None)

    def block(
        self, *args, **kwargs
    ) -> ProgressiveGeneratorBlock2d | ProgressiveGeneratorBlock3d:
        if self.n_dim == 2:
            return ProgressiveGeneratorBlock2d(
                *args,
                **kwargs,
                equalized_learning_rate=self.equalized_learning_rate,
            )
        else:
            return ProgressiveGeneratorBlock3d(
                *args,
                **kwargs,
                equalized_learning_rate=self.equalized_learning_rate,
            )

    def forward(
        self,
        x: torch.Tensor,
        alpha: float = 1.0,
        level: int = 0,
        prog_level: int | None = None,
    ) -> torch.Tensor:
        if alpha < 0 or alpha > 1:
            raise ValueError("Alpha must be between 0 and 1.")
        if level > self.n_levels:
            raise ValueError("Level must be less than the number of levels.")
        pseudo_level = self.n_levels - level
        if (prog_level is not None) and (alpha < 1.0):
            if prog_level + 1 != level:
                raise ValueError("prog_level must be one less than level.")
            pseudo_prog_level = self.n_levels - prog_level
            for block in self.blocks[:pseudo_prog_level]:
                x = block(x)
                x = self.lrn(x)
            progressive_x = self.blocks[pseudo_prog_level](x)
            progressive_x = self.output_blocks[pseudo_prog_level](progressive_x)
            x = F.interpolate(x, scale_factor=2, mode="nearest")
            x = self.output_blocks[pseudo_level](x)
            x = alpha * x + (1 - alpha) * progressive_x
        else:
            for block in self.blocks[: (pseudo_level + 1)]:
                x = block(x)
                x = self.lrn(x)
            x = self.output_blocks[pseudo_level](x)
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
        minibatch_std: bool = False,
        equalized_learning_rate: bool = False,
    ):
        super().__init__()

        self.n_dim = n_dim
        self.input_channels = input_channels
        self.output_channels = output_channels
        self.depths = depths
        self.mlp_dropout = mlp_dropout
        self.minibatch_std = minibatch_std
        self.equalized_learning_rate = equalized_learning_rate

        self.n_levels = len(depths) - 1

        # we instantiate all blocks to avoid initialising/allocating modules
        # half-way through training
        self.blocks = torch.nn.ModuleList()
        self.input_blocks = torch.nn.ModuleList()

        for i in range(len(depths)):
            is_last = i == len(depths) - 1
            out_d = depths[i + 1] if i + 1 < len(depths) else depths[-1]
            input_block = self.block(
                in_channels=input_channels,
                out_channels=depths[i],
                kernel_sizes=1,
                activation=("leaky_relu", {"negative_slope": 0.2}),
                downsample=None,
                minibatch_std=self.minibatch_std and is_last,
            )
            intermediate_block = self.block(
                in_channels=depths[i],
                out_channels=out_d,
                kernel_sizes=3,
                activation=("leaky_relu", {"negative_slope": 0.2}),
                downsample="conv",
                downsample_factor=2,
                minibatch_std=self.minibatch_std and is_last,
            )
            self.blocks.append(intermediate_block)
            self.input_blocks.append(input_block)

        self.classification_module = MLP(
            input_dim=depths[-1],
            output_dim=output_channels,
            structure=[],
            adn_fn=get_adn_fn(
                1,
                norm_fn="identity",
                act_fn="identity",
                dropout_param=mlp_dropout,
            ),
        )

    def block(
        self, *args, **kwargs
    ) -> ProgressiveGANBlock2d | ProgressiveGANBlock3d:
        if self.n_dim == 2:
            return ProgressiveGANBlock2d(
                *args,
                **kwargs,
                equalized_learning_rate=self.equalized_learning_rate,
            )
        else:
            return ProgressiveGANBlock3d(
                *args,
                **kwargs,
                equalized_learning_rate=self.equalized_learning_rate,
            )

    def forward(
        self,
        x: torch.Tensor,
        alpha: float = 1.0,
        level: int = 0,
        prog_level: int | None = None,
        split_minibatch_std: bool = False,
    ) -> torch.Tensor:
        if alpha < 0 or alpha > 1:
            raise ValueError("Alpha must be between 0 and 1.")
        if level > self.n_levels:
            raise ValueError("Level must be less than the number of levels.")
        if (prog_level is not None) and (alpha < 1.0):
            if prog_level + 1 != level:
                raise ValueError("prog_level must be one less than level.")
            progressive_x = self.input_blocks[prog_level](x)
            progressive_x = self.blocks[prog_level](
                progressive_x, split_minibatch_std=split_minibatch_std
            )
            x = F.interpolate(x, scale_factor=0.5, mode="nearest")
            x = self.input_blocks[level](x)
            x = alpha * x + (1 - alpha) * progressive_x
            x = self.blocks[level](x, split_minibatch_std=split_minibatch_std)
        else:
            x = self.input_blocks[level](x)
            x = self.blocks[level](x, split_minibatch_std=split_minibatch_std)
        for block in self.blocks[level + 1 :]:
            x = block(x, split_minibatch_std=split_minibatch_std)
        x = x.flatten(start_dim=2)
        if x.shape[-1] >= 1:
            x = x.amax(dim=-1)
        x = self.classification_module(x)
        return x
