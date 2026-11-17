"""
Strided adaptation of MONAI's AutoencoderKL.
"""

import math
from collections.abc import Sequence

import torch
from monai.networks.blocks import Convolution, SpatialAttentionBlock, UpSample
from monai.networks.nets.autoencoderkl import AEKLResBlock, AutoencoderKL
from monai.utils import ensure_tuple_rep
from torch import nn


def _resolve_stride(
    stride: int | Sequence[int], spatial_dims: int
) -> tuple[int, ...]:
    """
    Normalizes a stride specification to a tuple with one value per spatial
    dimension.

    Args:
        stride (int | Sequence[int]): stride specification.
        spatial_dims (int): number of spatial dimensions.

    Returns:
        tuple[int, ...]: stride per spatial dimension.

    Raises:
        ValueError: if the stride is not positive or has the wrong length.
    """
    if isinstance(stride, int):
        resolved = (stride,) * spatial_dims
    else:
        resolved = tuple(stride)
        if len(resolved) != spatial_dims:
            raise ValueError(
                f"stride should have {spatial_dims} elements but has "
                f"{len(resolved)}"
            )
    if any(value < 1 for value in resolved):
        raise ValueError("stride values should be greater than or equal to 1")
    return resolved


def _is_identity_stride(stride: Sequence[int]) -> bool:
    """
    Returns True if the stride does not change the spatial shape.
    """
    return all(value == 1 for value in stride)


class StridedDownsample(nn.Module):
    """
    Convolutional downsampling with an arbitrary stride.
    """

    def __init__(
        self,
        spatial_dims: int,
        in_channels: int,
        stride: int | Sequence[int],
    ) -> None:
        super().__init__()
        self.stride = _resolve_stride(stride, spatial_dims)
        self.conv = Convolution(
            spatial_dims=spatial_dims,
            in_channels=in_channels,
            out_channels=in_channels,
            strides=self.stride,
            kernel_size=3,
            padding=1,
            conv_only=True,
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.conv(x)


class StridedUpsample(nn.Module):
    """
    Nearest-neighbour upsampling with an arbitrary stride.
    """

    def __init__(
        self,
        spatial_dims: int,
        in_channels: int,
        stride: int | Sequence[int],
        use_convtranspose: bool = False,
    ) -> None:
        super().__init__()
        self.stride = _resolve_stride(stride, spatial_dims)
        if use_convtranspose:
            self.op = UpSample(
                spatial_dims=spatial_dims,
                mode="deconv",
                in_channels=in_channels,
                out_channels=in_channels,
                scale_factor=self.stride,
            )
        else:
            post_conv = Convolution(
                spatial_dims=spatial_dims,
                in_channels=in_channels,
                out_channels=in_channels,
                strides=1,
                kernel_size=3,
                padding=1,
                conv_only=True,
            )
            self.op = UpSample(
                spatial_dims=spatial_dims,
                mode="nontrainable",
                in_channels=in_channels,
                out_channels=in_channels,
                interp_mode="nearest",
                scale_factor=self.stride,
                post_conv=post_conv,
                align_corners=None,
            )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.op(x)


class StridedEncoder(nn.Module):
    """
    AutoencoderKL encoder with per-block strides.
    """

    def __init__(
        self,
        spatial_dims: int,
        in_channels: int,
        channels: Sequence[int],
        out_channels: int,
        num_res_blocks: Sequence[int],
        norm_num_groups: int,
        norm_eps: float,
        attention_levels: Sequence[bool],
        strides: Sequence[int | Sequence[int]],
        with_nonlocal_attn: bool = False,
        include_fc: bool = True,
        use_combined_linear: bool = False,
        use_flash_attention: bool = False,
    ) -> None:
        super().__init__()
        self.spatial_dims = spatial_dims
        self.in_channels = in_channels
        self.channels = channels
        self.out_channels = out_channels
        self.num_res_blocks = num_res_blocks
        self.norm_num_groups = norm_num_groups
        self.norm_eps = norm_eps
        self.attention_levels = attention_levels
        self.strides = [
            _resolve_stride(stride, spatial_dims) for stride in strides
        ]

        blocks: list[nn.Module] = [
            Convolution(
                spatial_dims=spatial_dims,
                in_channels=in_channels,
                out_channels=channels[0],
                strides=1,
                kernel_size=3,
                padding=1,
                conv_only=True,
            )
        ]

        output_channel = channels[0]
        for i in range(len(channels)):
            input_channel = output_channel
            output_channel = channels[i]

            for _ in range(num_res_blocks[i]):
                blocks.append(
                    AEKLResBlock(
                        spatial_dims=spatial_dims,
                        in_channels=input_channel,
                        norm_num_groups=norm_num_groups,
                        norm_eps=norm_eps,
                        out_channels=output_channel,
                    )
                )
                input_channel = output_channel
                if attention_levels[i]:
                    blocks.append(
                        SpatialAttentionBlock(
                            spatial_dims=spatial_dims,
                            num_channels=input_channel,
                            norm_num_groups=norm_num_groups,
                            norm_eps=norm_eps,
                            include_fc=include_fc,
                            use_combined_linear=use_combined_linear,
                            use_flash_attention=use_flash_attention,
                        )
                    )

            if not _is_identity_stride(self.strides[i]):
                blocks.append(
                    StridedDownsample(
                        spatial_dims=spatial_dims,
                        in_channels=input_channel,
                        stride=self.strides[i],
                    )
                )

        if with_nonlocal_attn:
            blocks.extend(
                [
                    AEKLResBlock(
                        spatial_dims=spatial_dims,
                        in_channels=channels[-1],
                        norm_num_groups=norm_num_groups,
                        norm_eps=norm_eps,
                        out_channels=channels[-1],
                    ),
                    SpatialAttentionBlock(
                        spatial_dims=spatial_dims,
                        num_channels=channels[-1],
                        norm_num_groups=norm_num_groups,
                        norm_eps=norm_eps,
                        include_fc=include_fc,
                        use_combined_linear=use_combined_linear,
                        use_flash_attention=use_flash_attention,
                    ),
                    AEKLResBlock(
                        spatial_dims=spatial_dims,
                        in_channels=channels[-1],
                        norm_num_groups=norm_num_groups,
                        norm_eps=norm_eps,
                        out_channels=channels[-1],
                    ),
                ]
            )

        blocks.append(
            nn.GroupNorm(
                num_groups=norm_num_groups,
                num_channels=channels[-1],
                eps=norm_eps,
                affine=True,
            )
        )
        blocks.append(
            Convolution(
                spatial_dims=spatial_dims,
                in_channels=channels[-1],
                out_channels=out_channels,
                strides=1,
                kernel_size=3,
                padding=1,
                conv_only=True,
            )
        )

        self.blocks = nn.ModuleList(blocks)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        for block in self.blocks:
            x = block(x)
        return x


class StridedDecoder(nn.Module):
    """
    AutoencoderKL decoder with per-block strides.
    """

    def __init__(
        self,
        spatial_dims: int,
        channels: Sequence[int],
        in_channels: int,
        out_channels: int,
        num_res_blocks: Sequence[int],
        norm_num_groups: int,
        norm_eps: float,
        attention_levels: Sequence[bool],
        strides: Sequence[int | Sequence[int]],
        with_nonlocal_attn: bool = False,
        use_convtranspose: bool = False,
        include_fc: bool = True,
        use_combined_linear: bool = False,
        use_flash_attention: bool = False,
    ) -> None:
        super().__init__()
        self.spatial_dims = spatial_dims
        self.channels = channels
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.num_res_blocks = num_res_blocks
        self.norm_num_groups = norm_num_groups
        self.norm_eps = norm_eps
        self.attention_levels = attention_levels
        self.strides = [
            _resolve_stride(stride, spatial_dims)
            for stride in reversed(strides)
        ]

        reversed_channels = list(reversed(channels))
        blocks: list[nn.Module] = [
            Convolution(
                spatial_dims=spatial_dims,
                in_channels=in_channels,
                out_channels=reversed_channels[0],
                strides=1,
                kernel_size=3,
                padding=1,
                conv_only=True,
            )
        ]

        if with_nonlocal_attn:
            blocks.extend(
                [
                    AEKLResBlock(
                        spatial_dims=spatial_dims,
                        in_channels=reversed_channels[0],
                        norm_num_groups=norm_num_groups,
                        norm_eps=norm_eps,
                        out_channels=reversed_channels[0],
                    ),
                    SpatialAttentionBlock(
                        spatial_dims=spatial_dims,
                        num_channels=reversed_channels[0],
                        norm_num_groups=norm_num_groups,
                        norm_eps=norm_eps,
                        include_fc=include_fc,
                        use_combined_linear=use_combined_linear,
                        use_flash_attention=use_flash_attention,
                    ),
                    AEKLResBlock(
                        spatial_dims=spatial_dims,
                        in_channels=reversed_channels[0],
                        norm_num_groups=norm_num_groups,
                        norm_eps=norm_eps,
                        out_channels=reversed_channels[0],
                    ),
                ]
            )

        reversed_attention_levels = list(reversed(attention_levels))
        reversed_num_res_blocks = list(reversed(num_res_blocks))
        block_out_ch = reversed_channels[0]
        for i in range(len(reversed_channels)):
            block_in_ch = block_out_ch
            block_out_ch = reversed_channels[i]

            for _ in range(reversed_num_res_blocks[i]):
                blocks.append(
                    AEKLResBlock(
                        spatial_dims=spatial_dims,
                        in_channels=block_in_ch,
                        norm_num_groups=norm_num_groups,
                        norm_eps=norm_eps,
                        out_channels=block_out_ch,
                    )
                )
                block_in_ch = block_out_ch
                if reversed_attention_levels[i]:
                    blocks.append(
                        SpatialAttentionBlock(
                            spatial_dims=spatial_dims,
                            num_channels=block_in_ch,
                            norm_num_groups=norm_num_groups,
                            norm_eps=norm_eps,
                            include_fc=include_fc,
                            use_combined_linear=use_combined_linear,
                            use_flash_attention=use_flash_attention,
                        )
                    )

            if not _is_identity_stride(self.strides[i]):
                blocks.append(
                    StridedUpsample(
                        spatial_dims=spatial_dims,
                        in_channels=block_in_ch,
                        stride=self.strides[i],
                        use_convtranspose=use_convtranspose,
                    )
                )

        blocks.append(
            nn.GroupNorm(
                num_groups=norm_num_groups,
                num_channels=block_in_ch,
                eps=norm_eps,
                affine=True,
            )
        )
        blocks.append(
            Convolution(
                spatial_dims=spatial_dims,
                in_channels=block_in_ch,
                out_channels=out_channels,
                strides=1,
                kernel_size=3,
                padding=1,
                conv_only=True,
            )
        )

        self.blocks = nn.ModuleList(blocks)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        for block in self.blocks:
            x = block(x)
        return x


class StridedAutoencoderKL(AutoencoderKL):
    """
    AutoencoderKL with per-block encoder/decoder strides.
    """

    def __init__(
        self,
        spatial_dims: int,
        in_channels: int = 1,
        out_channels: int = 1,
        num_res_blocks: Sequence[int] | int = (2, 2, 2, 2),
        num_channels: Sequence[int] = (32, 64, 64, 64),
        attention_levels: Sequence[bool] = (False, False, True, True),
        latent_channels: int = 3,
        norm_num_groups: int = 32,
        norm_eps: float = 1e-6,
        strides: Sequence[int | Sequence[int]] | None = None,
        with_encoder_nonlocal_attn: bool = False,
        with_decoder_nonlocal_attn: bool = False,
        use_checkpoint: bool = False,
        use_convtranspose: bool = False,
        include_fc: bool = True,
        use_combined_linear: bool = False,
        use_flash_attention: bool = False,
    ) -> None:
        if isinstance(num_res_blocks, int):
            num_res_blocks = ensure_tuple_rep(num_res_blocks, len(num_channels))
        if strides is None:
            strides = [2] * len(num_channels)
        elif len(strides) != len(num_channels):
            raise ValueError(
                "`strides` should have the same length as `num_channels`."
            )

        self.spatial_dims = spatial_dims
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.strides = [
            _resolve_stride(stride, spatial_dims) for stride in strides
        ]
        self.block_out_channels = tuple(num_channels)

        super().__init__(
            spatial_dims=spatial_dims,
            in_channels=in_channels,
            out_channels=out_channels,
            num_res_blocks=num_res_blocks,
            channels=num_channels,
            attention_levels=attention_levels,
            latent_channels=latent_channels,
            norm_num_groups=norm_num_groups,
            norm_eps=norm_eps,
            with_encoder_nonlocal_attn=with_encoder_nonlocal_attn,
            with_decoder_nonlocal_attn=with_decoder_nonlocal_attn,
            use_checkpoint=use_checkpoint,
            use_convtranspose=use_convtranspose,
            include_fc=include_fc,
            use_combined_linear=use_combined_linear,
            use_flash_attention=use_flash_attention,
        )

        self.encoder = StridedEncoder(
            spatial_dims=spatial_dims,
            in_channels=in_channels,
            channels=num_channels,
            out_channels=latent_channels,
            num_res_blocks=num_res_blocks,
            norm_num_groups=norm_num_groups,
            norm_eps=norm_eps,
            attention_levels=attention_levels,
            strides=self.strides,
            with_nonlocal_attn=with_encoder_nonlocal_attn,
            include_fc=include_fc,
            use_combined_linear=use_combined_linear,
            use_flash_attention=use_flash_attention,
        )
        self.decoder = StridedDecoder(
            spatial_dims=spatial_dims,
            channels=num_channels,
            in_channels=latent_channels,
            out_channels=out_channels,
            num_res_blocks=num_res_blocks,
            norm_num_groups=norm_num_groups,
            norm_eps=norm_eps,
            attention_levels=attention_levels,
            strides=self.strides,
            with_nonlocal_attn=with_decoder_nonlocal_attn,
            use_convtranspose=use_convtranspose,
            include_fc=include_fc,
            use_combined_linear=use_combined_linear,
            use_flash_attention=use_flash_attention,
        )
        self.latent_channels = latent_channels

    @property
    def latent_downsample_factor(self) -> int | tuple[int, ...]:
        """
        Spatial downsampling factor applied by the encoder.
        """
        factors = tuple(
            math.prod(stride[axis] for stride in self.strides)
            for axis in range(self.encoder.spatial_dims)
        )
        if len(set(factors)) == 1:
            return factors[0]
        return factors
