"""
Straightforward re-implementation of diffusion models in the MONAI generative
package but replacing mandatory time embeddings with optional class embeddings.

So this is essentially repackaging what the MONAI team did as a GAN generator.
"""

import importlib
import math
from collections.abc import Callable, Sequence

import torch
import torch.nn.functional as F
from monai.networks.blocks import Convolution, SpatialAttentionBlock
from monai.networks.layers.factories import Pool
from monai.networks.nets.diffusion_model_unet import (
    AttnDownBlock as MONAIAttnDownBlock,
)
from monai.networks.nets.diffusion_model_unet import (
    AttnMidBlock as MONAIAttnMidBlock,
)
from monai.networks.nets.diffusion_model_unet import (
    AttnUpBlock as MONAIAttnUpBlock,
)
from monai.networks.nets.diffusion_model_unet import (
    CrossAttnDownBlock as MONAICrossAttnDownBlock,
)
from monai.networks.nets.diffusion_model_unet import (
    CrossAttnMidBlock as MONAICrossAttnMidBlock,
)
from monai.networks.nets.diffusion_model_unet import (
    CrossAttnUpBlock as MONAICrossAttnUpBlock,
)
from monai.networks.nets.diffusion_model_unet import DiffusionUNetResnetBlock
from monai.networks.nets.diffusion_model_unet import DownBlock as MONAIDownBlock
from monai.networks.nets.diffusion_model_unet import SpatialTransformer
from monai.networks.nets.diffusion_model_unet import UpBlock as MONAIUpBlock
from monai.utils import ensure_tuple_rep

# To install xformers, use pip install xformers==0.0.16rc401
if importlib.util.find_spec("xformers") is not None:
    import xformers
    import xformers.ops

    has_xformers = True
else:
    xformers = None
    has_xformers = False


def zero_module(module: torch.nn.Module) -> torch.nn.Module:
    """
    Zero out the parameters of a module and return it.

    Args:
        module (torch.nn.Module): torch module.

    Returns:
        torch.nn.Module: module all parameters set to 0.
    """
    for p in module.parameters():
        p.detach().zero_()
    return module


def _resolve_stride(
    stride: int | Sequence[int], spatial_dims: int
) -> tuple[int, ...]:
    """
    Normalizes a stride specification to a tuple with one value per spatial
    dimension.

    Args:
        stride (int | Sequence[int]): stride specification. A single int is
            applied to every spatial dimension.
        spatial_dims (int): number of spatial dimensions.

    Raises:
        ValueError: if the stride is not positive or if the tuple length does
            not match ``spatial_dims``.

    Returns:
        tuple[int, ...]: stride per spatial dimension.
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


def _resolve_emb(
    emb: torch.Tensor | None, reference: torch.Tensor, emb_channels: int
) -> torch.Tensor:
    """
    Returns a zero conditioning tensor when no class embedding is provided.
    """
    if emb is not None:
        return emb
    return reference.new_zeros(reference.shape[0], emb_channels)


def _neutralize_unconditional(module: torch.nn.Module) -> None:
    """
    Disables the unused embedding projection of unconditional MONAI resnets.
    """
    for submodule in module.modules():
        if isinstance(submodule, DiffusionUNetResnetBlock):
            zero_module(submodule.time_emb_proj)
            for parameter in submodule.time_emb_proj.parameters():
                parameter.requires_grad_(False)


def get_timestep_embedding(
    timesteps: torch.Tensor, embedding_dim: int, max_period: int = 10000
) -> torch.Tensor:
    """
    Create sinusoidal timestep embeddings following the implementation in [1].

    [1] https://arxiv.org/abs/2006.11239

    Args:
        timesteps (int): a 1-D Tensor of N indices, one per batch element.
        embedding_dim (int): the dimension of the output.
        max_period (int, optional): controls the minimum frequency of the
            embeddings. Defaults to 10000.
    """
    if timesteps.ndim != 1:
        raise ValueError("Timesteps should be a 1d-array")

    half_dim = embedding_dim // 2
    exponent = -math.log(max_period) * torch.arange(
        start=0, end=half_dim, dtype=torch.float32, device=timesteps.device
    )
    freqs = torch.exp(exponent / half_dim)

    args = timesteps[:, None].float() * freqs[None, :]
    embedding = torch.cat([torch.cos(args), torch.sin(args)], dim=-1)

    # zero pad
    if embedding_dim % 2 == 1:
        embedding = torch.nn.functional.pad(embedding, (0, 1, 0, 0))

    return embedding


class Downsample(torch.nn.Module):
    """
    Downsampling layer.
    """

    def __init__(
        self,
        spatial_dims: int,
        num_channels: int,
        use_conv: bool,
        out_channels: int | None = None,
        padding: int = 1,
        stride: int | Sequence[int] = 2,
    ) -> None:
        """
        Args:
            spatial_dims (int): number of spatial dimensions.
            num_channels (int): number of input channels.
            use_conv (bool): if True uses Convolution instead of Pool average
                to perform downsampling. In case that use_conv is False, the
                number of output channels must be the same as the number of
                input channels.
            out_channels (int | None, optional): number of output channels.
                Defaults to None.
            padding (int, optional): controls the amount of implicit
                zero-paddings on both sides for padding number of points for
                each dimension. Defaults to 1.
            stride (int | Sequence[int], optional): downsampling stride. A
                single int is applied to every spatial dimension. Defaults to
                2.
        """
        super().__init__()
        self.num_channels = num_channels
        self.out_channels = out_channels or num_channels
        self.use_conv = use_conv
        self.stride = _resolve_stride(stride, spatial_dims)
        if use_conv:
            self.op = Convolution(
                spatial_dims=spatial_dims,
                in_channels=self.num_channels,
                out_channels=self.out_channels,
                strides=self.stride,
                kernel_size=3,
                padding=padding,
                conv_only=True,
            )
        else:
            if self.num_channels != self.out_channels:
                raise ValueError(
                    "num_channels and out_channels must be equal when use_conv=False"
                )
            self.op = Pool[Pool.AVG, spatial_dims](
                kernel_size=self.stride, stride=self.stride
            )

    def forward(
        self, x: torch.Tensor, emb: torch.Tensor | None = None
    ) -> torch.Tensor:
        del emb
        if x.shape[1] != self.num_channels:
            raise ValueError(
                f"Input number of channels ({x.shape[1]}) is not equal to expected number of channels "
                f"({self.num_channels})"
            )
        return self.op(x)


class Upsample(torch.nn.Module):
    """
    Upsampling layer with an optional convolution.
    """

    def __init__(
        self,
        spatial_dims: int,
        num_channels: int,
        use_conv: bool,
        out_channels: int | None = None,
        padding: int = 1,
        stride: int | Sequence[int] = 2,
    ) -> None:
        """

        Args:
            spatial_dims (int): number of spatial dimensions.
            num_channels (int): number of input channels.
            use_conv (bool): if True uses Convolution instead of Pool average
                to perform downsampling.
            out_channels (int | None, optional): number of output channels.
                Defaults to None.
            padding (int, optional): controls the amount of implicit
                zero-paddings on both sides for padding number of points for
                each dimension. Defaults to 1.
            stride (int | Sequence[int], optional): upsampling stride (scale
                factor). A single int is applied to every spatial dimension.
                Defaults to 2.
        """
        super().__init__()
        self.num_channels = num_channels
        self.out_channels = out_channels or num_channels
        self.use_conv = use_conv
        self.stride = _resolve_stride(stride, spatial_dims)
        if use_conv:
            self.conv = Convolution(
                spatial_dims=spatial_dims,
                in_channels=self.num_channels,
                out_channels=self.out_channels,
                strides=1,
                kernel_size=3,
                padding=padding,
                conv_only=True,
            )
        else:
            self.conv = None

    def forward(
        self, x: torch.Tensor, emb: torch.Tensor | None = None
    ) -> torch.Tensor:
        del emb
        if x.shape[1] != self.num_channels:
            raise ValueError("Input channels should be equal to num_channels")

        # Cast to float32 to as 'upsample_nearest2d_out_frame' op does not support bfloat16
        # https://github.com/pytorch/pytorch/issues/86679
        dtype = x.dtype
        if dtype == torch.bfloat16:
            x = x.to(torch.float32)

        x = F.interpolate(x, scale_factor=self.stride, mode="nearest")

        # If the input is bfloat16, we cast back to bfloat16
        if dtype == torch.bfloat16:
            x = x.to(dtype)

        if self.use_conv:
            x = self.conv(x)
        return x


class ResnetBlock(DiffusionUNetResnetBlock):
    """
    Residual block with optional class conditioning and arbitrary strides.

    This reuses MONAI's residual block and only adds:
    - optional class conditioning;
    - stride-aware up/down sampling for strides other than 2.
    """

    def __init__(
        self,
        spatial_dims: int,
        in_channels: int,
        class_emb_channels: int | None,
        out_channels: int | None = None,
        upsample_stride: int | Sequence[int] = 1,
        downsample_stride: int | Sequence[int] = 1,
        norm_num_groups: int = 32,
        norm_eps: float = 1e-6,
    ) -> None:
        upsample_stride = _resolve_stride(upsample_stride, spatial_dims)
        downsample_stride = _resolve_stride(downsample_stride, spatial_dims)
        if not _is_identity_stride(upsample_stride) and not (
            _is_identity_stride(downsample_stride)
        ):
            raise ValueError(
                "upsample_stride and downsample_stride cannot both be "
                "greater than 1"
            )

        self.class_emb_channels = class_emb_channels
        self.upsample_stride = upsample_stride
        self.downsample_stride = downsample_stride
        emb_channels = (
            class_emb_channels if class_emb_channels is not None else 1
        )
        self._emb_channels = emb_channels

        is_upsample = not _is_identity_stride(upsample_stride)
        is_downsample = not _is_identity_stride(downsample_stride)
        native_upsample = is_upsample and upsample_stride == (2,) * spatial_dims
        native_downsample = (
            is_downsample and downsample_stride == (2,) * spatial_dims
        )

        super().__init__(
            spatial_dims=spatial_dims,
            in_channels=in_channels,
            temb_channels=emb_channels,
            out_channels=out_channels,
            up=native_upsample,
            down=native_downsample,
            norm_num_groups=norm_num_groups,
            norm_eps=norm_eps,
        )

        if is_upsample and not native_upsample:
            self.upsample = Upsample(
                spatial_dims,
                in_channels,
                use_conv=False,
                stride=upsample_stride,
            )
        elif is_downsample and not native_downsample:
            self.downsample = Downsample(
                spatial_dims,
                in_channels,
                use_conv=False,
                stride=downsample_stride,
            )

        if class_emb_channels is None:
            zero_module(self.time_emb_proj)
            for parameter in self.time_emb_proj.parameters():
                parameter.requires_grad_(False)

    def forward(
        self, x: torch.Tensor, emb: torch.Tensor | None
    ) -> torch.Tensor:
        if emb is None:
            emb = x.new_zeros(x.shape[0], self._emb_channels)
        return super().forward(x, emb)


class DownBlock(MONAIDownBlock):
    """
    MONAI down block with class conditioning and arbitrary strides.
    """

    def __init__(
        self,
        spatial_dims: int,
        in_channels: int,
        out_channels: int,
        class_emb_channels: int | None,
        num_res_blocks: int = 1,
        norm_num_groups: int = 32,
        norm_eps: float = 1e-6,
        stride: int | Sequence[int] = 2,
        resblock_updown: bool = False,
        downsample_padding: int = 1,
    ) -> None:
        stride = _resolve_stride(stride, spatial_dims)
        emb_channels = (
            class_emb_channels if class_emb_channels is not None else 1
        )
        self._emb_channels = emb_channels

        super().__init__(
            spatial_dims=spatial_dims,
            in_channels=in_channels,
            out_channels=out_channels,
            temb_channels=emb_channels,
            num_res_blocks=num_res_blocks,
            norm_num_groups=norm_num_groups,
            norm_eps=norm_eps,
            add_downsample=not _is_identity_stride(stride),
            resblock_updown=resblock_updown,
            downsample_padding=downsample_padding,
        )

        if not _is_identity_stride(stride) and stride != (2,) * spatial_dims:
            self.downsampler = Downsample(
                spatial_dims=spatial_dims,
                num_channels=out_channels,
                use_conv=True,
                out_channels=out_channels,
                padding=downsample_padding,
                stride=stride,
            )

        if class_emb_channels is None:
            _neutralize_unconditional(self)

    def forward(
        self,
        hidden_states: torch.Tensor,
        emb: torch.Tensor | None,
        context: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, list[torch.Tensor]]:
        emb = _resolve_emb(emb, hidden_states, self._emb_channels)
        return super().forward(hidden_states, emb, context)


class AttnDownBlock(MONAIAttnDownBlock):
    """
    MONAI self-attention down block with class conditioning and strides.
    """

    def __init__(
        self,
        spatial_dims: int,
        in_channels: int,
        out_channels: int,
        class_emb_channels: int | None,
        num_res_blocks: int = 1,
        norm_num_groups: int = 32,
        norm_eps: float = 1e-6,
        stride: int | Sequence[int] = 2,
        resblock_updown: bool = False,
        downsample_padding: int = 1,
        num_head_channels: int = 1,
        use_flash_attention: bool = False,
    ) -> None:
        stride = _resolve_stride(stride, spatial_dims)
        emb_channels = (
            class_emb_channels if class_emb_channels is not None else 1
        )
        self._emb_channels = emb_channels

        super().__init__(
            spatial_dims=spatial_dims,
            in_channels=in_channels,
            out_channels=out_channels,
            temb_channels=emb_channels,
            num_res_blocks=num_res_blocks,
            norm_num_groups=norm_num_groups,
            norm_eps=norm_eps,
            add_downsample=not _is_identity_stride(stride),
            resblock_updown=resblock_updown,
            downsample_padding=downsample_padding,
            num_head_channels=num_head_channels,
            use_flash_attention=use_flash_attention,
        )

        if not _is_identity_stride(stride) and stride != (2,) * spatial_dims:
            self.downsampler = Downsample(
                spatial_dims=spatial_dims,
                num_channels=out_channels,
                use_conv=True,
                out_channels=out_channels,
                padding=downsample_padding,
                stride=stride,
            )

        if class_emb_channels is None:
            _neutralize_unconditional(self)

    def forward(
        self,
        hidden_states: torch.Tensor,
        emb: torch.Tensor | None,
        context: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, list[torch.Tensor]]:
        emb = _resolve_emb(emb, hidden_states, self._emb_channels)
        return super().forward(hidden_states, emb, context)


class CrossAttnDownBlock(MONAICrossAttnDownBlock):
    """
    MONAI cross-attention down block with class conditioning and strides.
    """

    def __init__(
        self,
        spatial_dims: int,
        in_channels: int,
        out_channels: int,
        class_emb_channels: int | None,
        num_res_blocks: int = 1,
        norm_num_groups: int = 32,
        norm_eps: float = 1e-6,
        stride: int | Sequence[int] = 2,
        resblock_updown: bool = False,
        downsample_padding: int = 1,
        num_head_channels: int = 1,
        transformer_num_layers: int = 1,
        cross_attention_dim: int | None = None,
        upcast_attention: bool = False,
        use_flash_attention: bool = False,
    ) -> None:
        stride = _resolve_stride(stride, spatial_dims)
        emb_channels = (
            class_emb_channels if class_emb_channels is not None else 1
        )
        self._emb_channels = emb_channels

        super().__init__(
            spatial_dims=spatial_dims,
            in_channels=in_channels,
            out_channels=out_channels,
            temb_channels=emb_channels,
            num_res_blocks=num_res_blocks,
            norm_num_groups=norm_num_groups,
            norm_eps=norm_eps,
            add_downsample=not _is_identity_stride(stride),
            resblock_updown=resblock_updown,
            downsample_padding=downsample_padding,
            num_head_channels=num_head_channels,
            transformer_num_layers=transformer_num_layers,
            cross_attention_dim=cross_attention_dim,
            upcast_attention=upcast_attention,
            use_flash_attention=use_flash_attention,
        )

        if not _is_identity_stride(stride) and stride != (2,) * spatial_dims:
            self.downsampler = Downsample(
                spatial_dims=spatial_dims,
                num_channels=out_channels,
                use_conv=True,
                out_channels=out_channels,
                padding=downsample_padding,
                stride=stride,
            )

        if class_emb_channels is None:
            _neutralize_unconditional(self)

    def forward(
        self,
        hidden_states: torch.Tensor,
        emb: torch.Tensor | None,
        context: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, list[torch.Tensor]]:
        emb = _resolve_emb(emb, hidden_states, self._emb_channels)
        return super().forward(hidden_states, emb, context)


class AttnMidBlock(MONAIAttnMidBlock):
    """
    MONAI self-attention mid block with optional class conditioning.
    """

    def __init__(
        self,
        spatial_dims: int,
        in_channels: int,
        class_emb_channels: int | None,
        norm_num_groups: int = 32,
        norm_eps: float = 1e-6,
        num_head_channels: int = 1,
        use_flash_attention: bool = False,
    ) -> None:
        emb_channels = (
            class_emb_channels if class_emb_channels is not None else 1
        )
        self._emb_channels = emb_channels

        super().__init__(
            spatial_dims=spatial_dims,
            in_channels=in_channels,
            temb_channels=emb_channels,
            norm_num_groups=norm_num_groups,
            norm_eps=norm_eps,
            num_head_channels=num_head_channels,
            use_flash_attention=use_flash_attention,
        )

        if class_emb_channels is None:
            _neutralize_unconditional(self)

    def forward(
        self,
        hidden_states: torch.Tensor,
        emb: torch.Tensor | None,
        context: torch.Tensor | None = None,
    ) -> torch.Tensor:
        emb = _resolve_emb(emb, hidden_states, self._emb_channels)
        return super().forward(hidden_states, emb, context)


class CrossAttnMidBlock(MONAICrossAttnMidBlock):
    """
    MONAI cross-attention mid block with optional class conditioning.
    """

    def __init__(
        self,
        spatial_dims: int,
        in_channels: int,
        class_emb_channels: int | None,
        norm_num_groups: int = 32,
        norm_eps: float = 1e-6,
        num_head_channels: int = 1,
        transformer_num_layers: int = 1,
        cross_attention_dim: int | None = None,
        upcast_attention: bool = False,
        use_flash_attention: bool = False,
    ) -> None:
        emb_channels = (
            class_emb_channels if class_emb_channels is not None else 1
        )
        self._emb_channels = emb_channels

        super().__init__(
            spatial_dims=spatial_dims,
            in_channels=in_channels,
            temb_channels=emb_channels,
            norm_num_groups=norm_num_groups,
            norm_eps=norm_eps,
            num_head_channels=num_head_channels,
            transformer_num_layers=transformer_num_layers,
            cross_attention_dim=cross_attention_dim,
            upcast_attention=upcast_attention,
            use_flash_attention=use_flash_attention,
        )

        if class_emb_channels is None:
            _neutralize_unconditional(self)

    def forward(
        self,
        hidden_states: torch.Tensor,
        emb: torch.Tensor | None,
        context: torch.Tensor | None = None,
    ) -> torch.Tensor:
        emb = _resolve_emb(emb, hidden_states, self._emb_channels)
        return super().forward(hidden_states, emb, context)


class UpBlock(MONAIUpBlock):
    """
    MONAI up block with class conditioning and arbitrary strides.
    """

    def __init__(
        self,
        spatial_dims: int,
        in_channels: int,
        prev_output_channel: int,
        out_channels: int,
        class_emb_channels: int | None,
        num_res_blocks: int = 1,
        norm_num_groups: int = 32,
        norm_eps: float = 1e-6,
        stride: int | Sequence[int] = 2,
        resblock_updown: bool = False,
    ) -> None:
        stride = _resolve_stride(stride, spatial_dims)
        emb_channels = (
            class_emb_channels if class_emb_channels is not None else 1
        )
        self._emb_channels = emb_channels

        super().__init__(
            spatial_dims=spatial_dims,
            in_channels=in_channels,
            prev_output_channel=prev_output_channel,
            out_channels=out_channels,
            temb_channels=emb_channels,
            num_res_blocks=num_res_blocks,
            norm_num_groups=norm_num_groups,
            norm_eps=norm_eps,
            add_upsample=not _is_identity_stride(stride),
            resblock_updown=resblock_updown,
        )

        if not _is_identity_stride(stride) and stride != (2,) * spatial_dims:
            self.upsampler = Upsample(
                spatial_dims=spatial_dims,
                num_channels=out_channels,
                use_conv=True,
                out_channels=out_channels,
                stride=stride,
            )

        if class_emb_channels is None:
            _neutralize_unconditional(self)

    def forward(
        self,
        hidden_states: torch.Tensor,
        emb: torch.Tensor | None,
        res_hidden_states_list: list[torch.Tensor] | None = None,
        context: torch.Tensor | None = None,
    ) -> torch.Tensor:
        emb = _resolve_emb(emb, hidden_states, self._emb_channels)
        return super().forward(
            hidden_states=hidden_states,
            res_hidden_states_list=res_hidden_states_list,
            temb=emb,
            context=context,
        )


class AttnUpBlock(MONAIAttnUpBlock):
    """
    MONAI self-attention up block with class conditioning and strides.
    """

    def __init__(
        self,
        spatial_dims: int,
        in_channels: int,
        prev_output_channel: int,
        out_channels: int,
        class_emb_channels: int | None,
        num_res_blocks: int = 1,
        norm_num_groups: int = 32,
        norm_eps: float = 1e-6,
        stride: int | Sequence[int] = 2,
        resblock_updown: bool = False,
        num_head_channels: int = 1,
        use_flash_attention: bool = False,
    ) -> None:
        stride = _resolve_stride(stride, spatial_dims)
        emb_channels = (
            class_emb_channels if class_emb_channels is not None else 1
        )
        self._emb_channels = emb_channels

        super().__init__(
            spatial_dims=spatial_dims,
            in_channels=in_channels,
            prev_output_channel=prev_output_channel,
            out_channels=out_channels,
            temb_channels=emb_channels,
            num_res_blocks=num_res_blocks,
            norm_num_groups=norm_num_groups,
            norm_eps=norm_eps,
            add_upsample=not _is_identity_stride(stride),
            resblock_updown=resblock_updown,
            num_head_channels=num_head_channels,
            use_flash_attention=use_flash_attention,
        )

        if not _is_identity_stride(stride) and stride != (2,) * spatial_dims:
            self.upsampler = Upsample(
                spatial_dims=spatial_dims,
                num_channels=out_channels,
                use_conv=True,
                out_channels=out_channels,
                stride=stride,
            )

        if class_emb_channels is None:
            _neutralize_unconditional(self)

    def forward(
        self,
        hidden_states: torch.Tensor,
        emb: torch.Tensor | None,
        res_hidden_states_list: list[torch.Tensor] | None = None,
        context: torch.Tensor | None = None,
    ) -> torch.Tensor:
        emb = _resolve_emb(emb, hidden_states, self._emb_channels)
        return super().forward(
            hidden_states=hidden_states,
            res_hidden_states_list=res_hidden_states_list,
            temb=emb,
            context=context,
        )


class CrossAttnUpBlock(MONAICrossAttnUpBlock):
    """
    MONAI cross-attention up block with class conditioning and strides.
    """

    def __init__(
        self,
        spatial_dims: int,
        in_channels: int,
        prev_output_channel: int,
        out_channels: int,
        class_emb_channels: int | None,
        num_res_blocks: int = 1,
        norm_num_groups: int = 32,
        norm_eps: float = 1e-6,
        stride: int | Sequence[int] = 2,
        resblock_updown: bool = False,
        num_head_channels: int = 1,
        transformer_num_layers: int = 1,
        cross_attention_dim: int | None = None,
        upcast_attention: bool = False,
        use_flash_attention: bool = False,
    ) -> None:
        stride = _resolve_stride(stride, spatial_dims)
        emb_channels = (
            class_emb_channels if class_emb_channels is not None else 1
        )
        self._emb_channels = emb_channels

        super().__init__(
            spatial_dims=spatial_dims,
            in_channels=in_channels,
            prev_output_channel=prev_output_channel,
            out_channels=out_channels,
            temb_channels=emb_channels,
            num_res_blocks=num_res_blocks,
            norm_num_groups=norm_num_groups,
            norm_eps=norm_eps,
            add_upsample=not _is_identity_stride(stride),
            resblock_updown=resblock_updown,
            num_head_channels=num_head_channels,
            transformer_num_layers=transformer_num_layers,
            cross_attention_dim=cross_attention_dim,
            upcast_attention=upcast_attention,
            use_flash_attention=use_flash_attention,
        )

        if not _is_identity_stride(stride) and stride != (2,) * spatial_dims:
            self.upsampler = Upsample(
                spatial_dims=spatial_dims,
                num_channels=out_channels,
                use_conv=True,
                out_channels=out_channels,
                stride=stride,
            )

        if class_emb_channels is None:
            _neutralize_unconditional(self)

    def forward(
        self,
        hidden_states: torch.Tensor,
        emb: torch.Tensor | None,
        res_hidden_states_list: list[torch.Tensor] | None = None,
        context: torch.Tensor | None = None,
    ) -> torch.Tensor:
        emb = _resolve_emb(emb, hidden_states, self._emb_channels)
        return super().forward(
            hidden_states=hidden_states,
            res_hidden_states_list=res_hidden_states_list,
            temb=emb,
            context=context,
        )


class NoSkipUpBlock(torch.nn.Module):
    """
    Up block without skip connections.

    This is the only up-block implementation that cannot reuse MONAI's block
    directly, because MONAI's up blocks always consume skip tensors.
    """

    def __init__(
        self,
        spatial_dims: int,
        in_channels: int,
        prev_output_channel: int,
        out_channels: int,
        class_emb_channels: int | None,
        num_res_blocks: int = 1,
        norm_num_groups: int = 32,
        norm_eps: float = 1e-6,
        stride: int | Sequence[int] = 2,
        resblock_updown: bool = False,
        with_attn: bool = False,
        with_cross_attn: bool = False,
        num_head_channels: int = 1,
        transformer_num_layers: int = 1,
        cross_attention_dim: int | None = None,
        upcast_attention: bool = False,
        use_flash_attention: bool = False,
    ) -> None:
        super().__init__()
        stride = _resolve_stride(stride, spatial_dims)
        emb_channels = (
            class_emb_channels if class_emb_channels is not None else 1
        )
        self._emb_channels = emb_channels
        self.no_skip_connection = True

        self.resnets = torch.nn.ModuleList()
        self.attentions = torch.nn.ModuleList()
        for i in range(num_res_blocks):
            resnet_in_channels = prev_output_channel if i == 0 else out_channels
            self.resnets.append(
                ResnetBlock(
                    spatial_dims=spatial_dims,
                    in_channels=resnet_in_channels,
                    out_channels=out_channels,
                    class_emb_channels=class_emb_channels,
                    norm_num_groups=norm_num_groups,
                    norm_eps=norm_eps,
                )
            )
            if with_attn:
                self.attentions.append(
                    SpatialAttentionBlock(
                        spatial_dims=spatial_dims,
                        num_channels=out_channels,
                        num_head_channels=num_head_channels,
                        norm_num_groups=norm_num_groups,
                        norm_eps=norm_eps,
                        use_flash_attention=use_flash_attention,
                    )
                )
            elif with_cross_attn:
                self.attentions.append(
                    SpatialTransformer(
                        spatial_dims=spatial_dims,
                        in_channels=out_channels,
                        num_attention_heads=out_channels // num_head_channels,
                        num_head_channels=num_head_channels,
                        num_layers=transformer_num_layers,
                        norm_num_groups=norm_num_groups,
                        norm_eps=norm_eps,
                        cross_attention_dim=cross_attention_dim,
                        upcast_attention=upcast_attention,
                        use_flash_attention=use_flash_attention,
                    )
                )

        self.upsampler = None
        if not _is_identity_stride(stride):
            if resblock_updown:
                self.upsampler = ResnetBlock(
                    spatial_dims=spatial_dims,
                    in_channels=out_channels,
                    out_channels=out_channels,
                    class_emb_channels=class_emb_channels,
                    norm_num_groups=norm_num_groups,
                    norm_eps=norm_eps,
                    upsample_stride=stride,
                )
            else:
                self.upsampler = Upsample(
                    spatial_dims=spatial_dims,
                    num_channels=out_channels,
                    use_conv=True,
                    out_channels=out_channels,
                    stride=stride,
                )

    def forward(
        self,
        hidden_states: torch.Tensor,
        emb: torch.Tensor | None,
        res_hidden_states_list: list[torch.Tensor] | None = None,
        context: torch.Tensor | None = None,
    ) -> torch.Tensor:
        del res_hidden_states_list
        emb = _resolve_emb(emb, hidden_states, self._emb_channels)
        for index, resnet in enumerate(self.resnets):
            hidden_states = resnet(hidden_states, emb)
            if len(self.attentions) > 0:
                attention = self.attentions[index]
                if isinstance(attention, SpatialTransformer):
                    hidden_states = attention(
                        hidden_states, context=context
                    ).contiguous()
                else:
                    hidden_states = attention(hidden_states).contiguous()

        if self.upsampler is not None:
            if isinstance(self.upsampler, ResnetBlock):
                hidden_states = self.upsampler(hidden_states, emb)
            else:
                hidden_states = self.upsampler(hidden_states, emb)

        return hidden_states


def get_down_block(
    spatial_dims: int,
    in_channels: int,
    out_channels: int,
    class_emb_channels: int,
    num_res_blocks: int,
    norm_num_groups: int,
    norm_eps: float,
    stride: int | Sequence[int],
    resblock_updown: bool,
    with_attn: bool,
    with_cross_attn: bool,
    num_head_channels: int,
    transformer_num_layers: int,
    cross_attention_dim: int | None,
    upcast_attention: bool = False,
    use_flash_attention: bool = False,
) -> torch.nn.Module:
    """
    Args:
        spatial_dims (int): The number of spatial dimensions.
        in_channels (int): number of input channels.
        out_channels (int): number of output channels.
        class_emb_channels (int): number of class embedding channels.
        num_res_blocks (int, optional): number of residual blocks. Defaults
            to 1.
        norm_num_groups (int, optional): number of groups for the group
            normalization. Defaults to 32.
        norm_eps (float, optional): epsilon for the group normalization.
            Defaults to 1e-6.
        stride (int | Sequence[int]): downsampling stride. A value of 1
            disables downsampling.
        resblock_updown (bool, optional): if True use residual blocks for
            downsampling. Defaults to False.
        with_attn (bool): if True adds attention.
        with_cross_attn (bool): if True adds cross-attention.
        num_head_channels (int, optional): number of channels in each
            attention head. Defaults to 1.
        transformer_num_layers (int, optional): number of layers of
            Transformer blocks to use. Defaults to 1.
        cross_attention_dim (int | None, optional): number of context
            dimensions to use. Defaults to None.
        upcast_attention (bool, optional): if True, upcast attention
            operations to full precision. Defaults to False.
        use_flash_attention (bool, optional): if True, use flash attention
            for a memory efficient attention mechanism. Defaults to False.

    Returns:
        torch.nn.Module: downsampling block.
    """
    if with_attn:
        return AttnDownBlock(
            spatial_dims=spatial_dims,
            in_channels=in_channels,
            out_channels=out_channels,
            class_emb_channels=class_emb_channels,
            num_res_blocks=num_res_blocks,
            norm_num_groups=norm_num_groups,
            norm_eps=norm_eps,
            stride=stride,
            resblock_updown=resblock_updown,
            num_head_channels=num_head_channels,
            use_flash_attention=use_flash_attention,
        )
    elif with_cross_attn:
        return CrossAttnDownBlock(
            spatial_dims=spatial_dims,
            in_channels=in_channels,
            out_channels=out_channels,
            class_emb_channels=class_emb_channels,
            num_res_blocks=num_res_blocks,
            norm_num_groups=norm_num_groups,
            norm_eps=norm_eps,
            stride=stride,
            resblock_updown=resblock_updown,
            num_head_channels=num_head_channels,
            transformer_num_layers=transformer_num_layers,
            cross_attention_dim=cross_attention_dim,
            upcast_attention=upcast_attention,
            use_flash_attention=use_flash_attention,
        )
    else:
        return DownBlock(
            spatial_dims=spatial_dims,
            in_channels=in_channels,
            out_channels=out_channels,
            class_emb_channels=class_emb_channels,
            num_res_blocks=num_res_blocks,
            norm_num_groups=norm_num_groups,
            norm_eps=norm_eps,
            stride=stride,
            resblock_updown=resblock_updown,
        )


def get_mid_block(
    spatial_dims: int,
    in_channels: int,
    class_emb_channels: int,
    norm_num_groups: int,
    norm_eps: float,
    with_conditioning: bool,
    num_head_channels: int,
    transformer_num_layers: int,
    cross_attention_dim: int | None,
    upcast_attention: bool = False,
    use_flash_attention: bool = False,
) -> torch.nn.Module:
    """
    Args:
        spatial_dims (int): The number of spatial dimensions.
        in_channels (int): number of input channels.
        out_channels (int): number of output channels.
        class_emb_channels (int): number of class embedding channels.
        norm_num_groups (int, optional): number of groups for the group
            normalization. Defaults to 32.
        norm_eps (float, optional): epsilon for the group normalization.
            Defaults to 1e-6.
        with_conditioning (bool): if True uses cross-attention blocks.
        num_head_channels (int, optional): number of channels in each
            attention head. Defaults to 1.
        transformer_num_layers (int, optional): number of layers of
            Transformer blocks to use. Defaults to 1.
        cross_attention_dim (int | None, optional): number of context
            dimensions to use. Defaults to None.
        upcast_attention (bool, optional): if True, upcast attention
            operations to full precision. Defaults to False.
        use_flash_attention (bool, optional): if True, use flash attention
            for a memory efficient attention mechanism. Defaults to False.

    Returns:
        torch.nn.Module: middle block.
    """
    if with_conditioning:
        return CrossAttnMidBlock(
            spatial_dims=spatial_dims,
            in_channels=in_channels,
            class_emb_channels=class_emb_channels,
            norm_num_groups=norm_num_groups,
            norm_eps=norm_eps,
            num_head_channels=num_head_channels,
            transformer_num_layers=transformer_num_layers,
            cross_attention_dim=cross_attention_dim,
            upcast_attention=upcast_attention,
            use_flash_attention=use_flash_attention,
        )
    else:
        return AttnMidBlock(
            spatial_dims=spatial_dims,
            in_channels=in_channels,
            class_emb_channels=class_emb_channels,
            norm_num_groups=norm_num_groups,
            norm_eps=norm_eps,
            num_head_channels=num_head_channels,
            use_flash_attention=use_flash_attention,
        )


def get_up_block(
    spatial_dims: int,
    in_channels: int,
    prev_output_channel: int,
    out_channels: int,
    class_emb_channels: int,
    num_res_blocks: int,
    norm_num_groups: int,
    norm_eps: float,
    stride: int | Sequence[int],
    resblock_updown: bool,
    with_attn: bool,
    with_cross_attn: bool,
    num_head_channels: int,
    transformer_num_layers: int,
    cross_attention_dim: int | None,
    upcast_attention: bool = False,
    use_flash_attention: bool = False,
    no_skip_connection: bool = False,
) -> torch.nn.Module:
    """
    Returns an upsampling block with optional attention and skip connections.
    """
    if no_skip_connection:
        return NoSkipUpBlock(
            spatial_dims=spatial_dims,
            in_channels=in_channels,
            prev_output_channel=prev_output_channel,
            out_channels=out_channels,
            class_emb_channels=class_emb_channels,
            num_res_blocks=num_res_blocks,
            norm_num_groups=norm_num_groups,
            norm_eps=norm_eps,
            stride=stride,
            resblock_updown=resblock_updown,
            with_attn=with_attn,
            with_cross_attn=with_cross_attn,
            num_head_channels=num_head_channels,
            transformer_num_layers=transformer_num_layers,
            cross_attention_dim=cross_attention_dim,
            upcast_attention=upcast_attention,
            use_flash_attention=use_flash_attention,
        )
    if with_attn:
        return AttnUpBlock(
            spatial_dims=spatial_dims,
            in_channels=in_channels,
            prev_output_channel=prev_output_channel,
            out_channels=out_channels,
            class_emb_channels=class_emb_channels,
            num_res_blocks=num_res_blocks,
            norm_num_groups=norm_num_groups,
            norm_eps=norm_eps,
            stride=stride,
            resblock_updown=resblock_updown,
            num_head_channels=num_head_channels,
            use_flash_attention=use_flash_attention,
        )
    elif with_cross_attn:
        return CrossAttnUpBlock(
            spatial_dims=spatial_dims,
            in_channels=in_channels,
            prev_output_channel=prev_output_channel,
            out_channels=out_channels,
            class_emb_channels=class_emb_channels,
            num_res_blocks=num_res_blocks,
            norm_num_groups=norm_num_groups,
            norm_eps=norm_eps,
            stride=stride,
            resblock_updown=resblock_updown,
            num_head_channels=num_head_channels,
            transformer_num_layers=transformer_num_layers,
            cross_attention_dim=cross_attention_dim,
            upcast_attention=upcast_attention,
            use_flash_attention=use_flash_attention,
        )
    else:
        return UpBlock(
            spatial_dims=spatial_dims,
            in_channels=in_channels,
            prev_output_channel=prev_output_channel,
            out_channels=out_channels,
            class_emb_channels=class_emb_channels,
            num_res_blocks=num_res_blocks,
            norm_num_groups=norm_num_groups,
            norm_eps=norm_eps,
            stride=stride,
            resblock_updown=resblock_updown,
        )


class Generator(torch.nn.Module):
    """
    A wrapper for most of DiffusionModelUNet with some modifications (no
    timestep-encoding stuff).
    """

    def __init__(
        self,
        spatial_dims: int,
        in_channels: int,
        out_channels: int,
        num_res_blocks: Sequence[int] | int = (2, 2, 2, 2),
        num_channels: Sequence[int] = (32, 64, 64, 64),
        attention_levels: Sequence[bool] = (False, False, True, True),
        norm_num_groups: int = 32,
        norm_eps: float = 1e-6,
        resblock_updown: bool = False,
        strides: Sequence[int | Sequence[int]] | None = None,
        num_head_channels: int | Sequence[int] = 8,
        with_conditioning: bool = False,
        transformer_num_layers: int = 1,
        cross_attention_dim: int | None = None,
        num_class_embeds: int | None = None,
        upcast_attention: bool = False,
        use_flash_attention: bool = False,
        no_skip_connection: bool = False,
        out_activation: str | Callable = torch.nn.Tanh,
    ) -> None:
        """
        Args:
            spatial_dims (int): number of spatial dimensions.
            in_channels (int): number of input channels.
            out_channels (int): number of output channels.
            num_res_blocks (Sequence[int]): number of residual blocks (see
                ResnetBlock) per level.
            num_channels (Sequence[int]): tuple of block output channels.
            attention_levels (Sequence[bool]): list of levels to add attention.
            norm_num_groups (int): number of groups for the normalization.
            norm_eps (float): epsilon for the normalization.
            resblock_updown (bool): if True use residual blocks for
                up/downsampling.
            strides (Sequence[int | Sequence[int]] | None): stride for each
                block. Must contain one entry per block, where each entry is
                either an int or a per-spatial-dimension tuple. For example,
                ``(2, 2, 2, 1)`` or ``((2, 2), (2, 2), (2, 2), 1)``. The
                upsampling blocks use the strides in reverse order. If None,
                a stride of 2 is used for every block. Defaults to None.
            num_head_channels (int | Sequence[int]): number of channels in each
                attention head.
            with_conditioning (bool): if True add spatial transformers to
                perform conditioning.
            transformer_num_layers (int): number of layers of Transformer
                blocks to use.
            cross_attention_dim (int | None): number of context dimensions to
                use.
            num_class_embeds (int | None): if specified (as an int), then this
                model will be class-conditional with `num_class_embeds`
                classes.
            upcast_attention (bool): if True, upcast attention operations to
                full precision.
            use_flash_attention (bool): if True, use flash attention for a
                memory efficient attention mechanism.
            no_skip_connection (bool): if True, do not add skip connections
                between up/down blocks.
            out_activation (str | Callable): the activation for the output
                layer. It can be specified as a string which is executed as is
                using `eval()`. Defaults to torch.nn.Tanh.
        """
        super().__init__()
        if with_conditioning is True:
            if cross_attention_dim is None and num_class_embeds is None:
                raise ValueError(
                    "Generator expects dimension of the cross-attention conditioning (cross_attention_dim) "
                    "or size of class embeddings (num_class_embeds) "
                    "when using with_conditioning."
                )
        if cross_attention_dim is not None and with_conditioning is False:
            raise ValueError(
                "Generator expects with_conditioning=True when specifying the cross_attention_dim."
            )

        # All number of channels should be multiple of num_groups
        if any(
            (out_channel % norm_num_groups) != 0 for out_channel in num_channels
        ):
            raise ValueError(
                "DiffusionModelUNet expects all num_channels being multiple of norm_num_groups"
            )

        if len(num_channels) != len(attention_levels):
            raise ValueError(
                "DiffusionModelUNet expects num_channels being same size of attention_levels"
            )

        if isinstance(num_head_channels, int):
            num_head_channels = ensure_tuple_rep(
                num_head_channels, len(attention_levels)
            )

        if len(num_head_channels) != len(attention_levels):
            raise ValueError(
                "num_head_channels should have the same length as attention_levels. For the i levels without attention,"
                " i.e. `attention_level[i]=False`, the num_head_channels[i] will be ignored."
            )

        if isinstance(num_res_blocks, int):
            num_res_blocks = ensure_tuple_rep(num_res_blocks, len(num_channels))

        if len(num_res_blocks) != len(num_channels):
            raise ValueError(
                "`num_res_blocks` should be a single integer or a tuple of integers with the same length as "
                "`num_channels`."
            )

        if strides is None:
            strides = [2] * len(num_channels)
        elif len(strides) != len(num_channels):
            raise ValueError(
                "`strides` should have the same length as `num_channels`."
            )
        strides = [
            _resolve_stride(block_stride, spatial_dims)
            for block_stride in strides
        ]

        if use_flash_attention and not has_xformers:
            raise ValueError(
                "use_flash_attention is True but xformers is not installed."
            )

        if use_flash_attention is True and not torch.cuda.is_available():
            raise ValueError(
                "torch.cuda.is_available() should be True but is False. Flash attention is only available for GPU."
            )

        self.in_channels = in_channels
        self.spatial_dims = spatial_dims
        self.strides = list(strides)
        self.block_out_channels = num_channels
        self.out_channels = out_channels
        self.num_res_blocks = num_res_blocks
        self.attention_levels = attention_levels
        self.num_head_channels = num_head_channels
        self.with_conditioning = with_conditioning
        self.cross_attention_dim = cross_attention_dim
        self.no_skip_connection = no_skip_connection

        self.last_bottleneck_shape = None

        # input
        self.conv_in = Convolution(
            spatial_dims=spatial_dims,
            in_channels=in_channels,
            out_channels=num_channels[0],
            strides=1,
            kernel_size=3,
            padding=1,
            conv_only=True,
        )

        # class embedding
        self.num_class_embeds = num_class_embeds
        class_embedding_dim = None
        if num_class_embeds is not None:
            class_embedding_dim = num_channels[0] * 4
            self.class_embedding = torch.nn.Embedding(
                num_class_embeds, class_embedding_dim
            )

        # down
        self.down_blocks = torch.nn.ModuleList([])
        output_channel = num_channels[0]
        for i in range(len(num_channels)):
            input_channel = output_channel
            output_channel = num_channels[i]

            down_block = get_down_block(
                spatial_dims=spatial_dims,
                in_channels=input_channel,
                out_channels=output_channel,
                class_emb_channels=class_embedding_dim,
                num_res_blocks=num_res_blocks[i],
                norm_num_groups=norm_num_groups,
                norm_eps=norm_eps,
                stride=self.strides[i],
                resblock_updown=resblock_updown,
                with_attn=(attention_levels[i] and not with_conditioning),
                with_cross_attn=(attention_levels[i] and with_conditioning),
                num_head_channels=num_head_channels[i],
                transformer_num_layers=transformer_num_layers,
                cross_attention_dim=cross_attention_dim,
                upcast_attention=upcast_attention,
                use_flash_attention=use_flash_attention,
            )

            self.down_blocks.append(down_block)

        # mid
        self.middle_block = get_mid_block(
            spatial_dims=spatial_dims,
            in_channels=num_channels[-1],
            class_emb_channels=class_embedding_dim,
            norm_num_groups=norm_num_groups,
            norm_eps=norm_eps,
            with_conditioning=with_conditioning,
            num_head_channels=num_head_channels[-1],
            transformer_num_layers=transformer_num_layers,
            cross_attention_dim=cross_attention_dim,
            upcast_attention=upcast_attention,
            use_flash_attention=use_flash_attention,
        )

        # up
        self.up_blocks = torch.nn.ModuleList([])
        reversed_block_out_channels = list(reversed(num_channels))
        reversed_num_res_blocks = list(reversed(num_res_blocks))
        reversed_attention_levels = list(reversed(attention_levels))
        reversed_num_head_channels = list(reversed(num_head_channels))
        output_channel = reversed_block_out_channels[0]
        for i in range(len(reversed_block_out_channels)):
            prev_output_channel = output_channel
            output_channel = reversed_block_out_channels[i]
            input_channel = reversed_block_out_channels[
                min(i + 1, len(num_channels) - 1)
            ]

            # Up blocks are built in reverse order. The up block at level j
            # consumes the residual samples from down level j plus the
            # downsampler output from level j - 1 (or, for j == 0, the input
            # convolution sample). Its upsampling stride is the one that
            # inverted the corresponding downsampler.
            down_index = len(num_channels) - 1 - i
            if down_index == 0:
                block_stride = (1,) * spatial_dims
                extra_res_block = 1
            else:
                block_stride = self.strides[down_index - 1]
                extra_res_block = 0 if _is_identity_stride(block_stride) else 1

            up_block = get_up_block(
                spatial_dims=spatial_dims,
                in_channels=input_channel,
                prev_output_channel=prev_output_channel,
                out_channels=output_channel,
                class_emb_channels=class_embedding_dim,
                num_res_blocks=reversed_num_res_blocks[i] + extra_res_block,
                norm_num_groups=norm_num_groups,
                norm_eps=norm_eps,
                stride=block_stride,
                resblock_updown=resblock_updown,
                with_attn=(
                    reversed_attention_levels[i] and not with_conditioning
                ),
                with_cross_attn=(
                    reversed_attention_levels[i] and with_conditioning
                ),
                num_head_channels=reversed_num_head_channels[i],
                transformer_num_layers=transformer_num_layers,
                cross_attention_dim=cross_attention_dim,
                upcast_attention=upcast_attention,
                use_flash_attention=use_flash_attention,
                no_skip_connection=self.no_skip_connection,
            )

            self.up_blocks.append(up_block)

        # If the final down block downsamples, the decoder first needs to
        # return to the final encoder level before consuming skip connections.
        if _is_identity_stride(self.strides[-1]):
            self.bottleneck_upsampler = None
        else:
            self.bottleneck_upsampler = Upsample(
                spatial_dims=spatial_dims,
                num_channels=num_channels[-1],
                use_conv=False,
                stride=self.strides[-1],
            )

        # out
        self.out = torch.nn.Sequential(
            torch.nn.GroupNorm(
                num_groups=norm_num_groups,
                num_channels=num_channels[0],
                eps=norm_eps,
                affine=True,
            ),
            torch.nn.SiLU(),
            zero_module(
                Convolution(
                    spatial_dims=spatial_dims,
                    in_channels=num_channels[0],
                    out_channels=out_channels,
                    strides=1,
                    kernel_size=3,
                    padding=1,
                    conv_only=True,
                )
            ),
        )
        if isinstance(out_activation, str):
            self.out_activation = eval(out_activation)()
        else:
            self.out_activation = out_activation()

    @property
    def latent_downsample_factor(self) -> int | tuple[int, ...]:
        """
        Spatial downsampling factor applied by the encoder.

        Returns:
            int | tuple[int, ...]: downsampling factor, e.g. 3 for a factor of
                8 per dimension. If the factor differs between spatial
                dimensions, a tuple is returned.
        """
        factors = tuple(
            math.prod(stride[axis] for stride in self.strides)
            for axis in range(self.spatial_dims)
        )
        if len(set(factors)) == 1:
            return factors[0]
        return factors

    def get_class_embeddings(
        self, x: torch.Tensor, class_labels: torch.Tensor | None
    ) -> torch.Tensor:
        """
        Returns the classification embeddings.

        Args:
            x (torch.Tensor): input tensor.
            class_labels (torch.Tensor): class labels.

        Raises:
            ValueError: if class labels are not provided and
                num_class_embeds > 0.

        Returns:
            torch.Tensor: class embeddings.
        """
        class_emb = None
        if self.num_class_embeds is not None:
            if class_labels is None:
                raise ValueError(
                    "class_labels should be provided when num_class_embeds > 0"
                )
            class_emb = self.class_embedding(class_labels)
            class_emb = class_emb.to(dtype=x.dtype)
        return class_emb

    def forward(
        self,
        x: torch.Tensor,
        context: torch.Tensor | None = None,
        class_labels: torch.Tensor | None = None,
        down_block_additional_residuals: tuple[torch.Tensor] | None = None,
        mid_block_additional_residual: torch.Tensor | None = None,
        return_features: bool = False,
    ) -> torch.Tensor | tuple[torch.Tensor, torch.Tensor]:
        """
        Args:
            x (torch.Tensor): input tensor (N, C, SpatialDims).
            context (torch.Tensor | None, optional): context tensor
                (N, 1, ContextDim).
            class_labels (torch.Tensor | None, optional): class label tensor
                tensor (N, ).
            down_block_additional_residuals (tuple[torch.Tensor] | None,
                optional): additional residual tensors for down blocks
                (N, C, FeatureMapsDims).
            mid_block_additional_residual (torch.Tensor | None, optional):
                additional residual tensor for mid block
                (N, C, FeatureMapsDims).
            return_features (bool, optional). Whether the features for the last
                layer should be returned. Defaults to False.

        Returns:
            torch.Tensor | tuple[torch.Tensor, torch.Tensor]: the output
                prediction map or the output prediction map and the bottleneck
                features.
        """
        if context is not None and self.with_conditioning is False:
            raise ValueError(
                "model should have with_conditioning = True if context is provided"
            )
        # 1. class
        class_emb = self.get_class_embeddings(x, class_labels)

        # 2. initial convolution

        h = self.conv_in(x)

        # 3. down
        down_block_res_samples: list[torch.Tensor] = [h]
        for downsample_block in self.down_blocks:
            h, res_samples = downsample_block(
                hidden_states=h, emb=class_emb, context=context
            )
            for residual in res_samples:
                down_block_res_samples.append(residual)

        # Additional residual conections for Controlnets
        if down_block_additional_residuals is not None:
            new_down_block_res_samples = ()
            for down_block_res_sample, down_block_additional_residual in zip(
                down_block_res_samples, down_block_additional_residuals
            ):
                down_block_res_sample = (
                    down_block_res_sample + down_block_additional_residual
                )
                new_down_block_res_samples += (down_block_res_sample,)

            down_block_res_samples = new_down_block_res_samples

        # 4. mid
        h = self.middle_block(hidden_states=h, emb=class_emb, context=context)
        self.last_bottleneck_shape = h.shape

        # Additional residual conections for Controlnets
        if mid_block_additional_residual is not None:
            h = h + mid_block_additional_residual

        # 5. up
        h, out_features = self._decode_blocks(
            h,
            class_emb,
            context,
            down_block_res_samples=down_block_res_samples,
        )

        if return_features is True:
            return h, out_features
        return h

    def _decode_blocks(
        self,
        h: torch.Tensor,
        class_emb: torch.Tensor | None,
        context: torch.Tensor | None,
        down_block_res_samples: list[torch.Tensor] | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Runs the decoder half of the network.

        Args:
            h (torch.Tensor): bottleneck features.
            class_emb (torch.Tensor | None): class embeddings.
            context (torch.Tensor | None): context tensor for cross-attention.
            down_block_res_samples (list[torch.Tensor] | None, optional):
                skip connection tensors produced by the encoder. If
                ``no_skip_connection`` is True these are ignored. Defaults to
                None.

        Returns:
            tuple[torch.Tensor, torch.Tensor]: the output image and the output
                features before the final layer.

        Raises:
            ValueError: if skip connections are required but not provided.
        """
        if down_block_res_samples is None:
            if self.no_skip_connection is False:
                raise ValueError(
                    "skip connections are required to decode from a bottleneck"
                )
            down_block_res_samples = []
        # The final downsampler output is the bottleneck input, not a skip
        # connection, so it should not be consumed by the up blocks.
        if len(down_block_res_samples) > 0 and not _is_identity_stride(
            self.strides[-1]
        ):
            down_block_res_samples = down_block_res_samples[:-1]
        if self.bottleneck_upsampler is not None:
            h = self.bottleneck_upsampler(h)
        for upsample_block in self.up_blocks:
            if len(down_block_res_samples) > 0:
                res_samples = down_block_res_samples[
                    -len(upsample_block.resnets) :
                ]
                down_block_res_samples = down_block_res_samples[
                    : -len(upsample_block.resnets)
                ]
            else:
                res_samples = None
            if self.no_skip_connection:
                res_samples = None
            h = upsample_block(
                hidden_states=h,
                emb=class_emb,
                res_hidden_states_list=res_samples,
                context=context,
            )

        out_features = h
        h = self.out(h)
        h = self.out_activation(h)
        return h, out_features

    def encode(
        self,
        x: torch.Tensor,
        context: torch.Tensor | None = None,
        class_labels: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """
        Encodes an input image into the bottleneck representation.

        Args:
            x (torch.Tensor): input tensor (N, C, SpatialDims).
            context (torch.Tensor | None, optional): context tensor.
            class_labels (torch.Tensor | None, optional): class labels.

        Returns:
            torch.Tensor: bottleneck features.
        """
        return self.forward_bottleneck(
            x, context=context, class_labels=class_labels
        )

    def decode(
        self,
        h: torch.Tensor,
        context: torch.Tensor | None = None,
        class_labels: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """
        Decodes a bottleneck representation into an image.

        Args:
            h (torch.Tensor): bottleneck features.
            context (torch.Tensor | None, optional): context tensor.
            class_labels (torch.Tensor | None, optional): class labels.

        Returns:
            torch.Tensor: output image.

        Raises:
            ValueError: if ``no_skip_connection`` is False.
        """
        if self.no_skip_connection is False:
            raise ValueError(
                "decode is only available when no_skip_connection is True"
            )
        if context is not None and self.with_conditioning is False:
            raise ValueError(
                "model should have with_conditioning = True if context is provided"
            )
        class_emb = self.get_class_embeddings(h, class_labels)
        h, _ = self._decode_blocks(h, class_emb, context)
        return h

    def forward_bottleneck(
        self,
        x: torch.Tensor,
        context: torch.Tensor | None = None,
        class_labels: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """
        Args:
            x (torch.Tensor): input tensor (N, C, SpatialDims).
            context (torch.Tensor | None, optional): context tensor
                (N, 1, ContextDim).
            class_labels (torch.Tensor | None, optional): class label tensor
                tensor (N, ).
            down_block_additional_residuals (tuple[torch.Tensor] | None,
                optional): additional residual tensors for down blocks
                (N, C, FeatureMapsDims).
        Returns:
            torch.Tensor: bottleneck features.
        """
        if context is not None and self.with_conditioning is False:
            raise ValueError(
                "model should have with_conditioning = True if context is provided"
            )
        # 1. class
        class_emb = self.get_class_embeddings(x, class_labels)

        # 2. initial convolution

        h = self.conv_in(x)

        # 3. down
        down_block_res_samples: list[torch.Tensor] = [h]
        for downsample_block in self.down_blocks:
            h, res_samples = downsample_block(
                hidden_states=h, emb=class_emb, context=context
            )
            for residual in res_samples:
                down_block_res_samples.append(residual)

        # 4. mid
        h = self.middle_block(hidden_states=h, emb=class_emb, context=context)
        self.last_bottleneck_shape = h.shape

        return h
