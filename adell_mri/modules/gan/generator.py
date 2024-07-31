"""
Straightforward re-implementation of diffusion models in the MONAI generative
package but replacing mandatory time embeddings with optional class embeddings.

So this is essentially repackaging what the MONAI team did as a GAN generator.
"""

import importlib
import torch
import torch.nn.functional as F
import math
from monai.networks.blocks import Convolution, MLPBlock
from monai.utils import ensure_tuple_rep
from monai.networks.layers.factories import Pool
from typing import Sequence

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


class CrossAttention(torch.nn.Module):
    """
    A cross attention layer.
    """

    def __init__(
        self,
        query_dim: int,
        cross_attention_dim: int | None = None,
        num_attention_heads: int = 8,
        num_head_channels: int = 64,
        dropout: float = 0.0,
        upcast_attention: bool = False,
        use_flash_attention: bool = False,
    ) -> None:
        """
        Args:
            query_dim (int): number of channels in the query.
            cross_attention_dim (int | None, optional): number of channels in
                the context. Defaults to None.
            num_attention_heads (int, optional): number of heads to use for
                multi-head attention. Defaults to 8.
            num_head_channels (int, optional): number of channels in each head.
                Defaults to 64.
            dropout (float, optional): dropout probability to use. Defaults to
                0.0.
            upcast_attention (bool, optional): if True, upcast attention
                operations to full precision. Defaults to False.
            use_flash_attention (bool, optional): if True, use flash attention
                for a memory efficient attention mechanism. Defaults to False.
        """
        super().__init__()
        self.use_flash_attention = use_flash_attention
        inner_dim = num_head_channels * num_attention_heads
        cross_attention_dim = (
            cross_attention_dim
            if cross_attention_dim is not None
            else query_dim
        )

        self.scale = 1 / math.sqrt(num_head_channels)
        self.num_heads = num_attention_heads

        self.upcast_attention = upcast_attention

        self.to_q = torch.nn.Linear(query_dim, inner_dim, bias=False)
        self.to_k = torch.nn.Linear(cross_attention_dim, inner_dim, bias=False)
        self.to_v = torch.nn.Linear(cross_attention_dim, inner_dim, bias=False)

        self.to_out = torch.nn.Sequential(
            torch.nn.Linear(inner_dim, query_dim), torch.nn.Dropout(dropout)
        )

    def reshape_heads_to_batch_dim(self, x: torch.Tensor) -> torch.Tensor:
        """
        Divide hidden state dimension to the multiple attention heads and
        reshape their input as instances in the batch.
        """
        batch_size, seq_len, dim = x.shape
        x = x.reshape(
            batch_size, seq_len, self.num_heads, dim // self.num_heads
        )
        x = x.permute(0, 2, 1, 3).reshape(
            batch_size * self.num_heads, seq_len, dim // self.num_heads
        )
        return x

    def reshape_batch_dim_to_heads(self, x: torch.Tensor) -> torch.Tensor:
        """
        Combine the output of the attention heads back into the hidden state
        dimension.
        """
        batch_size, seq_len, dim = x.shape
        x = x.reshape(
            batch_size // self.num_heads, self.num_heads, seq_len, dim
        )
        x = x.permute(0, 2, 1, 3).reshape(
            batch_size // self.num_heads, seq_len, dim * self.num_heads
        )
        return x

    def _memory_efficient_attention_xformers(
        self, query: torch.Tensor, key: torch.Tensor, value: torch.Tensor
    ) -> torch.Tensor:
        query = query.contiguous()
        key = key.contiguous()
        value = value.contiguous()
        x = xformers.ops.memory_efficient_attention(
            query, key, value, attn_bias=None
        )
        return x

    def _attention(
        self, query: torch.Tensor, key: torch.Tensor, value: torch.Tensor
    ) -> torch.Tensor:
        dtype = query.dtype
        if self.upcast_attention:
            query = query.float()
            key = key.float()

        attention_scores = torch.baddbmm(
            torch.empty(
                query.shape[0],
                query.shape[1],
                key.shape[1],
                dtype=query.dtype,
                device=query.device,
            ),
            query,
            key.transpose(-1, -2),
            beta=0,
            alpha=self.scale,
        )
        attention_probs = attention_scores.softmax(dim=-1)
        attention_probs = attention_probs.to(dtype=dtype)

        x = torch.bmm(attention_probs, value)
        return x

    def forward(
        self, x: torch.Tensor, context: torch.Tensor | None = None
    ) -> torch.Tensor:
        query = self.to_q(x)
        context = context if context is not None else x
        key = self.to_k(context)
        value = self.to_v(context)

        # Multi-Head Attention
        query = self.reshape_heads_to_batch_dim(query)
        key = self.reshape_heads_to_batch_dim(key)
        value = self.reshape_heads_to_batch_dim(value)

        if self.use_flash_attention:
            x = self._memory_efficient_attention_xformers(query, key, value)
        else:
            x = self._attention(query, key, value)

        x = self.reshape_batch_dim_to_heads(x)
        x = x.to(query.dtype)

        return self.to_out(x)


class BasicTransformerBlock(torch.nn.Module):
    """
    A basic Transformer block.
    """

    def __init__(
        self,
        num_channels: int,
        num_attention_heads: int,
        num_head_channels: int,
        dropout: float = 0.0,
        cross_attention_dim: int | None = None,
        upcast_attention: bool = False,
        use_flash_attention: bool = False,
    ) -> None:
        """
        Args:
            num_channels (int): number of channels in the input and output.
            num_attention_heads (int, optional): number of heads to use for
                multi-head attention. Defaults to 8.
            num_head_channels (int, optional): number of channels in each head.
                Defaults to 64.
            dropout (float, optional): dropout probability to use. Defaults to
                0.0.
            cross_attention_dim (int | None, optional): number of channels in
                the context. Defaults to None.
            upcast_attention (bool, optional): if True, upcast attention
                operations to full precision. Defaults to False.
            use_flash_attention (bool, optional): if True, use flash attention
                for a memory efficient attention mechanism. Defaults to False.
        """
        super().__init__()
        self.attn1 = CrossAttention(
            query_dim=num_channels,
            num_attention_heads=num_attention_heads,
            num_head_channels=num_head_channels,
            dropout=dropout,
            upcast_attention=upcast_attention,
            use_flash_attention=use_flash_attention,
        )  # is a self-attention
        self.ff = MLPBlock(
            hidden_size=num_channels,
            mlp_dim=num_channels * 4,
            act="GEGLU",
            dropout_rate=dropout,
        )
        self.attn2 = CrossAttention(
            query_dim=num_channels,
            cross_attention_dim=cross_attention_dim,
            num_attention_heads=num_attention_heads,
            num_head_channels=num_head_channels,
            dropout=dropout,
            upcast_attention=upcast_attention,
            use_flash_attention=use_flash_attention,
        )  # is a self-attention if context is None
        self.norm1 = torch.nn.LayerNorm(num_channels)
        self.norm2 = torch.nn.LayerNorm(num_channels)
        self.norm3 = torch.nn.LayerNorm(num_channels)

    def forward(
        self, x: torch.Tensor, context: torch.Tensor | None = None
    ) -> torch.Tensor:
        # 1. Self-Attention
        x = self.attn1(self.norm1(x)) + x

        # 2. Cross-Attention
        x = self.attn2(self.norm2(x), context=context) + x

        # 3. Feed-forward
        x = self.ff(self.norm3(x)) + x
        return x


class SpatialTransformer(torch.nn.Module):
    """
    Transformer block for image-like data. First, project the input (aka
    embedding) and reshape to b, t, d. Then apply standard transformer action.
    Finally, reshape to image.
    """

    def __init__(
        self,
        spatial_dims: int,
        in_channels: int,
        num_attention_heads: int,
        num_head_channels: int,
        num_layers: int = 1,
        dropout: float = 0.0,
        norm_num_groups: int = 32,
        norm_eps: float = 1e-6,
        cross_attention_dim: int | None = None,
        upcast_attention: bool = False,
        use_flash_attention: bool = False,
    ) -> None:
        """
        Args:
            spatial_dim (int): number of spatial dimensions.
            in_channels (int): number of channels in the input and output.
            num_attention_heads (int, optional): number of heads to use for
                multi-head attention. Defaults to 8.
            num_head_channels (int, optional): number of channels in each head.
                Defaults to 64.
            num_layers (int): number of layers of Transformer blocks to use.
            dropout (float, optional): dropout probability to use. Defaults to
                0.0.
            norm_num_groups (int, optional): number of groups for the
                normalization. Defaults to 32.
            norm_eps (float, optional): epsilon for the normalization. Defaults
                to 1e-6.
            cross_attention_dim (int | None, optional): number of channels in
                the context. Defaults to None.
            upcast_attention (bool, optional): if True, upcast attention
                operations to full precision. Defaults to False.
            use_flash_attention (bool, optional): if True, use flash attention
                for a memory efficient attention mechanism. Defaults to False.
        """
        super().__init__()
        self.spatial_dims = spatial_dims
        self.in_channels = in_channels
        inner_dim = num_attention_heads * num_head_channels

        self.norm = torch.nn.GroupNorm(
            num_groups=norm_num_groups,
            num_channels=in_channels,
            eps=norm_eps,
            affine=True,
        )

        self.proj_in = Convolution(
            spatial_dims=spatial_dims,
            in_channels=in_channels,
            out_channels=inner_dim,
            strides=1,
            kernel_size=1,
            padding=0,
            conv_only=True,
        )

        self.transformer_blocks = torch.nn.ModuleList(
            [
                BasicTransformerBlock(
                    num_channels=inner_dim,
                    num_attention_heads=num_attention_heads,
                    num_head_channels=num_head_channels,
                    dropout=dropout,
                    cross_attention_dim=cross_attention_dim,
                    upcast_attention=upcast_attention,
                    use_flash_attention=use_flash_attention,
                )
                for _ in range(num_layers)
            ]
        )

        self.proj_out = zero_module(
            Convolution(
                spatial_dims=spatial_dims,
                in_channels=inner_dim,
                out_channels=in_channels,
                strides=1,
                kernel_size=1,
                padding=0,
                conv_only=True,
            )
        )

    def forward(
        self, x: torch.Tensor, context: torch.Tensor | None = None
    ) -> torch.Tensor:
        # note: if no context is given, cross-attention defaults to self-attention
        batch = channel = height = width = depth = -1
        if self.spatial_dims == 2:
            batch, channel, height, width = x.shape
        if self.spatial_dims == 3:
            batch, channel, height, width, depth = x.shape

        residual = x
        x = self.norm(x)
        x = self.proj_in(x)

        inner_dim = x.shape[1]

        if self.spatial_dims == 2:
            x = x.permute(0, 2, 3, 1).reshape(batch, height * width, inner_dim)
        if self.spatial_dims == 3:
            x = x.permute(0, 2, 3, 4, 1).reshape(
                batch, height * width * depth, inner_dim
            )

        for block in self.transformer_blocks:
            x = block(x, context=context)

        if self.spatial_dims == 2:
            x = (
                x.reshape(batch, height, width, inner_dim)
                .permute(0, 3, 1, 2)
                .contiguous()
            )
        if self.spatial_dims == 3:
            x = (
                x.reshape(batch, height, width, depth, inner_dim)
                .permute(0, 4, 1, 2, 3)
                .contiguous()
            )

        x = self.proj_out(x)
        return x + residual


class AttentionBlock(torch.nn.Module):
    """
    An attention block that allows spatial positions to attend to each other.
    Uses three q, k, v linear layers to compute attention.
    """

    def __init__(
        self,
        spatial_dims: int,
        num_channels: int,
        num_head_channels: int | None = None,
        norm_num_groups: int = 32,
        norm_eps: float = 1e-6,
        use_flash_attention: bool = False,
    ) -> None:
        """
        Args:
            spatial_dims (int): number of spatial dimensions.
            num_channels (int): number of input channels.
            num_head_channels (int | None, optional): number of channels in
                each attention head. Defaults to None.
            norm_num_groups (int, optional): number of groups involved for the
                group normalisation layer. Ensure that your number of channels
                is divisible by this number. Defaults to 32.
            norm_eps (float, optional): epsilon value to use for the
                normalisation. Defaults to 1e-6.
            use_flash_attention (bool, optional): if True, use flash attention
                for a memory efficient attention mechanism. Defaults to True.
        """
        super().__init__()
        self.use_flash_attention = use_flash_attention
        self.spatial_dims = spatial_dims
        self.num_channels = num_channels

        self.num_heads = (
            num_channels // num_head_channels
            if num_head_channels is not None
            else 1
        )
        self.scale = 1 / math.sqrt(num_channels / self.num_heads)

        self.norm = torch.nn.GroupNorm(
            num_groups=norm_num_groups,
            num_channels=num_channels,
            eps=norm_eps,
            affine=True,
        )

        self.to_q = torch.nn.Linear(num_channels, num_channels)
        self.to_k = torch.nn.Linear(num_channels, num_channels)
        self.to_v = torch.nn.Linear(num_channels, num_channels)

        self.proj_attn = torch.nn.Linear(num_channels, num_channels)

    def reshape_heads_to_batch_dim(self, x: torch.Tensor) -> torch.Tensor:
        batch_size, seq_len, dim = x.shape
        x = x.reshape(
            batch_size, seq_len, self.num_heads, dim // self.num_heads
        )
        x = x.permute(0, 2, 1, 3).reshape(
            batch_size * self.num_heads, seq_len, dim // self.num_heads
        )
        return x

    def reshape_batch_dim_to_heads(self, x: torch.Tensor) -> torch.Tensor:
        batch_size, seq_len, dim = x.shape
        x = x.reshape(
            batch_size // self.num_heads, self.num_heads, seq_len, dim
        )
        x = x.permute(0, 2, 1, 3).reshape(
            batch_size // self.num_heads, seq_len, dim * self.num_heads
        )
        return x

    def _memory_efficient_attention_xformers(
        self, query: torch.Tensor, key: torch.Tensor, value: torch.Tensor
    ) -> torch.Tensor:
        query = query.contiguous()
        key = key.contiguous()
        value = value.contiguous()
        x = xformers.ops.memory_efficient_attention(
            query, key, value, attn_bias=None
        )
        return x

    def _attention(
        self, query: torch.Tensor, key: torch.Tensor, value: torch.Tensor
    ) -> torch.Tensor:
        attention_scores = torch.baddbmm(
            torch.empty(
                query.shape[0],
                query.shape[1],
                key.shape[1],
                dtype=query.dtype,
                device=query.device,
            ),
            query,
            key.transpose(-1, -2),
            beta=0,
            alpha=self.scale,
        )
        attention_probs = attention_scores.softmax(dim=-1)
        x = torch.bmm(attention_probs, value)
        return x

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        residual = x

        batch = channel = height = width = depth = -1
        if self.spatial_dims == 2:
            batch, channel, height, width = x.shape
        if self.spatial_dims == 3:
            batch, channel, height, width, depth = x.shape

        # norm
        x = self.norm(x)

        if self.spatial_dims == 2:
            x = x.view(batch, channel, height * width).transpose(1, 2)
        if self.spatial_dims == 3:
            x = x.view(batch, channel, height * width * depth).transpose(1, 2)

        # proj to q, k, v
        query = self.to_q(x)
        key = self.to_k(x)
        value = self.to_v(x)

        # Multi-Head Attention
        query = self.reshape_heads_to_batch_dim(query)
        key = self.reshape_heads_to_batch_dim(key)
        value = self.reshape_heads_to_batch_dim(value)

        if self.use_flash_attention:
            x = self._memory_efficient_attention_xformers(query, key, value)
        else:
            x = self._attention(query, key, value)

        x = self.reshape_batch_dim_to_heads(x)
        x = x.to(query.dtype)

        if self.spatial_dims == 2:
            x = x.transpose(-1, -2).reshape(batch, channel, height, width)
        if self.spatial_dims == 3:
            x = x.transpose(-1, -2).reshape(
                batch, channel, height, width, depth
            )

        return x + residual


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
        """
        super().__init__()
        self.num_channels = num_channels
        self.out_channels = out_channels or num_channels
        self.use_conv = use_conv
        if use_conv:
            self.op = Convolution(
                spatial_dims=spatial_dims,
                in_channels=self.num_channels,
                out_channels=self.out_channels,
                strides=2,
                kernel_size=3,
                padding=padding,
                conv_only=True,
            )
        else:
            if self.num_channels != self.out_channels:
                raise ValueError(
                    "num_channels and out_channels must be equal when use_conv=False"
                )
            self.op = Pool[Pool.AVG, spatial_dims](kernel_size=2, stride=2)

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
        """
        super().__init__()
        self.num_channels = num_channels
        self.out_channels = out_channels or num_channels
        self.use_conv = use_conv
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

        x = F.interpolate(x, scale_factor=2.0, mode="nearest")

        # If the input is bfloat16, we cast back to bfloat16
        if dtype == torch.bfloat16:
            x = x.to(dtype)

        if self.use_conv:
            x = self.conv(x)
        return x


class ResnetBlock(torch.nn.Module):
    """
    Residual block with class conditioning.
    """

    def __init__(
        self,
        spatial_dims: int,
        in_channels: int,
        class_emb_channels: int,
        out_channels: int | None = None,
        up: bool = False,
        down: bool = False,
        norm_num_groups: int = 32,
        norm_eps: float = 1e-6,
    ) -> None:
        """
        Args:
            spatial_dims (int): The number of spatial dimensions.
            in_channels (int): number of input channels.
            class_emb_channels (int): number of class embedding channels.
            out_channels (int | None, optional): number of output channels.
                Defaults to None.
            up (bool, optional): if True, performs upsampling. Defaults to
                False.
            down (bool, optional): if True, performs downsampling. Defaults to
                False.
            norm_num_groups (int, optional): number of groups for the group
                normalization. Defaults to 32.
            norm_eps (float, optional): epsilon for the group normalization.
                Defaults to 1e-6.
        """
        super().__init__()
        self.spatial_dims = spatial_dims
        self.channels = in_channels
        self.class_emb_channels = class_emb_channels
        self.out_channels = out_channels or in_channels
        self.up = up
        self.down = down

        self.norm1 = torch.nn.GroupNorm(
            num_groups=norm_num_groups,
            num_channels=in_channels,
            eps=norm_eps,
            affine=True,
        )
        self.nonlinearity = torch.nn.SiLU()
        self.conv1 = Convolution(
            spatial_dims=spatial_dims,
            in_channels=in_channels,
            out_channels=self.out_channels,
            strides=1,
            kernel_size=3,
            padding=1,
            conv_only=True,
        )

        self.upsample = self.downsample = None
        if self.up:
            self.upsample = Upsample(spatial_dims, in_channels, use_conv=False)
        elif down:
            self.downsample = Downsample(
                spatial_dims, in_channels, use_conv=False
            )

        if self.class_emb_channels is not None:
            self.class_emb_proj = torch.nn.Linear(
                class_emb_channels, self.out_channels
            )
        else:
            self.class_emb_proj = None

        self.norm2 = torch.nn.GroupNorm(
            num_groups=norm_num_groups,
            num_channels=self.out_channels,
            eps=norm_eps,
            affine=True,
        )
        self.conv2 = zero_module(
            Convolution(
                spatial_dims=spatial_dims,
                in_channels=self.out_channels,
                out_channels=self.out_channels,
                strides=1,
                kernel_size=3,
                padding=1,
                conv_only=True,
            )
        )

        if self.out_channels == in_channels:
            self.skip_connection = torch.nn.Identity()
        else:
            self.skip_connection = Convolution(
                spatial_dims=spatial_dims,
                in_channels=in_channels,
                out_channels=self.out_channels,
                strides=1,
                kernel_size=1,
                padding=0,
                conv_only=True,
            )

    def forward(self, x: torch.Tensor, emb: torch.Tensor) -> torch.Tensor:
        h = x
        h = self.norm1(h)
        h = self.nonlinearity(h)

        if self.upsample is not None:
            if h.shape[0] >= 64:
                x = x.contiguous()
                h = h.contiguous()
            x = self.upsample(x)
            h = self.upsample(h)
        elif self.downsample is not None:
            x = self.downsample(x)
            h = self.downsample(h)

        h = self.conv1(h)

        if self.class_emb_proj is not None:
            if self.spatial_dims == 2:
                class_emb = self.class_emb_proj(self.nonlinearity(emb))[
                    :, :, None, None
                ]
            else:
                class_emb = self.class_emb_proj(self.nonlinearity(emb))[
                    :, :, None, None, None
                ]
            h = h + class_emb

        h = self.norm2(h)
        h = self.nonlinearity(h)
        h = self.conv2(h)

        return self.skip_connection(x) + h


class DownBlock(torch.nn.Module):
    """
    Unet's down block containing resnet and downsamplers blocks.
    """

    def __init__(
        self,
        spatial_dims: int,
        in_channels: int,
        out_channels: int,
        class_emb_channels: int,
        num_res_blocks: int = 1,
        norm_num_groups: int = 32,
        norm_eps: float = 1e-6,
        add_downsample: bool = True,
        resblock_updown: bool = False,
        downsample_padding: int = 1,
    ) -> None:
        """
        Args:
            spatial_dims (int): The number of spatial dimensions.
            in_channels  (int): number of input channels.
            out_channels (int): number of output channels.
            class_emb_channels (int): number of class embedding channels.
            num_res_blocks (int, optional): number of residual blocks. Defaults
                to 1.
            norm_num_groups (int, optional): number of groups for the group
                normalization. Defaults to 32.
            norm_eps (float, optional): epsilon for the group normalization.
                Defaults to 1e-6.
            add_downsample (bool, True): if True add downsample block. Defaults
                to True.
            resblock_updown (bool, False): if True use residual blocks for
                downsampling. Defaults to False.
            downsample_padding (int, optional): padding used in the
                downsampling block. Defaults to 1.
        """
        super().__init__()
        self.resblock_updown = resblock_updown

        resnets = []

        for i in range(num_res_blocks):
            in_channels = in_channels if i == 0 else out_channels
            resnets.append(
                ResnetBlock(
                    spatial_dims=spatial_dims,
                    in_channels=in_channels,
                    out_channels=out_channels,
                    class_emb_channels=class_emb_channels,
                    norm_num_groups=norm_num_groups,
                    norm_eps=norm_eps,
                )
            )

        self.resnets = torch.nn.ModuleList(resnets)

        if add_downsample:
            if resblock_updown:
                self.downsampler = ResnetBlock(
                    spatial_dims=spatial_dims,
                    in_channels=out_channels,
                    out_channels=out_channels,
                    class_emb_channels=class_emb_channels,
                    norm_num_groups=norm_num_groups,
                    norm_eps=norm_eps,
                    down=True,
                )
            else:
                self.downsampler = Downsample(
                    spatial_dims=spatial_dims,
                    num_channels=out_channels,
                    use_conv=True,
                    out_channels=out_channels,
                    padding=downsample_padding,
                )
        else:
            self.downsampler = None

    def forward(
        self,
        hidden_states: torch.Tensor,
        emb: torch.Tensor,
        context: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, list[torch.Tensor]]:
        del context
        output_states = []

        for resnet in self.resnets:
            hidden_states = resnet(hidden_states, emb)
            output_states.append(hidden_states)

        if self.downsampler is not None:
            hidden_states = self.downsampler(hidden_states, emb)
            output_states.append(hidden_states)

        return hidden_states, output_states


class AttnDownBlock(torch.nn.Module):
    """
    Unet's down block containing resnet, downsamplers and self-attention blocks.
    """

    def __init__(
        self,
        spatial_dims: int,
        in_channels: int,
        out_channels: int,
        class_emb_channels: int,
        num_res_blocks: int = 1,
        norm_num_groups: int = 32,
        norm_eps: float = 1e-6,
        add_downsample: bool = True,
        resblock_updown: bool = False,
        downsample_padding: int = 1,
        num_head_channels: int = 1,
        use_flash_attention: bool = False,
    ) -> None:
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
            add_downsample (bool, optional): if True add downsample block.
                Defaults to True.
            resblock_updown (bool, optional): if True use residual blocks for
                downsampling. Defaults to False.
            downsample_padding (int, optional): padding used in the downsampling
                block. Defaults to 1.
            num_head_channels (int, optional): number of channels in each
                attention head. Defaults to 1.
            use_flash_attention (bool, optional): if True, use flash attention
                for a memory efficient attention mechanism. Defaults to False.
        """
        super().__init__()
        self.resblock_updown = resblock_updown

        resnets = []
        attentions = []

        for i in range(num_res_blocks):
            in_channels = in_channels if i == 0 else out_channels
            resnets.append(
                ResnetBlock(
                    spatial_dims=spatial_dims,
                    in_channels=in_channels,
                    out_channels=out_channels,
                    class_emb_channels=class_emb_channels,
                    norm_num_groups=norm_num_groups,
                    norm_eps=norm_eps,
                )
            )
            attentions.append(
                AttentionBlock(
                    spatial_dims=spatial_dims,
                    num_channels=out_channels,
                    num_head_channels=num_head_channels,
                    norm_num_groups=norm_num_groups,
                    norm_eps=norm_eps,
                    use_flash_attention=use_flash_attention,
                )
            )

        self.attentions = torch.nn.ModuleList(attentions)
        self.resnets = torch.nn.ModuleList(resnets)

        if add_downsample:
            if resblock_updown:
                self.downsampler = ResnetBlock(
                    spatial_dims=spatial_dims,
                    in_channels=out_channels,
                    out_channels=out_channels,
                    class_emb_channels=class_emb_channels,
                    norm_num_groups=norm_num_groups,
                    norm_eps=norm_eps,
                    down=True,
                )
            else:
                self.downsampler = Downsample(
                    spatial_dims=spatial_dims,
                    num_channels=out_channels,
                    use_conv=True,
                    out_channels=out_channels,
                    padding=downsample_padding,
                )
        else:
            self.downsampler = None

    def forward(
        self,
        hidden_states: torch.Tensor,
        emb: torch.Tensor,
        context: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, list[torch.Tensor]]:
        del context
        output_states = []

        for resnet, attn in zip(self.resnets, self.attentions):
            hidden_states = resnet(hidden_states, emb)
            hidden_states = attn(hidden_states)
            output_states.append(hidden_states)

        if self.downsampler is not None:
            hidden_states = self.downsampler(hidden_states, emb)
            output_states.append(hidden_states)

        return hidden_states, output_states


class CrossAttnDownBlock(torch.nn.Module):
    """
    Unet's down block containing resnet, downsamplers and cross-attention blocks.
    """

    def __init__(
        self,
        spatial_dims: int,
        in_channels: int,
        out_channels: int,
        class_emb_channels: int,
        num_res_blocks: int = 1,
        norm_num_groups: int = 32,
        norm_eps: float = 1e-6,
        add_downsample: bool = True,
        resblock_updown: bool = False,
        downsample_padding: int = 1,
        num_head_channels: int = 1,
        transformer_num_layers: int = 1,
        cross_attention_dim: int | None = None,
        upcast_attention: bool = False,
        use_flash_attention: bool = False,
    ) -> None:
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
            add_downsample (bool, optional): if True add downsample block.
                Defaults to True.
            resblock_updown (bool, optional): if True use residual blocks for
                downsampling. Defaults to False.
            downsample_padding (int, optional): padding used in the downsampling
                block. Defaults to 1.
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
        """
        super().__init__()
        self.resblock_updown = resblock_updown

        resnets = []
        attentions = []

        for i in range(num_res_blocks):
            in_channels = in_channels if i == 0 else out_channels
            resnets.append(
                ResnetBlock(
                    spatial_dims=spatial_dims,
                    in_channels=in_channels,
                    out_channels=out_channels,
                    class_emb_channels=class_emb_channels,
                    norm_num_groups=norm_num_groups,
                    norm_eps=norm_eps,
                )
            )

            attentions.append(
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

        self.attentions = torch.nn.ModuleList(attentions)
        self.resnets = torch.nn.ModuleList(resnets)

        if add_downsample:
            if resblock_updown:
                self.downsampler = ResnetBlock(
                    spatial_dims=spatial_dims,
                    in_channels=out_channels,
                    out_channels=out_channels,
                    class_emb_channels=class_emb_channels,
                    norm_num_groups=norm_num_groups,
                    norm_eps=norm_eps,
                    down=True,
                )
            else:
                self.downsampler = Downsample(
                    spatial_dims=spatial_dims,
                    num_channels=out_channels,
                    use_conv=True,
                    out_channels=out_channels,
                    padding=downsample_padding,
                )
        else:
            self.downsampler = None

    def forward(
        self,
        hidden_states: torch.Tensor,
        emb: torch.Tensor,
        context: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, list[torch.Tensor]]:
        output_states = []

        for resnet, attn in zip(self.resnets, self.attentions):
            hidden_states = resnet(hidden_states, emb)
            hidden_states = attn(hidden_states, context=context)
            output_states.append(hidden_states)

        if self.downsampler is not None:
            hidden_states = self.downsampler(hidden_states, emb)
            output_states.append(hidden_states)

        return hidden_states, output_states


class AttnMidBlock(torch.nn.Module):
    """
    Unet's mid block containing resnet and self-attention blocks.
    """

    def __init__(
        self,
        spatial_dims: int,
        in_channels: int,
        class_emb_channels: int,
        norm_num_groups: int = 32,
        norm_eps: float = 1e-6,
        num_head_channels: int = 1,
        use_flash_attention: bool = False,
    ) -> None:
        """
        Args:
            spatial_dims (int): The number of spatial dimensions.
            in_channels (int): number of input channels.
            class_emb_channels (int): number of class embedding channels.
            norm_num_groups (int, optional): number of groups for the group
                normalization. Defaults to 32.
            norm_eps (float, optional): epsilon for the group normalization.
                Defaults to 1e-6.
            num_head_channels (int, optional): number of channels in each
                attention head. Defaults to 1.
            use_flash_attention (bool, optional): if True, use flash attention
                for a memory efficient attention mechanism. Defaults to False.
        """
        super().__init__()
        self.attention = None

        self.resnet_1 = ResnetBlock(
            spatial_dims=spatial_dims,
            in_channels=in_channels,
            out_channels=in_channels,
            class_emb_channels=class_emb_channels,
            norm_num_groups=norm_num_groups,
            norm_eps=norm_eps,
        )
        self.attention = AttentionBlock(
            spatial_dims=spatial_dims,
            num_channels=in_channels,
            num_head_channels=num_head_channels,
            norm_num_groups=norm_num_groups,
            norm_eps=norm_eps,
            use_flash_attention=use_flash_attention,
        )

        self.resnet_2 = ResnetBlock(
            spatial_dims=spatial_dims,
            in_channels=in_channels,
            out_channels=in_channels,
            class_emb_channels=class_emb_channels,
            norm_num_groups=norm_num_groups,
            norm_eps=norm_eps,
        )

    def forward(
        self,
        hidden_states: torch.Tensor,
        emb: torch.Tensor,
        context: torch.Tensor | None = None,
    ) -> torch.Tensor:
        del context
        hidden_states = self.resnet_1(hidden_states, emb)
        hidden_states = self.attention(hidden_states)
        hidden_states = self.resnet_2(hidden_states, emb)

        return hidden_states


class CrossAttnMidBlock(torch.nn.Module):
    """
    Unet's mid block containing resnet and cross-attention blocks.
    """

    def __init__(
        self,
        spatial_dims: int,
        in_channels: int,
        class_emb_channels: int,
        norm_num_groups: int = 32,
        norm_eps: float = 1e-6,
        num_head_channels: int = 1,
        transformer_num_layers: int = 1,
        cross_attention_dim: int | None = None,
        upcast_attention: bool = False,
        use_flash_attention: bool = False,
    ) -> None:
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
        """
        super().__init__()
        self.attention = None

        self.resnet_1 = ResnetBlock(
            spatial_dims=spatial_dims,
            in_channels=in_channels,
            out_channels=in_channels,
            class_emb_channels=class_emb_channels,
            norm_num_groups=norm_num_groups,
            norm_eps=norm_eps,
        )
        self.attention = SpatialTransformer(
            spatial_dims=spatial_dims,
            in_channels=in_channels,
            num_attention_heads=in_channels // num_head_channels,
            num_head_channels=num_head_channels,
            num_layers=transformer_num_layers,
            norm_num_groups=norm_num_groups,
            norm_eps=norm_eps,
            cross_attention_dim=cross_attention_dim,
            upcast_attention=upcast_attention,
            use_flash_attention=use_flash_attention,
        )
        self.resnet_2 = ResnetBlock(
            spatial_dims=spatial_dims,
            in_channels=in_channels,
            out_channels=in_channels,
            class_emb_channels=class_emb_channels,
            norm_num_groups=norm_num_groups,
            norm_eps=norm_eps,
        )

    def forward(
        self,
        hidden_states: torch.Tensor,
        emb: torch.Tensor,
        context: torch.Tensor | None = None,
    ) -> torch.Tensor:
        hidden_states = self.resnet_1(hidden_states, emb)
        hidden_states = self.attention(hidden_states, context=context)
        hidden_states = self.resnet_2(hidden_states, emb)

        return hidden_states


class UpBlock(torch.nn.Module):
    """
    Unet's up block containing resnet and upsamplers blocks.
    """

    def __init__(
        self,
        spatial_dims: int,
        in_channels: int,
        prev_output_channel: int,
        out_channels: int,
        class_emb_channels: int,
        num_res_blocks: int = 1,
        norm_num_groups: int = 32,
        norm_eps: float = 1e-6,
        add_upsample: bool = True,
        resblock_updown: bool = False,
        no_skip_connection: bool = False,
    ) -> None:
        """
        Args:
            spatial_dims (int): The number of spatial dimensions.
            in_channels (int): number of input channels.
            prev_output_channel (int): number of channels from residual
                connection.
            out_channels (int): number of output channels.
            class_emb_channels (int): number of class embedding channels.
            num_res_blocks (int): number of residual blocks. Defaults to 1.
            norm_num_groups (int, optional): number of groups for the group
                normalization. Defaults to 32.
            norm_eps (float, optional): epsilon for the group normalization.
                Defaults to 1e-6.
            add_upsample (bool, optional): if True add downsample block.
                Defaults to False.
            resblock_updown (bool, optional): if True use residual blocks for
                upsampling. Defaults to False.
        """
        super().__init__()
        self.resblock_updown = resblock_updown
        self.no_skip_connection = no_skip_connection
        resnets = []

        for i in range(num_res_blocks):
            res_skip_channels = (
                in_channels if (i == num_res_blocks - 1) else out_channels
            )
            if self.no_skip_connection:
                res_skip_channels = 0
            resnet_in_channels = (
                prev_output_channel if i == 0 else out_channels
            )

            resnets.append(
                ResnetBlock(
                    spatial_dims=spatial_dims,
                    in_channels=resnet_in_channels + res_skip_channels,
                    out_channels=out_channels,
                    class_emb_channels=class_emb_channels,
                    norm_num_groups=norm_num_groups,
                    norm_eps=norm_eps,
                )
            )

        self.resnets = torch.nn.ModuleList(resnets)

        if add_upsample:
            if resblock_updown:
                self.upsampler = ResnetBlock(
                    spatial_dims=spatial_dims,
                    in_channels=out_channels,
                    out_channels=out_channels,
                    class_emb_channels=class_emb_channels,
                    norm_num_groups=norm_num_groups,
                    norm_eps=norm_eps,
                    up=True,
                )
            else:
                self.upsampler = Upsample(
                    spatial_dims=spatial_dims,
                    num_channels=out_channels,
                    use_conv=True,
                    out_channels=out_channels,
                )
        else:
            self.upsampler = None

    def forward(
        self,
        hidden_states: torch.Tensor,
        emb: torch.Tensor,
        res_hidden_states_list: list[torch.Tensor] | None = None,
        context: torch.Tensor | None = None,
    ) -> torch.Tensor:
        del context
        for resnet in self.resnets:
            # pop res hidden states
            if self.no_skip_connection is False:
                res_hidden_states = res_hidden_states_list[-1]
                res_hidden_states_list = res_hidden_states_list[:-1]
                hidden_states = torch.cat(
                    [hidden_states, res_hidden_states], dim=1
                )

            hidden_states = resnet(hidden_states, emb)

        if self.upsampler is not None:
            hidden_states = self.upsampler(hidden_states, emb)

        return hidden_states


class AttnUpBlock(torch.nn.Module):
    """
    Unet's up block containing resnet, upsamplers, and self-attention blocks.
    """

    def __init__(
        self,
        spatial_dims: int,
        in_channels: int,
        prev_output_channel: int,
        out_channels: int,
        class_emb_channels: int,
        num_res_blocks: int = 1,
        norm_num_groups: int = 32,
        norm_eps: float = 1e-6,
        add_upsample: bool = True,
        resblock_updown: bool = False,
        num_head_channels: int = 1,
        use_flash_attention: bool = False,
        no_skip_connection: bool = False,
    ) -> None:
        """
        Args:
            spatial_dims (int): The number of spatial dimensions.
            in_channels (int): number of input channels.
            prev_output_channel (int): number of channels from residual
                connection.
            out_channels (int): number of output channels.
            class_emb_channels (int): number of class embedding channels.
            num_res_blocks (int, optional): number of residual blocks. Defaults
                to 1.
            norm_num_groups (int, optional): number of groups for the group
                normalization. Defaults to 32.
            norm_eps (float, optional): epsilon for the group normalization.
                Defaults to 1e-6.
            add_upsample (bool, optional): if True add upsample block. Defaults
                to True.
            resblock_updown (bool, optional): if True use residual blocks for
                downsampling. Defaults to False.
            num_head_channels (int, optional): number of channels in each
                attention head. Defaults to 1.
            use_flash_attention (bool, optional): if True, use flash attention
                for a memory efficient attention mechanism. Defaults to False.
        """
        super().__init__()
        self.resblock_updown = resblock_updown
        self.no_skip_connection = no_skip_connection

        resnets = []
        attentions = []

        for i in range(num_res_blocks):
            res_skip_channels = (
                in_channels if (i == num_res_blocks - 1) else out_channels
            )
            if self.no_skip_connection:
                res_skip_channels = 0
            resnet_in_channels = (
                prev_output_channel if i == 0 else out_channels
            )

            resnets.append(
                ResnetBlock(
                    spatial_dims=spatial_dims,
                    in_channels=resnet_in_channels + res_skip_channels,
                    out_channels=out_channels,
                    class_emb_channels=class_emb_channels,
                    norm_num_groups=norm_num_groups,
                    norm_eps=norm_eps,
                )
            )
            attentions.append(
                AttentionBlock(
                    spatial_dims=spatial_dims,
                    num_channels=out_channels,
                    num_head_channels=num_head_channels,
                    norm_num_groups=norm_num_groups,
                    norm_eps=norm_eps,
                    use_flash_attention=use_flash_attention,
                )
            )

        self.resnets = torch.nn.ModuleList(resnets)
        self.attentions = torch.nn.ModuleList(attentions)

        if add_upsample:
            if resblock_updown:
                self.upsampler = ResnetBlock(
                    spatial_dims=spatial_dims,
                    in_channels=out_channels,
                    out_channels=out_channels,
                    class_emb_channels=class_emb_channels,
                    norm_num_groups=norm_num_groups,
                    norm_eps=norm_eps,
                    up=True,
                )
            else:
                self.upsampler = Upsample(
                    spatial_dims=spatial_dims,
                    num_channels=out_channels,
                    use_conv=True,
                    out_channels=out_channels,
                )
        else:
            self.upsampler = None

    def forward(
        self,
        hidden_states: torch.Tensor,
        emb: torch.Tensor,
        res_hidden_states_list: list[torch.Tensor] | None = None,
        context: torch.Tensor | None = None,
    ) -> torch.Tensor:
        del context
        for resnet, attn in zip(self.resnets, self.attentions):
            # pop res hidden states
            if self.no_skip_connection is False:
                res_hidden_states = res_hidden_states_list[-1]
                res_hidden_states_list = res_hidden_states_list[:-1]
                hidden_states = torch.cat(
                    [hidden_states, res_hidden_states], dim=1
                )

            hidden_states = resnet(hidden_states, emb)
            hidden_states = attn(hidden_states)

        if self.upsampler is not None:
            hidden_states = self.upsampler(hidden_states, emb)

        return hidden_states


class CrossAttnUpBlock(torch.nn.Module):
    """
    Unet's up block containing resnet, upsamplers, and self-attention blocks.
    """

    def __init__(
        self,
        spatial_dims: int,
        in_channels: int,
        prev_output_channel: int,
        out_channels: int,
        class_emb_channels: int,
        num_res_blocks: int = 1,
        norm_num_groups: int = 32,
        norm_eps: float = 1e-6,
        add_upsample: bool = True,
        resblock_updown: bool = False,
        num_head_channels: int = 1,
        transformer_num_layers: int = 1,
        cross_attention_dim: int | None = None,
        upcast_attention: bool = False,
        use_flash_attention: bool = False,
        no_skip_connection: bool = False,
    ) -> None:
        """
        Args:
            spatial_dims (int): The number of spatial dimensions.
            in_channels (int): number of input channels.
            prev_output_channel (int): number of channels from residual
                connection.
            out_channels (int): number of output channels.
            class_emb_channels (int): number of class embedding channels.
            num_res_blocks (int, optional): number of residual blocks. Defaults
                to 1.
            norm_num_groups (int, optional): number of groups for the group
                normalization. Defaults to 32.
            norm_eps (float, optional): epsilon for the group normalization.
                Defaults to 1e-6.
            add_upsample (bool, optional): if True add upsample block. Defaults
                to True.
            resblock_updown (bool, optional): if True use residual blocks for
                downsampling. Defaults to False.
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
        """
        super().__init__()
        self.resblock_updown = resblock_updown
        self.no_skip_connection = no_skip_connection

        resnets = []
        attentions = []

        for i in range(num_res_blocks):
            res_skip_channels = (
                in_channels if (i == num_res_blocks - 1) else out_channels
            )
            if self.no_skip_connection:
                res_skip_channels = 0
            resnet_in_channels = (
                prev_output_channel if i == 0 else out_channels
            )

            resnets.append(
                ResnetBlock(
                    spatial_dims=spatial_dims,
                    in_channels=resnet_in_channels + res_skip_channels,
                    out_channels=out_channels,
                    class_emb_channels=class_emb_channels,
                    norm_num_groups=norm_num_groups,
                    norm_eps=norm_eps,
                )
            )
            attentions.append(
                SpatialTransformer(
                    spatial_dims=spatial_dims,
                    in_channels=out_channels,
                    num_attention_heads=out_channels // num_head_channels,
                    num_head_channels=num_head_channels,
                    norm_num_groups=norm_num_groups,
                    norm_eps=norm_eps,
                    num_layers=transformer_num_layers,
                    cross_attention_dim=cross_attention_dim,
                    upcast_attention=upcast_attention,
                    use_flash_attention=use_flash_attention,
                )
            )

        self.attentions = torch.nn.ModuleList(attentions)
        self.resnets = torch.nn.ModuleList(resnets)

        if add_upsample:
            if resblock_updown:
                self.upsampler = ResnetBlock(
                    spatial_dims=spatial_dims,
                    in_channels=out_channels,
                    out_channels=out_channels,
                    class_emb_channels=class_emb_channels,
                    norm_num_groups=norm_num_groups,
                    norm_eps=norm_eps,
                    up=True,
                )
            else:
                self.upsampler = Upsample(
                    spatial_dims=spatial_dims,
                    num_channels=out_channels,
                    use_conv=True,
                    out_channels=out_channels,
                )
        else:
            self.upsampler = None

    def forward(
        self,
        hidden_states: torch.Tensor,
        emb: torch.Tensor,
        res_hidden_states_list: list[torch.Tensor] | None = None,
        context: torch.Tensor | None = None,
    ) -> torch.Tensor:
        for resnet, attn in zip(self.resnets, self.attentions):
            # pop res hidden states
            if self.no_skip_connection is False:
                res_hidden_states = res_hidden_states_list[-1]
                res_hidden_states_list = res_hidden_states_list[:-1]
                hidden_states = torch.cat(
                    [hidden_states, res_hidden_states], dim=1
                )

            hidden_states = resnet(hidden_states, emb)
            hidden_states = attn(hidden_states, context=context)

        if self.upsampler is not None:
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
    add_downsample: bool,
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
        add_downsample (bool, optional): if True add downsample block. Defaults
            to True.
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
            add_downsample=add_downsample,
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
            add_downsample=add_downsample,
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
            add_downsample=add_downsample,
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
    add_upsample: bool,
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
    Args:
        spatial_dims (int): The number of spatial dimensions.
        in_channels (int): number of input channels.
        prev_output_channel (int): number of channels from residual
            connection.
        out_channels (int): number of output channels.
        class_emb_channels (int): number of class embedding channels.
        num_res_blocks (int, optional): number of residual blocks. Defaults
            to 1.
        norm_num_groups (int, optional): number of groups for the group
            normalization. Defaults to 32.
        norm_eps (float, optional): epsilon for the group normalization.
            Defaults to 1e-6.
        add_upsample (bool, optional): if True add upsample block. Defaults
            to True.
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
        no_skip_connection (bool): if True does not consider prev_output_channel
            features.

    Returns:
        torch.nn.Module: downsampling block.
    """
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
            add_upsample=add_upsample,
            resblock_updown=resblock_updown,
            num_head_channels=num_head_channels,
            use_flash_attention=use_flash_attention,
            no_skip_connection=no_skip_connection,
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
            add_upsample=add_upsample,
            resblock_updown=resblock_updown,
            num_head_channels=num_head_channels,
            transformer_num_layers=transformer_num_layers,
            cross_attention_dim=cross_attention_dim,
            upcast_attention=upcast_attention,
            use_flash_attention=use_flash_attention,
            no_skip_connection=no_skip_connection,
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
            add_upsample=add_upsample,
            resblock_updown=resblock_updown,
            no_skip_connection=no_skip_connection,
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
        num_head_channels: int | Sequence[int] = 8,
        with_conditioning: bool = False,
        transformer_num_layers: int = 1,
        cross_attention_dim: int | None = None,
        num_class_embeds: int | None = None,
        upcast_attention: bool = False,
        use_flash_attention: bool = False,
        no_skip_connection: bool = False,
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
            (out_channel % norm_num_groups) != 0
            for out_channel in num_channels
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
            num_res_blocks = ensure_tuple_rep(
                num_res_blocks, len(num_channels)
            )

        if len(num_res_blocks) != len(num_channels):
            raise ValueError(
                "`num_res_blocks` should be a single integer or a tuple of integers with the same length as "
                "`num_channels`."
            )

        if use_flash_attention and not has_xformers:
            raise ValueError(
                "use_flash_attention is True but xformers is not installed."
            )

        if use_flash_attention is True and not torch.cuda.is_available():
            raise ValueError(
                "torch.cuda.is_available() should be True but is False. Flash attention is only available for GPU."
            )

        self.in_channels = in_channels
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
            is_final_block = i == len(num_channels) - 1

            down_block = get_down_block(
                spatial_dims=spatial_dims,
                in_channels=input_channel,
                out_channels=output_channel,
                class_emb_channels=class_embedding_dim,
                num_res_blocks=num_res_blocks[i],
                norm_num_groups=norm_num_groups,
                norm_eps=norm_eps,
                add_downsample=not is_final_block,
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

            is_final_block = i == len(num_channels) - 1

            up_block = get_up_block(
                spatial_dims=spatial_dims,
                in_channels=input_channel,
                prev_output_channel=prev_output_channel,
                out_channels=output_channel,
                class_emb_channels=class_embedding_dim,
                num_res_blocks=reversed_num_res_blocks[i] + 1,
                norm_num_groups=norm_num_groups,
                norm_eps=norm_eps,
                add_upsample=not is_final_block,
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
        self.out_activation = torch.nn.Tanh()

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
        for upsample_block in self.up_blocks:
            res_samples = down_block_res_samples[
                -len(upsample_block.resnets) :
            ]
            down_block_res_samples = down_block_res_samples[
                : -len(upsample_block.resnets)
            ]
            if self.no_skip_connection:
                res_samples = None
            h = upsample_block(
                hidden_states=h,
                emb=class_emb,
                res_hidden_states_list=res_samples,
                context=context,
            )

        # 6. output block
        out_features = h
        h = self.out(h)
        h = self.out_activation(h)

        if return_features is True:
            return h, out_features
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
