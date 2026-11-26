"""
U-Net with an arbitrary pretrained encoder.

The encoder only needs to implement ``forward_with_intermediate`` and return a
``(bottleneck, [skip_1, ..., skip_n])`` tuple, where the skip tensors are
ordered from the highest to the lowest resolution.
"""

from __future__ import annotations

from itertools import pairwise

import torch
import torch.nn.functional as F

from adell_mri.modules.layers.vit import ViT
from adell_mri.modules.segmentation.unet import UNet


class EncoderUNet(UNet):
    """
    U-Net that consumes an arbitrary encoder returning intermediate feature
    maps.
    """

    def __init__(
        self,
        encoder: torch.nn.Module,
        encoder_in_channels: int | None = None,
        input_shape: list[int] | None = None,
        depth: list[int] | None = None,
        kernel_sizes: list[int] | None = None,
        encoder_config: dict | None = None,
        *args,
        **kwargs,
    ):
        """
        Args:
            encoder (torch.nn.Module): encoder implementing
                ``forward_with_intermediate``.
            encoder_in_channels (int, optional): number of channels expected by
                the encoder. An input adapter is created when this differs from
                ``in_channels``. Defaults to None (same as ``in_channels``).
            input_shape (list[int], optional): spatial shape used to probe the
                encoder. Defaults to None (inferred from the encoder or set to
                a sensible default).
            depth (list[int], optional): decoder channel widths, ordered from
                the highest to the lowest resolution. Defaults to None (the
                encoder channels are used).
            kernel_sizes (list[int], optional): decoder kernel sizes. Defaults
                to None (3 in every level).
            encoder_config (dict, optional): manifest describing the encoder.
                Stored in checkpoints so that the encoder can be rebuilt at
                inference time. Defaults to None.
            args: positional arguments forwarded to :class:`UNet`.
            kwargs: keyword arguments forwarded to :class:`UNet`.
        """
        super().__init__(*args, parent_class=True, **kwargs)

        self.encoder = encoder
        self.encoder_config = encoder_config
        self.encoder_in_channels = encoder_in_channels or self.in_channels
        self.probe_input_shape = input_shape
        if depth is not None:
            self.depth = list(depth)
        if kernel_sizes is not None:
            self.kernel_sizes = list(kernel_sizes)

        self._init_input_adapter()
        self._infer_encoder_levels()
        self._init_decoder_layers()
        self._init_channel_adapters()

    def _init_input_adapter(self) -> None:
        if self.encoder_in_channels == self.in_channels:
            self.input_adapter = torch.nn.Identity()
            return
        conv_op = self._conv_op()
        self.input_adapter = conv_op(
            self.in_channels, self.encoder_in_channels, 1
        )

    def _conv_op(self) -> type[torch.nn.Module]:
        if self.spatial_dimensions == 2:
            return torch.nn.Conv2d
        return torch.nn.Conv3d

    def _probe_shape(self) -> list[int]:
        if self.probe_input_shape is not None:
            return list(self.probe_input_shape)
        if isinstance(self.encoder, ViT):
            image_size = getattr(self.encoder.embedding, "image_size", None)
            if image_size is not None:
                return list(image_size)
        image_size = getattr(self.encoder, "image_size", None)
        if image_size is not None:
            return list(image_size)
        return [64 for _ in range(self.spatial_dimensions)]

    def _encoder_forward(
        self, X: torch.Tensor
    ) -> tuple[torch.Tensor, list[torch.Tensor]]:
        if isinstance(self.encoder, ViT):
            return self.encoder.forward_with_intermediate(
                X, n_levels=self.encoder_levels
            )
        return self.encoder.forward_with_intermediate(X)

    def _infer_encoder_levels(self) -> None:
        requested_levels = (
            len(self.depth) - 1 if getattr(self, "depth", None) else 4
        )
        self.encoder_levels = requested_levels
        was_training = self.encoder.training
        self.encoder.eval()
        with torch.no_grad():
            dummy = torch.zeros(
                1, self.encoder_in_channels, *self._probe_shape()
            )
            bottleneck, skips = self._encoder_forward(dummy)
        self.encoder.train(was_training)

        self.encoder_channels = [skip.shape[1] for skip in skips]
        self.encoder_spatial = [tuple(skip.shape[2:]) for skip in skips]
        self.bottleneck_channels = bottleneck.shape[1]
        self.bottleneck_spatial = tuple(bottleneck.shape[2:])
        self.input_stride = self._spatial_stride(
            self._probe_shape(), self.encoder_spatial[0]
        )

    def _spatial_stride(
        self, high: list[int] | tuple[int, ...], low: tuple[int, ...]
    ) -> list[int]:
        return [max(1, round(h / l)) for h, l in zip(high, low)]

    def _init_decoder_layers(self) -> None:
        n_levels = len(self.encoder_channels) + 1
        if getattr(self, "depth", None) is not None and (
            len(self.depth) == n_levels
        ):
            depth = list(self.depth)
        else:
            depth = [*self.encoder_channels, self.bottleneck_channels]
        self.depth = depth

        if getattr(self, "kernel_sizes", None) is not None and len(
            self.kernel_sizes
        ) == len(depth):
            self.kernel_sizes = list(self.kernel_sizes)
        else:
            self.kernel_sizes = [3 for _ in depth]

        spatial_levels = [
            *self.encoder_spatial,
            self.bottleneck_spatial,
        ]
        strides = [
            self._spatial_stride(high, low)
            for high, low in pairwise(spatial_levels)
        ]
        # ``UNet`` expects one stride entry per depth level even though the
        # deepest entry is not used for upsampling.
        self.strides = [*strides, [1 for _ in range(self.spatial_dimensions)]]

        self.get_norm_op()
        self.get_drop_op()
        self.get_conv_op()
        self.init_upscale_ops()
        self.init_link_ops()
        self.init_decoder()
        self.init_final_layer()
        if self.bottleneck_classification is True:
            self.init_bottleneck_classifier()
        if self.feature_conditioning is not None:
            self.init_feature_conditioning_operations()

    def _init_channel_adapters(self) -> None:
        conv_op = self._conv_op()
        adapters = []
        for in_channels, out_channels in zip(self.encoder_channels, self.depth):
            if in_channels == out_channels:
                adapters.append(torch.nn.Identity())
            else:
                adapters.append(conv_op(in_channels, out_channels, 1))
        self.encoder_adapters = torch.nn.ModuleList(adapters)

        if self.bottleneck_channels == self.depth[-1]:
            self.bottleneck_adapter = torch.nn.Identity()
        else:
            self.bottleneck_adapter = conv_op(
                self.bottleneck_channels, self.depth[-1], 1
            )

    def _interpolation_mode(self) -> str:
        mode = self.interpolation
        if self.spatial_dimensions == 3 and mode == "bilinear":
            return "trilinear"
        return mode

    def _upsample_to_input(
        self, X: torch.Tensor, target: torch.Size
    ) -> torch.Tensor:
        if list(X.shape[2:]) == list(target):
            return X
        mode = self._interpolation_mode()
        kwargs = {}
        if mode in {"linear", "bilinear", "bicubic", "trilinear"}:
            kwargs["align_corners"] = False
        return F.interpolate(X, size=target, mode=mode, **kwargs)

    def forward(
        self,
        X: torch.Tensor,
        X_skip_layer: torch.Tensor = None,
        X_feature_conditioning: torch.Tensor = None,
        return_features: bool = False,
        return_bottleneck: bool = False,
        return_logits: bool = False,
    ) -> torch.Tensor | tuple[torch.Tensor, ...]:
        """
        Forward pass.

        Args:
            X (torch.Tensor): input tensor.
            X_skip_layer (torch.Tensor, optional): skip conditioning tensor.
                Defaults to None.
            X_feature_conditioning (torch.Tensor, optional): feature
                conditioning tensor. Defaults to None.
            return_features (bool, optional): returns the decoder features
                alongside the output. Defaults to False.
            return_bottleneck (bool, optional): returns the bottleneck
                alongside the output. Defaults to False.
            return_logits (bool, optional): returns logits instead of
                probabilities. Defaults to False.

        Returns:
            torch.Tensor | tuple[torch.Tensor, ...]: the output tensor or a
                tuple with the requested auxiliary outputs.
        """
        if X_skip_layer is not None and len(X_skip_layer.shape) < len(X.shape):
            X_skip_layer = X_skip_layer.unsqueeze(1)

        if X_feature_conditioning is not None:
            X_feature_conditioning = X_feature_conditioning - self.f_mean
            X_feature_conditioning = X_feature_conditioning / self.f_std

        bottleneck, skips = self._encoder_forward(self.input_adapter(X))
        bottleneck = self.bottleneck_adapter(bottleneck)
        skips = [
            adapter(skip) for adapter, skip in zip(self.encoder_adapters, skips)
        ]

        if return_bottleneck is True:
            return None, None, bottleneck

        encoding_out = [*skips, bottleneck]
        curr, deep_outputs = self._forward_decoder(
            bottleneck,
            encoding_out,
            X_skip_layer=X_skip_layer,
            X_feature_conditioning=X_feature_conditioning,
        )
        final_features = curr

        target = X.shape[2:]
        needs_upsample = list(curr.shape[2:]) != list(target)
        if return_logits is True:
            if needs_upsample:
                curr = self.final_layer[:-1](curr)
                curr = self._upsample_to_input(curr, target)
            else:
                curr = self.final_layer[:-1](curr)
        else:
            if needs_upsample:
                curr = self.final_layer[:-1](curr)
                curr = self._upsample_to_input(curr, target)
                curr = self.final_layer[-1](curr)
            else:
                curr = self.final_layer(curr)

        if return_features is True:
            return curr, final_features, bottleneck

        if self.bottleneck_classification is True:
            bottleneck_features = bottleneck.flatten(start_dim=2).max(-1).values
            bn_out = self.bottleneck_classifier(bottleneck_features)
        else:
            bn_out = None

        if self.deep_supervision is True:
            for i in range(len(deep_outputs)):
                deep_outputs[i] = self.deep_supervision_ops[i](deep_outputs[i])
                deep_outputs[i] = self._upsample_to_input(
                    deep_outputs[i], target
                )
            return curr, bn_out, deep_outputs

        return curr, bn_out
