"""
Codebase for modifiable input module U-Net (MIMU-Net).
"""

import torch
import einops
import numpy as np
from .unet import crop_to_size
from ..layers.adn_fn import get_adn_fn
from ..layers.res_blocks import ResidualBlock3d
from typing import List, Callable, Any


def is_if_none(a: Any, b: Any) -> Any:
    if a is None:
        return b
    return a


def zeros_like(X: torch.Tensor, shape: List[int]):
    X_shape = X.shape
    for i in range(len(shape)):
        shape[i] = shape[i] if shape[i] != -1 else X_shape[i]
    return torch.zeros(shape).to(X)


class MIMUNet(torch.nn.Module):
    """
    Modifiable input module U-Net (MIMU-Net).
    """

    def __init__(
        self,
        module: torch.nn.Module,
        n_classes: int,
        depth: List[int] = None,
        padding: List[int] = None,
        adn_fn: Callable = get_adn_fn(3, "instance", "relu", 0.1),
        n_channels: int = None,
        n_slices: int = None,
        deep_supervision: bool = False,
        upscale_type: str = "upsample",
        link_type: str = "conv",
    ):
        """
        Args:
            module (torch.nn.Module): end-to-end module that takes a batched
                set of 2D images and output a vector for each image.
            depth (int): output sizes for module.
            n_classes (int): number of output classes.
            adn_fn (Callable, optional): ADN function for the feature extraction
                module. Defaults to get_adn_fn( 1,"layer","gelu",0.1).
            n_channels (int, optional): number of channels. Defaults to None.
            n_slices (int, optional): number of slices. Defaults to None.
            deep_supervision (bool, optional): forward method returns
                segmentation predictions obtained from each decoder block.
            link_type (str, optional): link type for the skip connections.
                Can be a regular convolution ("conv"), residual block ("residual) or
                the identity ("identity"). Defaults to "identity".
        """

        super().__init__()
        self.module = module
        self.n_classes = n_classes
        self.depth = depth
        self.padding = padding
        self.adn_fn = adn_fn
        self.n_channels = n_channels
        self.n_slices = n_slices
        self.deep_supervision = deep_supervision
        self.upscale_type = upscale_type
        self.link_type = link_type

        if any([self.depth is None, self.padding is None]):
            inf_par = self.infer_depth_padding_kernel_size(self.module)
            self.depth = is_if_none(self.depth, inf_par[0])
            self.padding = is_if_none(self.padding, inf_par[1])

        self.vol_to_slice = einops.layers.torch.Rearrange(
            "b c h w s -> (b s c) h w"
        )
        self.slice_to_vol = torch.nn.ModuleList(
            [
                einops.layers.torch.Rearrange(
                    "(b s c) D h w -> b (c D) h w s",
                    s=self.n_slices,
                    c=self.n_channels,
                    D=d,
                )
                for d in self.depth
            ]
        )

        self.init_feature_reduction()
        self.init_upscale_ops()
        self.init_link_ops()
        self.init_decoder()
        self.init_final_layer()

    @staticmethod
    def infer_depth_padding_kernel_size(
        module: torch.nn.Module, test_shape: List[int] = [1, 1, 256, 256]
    ):
        X = torch.randn(*test_shape).to(next(module.parameters()))
        output = module(X)
        depth = [x.shape[1] for x in output]
        diff_from_same = []
        prev = output[0].shape[2]
        for output_tensor in output[1:]:
            diff_from_same.append((prev // 2) - output_tensor.shape[2])
            prev = output_tensor.shape[2]
        minimum_kernel_size = [3 for _ in output]
        kernel_size = [
            x + y for x, y in zip(minimum_kernel_size, diff_from_same)
        ]
        padding = [
            0 if (pad > 1) else (ks // 2)
            for ks, pad in zip(kernel_size, diff_from_same)
        ]
        padding = [*padding, padding[-1]]
        return depth, padding, kernel_size

    def init_feature_reduction(self):
        if self.n_channels > 1:
            self.feature_reduction = torch.nn.ModuleList(
                [
                    torch.nn.Sequential(
                        torch.nn.Conv3d(d * self.n_channels, d, 1),
                        self.adn_fn(d),
                    )
                    for d in self.depth
                ]
            )
        else:
            self.feature_reduction = torch.nn.ModuleList(
                [
                    torch.nn.Sequential(torch.nn.Identity(), self.adn_fn(d))
                    for d in self.depth
                ]
            )

    def v_module(self, X: torch.Tensor) -> torch.Tensor:
        n = self.n_slices
        b, c, h, w, n_X = X.shape
        if n_X == self.n_slices:
            X = self.vol_to_slice(X).unsqueeze(
                1
            )  # unsqueeze a channel dimension
            X = self.module(X)
            output = [S(x) for x, S in zip(X, self.slice_to_vol)]
        else:  # so that inference runs for arbitrarily sized tensors
            output = None
            denominator = []
            slice_ranges = [
                slice(i, i + n) if (i + n) < n_X else slice(n_X - n, n_X)
                for i in range(0, n_X, self.n_slices)
            ]
            for sr in slice_ranges:
                sliced_X = X[..., sr]
                sliced_X = self.vol_to_slice(sliced_X).unsqueeze(1)
                sliced_X = self.module(sliced_X)
                sliced_X = [S(x) for x, S in zip(sliced_X, self.slice_to_vol)]
                if output is None:
                    output = [
                        zeros_like(sx, [-1, -1, -1, -1, n_X])
                        for sx in sliced_X
                    ]
                    denominator = [
                        zeros_like(sx, [-1, -1, -1, -1, n_X])
                        for sx in sliced_X
                    ]
                for i in range(len(output)):
                    output[i][..., sr] += sliced_X[i]
                    denominator[i][..., sr] += 1
            output = [out / den for out, den in zip(output, denominator)]
        return output

    def init_link_ops(self):
        """Initializes linking (skip) operations."""
        if self.link_type == "identity":
            self.link_ops = torch.nn.ModuleList(
                [torch.nn.Identity() for _ in self.depth[:-1]]
            )
        elif self.link_type == "conv":
            self.link_ops = torch.nn.ModuleList(
                [
                    torch.nn.Sequential(
                        torch.nn.Conv3d(d, d, 3, padding=p), self.adn_fn(d)
                    )
                    for d, p in zip(self.depth[-2::-1], self.padding[-2::-1])
                ]
            )
        elif self.link_type == "residual":
            self.link_ops = torch.nn.ModuleList(
                [
                    ResidualBlock3d(d, 3, out_channels=d, adn_fn=self.adn_fn)
                    for d in self.depth[-2::-1]
                ]
            )

    def init_decoder(self):
        """Initializes the decoder operations."""
        self.decoding_operations = torch.nn.ModuleList([])
        depths = self.depth[-2::-1]
        padding = self.padding[-2::-1]
        kernel_sizes = [3 for _ in depths]
        self.deep_supervision_ops = torch.nn.ModuleList([])
        for i in range(len(depths)):
            d, k, p = depths[i], kernel_sizes[i], padding[i]
            op = torch.nn.Sequential(
                torch.nn.Conv3d(d * 2, d, kernel_size=k, stride=1, padding=p),
                self.adn_fn(d),
            )
            self.decoding_operations.append(op)
            if self.deep_supervision is True:
                self.deep_supervision_ops.append(self.get_ds_final_layer(d))

    def init_upscale_ops(self):
        """Initializes upscaling operations."""
        depths_a = self.depth[:0:-1]
        depths_b = self.depth[-2::-1]
        self.strides = [2 for _ in self.depth]
        if self.upscale_type == "upsample":
            upscale_ops = [
                torch.nn.Sequential(
                    torch.nn.Conv3d(d1, d2, 1),
                    torch.nn.Upsample(
                        scale_factor=[s, s, 1], mode=self.interpolation
                    ),
                )
                for d1, d2, s in zip(
                    depths_a, depths_b, self.strides[::-1][1:]
                )
            ]
        elif self.upscale_type == "transpose":
            upscale_ops = []
            for d1, d2, s in zip(depths_a, depths_b, self.strides[::-1][1:]):
                if isinstance(s, int) is True:
                    s = [s, s, 1]
                p = [np.maximum(i - 2, 0) for i in s]
                p[2] = 0
                upscale_ops.append(
                    torch.nn.ConvTranspose3d(d1, d2, s, stride=s, padding=p)
                )
        self.upscale_ops = torch.nn.ModuleList(upscale_ops)

    def get_final_layer(self, d: int) -> torch.nn.Module:
        """Returns the final layer.

        Args:
            d (int): depth.

        Returns:
            torch.nn.Module: final classification layer.
        """
        op = torch.nn.Conv3d
        if self.n_classes > 2:
            return torch.nn.Sequential(
                op(d, d, 1),
                self.adn_fn(d),
                op(d, self.n_classes, 1),
                torch.nn.Softmax(dim=1),
            )
        else:
            # coherces to a binary classification problem rather than
            # to a multiclass problem with two classes
            return torch.nn.Sequential(
                op(d, d, 3, padding=1),
                self.adn_fn(d),
                op(d, 1, 1),
                torch.nn.Sigmoid(),
            )

    def get_ds_final_layer(self, d: int) -> torch.nn.Module:
        """Returns the final layer for deep supervision.

        Args:
            d (int): depth.

        Returns:
            torch.nn.Module: final classification layer.
        """
        op = torch.nn.Conv3d
        if self.n_classes > 2:
            return torch.nn.Sequential(
                op(d, d, 1),
                self.adn_fn(d),
                op(d, self.n_classes, 1),
                torch.nn.Softmax(dim=1),
            )
        else:
            # coherces to a binary classification problem rather than
            # to a multiclass problem with two classes
            return torch.nn.Sequential(
                op(d, d, 1), self.adn_fn(d), op(d, 1, 1), torch.nn.Sigmoid()
            )

    def init_final_layer(self):
        """Initializes the classification layer (simple linear layer)."""
        o = self.depth[0]
        self.final_layer = self.get_final_layer(o)

    def forward(self, X: torch.Tensor) -> torch.Tensor:
        """Forward pass for this class.

        Args:
            X (torch.Tensor)

        Returns:
            torch.Tensor
        """
        encoding_out = self.v_module(X)
        encoding_out = [
            op(x) for op, x in zip(self.feature_reduction, encoding_out)
        ]
        curr = encoding_out[-1]

        deep_outputs = []
        for i in range(len(self.decoding_operations)):
            op = self.decoding_operations[i]
            link_op = self.link_ops[i]
            up = self.upscale_ops[i]
            link_op_input = encoding_out[-i - 2]
            encoded = link_op(link_op_input)
            curr = up(curr)
            sh = list(curr.shape)[2:]
            sh2 = list(encoded.shape)[2:]
            if np.prod(sh) < np.prod(sh2):
                encoded = crop_to_size(encoded, sh)
            if np.prod(sh) > np.prod(sh2):
                curr = crop_to_size(curr, sh2)
            curr = torch.concat((curr, encoded), dim=1)
            curr = op(curr)
            deep_outputs.append(curr)

        curr = self.final_layer(curr)
        if self.deep_supervision is True:
            for i in range(len(deep_outputs)):
                o = deep_outputs[i]
                op = self.deep_supervision_ops[i]
                deep_outputs[i] = op(o)
            return curr, deep_outputs

        return curr
