"""
Standard U-Net implementation.
"""

from typing import Dict, List

import numpy as np
import torch
import torch.nn.functional as F

from adell_mri.custom_types import ModuleList
from adell_mri.modules.layers.adn_fn import (
    ActDropNorm,
    get_adn_fn,
    norm_fn_dict,
)
from adell_mri.modules.layers.multi_resolution import (
    AtrousSpatialPyramidPooling2d,
    AtrousSpatialPyramidPooling3d,
)
from adell_mri.modules.layers.regularization import UOut
from adell_mri.modules.layers.res_blocks import ResidualBlock2d, ResidualBlock3d
from adell_mri.modules.layers.self_attention import (
    ConcurrentSqueezeAndExcite2d,
    ConcurrentSqueezeAndExcite3d,
    SelfAttentionBlock,
)
from adell_mri.modules.layers.utils import crop_to_size


class UNet(torch.nn.Module):
    """
    Standard U-Net [1] implementation. Features some useful additions
    such as residual links, different upsampling types, normalizations
    (batch or instance) and ropouts (dropout and U-out). This version of
    the U-Net has been implemented in such a way that it can be easily
    expanded.

    [1] https://www.nature.com/articles/s41592-018-0261-2
    [2] https://openaccess.thecvf.com/content_CVPR_2019/papers/Li_Understanding_the_Disharmony_Between_Dropout_and_Batch_Normalization_by_Variance_CVPR_2019_paper.pdf
    """

    def __init__(
        self,
        spatial_dimensions: int = 2,
        encoding_operations: List[ModuleList] = None,
        conv_type: str = "regular",
        link_type: str = "identity",
        upscale_type: str = "upsample",
        interpolation: str = "bilinear",
        norm_type: str = "batch",
        dropout_type: str = "dropout",
        padding: str = "same",
        dropout_param: float = 0.1,
        activation_fn: torch.nn.Module = torch.nn.PReLU,
        in_channels: int = 1,
        n_classes: int = 2,
        depth: list = [16, 32, 64],
        kernel_sizes: list = [3, 3, 3],
        strides: list = [2, 2, 2],
        bottleneck_classification: bool = False,
        skip_conditioning: int = None,
        feature_conditioning: int = None,
        feature_conditioning_params: Dict[str, torch.Tensor] = None,
        deep_supervision: bool = False,
        parent_class: bool = False,
        encoder_only: bool = False,
    ):
        """
        Args:
        spatial_dimensions (int, optional): number of dimensions for the
            input (not counting batch or channels). Defaults to 2.
        encoding_operations (List[ModuleList], optional): backbone operations
            (uses these rather than a standard U-Net encoder).
            Must be a list where each element is a list containing a
            convolutional operation and a downsampling operation.
        conv_type (str, optional): types of base convolutional operations.
            For now it supports regular convolutions ("regular"), residual
            convolutions ("resnet") and convolutions followed by squeeze
            and excite modules ("sae"). Defaults to "regular".
        link_type (str, optional): link type for the skip connections.
            Can be a regular convolution ("conv"), residual block ("residual) or
            the identity ("identity"). Defaults to "identity".
        upscale_type (str, optional): upscaling type for decoder. Can be
            regular interpolate upsampling ("upsample") or transpose
            convolutions ("transpose"). Defaults to "upsample".
        interpolation (str, optional): interpolation for the upsampling
            operation (if `upscale_type="upsample"`). Defaults to "bilinear".
        norm_type (str, optional): type of normalization. Can be batch
            normalization ("batch") or instance normalization ("instance").
            Defaults to "batch".
        dropout_type (str, optional): type of dropout. Can be either
            regular dropout ("dropout") or U-out [2] ("uout"). Defaults to
            "dropout".
        padding (str, optional): padding for convolutions. Should be either
            "same" or "valid". Defaults to "same".
        dropout_param (float, optional): parameter for dropout layers.
            Sets the dropout rate for "dropout" and beta for "uout". Defaults
            to 0.1.
        activation_fn (torch.nn.Module, optional): activation function to
            be applied after normalizing. Defaults to torch.nn.PReLU.
        in_channels (int, optional): number of channels in input. Defaults
            to 1.
        n_classes (int, optional): number of output classes. Defaults to 2.
        depth (list, optional): defines the depths of each layer of the
            U-Net (the decoder will be the opposite). Defaults to [16,32,64].
        kernel_sizes (list, optional): defines the kernels of each layer
            of the U-Net. Defaults to [3,3,3].
        strides (list, optional): defines the strides of each layer of the
            U-Net. Defaults to [2,2,2].
        bottleneck_classification (bool, optional): sets up a
            classification task using the channel-wise maximum of the
            bottleneck layer. Defaults to False.
        skip_conditioning (int, optional): assumes that the skip
            layers will be conditioned by an image provided as the
            second argument of forward. This parameter specifies the
            number of channels in that image. Useful if any priors
            (complementary segmentation masks) are available.
        feature_conditioning (int,optional): linearly transforms tabular
            features and adds them to each channel of the skip connections.
            Useful to include tabular features in the prediction algorithm.
            Defaults to None.
        feature_conditioning_params (Dict[str,torch.Tensor], optional):
            dictionary with keys "mean" and "std" to normalize the tabular
            features. Must be present if feature conditioning is used.
            Defaults to None.
        deep_supervision (bool, optional): forward method returns
            segmentation predictions obtained from each decoder block.
        parent_class (bool, optional): does not initialise any layer, only
            sets constants. Helpful for inheritance.
        encoder_only (bool, optional): makes only encoder.
        """

        super().__init__()
        self.spatial_dimensions = spatial_dimensions
        self.encoding_operations = encoding_operations
        self.conv_type = conv_type
        self.link_type = link_type
        self.upscale_type = upscale_type
        self.interpolation = interpolation
        self.norm_type = norm_type
        self.dropout_type = dropout_type
        self.padding = padding
        self.dropout_param = dropout_param
        self.activation_fn = activation_fn
        self.in_channels = in_channels
        self.n_classes = n_classes
        self.depth = depth
        self.kernel_sizes = kernel_sizes
        self.strides = strides
        self.bottleneck_classification = bottleneck_classification
        self.skip_conditioning = skip_conditioning
        self.feature_conditioning = feature_conditioning
        self.feature_conditioning_params = feature_conditioning_params
        self.deep_supervision = deep_supervision
        self.encoder_only = encoder_only

        if self.encoder_only is True:
            # initialize all layers
            self.get_norm_op()
            self.get_drop_op()

            self.get_conv_op()
            if self.encoding_operations is None:
                self.init_encoder()
            else:
                self.init_encoder_backbone()

        elif parent_class is False:
            # initialize all layers
            self.get_norm_op()
            self.get_drop_op()

            self.get_conv_op()
            if self.encoding_operations is None:
                self.init_encoder()
            else:
                self.init_encoder_backbone()
            self.init_upscale_ops()
            self.init_link_ops()
            self.init_decoder()
            self.init_final_layer()
            if self.bottleneck_classification is True:
                self.init_bottleneck_classifier()
            if self.feature_conditioning == 0:
                self.feature_conditioning = None
            if self.feature_conditioning is not None:
                self.init_feature_conditioning_operations()

    def get_norm_op(self):
        """
        Retrieves the normalization operation using `self.norm_type`."""
        if self.norm_type is None:
            self.norm_op = torch.nn.Identity

        self.norm_op = norm_fn_dict[self.norm_type][self.spatial_dimensions]

    def get_drop_op(self):
        """
        Retrieves the dropout operations using `self.dropout_type`."""
        if self.dropout_type is None:
            self.drop_op = torch.nn.Identity

        elif self.dropout_type == "dropout":
            self.drop_op = torch.nn.Dropout
        elif self.dropout_type == "uout":
            self.drop_op = UOut

    def get_conv_op(self):
        """
        Retrieves the convolutional operations using `self.conv_type`."""
        if self.spatial_dimensions == 2:
            if self.conv_type == "regular":
                self.conv_op_enc = self.conv_block_2d
                self.conv_op_dec = self.conv_block_2d
            elif self.conv_type == "depthwise":
                self.conv_op_enc = self.depthwise_conv_block_2d
                self.conv_op_dec = self.depthwise_conv_block_2d
            elif self.conv_type == "resnet":
                self.conv_op_enc = self.res_block_conv_2d
                self.conv_op_dec = self.conv_block_2d
            elif self.conv_type == "sae":
                self.conv_op_enc = self.sae_2d
                self.conv_op_dec = self.sae_2d
            elif self.conv_type == "asp":
                self.conv_op_enc = self.asp_2d
                self.conv_op_dec = self.sae_2d
        if self.spatial_dimensions == 3:
            if self.conv_type == "regular":
                self.conv_op_enc = self.conv_block_3d
                self.conv_op_dec = self.conv_block_3d
            elif self.conv_type == "depthwise":
                self.conv_op_enc = self.depthwise_conv_block_3d
                self.conv_op_dec = self.depthwise_conv_block_3d
            elif self.conv_type == "resnet":
                self.conv_op_enc = self.res_block_conv_3d
                self.conv_op_dec = self.conv_block_3d
            elif self.conv_type == "sae":
                self.conv_op_enc = self.sae_3d
                self.conv_op_dec = self.sae_3d
            elif self.conv_type == "asp":
                self.conv_op_enc = self.asp_3d
                self.conv_op_dec = self.sae_3d

    def conv_block_2d(
        self, in_d, out_d, kernel_size, stride=None, padding=None
    ):
        """
        Convenience wrapper for 2d convolutional block."""
        if padding is None:
            padding = 0
        if stride is None:
            stride = 1
        return torch.nn.Sequential(
            torch.nn.Conv2d(in_d, in_d, kernel_size, stride, padding),
            self.adn_fn(in_d),
            torch.nn.Conv2d(in_d, out_d, kernel_size, 1, padding),
        )

    def conv_block_3d(
        self, in_d, out_d, kernel_size, stride=None, padding=None
    ):
        """
        Convenience wrapper for 3d convolutional block."""
        if padding is None:
            padding = 0
        if stride is None:
            stride = 1
        return torch.nn.Sequential(
            torch.nn.Conv3d(in_d, in_d, kernel_size, stride, padding),
            self.adn_fn(in_d),
            torch.nn.Conv3d(in_d, out_d, kernel_size, 1, padding),
        )

    def depthwise_conv_block_2d(
        self, in_d, out_d, kernel_size, stride=None, padding=None
    ):
        """
        Convenience wrapper for 2d convolutional block."""
        if padding is None:
            padding = 0
        if stride is None:
            stride = 1
        return torch.nn.Sequential(
            torch.nn.Conv2d(
                in_d, in_d, kernel_size, stride, padding, groups=in_d
            ),
            self.adn_fn(in_d),
            torch.nn.Conv2d(in_d, out_d, 1, 1, padding),
        )

    def depthwise_conv_block_3d(
        self, in_d, out_d, kernel_size, stride=None, padding=None
    ):
        """
        Convenience wrapper for 3d convolutional block."""
        if padding is None:
            padding = 0
        if stride is None:
            stride = 1
        return torch.nn.Sequential(
            torch.nn.Conv3d(
                in_d, in_d, kernel_size, stride, padding, groups=in_d
            ),
            self.adn_fn(in_d),
            torch.nn.Conv3d(in_d, out_d, 1, 1, padding),
        )

    def res_block_conv_2d(
        self, in_d, out_d, kernel_size, stride=None, padding=None
    ):
        """
        Convenience wrapper for ResidualBlock2d."""
        if in_d > 32:
            inter_d = int(in_d)
        else:
            inter_d = None
        if stride is None:
            stride = 1
        if isinstance(stride, int):
            stride = [stride for _ in range(self.spatial_dimensions)]
        if isinstance(padding, int):
            padding = [padding for _ in range(self.spatial_dimensions)]
        if any([x > 1 for x in stride]):
            new_padding = []
            for p, s in zip(padding, stride):
                if p > s // 2:
                    new_padding.append(p // 2)
                else:
                    new_padding.append(p)
            return torch.nn.Sequential(
                ResidualBlock2d(
                    in_d, kernel_size, inter_d, out_d, adn_fn=self.adn_fn
                ),
                torch.nn.MaxPool2d(stride, stride, padding=new_padding),
            )
        else:
            return ResidualBlock2d(
                in_d, kernel_size, inter_d, out_d, adn_fn=self.adn_fn
            )

    def res_block_conv_3d(
        self, in_d, out_d, kernel_size, stride=None, padding=None
    ):
        """
        Convenience wrapper for ResidualBlock3d."""
        if in_d > 32:
            inter_d = int(in_d)
        else:
            inter_d = None
        if stride is None:
            stride = 1
        if isinstance(stride, int):
            stride = [stride for _ in range(self.spatial_dimensions)]
        if isinstance(padding, int):
            padding = [padding for _ in range(self.spatial_dimensions)]
        if any([x > 1 for x in stride]):
            new_padding = []
            for p, s in zip(padding, stride):
                if p > s // 2:
                    new_padding.append(p // 2)
                else:
                    new_padding.append(p)
            return torch.nn.Sequential(
                ResidualBlock3d(
                    in_d, kernel_size, inter_d, out_d, adn_fn=self.adn_fn
                ),
                torch.nn.MaxPool3d(stride, stride, padding=new_padding),
            )
        else:
            return ResidualBlock3d(
                in_d, kernel_size, inter_d, out_d, adn_fn=self.adn_fn
            )

    def sae_2d(self, in_d, out_d, kernel_size, stride=1, padding=0):
        return torch.nn.Sequential(
            self.conv_block_2d(
                in_d,
                out_d,
                kernel_size=kernel_size,
                stride=stride,
                padding=padding,
            ),
            ConcurrentSqueezeAndExcite2d(out_d),
        )

    def sae_3d(self, in_d, out_d, kernel_size, stride=None, padding=None):
        return torch.nn.Sequential(
            self.conv_block_3d(
                in_d,
                out_d,
                kernel_size=kernel_size,
                stride=stride,
                padding=padding,
            ),
            ConcurrentSqueezeAndExcite3d(out_d),
        )

    def asp_2d(self, in_d, out_d, kernel_size, stride=1, padding=0):
        return AtrousSpatialPyramidPooling2d(
            in_d,
            out_d,
            [1, 2],
            get_adn_fn(2, "instance", self.activation_fn, self.dropout_param),
        )

    def asp_3d(self, in_d, out_d, kernel_size, stride=None, padding=None):
        return AtrousSpatialPyramidPooling3d(
            in_d,
            out_d,
            [1, 2],
            get_adn_fn(3, "instance", self.activation_fn, self.dropout_param),
        )

    def init_upscale_ops(self):
        """
        Initializes upscaling operations."""
        depths_a = self.depth[:0:-1]
        depths_b = self.depth[-2::-1]
        if self.upscale_type == "upsample":
            if self.spatial_dimensions == 2:
                upscale_ops = [
                    torch.nn.Sequential(
                        torch.nn.Conv2d(d1, d2, 1),
                        torch.nn.Upsample(
                            scale_factor=s, mode=self.interpolation
                        ),
                    )
                    for d1, d2, s in zip(
                        depths_a, depths_b, self.strides[::-1][1:]
                    )
                ]
            if self.spatial_dimensions == 3:
                upscale_ops = [
                    torch.nn.Sequential(
                        torch.nn.Conv3d(d1, d2, 1),
                        torch.nn.Upsample(
                            scale_factor=s, mode=self.interpolation
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
                    s = [s for _ in range(self.spatial_dimensions)]
                p = [np.maximum(i - 2, 0) for i in s]
                if self.spatial_dimensions == 2:
                    upscale_ops.append(
                        torch.nn.ConvTranspose2d(d1, d2, s, stride=s, padding=p)
                    )
                if self.spatial_dimensions == 3:
                    upscale_ops.append(
                        torch.nn.ConvTranspose3d(d1, d2, s, stride=s, padding=p)
                    )
        self.upscale_ops = torch.nn.ModuleList(upscale_ops)

    def init_link_ops(self):
        """
        Initializes linking (skip) operations."""
        if self.skip_conditioning is not None:
            ex = self.skip_conditioning
        else:
            ex = 0
        rev_depth = self.depth[-2::-1]
        if self.link_type == "identity":
            self.link_ops = torch.nn.ModuleList(
                [torch.nn.Identity() for _ in self.depth[:-1]]
            )
        elif self.link_type == "attention":
            self.link_ops = torch.nn.ModuleList(
                [
                    SelfAttentionBlock(
                        self.spatial_dimensions, d, d, [16, 16, 1]
                    )
                    for d in self.depth[-2::-1]
                ]
            )
        elif self.link_type == "conv":
            if self.spatial_dimensions == 2:
                self.link_ops = torch.nn.ModuleList(
                    [
                        torch.nn.Sequential(
                            torch.nn.Conv2d(d + ex, d, 3, padding=self.padding),
                            self.adn_fn(d),
                        )
                        for d in rev_depth
                    ]
                )
            elif self.spatial_dimensions == 3:
                self.link_ops = torch.nn.ModuleList(
                    [
                        torch.nn.Sequential(
                            torch.nn.Conv3d(d + ex, d, 3, padding=self.padding),
                            self.adn_fn(d),
                        )
                        for d in rev_depth
                    ]
                )
        elif self.link_type == "residual":
            if self.spatial_dimensions == 2:
                self.link_ops = torch.nn.ModuleList(
                    [
                        ResidualBlock2d(
                            d + ex,
                            3,
                            out_channels=d,
                            adn_fn=self.adn_fn,
                        )
                        for d in rev_depth
                    ]
                )
            elif self.spatial_dimensions == 3:
                self.link_ops = torch.nn.ModuleList(
                    [
                        ResidualBlock3d(
                            d + ex,
                            3,
                            out_channels=d,
                            adn_fn=self.adn_fn,
                        )
                        for d in rev_depth
                    ]
                )

    def interpolate_depths(self, a: int, b: int, n=3) -> List[int]:
        """
                Interpolates between two whole numbers. Not really used.

        Args:
            a (int): start integer
            b (int): final integer
            n (int, optional): number of points. Defaults to 3.

        Returns:
            (List[int]): list of `n` integers sampled between `a` and `b`.
        """
        return list(np.linspace(a, b, n, dtype=np.int32))

    def init_encoder(self):
        """
        Initializes the encoder operations."""
        self.encoding_operations = torch.nn.ModuleList([])
        previous_d = self.in_channels
        for i in range(len(self.depth) - 1):
            d, k, s = self.depth[i], self.kernel_sizes[i], self.strides[i]
            if isinstance(s, int) is True:
                s = [s for _ in range(self.spatial_dimensions)]
            if isinstance(k, int) is True:
                k = [k for _ in range(self.spatial_dimensions)]
            p = [int(i // 2) for i in k]
            op = torch.nn.Sequential(
                self.conv_op_enc(
                    previous_d,
                    d,
                    kernel_size=k,
                    stride=1,
                    padding=self.padding,
                ),
                self.adn_fn(d),
            )
            op_downsample = torch.nn.Sequential(
                self.conv_op_enc(d, d, kernel_size=k, stride=s, padding=p),
                self.adn_fn(d),
            )
            self.encoding_operations.append(
                torch.nn.ModuleList([op, op_downsample])
            )
            previous_d = d
        op = torch.nn.Sequential(
            self.conv_op_enc(
                self.depth[-2],
                self.depth[-1],
                kernel_size=k,
                stride=1,
                padding=self.padding,
            ),
            self.adn_fn(self.depth[-1]),
        )
        op_downsample = torch.nn.Identity()
        self.encoding_operations.append(
            torch.nn.ModuleList([op, op_downsample])
        )

    def init_encoder_backbone(self):
        """
        Initializes the encoder operations."""
        if self.spatial_dimensions == 2:
            mp = torch.nn.MaxPool2d
        elif self.spatial_dimensions == 3:
            mp = torch.nn.MaxPool3d
        for i in range(len(self.encoding_operations)):
            s = self.strides[i]
            if isinstance(s, int):
                s = [s for _ in range(self.spatial_dimensions)]
            s = np.array(s)
            self.encoding_operations[i][1] = mp(
                kernel_size=tuple(s), stride=tuple(s), padding=tuple(s // 2)
            )
        self.encoding_operations[-1][1] = torch.nn.Identity()

    def init_decoder(self):
        """
        Initializes the decoder operations."""
        self.decoding_operations = torch.nn.ModuleList([])
        depths = self.depth[-2::-1]
        kernel_sizes = self.kernel_sizes[-2::-1]
        self.deep_supervision_ops = torch.nn.ModuleList([])
        for i in range(len(depths)):
            d, k = depths[i], kernel_sizes[i]
            if isinstance(k, int) is True:
                k = [k for _ in range(self.spatial_dimensions)]
            op = torch.nn.Sequential(
                self.conv_op_dec(
                    d * 2, d, kernel_size=k, stride=1, padding=self.padding
                ),
                self.adn_fn(d),
            )
            self.decoding_operations.append(op)
            if self.deep_supervision is True:
                self.deep_supervision_ops.append(self.get_ds_final_layer(d))

    def get_final_layer(self, d: int) -> torch.nn.Module:
        """
                Returns the final layer.

        Args:
            d (int): depth.

        Returns:
            torch.nn.Module: final classification layer.
        """
        if self.spatial_dimensions == 2:
            op = torch.nn.Conv2d
        elif self.spatial_dimensions == 3:
            op = torch.nn.Conv3d
        if self.n_classes > 2:
            return torch.nn.Sequential(
                op(d, d, 3, padding="same"),
                self.adn_fn(d),
                op(d, self.n_classes, 1),
                torch.nn.Softmax(dim=1),
            )
        else:
            # coherces to a binary classification problem rather than
            # to a multiclass problem with two classes
            return torch.nn.Sequential(
                op(d, d, 3, padding="same"),
                self.adn_fn(d),
                op(d, 1, 1),
                torch.nn.Sigmoid(),
            )

    def get_ds_final_layer(self, d: int) -> torch.nn.Module:
        """
                Returns the final layer for deep supervision.

        Args:
            d (int): depth.

        Returns:
            torch.nn.Module: final classification layer.
        """
        if self.spatial_dimensions == 2:
            op = torch.nn.Conv2d
        elif self.spatial_dimensions == 3:
            op = torch.nn.Conv3d
        if self.n_classes > 2:
            return torch.nn.Sequential(
                op(d, d, 3),
                self.adn_fn(d),
                op(d, self.n_classes, 1),
                torch.nn.Softmax(dim=1),
            )
        else:
            # coherces to a binary classification problem rather than
            # to a multiclass problem with two classes
            return torch.nn.Sequential(
                op(d, d, 3), self.adn_fn(d), op(d, 1, 1), torch.nn.Sigmoid()
            )

    def init_final_layer(self):
        """
        Initializes the classification layer (simple linear layer)."""
        o = self.depth[0]
        self.final_layer = self.get_final_layer(o)

    def init_bottleneck_classifier(self):
        """
        Initiates the layers for bottleneck classification."""
        nc = self.n_classes if self.n_classes > 2 else 1
        self.bottleneck_classifier = torch.nn.Linear(self.depth[-1], nc)

    def adn_fn(self, s: int) -> torch.Tensor:
        """
                Convenience wrapper for ADN function.

        Args:
            s (int): number of layers.

        Returns:
            torch.Tensor: ActDropNorm module
        """
        return ActDropNorm(
            in_channels=s,
            ordering="NDA",
            norm_fn=self.norm_op,
            act_fn=self.activation_fn,
            dropout_fn=self.drop_op,
            dropout_param=self.dropout_param,
        )

    def init_feature_conditioning_operations(self):
        depths = self.depth[-2::-1]
        self.feature_conditioning_ops = torch.nn.ModuleList([])
        if self.feature_conditioning_params is not None:
            self.f_mean = torch.nn.parameter.Parameter(
                self.feature_conditioning_params["mean"], requires_grad=False
            )
            self.f_std = torch.nn.parameter.Parameter(
                self.feature_conditioning_params["std"], requires_grad=False
            )
        else:
            self.f_mean = torch.nn.parameter.Parameter(
                torch.zeros([self.feature_conditioning], requires_grad=False)
            )
            self.f_std = torch.nn.parameter.Parameter(
                torch.ones([self.feature_conditioning], requires_grad=False)
            )
        for d in depths:
            op = torch.nn.Sequential(
                torch.nn.Linear(self.feature_conditioning, d),
                get_adn_fn(1, "batch", "swish", self.dropout_param)(d),
                torch.nn.Linear(d, d),
                get_adn_fn(1, "batch", "sigmoid", self.dropout_param)(d),
            )
            self.feature_conditioning_ops.append(op)

    def unsqueeze_to_dim(
        self, X: torch.Tensor, target: torch.Tensor
    ) -> torch.Tensor:
        n = len(target.shape) - len(X.shape)
        if n > 0:
            for _ in range(n):
                X = X.unsqueeze(-1)
        return X

    def forward(
        self,
        X: torch.Tensor,
        X_skip_layer: torch.Tensor = None,
        X_feature_conditioning: torch.Tensor = None,
        return_features=False,
        return_bottleneck=False,
        return_logits=False,
    ) -> torch.Tensor:
        """
                Forward pass for this class.

        Args:
            X (torch.Tensor)

        Returns:
            torch.Tensor
        """
        # check if channel dim is available and if not include it
        if X_skip_layer is not None:
            if len(X_skip_layer.shape) < len(X.shape):
                X_skip_layer = X_skip_layer.unsqueeze(1)

        # normalise features
        if X_feature_conditioning is not None:
            X_feature_conditioning = X_feature_conditioning - self.f_mean
            X_feature_conditioning = X_feature_conditioning / self.f_std

        encoding_out = []
        curr = X
        for op, op_ds in self.encoding_operations:
            curr = op(curr)
            encoding_out.append(curr)
            curr = op_ds(curr)
        bottleneck = curr
        if return_bottleneck is True:
            return None, None, bottleneck
        elif self.encoder_only is True:
            return bottleneck

        deep_outputs = []
        for i in range(len(self.decoding_operations)):
            op = self.decoding_operations[i]
            link_op = self.link_ops[i]
            up = self.upscale_ops[i]
            if X_skip_layer is not None:
                S = encoding_out[-i - 2].shape[2:]
                xfl = F.interpolate(X_skip_layer, S, mode="nearest")
                link_op_input = torch.cat([encoding_out[-i - 2], xfl], axis=1)
            else:
                link_op_input = encoding_out[-i - 2]
            encoded = link_op(link_op_input)
            if X_feature_conditioning is not None:
                feat_op = self.feature_conditioning_ops[i]
                transformed_features = feat_op(X_feature_conditioning)
                transformed_features = self.unsqueeze_to_dim(
                    transformed_features, encoded
                )
                encoded = torch.multiply(encoded, transformed_features)
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

        final_features = curr

        if return_logits is True:
            curr = self.final_layer[:-1](curr)
        else:
            curr = self.final_layer(curr)
        if return_features is True:
            return curr, final_features, bottleneck

        if self.bottleneck_classification is True:
            bottleneck = bottleneck.flatten(start_dim=2).max(-1).values
            bn_out = self.bottleneck_classifier(bottleneck)
        else:
            bn_out = None

        if self.deep_supervision is True:
            for i in range(len(deep_outputs)):
                o = deep_outputs[i]
                op = self.deep_supervision_ops[i]
                deep_outputs[i] = op(o)
            return curr, bn_out, deep_outputs

        return curr, bn_out


class BrUNet(UNet, torch.nn.Module):
    """
    BrUNet - UNet module supporting multiple inputs. Rather than
    constructing a multi-channel input, we process each channel separately
    and merge them before applying link operations and at the end of the
    encoder. This allows us to pre-train different encoders in situations
    where data is not equally available for all channels.

    BrUNet merges the input by applying squeeze and excite (SAE) operations
    at each merging point and summing them. SAE are self-attention
    mechanisms, and here they serve as a way of performing a weighted sum of
    each input.

    [1] https://www.nature.com/articles/s41592-018-0261-2
    [2] https://openaccess.thecvf.com/content_CVPR_2019/papers/Li_Understanding_the_Disharmony_Between_Dropout_and_Batch_Normalization_by_Variance_CVPR_2019_paper.pdf
    """

    def __init__(
        self,
        spatial_dimensions: int = 2,
        n_input_branches: int = 1,
        encoders: List[ModuleList] = None,
        conv_type: str = "regular",
        link_type: str = "identity",
        upscale_type: str = "upsample",
        interpolation: str = "bilinear",
        norm_type: str = "batch",
        dropout_type: str = "dropout",
        padding: str = "same",
        dropout_param: float = 0.1,
        activation_fn: torch.nn.Module = torch.nn.PReLU,
        in_channels: int = 1,
        n_classes: int = 2,
        depth: list = [16, 32, 64],
        kernel_sizes: list = [3, 3, 3],
        strides: list = [2, 2, 2],
        bottleneck_classification: bool = False,
        skip_conditioning: int = None,
        feature_conditioning: int = None,
        feature_conditioning_params: Dict[str, torch.Tensor] = None,
        deep_supervision: bool = False,
        encoder_only: bool = False,
    ) -> torch.nn.Module:
        """
        Args:
        spatial_dimensions (int, optional): number of dimensions for the
            input (not counting batch or channels). Defaults to 2.
        encoder (List[ModuleList], optional): a list of backbone operations
            which will be executed in parallel and whose output at each
            layer must be conformant (have the same channels).
        conv_type (str, optional): types of base convolutional operations.
            For now it supports regular convolutions ("regular"), residual
            convolutions ("resnet") and convolutions followed by squeeze
            and excite modules ("sae"). Defaults to "regular".
        link_type (str, optional): link type for the skip connections.
            Can be a regular convolution ("conv"), residual block ("residual) or
            the identity ("identity"). Defaults to "identity".
        upscale_type (str, optional): upscaling type for decoder. Can be
            regular interpolate upsampling ("upsample") or transpose
            convolutions ("transpose"). Defaults to "upsample".
        interpolation (str, optional): interpolation for the upsampling
            operation (if `upscale_type="upsample"`). Defaults to "bilinear".
        norm_type (str, optional): type of normalization. Can be batch
            normalization ("batch") or instance normalization ("instance").
            Defaults to "batch".
        dropout_type (str, optional): type of dropout. Can be either
            regular dropout ("dropout") or U-out [2] ("uout"). Defaults to
            "dropout".
        padding (str, optional): padding for convolutions. Should be either
            "same" or "valid". Defaults to "same".
        dropout_param (float, optional): parameter for dropout layers.
            Sets the dropout rate for "dropout" and beta for "uout". Defaults
            to 0.1.
        activation_fn (torch.nn.Module, optional): activation function to
            be applied after normalizing. Defaults to torch.nn.PReLU.
        in_channels (int, optional): number of channels in input. Defaults
            to 1.
        n_classes (int, optional): number of output classes. Defaults to 2.
        depth (list, optional): defines the depths of each layer of the
            U-Net (the decoder will be the opposite). Defaults to [16,32,64].
        kernel_sizes (list, optional): defines the kernels of each layer
            of the U-Net. Defaults to [3,3,3].
        strides (list, optional): defines the strides of each layer of the
            U-Net. Defaults to [2,2,2].
        bottleneck_classification (bool, optional): sets up a
            classification task using the channel-wise maximum of the
            bottleneck layer. Defaults to False.
        skip_conditioning (int, optional): assumes that the skip
            layers will be conditioned by an image provided as the
            second argument of forward. This parameter specifies the
            number of channels in that image. Useful if any priors
            (complementary segmentation masks) are available.
        feature_conditioning (int,optional): linearly transforms tabular
            features and adds them to each channel of the skip connections.
            Useful to include tabular features in the prediction algorithm.
            Defaults to None.
        feature_conditioning_params (Dict[str,torch.Tensor], optional):
            dictionary with keys "mean" and "std" to normalize the tabular
            features. Must be present if feature conditioning is used.
            Defaults to None.
        deep_supervision (bool, optional): forward method returns
            segmentation predictions obtained from each decoder block.
        encoder_only (bool, optional): makes only encoders.
        """

        super().__init__(parent_class=True)
        self.spatial_dimensions = spatial_dimensions
        self.n_input_branches = n_input_branches
        self.encoders = encoders
        self.conv_type = conv_type
        self.link_type = link_type
        self.upscale_type = upscale_type
        self.interpolation = interpolation
        self.norm_type = norm_type
        self.dropout_type = dropout_type
        self.padding = padding
        self.dropout_param = dropout_param
        self.activation_fn = activation_fn
        self.in_channels = in_channels
        self.n_classes = n_classes
        self.depth = depth
        self.kernel_sizes = kernel_sizes
        self.strides = strides
        self.bottleneck_classification = bottleneck_classification
        self.skip_conditioning = skip_conditioning
        self.feature_conditioning = feature_conditioning
        self.feature_conditioning_params = feature_conditioning_params
        self.deep_supervision = deep_supervision
        self.encoder_only = encoder_only

        # initialize all layers
        self.get_norm_op()
        self.get_drop_op()

        self.get_conv_op()
        if self.encoders is None:
            self.init_encoders()
        else:
            self.init_backbone_encoders()
        self.init_merge_ops()

        if self.encoder_only is not True:
            self.init_upscale_ops()
            self.init_link_ops()
            self.init_decoder()
            self.init_final_layer()
            if self.bottleneck_classification is True:
                self.init_bottleneck_classifier()
            if self.feature_conditioning is not None:
                self.init_feature_conditioning_operations()

    def init_encoders(self):
        """
        Initializes the encoder operations."""
        self.encoders = torch.nn.ModuleList([])
        for _ in range(self.n_input_branches):
            encoding_operations = torch.nn.ModuleList([])
            previous_d = self.in_channels
            for i in range(len(self.depth) - 1):
                d, k, s = self.depth[i], self.kernel_sizes[i], self.strides[i]
                if isinstance(s, int) is True:
                    s = [s for _ in range(self.spatial_dimensions)]
                if isinstance(k, int) is True:
                    k = [k for _ in range(self.spatial_dimensions)]
                p = [int(i // 2) for i in k]
                op = torch.nn.Sequential(
                    self.conv_op_enc(
                        previous_d,
                        d,
                        kernel_size=k,
                        stride=1,
                        padding=self.padding,
                    ),
                    self.adn_fn(d),
                )
                op_downsample = torch.nn.Sequential(
                    self.conv_op_enc(d, d, kernel_size=k, stride=s, padding=p),
                    self.adn_fn(d),
                )
                encoding_operations.append(
                    torch.nn.ModuleList([op, op_downsample])
                )
                previous_d = d
            op = torch.nn.Sequential(
                self.conv_op_enc(
                    self.depth[-2],
                    self.depth[-1],
                    kernel_size=k,
                    stride=1,
                    padding=self.padding,
                ),
                self.adn_fn(self.depth[-1]),
            )
            op_downsample = torch.nn.Identity()
            encoding_operations.append(torch.nn.ModuleList([op, op_downsample]))
            self.encoders.append(encoding_operations)

    def init_backbone_encoders(self):
        """
        Initializes the encoder operations."""
        n_encoders = len(self.encoders)
        assert (
            n_encoders == self.n_input_branches
        ), "n_input_branches and \
            len(self.encoders) must be the same"
        if self.spatial_dimensions == 2:
            mp = torch.nn.MaxPool2d
        elif self.spatial_dimensions == 3:
            mp = torch.nn.MaxPool3d
        for i in range(self.n_input_branches):
            for j in range(len(self.encoders[i])):
                self.encoders[i][j][1] = mp(
                    self.kernel_sizes[j], 2, self.kernel_sizes[j] // 2
                )
            self.encoders[i][-1][1] = torch.nn.Identity()

    def init_merge_ops(self):
        """
        Initializes operations that merge the outputs of each branch.
        For this we use concurrent squeeze and excite layers as they allow
        us to learn a self-attention mechanism that weighs different inputs
        in both spatial and channel terms.
        """
        if self.encoder_only is True:
            D = [self.depth[-1]]
        else:
            D = self.depth
        if self.spatial_dimensions == 2:
            self.merge_ops = torch.nn.ModuleList(
                [
                    torch.nn.ModuleList(
                        [
                            ConcurrentSqueezeAndExcite2d(d)
                            for _ in range(self.n_input_branches)
                        ]
                    )
                    for d in D
                ]
            )
        elif self.spatial_dimensions == 3:
            self.merge_ops = torch.nn.ModuleList(
                [
                    torch.nn.ModuleList(
                        [
                            ConcurrentSqueezeAndExcite3d(d)
                            for _ in range(self.n_input_branches)
                        ]
                    )
                    for d in D
                ]
            )

    @staticmethod
    def fix_input(X: List[List[torch.Tensor]]):
        shapes = [[] for _ in X]
        for i, item_list in enumerate(X):
            for x in item_list:
                if x is not None:
                    shapes[i].append(x.shape)
        shapes = [list(set(sh)) for sh in shapes]
        assert all(
            [len(sh) == 1 for sh in shapes]
        ), "all tensors must have the same shape"
        shapes = [sh[0] for sh in shapes]
        weights = [torch.ones(len(X[0])) for _ in X]
        for i in range(len(X)):
            for j in range(len(X[i])):
                if X[i][j] is None:
                    X[i][j] = torch.zeros(shapes[i])
                    weights[i][j] = 0.0
        X = [torch.stack(x, 0) for x in X]
        return X, weights

    def forward(
        self,
        X: List[torch.Tensor],
        X_weights: List[torch.Tensor] = None,
        X_skip_layer: torch.Tensor = None,
        X_feature_conditioning: torch.Tensor = None,
        return_features=False,
        return_bottleneck=False,
        return_logits=False,
    ) -> torch.Tensor:
        """
        Forward pass for this class.

        Args:
            X (List[torch.Tensor]): list of tensors.
            X_weights (List[torch.Tensor],optional): should be the same size as X and
                each element of X_weights should have size b, where b is the
                batch size of each element of X. Defaults to None (no weights).

        Returns:
            torch.Tensor
        """
        if X_weights is None:
            X_weights = [torch.ones(X[0].shape[0]).to(X[0].device) for _ in X]
        assert len(X) == len(
            X_weights
        ), "X and X_weights should have identical length"
        assert all(
            [x.shape[0] == xw.shape[0] for x, xw in zip(X, X_weights)]
        ), "The elements of X and X_weights should have identical batch sizes"
        # check if channel dim is available and if not include it
        if X_skip_layer is not None:
            if len(X_skip_layer.shape) < len(X[0].shape):
                X_skip_layer = X_skip_layer.unsqueeze(1)

        # normalise features
        if X_feature_conditioning is not None:
            X_feature_conditioning = X_feature_conditioning - self.f_mean
            X_feature_conditioning = X_feature_conditioning / self.f_std

        encoding_out_pre_merge = [[] for _ in self.encoders]
        bottleneck_features_pre_merge = []
        for i in range(self.n_input_branches):
            curr = X[i]
            w = self.unsqueeze_to_dim(X_weights[i], curr)
            encoding_operations = self.encoders[i]
            for op, op_ds in encoding_operations:
                curr = op(curr)
                encoding_out_pre_merge[i].append(curr * w)
                curr = op_ds(curr)
            bottleneck_features_pre_merge.append(curr * w)
        w_sum = self.unsqueeze_to_dim(sum(X_weights), X[0])

        # start by merging bottleneck with the last merge operation and
        # redefine curr
        bottleneck = sum(
            [
                self.merge_ops[-1][j](bottleneck_features_pre_merge[j]) / w_sum
                for j in range(self.n_input_branches)
            ]
        )
        if self.encoder_only is True:
            return bottleneck
        if return_bottleneck is True:
            return None, None, bottleneck
        curr = bottleneck

        # merge outputs from different branches as the weighted sum of each
        # for the skip connections (link_ops).
        # the rest of the U-Net is essentially the same now that the inputs
        # are merged.
        encoding_out = []
        for i, tensors in enumerate(zip(*encoding_out_pre_merge)):
            merge_op = self.merge_ops[i]
            merged_tensor = sum(
                [
                    merge_op[j](tensors[j]) / w_sum
                    for j in range(self.n_input_branches)
                ]
            )
            encoding_out.append(merged_tensor)

        deep_outputs = []
        for i in range(len(self.decoding_operations)):
            op = self.decoding_operations[i]
            link_op = self.link_ops[i]
            up = self.upscale_ops[i]
            if X_skip_layer is not None:
                S = encoding_out[-i - 2].shape[2:]
                xfl = F.interpolate(X_skip_layer, S, mode="nearest")
                link_op_input = torch.cat([encoding_out[-i - 2], xfl], axis=1)
            else:
                link_op_input = encoding_out[-i - 2]
            encoded = link_op(link_op_input)
            if X_feature_conditioning is not None:
                feat_op = self.feature_conditioning_ops[i]
                transformed_features = feat_op(X_feature_conditioning)
                transformed_features = self.unsqueeze_to_dim(
                    transformed_features, encoded
                )
                encoded = torch.multiply(encoded, transformed_features)
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

        final_features = curr

        if return_logits is True:
            curr = self.final_layer[:-1](curr)
        else:
            curr = self.final_layer(curr)
        if return_features is True:
            return curr, final_features, bottleneck

        if self.bottleneck_classification is True:
            bottleneck = bottleneck.flatten(start_dim=2).max(-1).values
            bn_out = self.bottleneck_classifier(bottleneck)
        else:
            bn_out = None

        if self.deep_supervision is True:
            for i in range(len(deep_outputs)):
                o = deep_outputs[i]
                op = self.deep_supervision_ops[i]
                deep_outputs[i] = op(o)
            return curr, bn_out, deep_outputs

        return curr, bn_out
