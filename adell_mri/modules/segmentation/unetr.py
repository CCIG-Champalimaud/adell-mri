"""
Standard UNETR implementation.
"""

from typing import Callable, Dict, List, Union

import numpy as np
import torch
import torch.nn.functional as F

from adell_mri.custom_types import Size2dOr3d
from adell_mri.modules.layers.adn_fn import get_adn_fn
from adell_mri.modules.layers.vit import (
    LinearEmbedding,
    SWINTransformerBlockStack,
    ViT,
)
from adell_mri.modules.segmentation.unet import UNet


class UNETR(UNet, torch.nn.Module):
    """
    UNETR module. Implementation as close as possible to that presented
    in the original paper [1] and integrated with other UNet-like
    architectures in this library.

    [1] https://arxiv.org/abs/2103.10504
    """
    def __init__(
        self,
        # parametrize linear embedding and transformer
        image_size: Size2dOr3d,
        patch_size: Size2dOr3d,
        number_of_blocks: int,
        return_at: List[int],
        attention_dim: int = None,
        hidden_dim: int = None,
        embedding_size: int = None,
        n_heads: int = 4,
        dropout_rate: float = 0.0,
        embed_method: str = "linear",
        mlp_structure: List[int] = [256, 256],
        adn_fn_mlp: Callable = get_adn_fn(1, "identity", "gelu"),
        # regular u-net parametrization
        spatial_dimensions: int = 2,
        conv_type: str = "regular",
        link_type: str = "identity",
        upscale_type: str = "upsample",
        interpolation: str = "bilinear",
        norm_type: str = "batch",
        dropout_type: str = "dropout",
        padding: int = 0,
        dropout_param: float = 0.0,
        activation_fn: torch.nn.Module = torch.nn.PReLU,
        in_channels: int = 1,
        n_classes: int = 2,
        depth: list = [16, 32, 64],
        kernel_sizes: list = [3, 3, 3],
        bottleneck_classification: bool = False,
        skip_conditioning: int = None,
        feature_conditioning: int = None,
        feature_conditioning_params: Dict[str, torch.Tensor] = None,
        deep_supervision: bool = False,
        encoder_only: bool = False,
    ):
        """
        Args:
            image_size (Size2dOr3d): size of the input image.
            patch_size (Size2dOr3d): size of the patches.
            number_of_blocks (int): number of total transformer blocks.
            return_at (List[int]): blocks that are returned and reshaped for
                the skip connections.
            attention_dim (int, optional): total dimension of the attention in
                the MHA. Defaults to None (same as inferred primary input
                dimension).
            hidden_dim (int, optional): output dimension for the attention
                layers in the MHA. Defaults to None (same as inferred primary
                input dimension).
            embedding_size (int, optional): size of the embedding. Defaults to
                None (same as inferred output dimension).
            n_heads (int, optional): Number of heads in the MHA (the actual
                dimension of each MHA is attention_dim // MHA). Defaults to 4.
            dropout_rate (float, optional): dropout rate for all of the
                modules in the network. Defaults to 0.0 (not recommended).
            embed_method (str, optional): embedding method for the linear
                embedding of input images/volumes. More details in
                `adell_mri.modules.layers.vit.LinearEmbedding`. Defaults to
                "linear".
            mlp_structure (List[int], optional): MLP structure (number of units
                in each hidden unit) for the transformers. Defaults to
                [256,256].
            adn_fn_mlp (Callable, optional): ADN function for the MLP in the
                Transformer. Defaults to torch.nn.Identity.
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
            padding (int, optional): amount of padding for convolutions.
                Defaults to 0.
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
            encoder_only (bool, optional): makes only encoder.
        """
        super().__init__(parent_class=True)
        # parametrize linear embedding and transformer
        self.image_size = image_size
        self.patch_size = patch_size
        self.number_of_blocks = number_of_blocks
        self.attention_dim = attention_dim
        self.hidden_dim = hidden_dim
        self.embedding_size = embedding_size
        self.return_at = return_at
        self.n_heads = n_heads
        self.dropout_rate = dropout_rate
        self.embed_method = embed_method
        self.mlp_structure = mlp_structure
        self.adn_fn_mlp = adn_fn_mlp
        # regular u-net parametrization
        self.spatial_dimensions = spatial_dimensions
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
        self.bottleneck_classification = bottleneck_classification
        self.skip_conditioning = skip_conditioning
        self.feature_conditioning = feature_conditioning
        self.feature_conditioning_params = feature_conditioning_params
        self.deep_supervision = deep_supervision

        self.strides = [2 for _ in self.depth]

        # define the scale of the reconstructions
        self.scale = int(2 ** len(self.return_at))
        # define the number of channels in the reconstructed ViT outputs
        self.in_channels_rec = np.prod(
            [self.scale**self.spatial_dimensions, self.in_channels]
        )

        # check parameters
        self.assertions()

        # initialize all layers
        self.get_norm_op()
        self.get_drop_op()

        self.get_conv_op()
        self.init_vit()

        if self.encoder_only is False:
            self.init_first_encoder()
            self.init_reconstruction_ops()
            self.init_upscale_ops()
            self.init_link_ops()
            self.init_decoder()
            self.init_final_layer()
            self.init_rescalers()
            if self.bottleneck_classification is True:
                self.init_bottleneck_classifier()
            if self.feature_conditioning is not None:
                self.init_feature_conditioning_operations()

    def assertions(self):
        assert (len(self.depth) - 1) == len(
            self.return_at
        ), "(len(depth)-1) must be the same as len(return_at)"
        assert (
            max(self.return_at) <= self.number_of_blocks
        ), "len(depth) must be smaller than number_of_blocks"
        assert len(self.depth) == len(
            self.kernel_sizes
        ), "len(depth) must be the same as len(kernel_sizes)"

    def init_vit(self):
        """
        Initialises ViT and infers the number of channels at
        (intermediary) output reconstruction.
        """
        self.vit = ViT(
            image_size=self.image_size,
            patch_size=self.patch_size,
            in_channels=self.in_channels,
            number_of_blocks=self.number_of_blocks,
            attention_dim=self.attention_dim,
            hidden_dim=self.hidden_dim,
            embedding_size=self.embedding_size,
            n_heads=self.n_heads,
            dropout_rate=self.dropout_rate,
            embed_method=self.embed_method,
            mlp_structure=self.mlp_structure,
            adn_fn=self.adn_fn_mlp,
        )
        # expose this function for convenience
        self.rearrange_rescale = self.vit.embedding.rearrange_rescale

    def init_rescalers(self):
        """
        Initialises a set of linear embeddings which transform transformer
        representations to volumetric representations.
        """
        self.rescalers = torch.nn.ModuleList(
            [
                LinearEmbedding(
                    image_size=self.image_size,
                    patch_size=self.patch_size,
                    in_channels=self.in_channels,
                    out_dim=self.embedding_size,
                    dropout_rate=0.0,
                    embed_method="linear",
                    use_pos_embed=False,
                    use_class_token=False,
                )
                for _ in self.depth[1:]
            ]
        )

    def init_first_encoder(self):
        """
        Initialises the first encoder.
        """
        self.first_encoder = torch.nn.Sequential(
            self.adn_fn(self.in_channels),
            self.conv_op_enc(
                self.in_channels, self.depth[0], 3, padding="same"
            ),
            self.adn_fn(self.depth[0]),
        )

    def unetr_transp_op(
        self, in_d: int, out_d: int, kernel_size: int = 3
    ) -> torch.nn.Module:
        if self.spatial_dimensions == 2:
            transp_conv = torch.nn.ConvTranspose2d
        else:
            transp_conv = torch.nn.ConvTranspose3d
        return torch.nn.Sequential(
            transp_conv(in_d, out_d, 2, 2),
            self.adn_fn(out_d),
            self.conv_op_enc(out_d, out_d, kernel_size, padding="same"),
            self.adn_fn(out_d),
        )

    def unetr_transp_block(
        self, in_d: int, out_d: int, n_ops: int, kernel_size: int = 3
    ) -> torch.nn.Module:
        out = torch.nn.ModuleList(
            [self.unetr_transp_op(in_d, out_d, kernel_size)]
        )
        for _ in range(n_ops):
            out.append(self.unetr_transp_op(out_d, out_d, 3))
        return torch.nn.Sequential(*out)

    def init_reconstruction_ops(self):
        """
Initialises the operations that will reconstruct the ViT outputs
        to be U-Net compliant.
        """
        self.reconstructed_dim = [
            self.in_channels_rec,
            *[x // self.scale for x in self.image_size],
        ]
        self.reconstruction_ops = torch.nn.ModuleList([])
        self.n_skip_connections = len(self.depth) - 1
        for i, d in enumerate(self.depth[1:-1]):
            i = i + 1
            n_ops = self.n_skip_connections - i
            rec_op_seq = self.unetr_transp_block(
                self.in_channels_rec, d, n_ops - 1, 3
            )
            self.reconstruction_ops.append(rec_op_seq)
        self.bottleneck_reconstruction = self.conv_op_enc(
            self.in_channels_rec, self.depth[-1], 1, 1
        )

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

        # run vit
        curr, encoding_out = self.vit(X, return_at=self.return_at)
        X_encoded_first = self.first_encoder(X)
        # rearrange outputs using einops
        curr = self.rearrange_rescale(curr, self.scale)
        encoding_out = [
            rescaler.rearrange_rescale(x, self.scale)
            for x, rescaler in zip(encoding_out, self.rescalers)
        ]

        # apply reconstruction (deconv) ops
        curr = self.bottleneck_reconstruction(curr)
        encoding_out = [
            X_encoded_first,
            *[
                rec_op(x)
                for x, rec_op in zip(encoding_out, self.reconstruction_ops)
            ],
        ]

        encoding_out.append(curr)
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


class MonaiUNETR(UNet, torch.nn.Module):
    """
    UNETR module from MONAI.
    """

    def __init__(
        self,
        # parametrize linear embedding and transformer
        image_size: Size2dOr3d,
        patch_size: Size2dOr3d,
        number_of_blocks: int,
        attention_dim: int = 128,
        return_at: List[int] = [2, 4, 6],
        hidden_dim: int = None,
        n_heads: int = 4,
        dropout_rate: float = 0.0,
        embed_method: str = "linear",
        mlp_structure: List[int] = [256, 256],
        embedding_size: int = 768,
        adn_fn_mlp: Callable = get_adn_fn(1, "identity", "gelu"),
        # regular u-net parametrization
        spatial_dimensions: int = 2,
        conv_type: str = "regular",
        link_type: str = "identity",
        upscale_type: str = "upsample",
        interpolation: str = "bilinear",
        norm_type: str = "batch",
        dropout_type: str = "dropout",
        padding: int = 0,
        dropout_param: float = 0.0,
        activation_fn: torch.nn.Module = torch.nn.PReLU,
        in_channels: int = 1,
        n_classes: int = 2,
        depth: list = [16, 32, 64],
        kernel_sizes: list = [3, 3, 3],
        bottleneck_classification: bool = False,
        skip_conditioning: int = None,
        feature_conditioning: int = None,
        feature_conditioning_params: Dict[str, torch.Tensor] = None,
        deep_supervision: bool = False,
    ):
        """
        Args:
            image_size (Size2dOr3d): size of the input image.
            patch_size (Size2dOr3d): size of the patches.
            number_of_blocks (int): number of total transformer blocks.
            attention_dim (int): total dimension of the attention in the
                MHA.
            return_at (List[int]): blocks that are returned and reshaped for
                the skip connections.
            hidden_dim (int, optional): output dimension for the attention
                layers in the MHA. Defaults to None (same as inferred primary
                input dimension).
            n_heads (int, optional): Number of heads in the MHA (the actual
                dimension of each MHA is attention_dim // MHA). Defaults to 4.
            dropout_rate (float, optional): dropout rate for all of the
                modules in the network. Defaults to 0.0 (not recommended).
            embed_method (str, optional): embedding method for the linear
                embedding of input images/volumes. More details in
                `adell_mri.modules.layers.vit.LinearEmbedding`. Defaults to
                "linear".
            mlp_structure (List[int], optional): MLP structure (number of units
                in each hidden unit) for the transformers. Defaults to
                [256,256].
            adn_fn_mlp (Callable, optional): ADN function for the MLP in the
                Transformer. Defaults to torch.nn.Identity.
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
            padding (int, optional): amount of padding for convolutions.
                Defaults to 0.
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
        super().__init__(parent_class=True)
        # parametrize linear embedding and transformer
        self.image_size = image_size
        self.patch_size = patch_size
        self.number_of_blocks = number_of_blocks
        self.attention_dim = attention_dim
        self.hidden_dim = hidden_dim
        self.return_at = return_at
        self.n_heads = n_heads
        self.dropout_rate = dropout_rate
        self.embed_method = embed_method
        self.mlp_structure = mlp_structure
        self.embedding_size = embedding_size
        self.adn_fn_mlp = adn_fn_mlp
        # regular u-net parametrization
        self.spatial_dimensions = spatial_dimensions
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
        self.bottleneck_classification = bottleneck_classification
        self.skip_conditioning = skip_conditioning
        self.feature_conditioning = feature_conditioning
        self.feature_conditioning_params = feature_conditioning_params
        self.deep_supervision = False

        from monai.networks.nets import UNETR

        self.network = UNETR(
            self.in_channels,
            1 if self.n_classes == 2 else self.n_classes,
            [int(x) for x in self.image_size],
        )

        self.last_activation = (
            torch.nn.Sigmoid() if self.n_classes == 2 else torch.nn.Softmax(1)
        )
        self.final_layer = torch.nn.Sequential(
            self.last_activation,
        )

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
        if return_logits is True:
            return self.network(X), None
        else:
            return self.last_activation(self.network(X)), None


class SWINUNet(UNet):
    """
    SWINUNet module. Implementation as close as possible to that
    presented in the original paper [1] and integrated with other UNet-like
    architectures in this library. To make similar architectures with
    different depths more compliant and easier to implement, I have
    included 1x1 convolutions *after* each stage of SWINUNet.

    [1] https://arxiv.org/pdf/2103.14030.pdf
    """

    def __init__(
        self,
        # parametrize linear embedding and transformer
        image_size: Size2dOr3d,
        patch_size: Size2dOr3d,
        window_size: Size2dOr3d,
        shift_sizes: Union[List[int], List[List[int]]],
        embedding_size: int = None,
        n_heads: int = 4,
        dropout_rate: float = 0.0,
        embed_method: str = "linear",
        mlp_structure: List[int] = [256, 256],
        adn_fn_mlp: Callable = get_adn_fn(1, "identity", "gelu"),
        # regular u-net parametrization
        spatial_dimensions: int = 2,
        conv_type: str = "regular",
        link_type: str = "identity",
        upscale_type: str = "upsample",
        interpolation: str = "bilinear",
        norm_type: str = "batch",
        dropout_type: str = "dropout",
        padding: int = 0,
        dropout_param: float = 0.0,
        activation_fn: torch.nn.Module = torch.nn.PReLU,
        in_channels: int = 1,
        n_classes: int = 2,
        depth: list = [16, 32, 64],
        kernel_sizes: list = [3, 3, 3],
        strides: list[int] = None,
        bottleneck_classification: bool = False,
        skip_conditioning: int = None,
        feature_conditioning: int = None,
        feature_conditioning_params: Dict[str, torch.Tensor] = None,
        deep_supervision: bool = False,
    ):
        """
        Args:
            image_size (Size2dOr3d): size of the input image.
            patch_size (Size2dOr3d): size of the patches.
            window_size (Size2dOr3d): size of image window.
            shift_sizes (Union[List[int],List[List[int]]]): size of shifts for
                the shifted-window MHA blocks.
            embedding_size (int, optional): size of the embedding. Defaults to
                None (same as inferred output dimension).
            n_heads (int, optional): Number of heads in the MHA (the actual
                dimension of each MHA is attention_dim // MHA). Defaults to 4.
            dropout_rate (float, optional): dropout rate for all of the
                transformer modules in the network. Defaults to 0.0.
            embed_method (str, optional): embedding method for the linear
                embedding of input images/volumes. More details in
                `adell_mri.modules.layers.vit.LinearEmbedding`. Defaults to
                "linear".
            mlp_structure (List[int], optional): MLP structure (number of units
                in each hidden unit) for the transformers. Defaults to
                [256,256].
            adn_fn_mlp (Callable, optional): ADN function for the MLP in the
                Transformer. Defaults to torch.nn.Identity.
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
            padding (int, optional): amount of padding for convolutions.
                Defaults to 0.
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
                U-Net. Defaults to None (2 for every instance).
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
        super().__init__(parent_class=True)
        # parametrize linear embedding and transformer
        self.image_size = image_size
        self.patch_size = patch_size
        self.window_size = window_size
        self.shift_sizes = shift_sizes
        self.embedding_size = embedding_size
        self.n_heads = n_heads
        self.dropout_rate = dropout_rate
        self.embed_method = embed_method
        self.mlp_structure = mlp_structure
        self.adn_fn_mlp = adn_fn_mlp
        # regular u-net parametrization
        self.spatial_dimensions = spatial_dimensions
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

        self.number_of_blocks = len(self.depth)

        # check parameters
        self.arg_compliance()

        # initialize all layers
        self.get_norm_op()
        self.get_drop_op()

        self.get_conv_op()
        self.init_first_encoder()
        self.init_swin_blocks()
        self.init_reconstruction_ops()
        self.init_upscale_ops()
        self.init_link_ops()
        self.init_decoder()
        self.init_final_layer()
        if self.bottleneck_classification is True:
            self.init_bottleneck_classifier()
        if self.feature_conditioning is not None:
            self.init_feature_conditioning_operations()

    def arg_compliance(self):
        shift_size_msg = "shift_sizes must be list of ints or list of list \
            of ints"
        assert (
            len(self.depth) == self.number_of_blocks
        ), "(len(depth)-1) must be the same as number_of_blocks"
        assert isinstance(self.shift_sizes, list), shift_size_msg
        # if shift sizes is a list of ints convert to list of list of ints
        if isinstance(self.shift_sizes[0], int):
            self.shift_sizes = [self.shift_sizes for _ in self.depth]
        if isinstance(self.n_heads, int):
            self.n_heads = [self.n_heads for _ in self.depth]
        if isinstance(self.embedding_size, int) or self.embedding_size is None:
            self.embedding_size = [self.embedding_size for _ in self.depth]
        elif isinstance(self.shift_sizes[0], list):
            assert isinstance(self.shift_sizes[0][0], int), shift_size_msg
        else:
            raise AssertionError(shift_size_msg)
        if self.strides is None:
            self.strides = [2 for _ in range(len(self.depth))]
        for i in range(len(self.strides)):
            if isinstance(self.strides[i], int):
                self.strides[i] = [
                    self.strides[i] for _ in range(self.spatial_dimensions)
                ]

    def init_first_encoder(self):
        """
        Initialises the first encoder.
        """
        self.first_encoder = torch.nn.Sequential(
            self.adn_fn(self.in_channels),
            self.conv_op_enc(
                self.in_channels, self.depth[0], 3, padding="same"
            ),
            self.adn_fn(self.depth[0]),
        )

    def init_final_layer(self):
        """
        Initializes the classification layer (simple linear layer)."""
        o = self.depth[0] * 2
        self.final_layer = self.get_final_layer(o)

    def init_swin_blocks(self):
        """
Initialises SWin and infers the number of channels at
        (intermediary) output reconstruction.
        """
        self.in_channels_rec = []
        self.first_swin_block = SWINTransformerBlockStack(
            image_size=self.image_size,
            patch_size=self.patch_size,
            window_size=self.window_size,
            in_channels=self.in_channels,
            shift_sizes=self.shift_sizes[0],
            attention_dim=self.embedding_size[0],
            hidden_dim=self.embedding_size[0],
            embedding_size=self.embedding_size[0],
            n_heads=self.n_heads[0],
            dropout_rate=self.dropout_rate,
            dropout_rate_embedding=self.dropout_rate,
            embed_method=self.embed_method,
            mlp_structure=self.mlp_structure,
            adn_fn=self.adn_fn_mlp,
            use_pos_embed=True,
        )
        self.swin_blocks = torch.nn.ModuleList([])
        image_size = self.image_size
        for i in range(self.number_of_blocks - 1):
            in_channels = self.in_channels
            if i > 0:
                in_channels *= np.prod([np.prod(s) for s in self.strides[:i]])
                self.in_channels_rec.append(in_channels)
            swin_block = SWINTransformerBlockStack(
                image_size=image_size,
                patch_size=self.patch_size,
                window_size=self.window_size,
                in_channels=in_channels,
                shift_sizes=self.shift_sizes[i + 1],
                attention_dim=self.embedding_size[i + 1],
                hidden_dim=self.embedding_size[i + 1],
                embedding_size=self.embedding_size[i + 1],
                n_heads=self.n_heads[i + 1],
                dropout_rate=self.dropout_rate,
                dropout_rate_embedding=0.0,
                embed_method="linear",
                mlp_structure=self.mlp_structure,
                adn_fn=self.adn_fn_mlp,
                use_pos_embed=False,
            )
            self.swin_blocks.append(swin_block)
            image_size = [x // s for x, s in zip(image_size, self.strides[i])]
        self.in_channels_rec.append(
            self.in_channels * np.prod([np.prod(s) for s in self.strides[:-1]])
        )

    def init_reconstruction_ops(self):
        """
Initialises the operations that will resize the SWin outputs
        to be U-Net compliant.
        """
        layer_norm = get_adn_fn(self.spatial_dimensions, "layer", None, 0.0)
        self.first_rec_op = torch.nn.Sequential(
            layer_norm(self.in_channels),
            self.conv_op_enc(
                self.in_channels, self.depth[0], 3, padding="same"
            ),
            self.adn_fn(self.depth[0]),
        )
        self.reconstruction_ops = torch.nn.ModuleList([])
        for i, d in enumerate(self.depth[1:]):
            rec_op_seq = torch.nn.Sequential(
                layer_norm(self.in_channels_rec[i]),
                self.conv_op_enc(self.in_channels_rec[i], d, 1, padding="same"),
                self.conv_op_enc(d, d, 3, padding="same"),
                self.adn_fn(d),
            )
            self.reconstruction_ops.append(rec_op_seq)

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

        X_encoded_first = self.first_encoder(X)
        # run swin blocks
        curr = self.first_swin_block(X)
        encoding_out = [self.first_rec_op(curr)]
        for i in range(len(self.swin_blocks)):
            swin_block = self.swin_blocks[i]
            rec_op = self.reconstruction_ops[i]
            curr = swin_block(curr, scale=self.strides[i])
            encoding_out.append(rec_op(curr))

        curr = encoding_out[-1]
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
            curr = torch.concat((curr, encoded), dim=1)
            curr = op(curr)
            deep_outputs.append(curr)

        final_features = curr
        curr = torch.cat([X_encoded_first, curr], 1)

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


class MonaiSWINUNet(UNet):
    """
    SWINUNet module from MONAI.
    """

    def __init__(
        self,
        # parametrize linear embedding and transformer
        image_size: Size2dOr3d,
        patch_size: Size2dOr3d,
        window_size: Size2dOr3d,
        shift_sizes: Union[List[int], List[List[int]]],
        attention_dim: int = None,
        hidden_dim: int = None,
        embedding_size: int = None,
        n_heads: int = 4,
        dropout_rate: float = 0.0,
        embed_method: str = "linear",
        mlp_structure: List[int] = [256, 256],
        adn_fn_mlp: Callable = get_adn_fn(1, "identity", "gelu"),
        # regular u-net parametrization
        spatial_dimensions: int = 2,
        conv_type: str = "regular",
        link_type: str = "identity",
        upscale_type: str = "upsample",
        interpolation: str = "bilinear",
        norm_type: str = "batch",
        dropout_type: str = "dropout",
        padding: int = 0,
        dropout_param: float = 0.0,
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
    ):
        """
        Args:
            image_size (Size2dOr3d): size of the input image.
            patch_size (Size2dOr3d): size of the patches.
            window_size (Size2dOr3d): size of image window.
            shift_sizes (Union[List[int],List[List[int]]]): size of shifts for
                the shifted-window MHA blocks.
            attention_dim (int): total dimension of the attention in the
                MHA. Defaults to None (same as inferred output dimension).
            hidden_dim (int): output dimension for the attention layers in the
                MHA. Defaults to None (same as inferred output dimension).
            embedding_size (int, optional): size of the embedding. Defaults to
                None (same as inferred output dimension).
            n_heads (int, optional): Number of heads in the MHA (the actual
                dimension of each MHA is attention_dim // MHA). Defaults to 4.
            dropout_rate (float, optional): dropout rate for all of the
                transformer modules in the network. Defaults to 0.0.
            embed_method (str, optional): embedding method for the linear
                embedding of input images/volumes. More details in
                `adell_mri.modules.layers.vit."LinearEmbedding"`. Defaults to
                "linear".
            mlp_structure (List[int], optional): MLP structure (number of units
                in each hidden unit) for the transformers. Defaults to
                [256,256].
            adn_fn_mlp (Callable, optional): ADN function for the MLP in the
                Transformer. Defaults to torch.nn.Identity.
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
            padding (int, optional): amount of padding for convolutions.
                Defaults to 0.
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
        super().__init__(parent_class=True)
        # parametrize linear embedding and transformer
        self.image_size = image_size
        self.patch_size = patch_size
        self.window_size = window_size
        self.shift_sizes = shift_sizes
        self.attention_dim = attention_dim
        self.hidden_dim = hidden_dim
        self.embedding_size = embedding_size
        self.n_heads = n_heads
        self.dropout_rate = dropout_rate
        self.embed_method = embed_method
        self.mlp_structure = mlp_structure
        self.adn_fn_mlp = adn_fn_mlp
        # regular u-net parametrization
        self.spatial_dimensions = spatial_dimensions
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

        self.number_of_blocks = len(self.depth)

        from monai.networks.nets.swin_unetr import SwinUNETR

        feature_size = 24 if self.attention_dim is None else self.attention_dim
        self.network = SwinUNETR(
            img_size=[int(x) for x in self.image_size],
            in_channels=self.in_channels,
            out_channels=1 if self.n_classes == 2 else self.n_classes,
            feature_size=feature_size,
            drop_rate=self.dropout_rate,
            spatial_dims=self.spatial_dimensions,
        )
        self.last_activation = (
            torch.nn.Sigmoid() if self.n_classes == 2 else torch.nn.Softmax(1)
        )
        self.final_layer = torch.nn.Sequential(
            self.last_activation,
        )

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
        if return_logits is True:
            return self.network(X), None
        else:
            return self.last_activation(self.network(X)), None
