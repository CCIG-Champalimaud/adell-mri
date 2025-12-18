import time
from typing import Callable, List, Tuple

import torch
import torch.nn.functional as F

from adell_mri.custom_types import TensorList
from adell_mri.modules.layers.adn_fn import ActDropNorm, get_adn_fn
from adell_mri.modules.layers.linear_blocks import MLP, SeqPool
from adell_mri.modules.layers.res_net import (
    ProjectionHead,
    ResNet,
    ResNetBackbone,
)
from adell_mri.modules.layers.standard_blocks import (
    GlobalPooling,
    VGGConvolution3d,
)
from adell_mri.modules.layers.vit import FactorizedViT, ViT
from adell_mri.modules.segmentation.unet import UNet

resnet_default = [(64, 128, 5, 2), (128, 256, 3, 5)]
maxpool_default = [(2, 2, 2), (2, 2, 2)]


def label_to_ordinal(
    label: torch.Tensor, n_classes: int, ignore_0: bool = True
) -> torch.Tensor:
    """
    Converts a label to an ordinal classification tensor. In essence, for a
    given label L, this function returns an n-hot encoded version of L that
    is 0 for indices < L and 1 otherwise.

    Args:
        label (torch.Tensor): one dimensional tensor with labels.
        n_classes (int): number of classes.
        ignore_0 (bool, optional): whether 0 should not be considered a class.
            Defaults to True.

    Returns:
        torch.Tensor: ordinal classes.
    """
    label = torch.squeeze(label, 1)
    if ignore_0 is True:
        label = torch.clamp(label - 1, min=0)
    one_hot = F.one_hot(label, n_classes)
    one_hot = one_hot.unsqueeze(1).swapaxes(1, -1).squeeze(-1)
    one_hot = torch.clamp(one_hot, max=1)
    one_hot_cumsum = torch.cumsum(one_hot, axis=1)
    output = torch.ones_like(one_hot_cumsum, device=one_hot_cumsum.device)
    return output - one_hot_cumsum


def ordinal_prediction_to_class(x: torch.Tensor) -> torch.Tensor:
    """
    Converts an ordinal prediction to a specific class.

    Args:
        x (torch.Tensor): ordinal prediction tensor.

    Returns:
        torch.Tensor: categorical prediction.
    """
    x_thresholded = F.threshold(x, 0.5, 1)
    output = x_thresholded.sum(dim=1)
    return output


class VGGBackbone(torch.nn.Module):
    """
    Very simple and naive VGG backbone for standard categorical classification.
    """

    def __init__(
        self,
        spatial_dimensions: int = 3,
        in_channels: int = 1,
        depth_mult: float = 1.0,
        feature_extraction=None,
        resnet_structure=None,
        maxpool_structure=None,
        adn_fn=None,
        res_type=None,
        batch_ensemble: int = 0,
    ):
        """
        Args:
            spatial_dimensions (int, optional): number of spatial dimensions.
                Defaults to 3.
           in_channels (int, optional): number of input channels. Defaults to
                1.
            depth_mult (float, optional): multiplies the feature extractor
                depths. Defaults to 1.0 (64, 128, 256).
            batch_ensemble (int, optional): number of batch ensemble modules.
                Defautls to 0.
        """
        super().__init__()
        self.in_channels = in_channels
        self.depth_mult = depth_mult
        self.batch_ensemble = batch_ensemble

        depths = [int(x * self.depth_mult) for x in [64, 128, 256]]

        self.conv1 = VGGConvolution3d(self.in_channels, depths[0])
        self.conv2 = VGGConvolution3d(depths[1], depths[1])
        self.conv3 = VGGConvolution3d(
            depths[2], depths[2], batch_ensemble=batch_ensemble
        )
        self.output_features = int(self.output_features * self.depth_mult)

    def forward_features(
        self, X: torch.Tensor, batch_idx: int = None
    ) -> torch.Tensor:
        """
        Forward method for features only.

        Args:
            X (torch.Tensor): input tensor
            batch_idx (int, optional): uses a specific batch ensemble
                transformation. Defaults to None (usually random).

        Returns:
            torch.Tensor: output (classification)
        """
        return self.forward(X, batch_idx=batch_idx)

    def forward(
        self,
        X: torch.Tensor,
        batch_idx: int = None,
    ) -> torch.Tensor:
        """
        Forward pass.

        Args:
            X (torch.Tensor): input tensor
            batch_idx (int, optional): uses a specific batch ensemble
                transformation. Defaults to None (usually random).

        Returns:
            torch.Tensor: output (classification)
        """
        X = self.conv1(X)
        X = self.conv2(X)
        return self.conv3(X, batch_idx=batch_idx)


class VGG(torch.nn.Module):
    """
    Very simple and naive VGG net for standard categorical classification.
    """

    def __init__(
        self,
        spatial_dimensions: int = 3,
        in_channels: int = 1,
        n_classes: int = 2,
        depth_mult: float = 1.0,
        feature_extraction=None,
        resnet_structure=None,
        maxpool_structure=None,
        adn_fn=None,
        res_type=None,
        classification_structure: List[int] = [512, 512, 512],
        batch_ensemble: int = 0,
        output_features: int = None,
    ):
        """
        Args:
            spatial_dimensions (int, optional): number of spatial dimensions.
                Defaults to 3.
            in_channels (int, optional): number of input channels. Defaults to
                1.
            n_classes (int, optional): number of classes. Defaults to 2.
            depth_mult (float, optional): multiplies the feature extractor
                depths. Defaults to 1.0 (64, 128, 256).
            classification_structure (List[int], optional): structure of the
                classifier. Defaults to [512,512,512].
            batch_ensemble (int, optional): number of batch ensemble modules.
                Defautls to 0.
            output_features (int, optional): number of output features. Used
                only when adapting this class to other methods. Defaults to
                None (512).
        """
        super().__init__()
        self.in_channels = in_channels
        self.n_classes = n_classes
        self.depth_mult = depth_mult
        self.batch_ensemble = batch_ensemble
        self.classification_structure = classification_structure
        self.output_features = output_features
        if output_features is None:
            self.output_features = 512

        depths = [int(x * self.depth_mult) for x in [64, 128, 256]]
        self.output_features = int(self.output_features * self.depth_mult)

        self.conv1 = VGGConvolution3d(self.in_channels, depths[0])
        self.conv2 = VGGConvolution3d(depths[1], depths[1])
        self.conv3 = VGGConvolution3d(
            depths[2], depths[2], batch_ensemble=batch_ensemble
        )

        if self.n_classes == 2:
            final_n = 1
            self.last_act = torch.nn.Sigmoid()
        else:
            final_n = self.n_classes
            self.last_act = torch.nn.Softmax(-1)

        self.classification_layer = torch.nn.Sequential(
            GlobalPooling(),
            MLP(
                self.output_features,
                final_n,
                classification_structure,
                adn_fn=get_adn_fn(1, "batch", "gelu"),
            ),
        )

    def forward_features(
        self, X: torch.Tensor, batch_idx: int = None
    ) -> torch.Tensor:
        """
        Forward method for features only.

        Args:
            X (torch.Tensor): input tensor
            batch_idx (int, optional): uses a specific batch ensemble
                transformation. Defaults to None (usually random).

        Returns:
            torch.Tensor: output (classification)
        """
        return self.forward(X, return_features=True, batch_idx=batch_idx)

    def forward(
        self,
        X: torch.Tensor,
        return_features: bool = False,
        batch_idx: int = None,
    ) -> torch.Tensor:
        """
        Forward pass.

        Args:
            X (torch.Tensor): input tensor
            return_features (bool, optional): returns the features before
                applying classification layer. Defaults to False.
            batch_idx (int, optional): uses a specific batch ensemble
                transformation. Defaults to None (usually random).

        Returns:
            torch.Tensor: output (classification)
        """
        X = self.conv1(X)
        X = self.conv2(X)
        X = self.conv3(X, batch_idx=batch_idx)
        if return_features is True:
            return X

        return self.classification_layer(X)


class CatNet(torch.nn.Module):
    """
    Case class for standard categorical classification. Defaults to
    feature extraction using ResNet.
    """

    def __init__(
        self,
        spatial_dimensions: int = 3,
        in_channels: int = 1,
        n_classes: int = 2,
        feature_extraction: torch.nn.Module = None,
        resnet_structure: List[Tuple[int, int, int, int]] = resnet_default,
        maxpool_structure: List[Tuple[int, int, int]] = maxpool_default,
        adn_fn: torch.nn.Module = None,
        res_type: str = "resnet",
        classification_structure: list[int] = None,
        batch_ensemble: bool = False,
        skip_last_activation: bool = False,
    ):
        """
        Args:
            spatial_dimensions (int, optional): number of spatial dimensions.
                Defaults to 3.
            in_channels (int, optional): number of input channels. Defaults to
                1.
            n_classes (int, optional): number of classes. Defaults to 2.
            feature_extraction (torch.nn.Module, optional): module to use for
                feature extraction. Defaults to None (builds a ResNet using
                `resnet_structure` and `maxpool_structure`).
            resnet_structure (List[Tuple[int,int,int,int]], optional):
                structure for ResNet should be a list of tuples with 4
                elements, corresponding to input size, intermediate size,
                kernel size and number of consecutive residual operations.
                Defaults to [(64,128,5,2),(128,256,3,5)].
            maxpool_structure (List[Tuple[int,int,int]], optional): structure
                for the max pooling operations. Must be a list with the same
                length as resnet_structure and defining the kernel size and
                stride (these will be identical). Defaults to
                [(2,2,2),(2,2,2)].
            adn_fn (torch.nn.Module, optional): activation dropout
                normalization function. Must be a function that takes an
                argument (number of channels) and returns a torch Module.
                Defaults to None (batch normalization).
            res_type (str, optional): type of residual operation, can be either
                "resnet" or "resnext". Defaults to "resnet".
            classification_structure (List[int], optional): structure of the
                classifier. Defaults to None (last layer size repeated three
                times).
            batch_ensemble (bool, optional): uses batch ensemble layers.
                Defaults to False.
            skip_last_activation (bool, optional): skips the last activation in
                the ResNet backbone. Defaults to False.
        """
        super().__init__()
        self.spatial_dim = spatial_dimensions
        self.in_channels = in_channels
        self.n_classes = n_classes
        self.feature_extraction = feature_extraction
        self.resnet_structure = resnet_structure
        self.maxpool_structure = maxpool_structure
        self.adn_fn = adn_fn
        self.res_type = res_type
        self.classification_structure = classification_structure
        self.batch_ensemble = batch_ensemble
        self.skip_last_activation = skip_last_activation

        if self.adn_fn is None:
            if self.spatial_dim == 2:
                self.adn_fn = lambda s: ActDropNorm(
                    s, norm_fn=torch.nn.BatchNorm2d
                )
            if self.spatial_dim == 3:
                self.adn_fn = lambda s: ActDropNorm(
                    s, norm_fn=torch.nn.BatchNorm3d
                )

        self.init_layers()
        self.init_classification_layer()

    def init_layers(self):
        if self.feature_extraction is None:
            self.res_net = ResNetBackbone(
                self.spatial_dim,
                self.in_channels,
                self.resnet_structure,
                adn_fn=self.adn_fn,
                maxpool_structure=self.maxpool_structure,
                res_type=self.res_type,
                batch_ensemble=self.batch_ensemble,
                skip_last_activation=self.skip_last_activation,
            )
            self.feature_extraction = self.res_net
            self.last_size = self.resnet_structure[-1][0]
        else:
            input_shape = [2, self.in_channels, 128, 128]
            if self.spatial_dim == 3:
                input_shape.append(32)
            example_tensor = torch.ones(input_shape)
            self.last_size = self.feature_extraction(example_tensor).shape[1]
        self.output_features = self.last_size

    def init_classification_layer(self):
        if self.n_classes == 2:
            final_n = 1
            self.last_act = torch.nn.Sigmoid()
        else:
            final_n = self.n_classes
            self.last_act = torch.nn.Softmax(1)
        if self.classification_structure is None:
            self.classification_structure = [self.last_size for _ in range(3)]
        self.gp = GlobalPooling()
        self.classification_layer = torch.nn.Sequential(
            MLP(
                self.last_size,
                final_n,
                self.classification_structure,
                adn_fn=get_adn_fn(1, "batch", "gelu"),
            )
        )

    def forward_features(
        self, X: torch.Tensor, *args, **kwargs
    ) -> torch.Tensor:
        """
        Forward method for features.

        Args:
            X (torch.Tensor): input tensor
            args, kwargs: features passed to the feature extraction module.

        Returns:
            torch.Tensor: output (classification or features)
        """
        return self.forward(X, *args, **kwargs, return_features=True)

    def forward(
        self, X: torch.Tensor, return_features: bool = False, *args, **kwargs
    ) -> torch.Tensor:
        """
        Forward method.

        Args:
            X (torch.Tensor): input tensor
            return_features (bool, optional): returns the features rather than
                the classification_head output. Defaults to False.
            args, kwargs: features passed to the feature extraction module.

        Returns:
            torch.Tensor: output (classification or features)
        """
        features = self.gp(self.feature_extraction(X, *args, **kwargs))
        if return_features is True:
            return features
        classification = self.classification_layer(features)
        return classification


class OrdNet(CatNet):
    """
    Same as CatNet but the output is ordinal.
    """

    def __init__(self, *args, **kwargs):
        """
        Args:
            args, kwargs: same arguments as CatNet.
        """
        super().__init__(*args, **kwargs)

    def init_classification_layer(self):
        self.gp = GlobalPooling()
        self.classification_layer = MLP(
            self.last_size,
            1,
            self.classification_structure,
            adn_fn=get_adn_fn(1, "batch", "gelu"),
        )
        # remove bias from last layer
        self.classification_layer.op[-1].bias.data.zero_()
        self.classification_layer.op[-1].bias.data.requires_grad = False

        # initialize bias as described in the CORAL paper
        self.bias = torch.nn.parameter.Parameter(
            torch.arange(self.n_classes - 1, 0, -1)
            .float()
            .divide(self.n_classes - 1),
            requires_grad=True,
        )
        self.last_act = torch.nn.Sigmoid()

    def forward_features(self, X: torch.Tensor) -> torch.Tensor:
        """
        Forward method for features.

        Args:
            X (torch.Tensor): input tensor

        Returns:
            torch.Tensor: output (classification or features)
        """
        return self.forward(X, return_features=True)

    def forward(
        self, X: torch.Tensor, return_features: bool = False
    ) -> torch.Tensor:
        """
        Forward method.

        Args:
            X (torch.Tensor): input tensor
            return_features (bool, optional): returns the features rather than
                the classification_head output. Defaults to False.

        Returns:
            torch.Tensor: output (classification or features)
        """
        features = self.gp(self.feature_extraction(X))
        if return_features is True:
            return features
        p_general = self.classification_layer(features)
        p_ordinal = p_general + self.bias
        return p_ordinal


class SegCatNet(torch.nn.Module):
    """
    Uses the bottleneck and final layer features from a U-Net module
    to train a classifier. The `u_net` module should have a `forward`
    method that can take a `return_features` argument that, when set to
    True, returns a tuple of tensors: prediction, final layer features
    (before prediction) and bottleneck features.
    """

    def __init__(
        self,
        spatial_dim: int,
        u_net: torch.nn.Module,
        n_input_channels: int,
        n_features_backbone: int,
        n_features_final_layer: int,
        n_classes: int,
    ):
        """
        Args:
            spatial_dim (int): number of spatial dimensions.
            u_net (torch.nn.Module): U-Net module.
            n_input_channels (int): number of input channels.
            n_features_backbone (int): number of channels in the U-Net
                backbone.
            n_features_final_layer (int): number of features in the U-Net final
                layer.
            n_classes (int): number of classes.
        """
        super().__init__()
        self.spatial_dim = spatial_dim
        self.u_net = u_net
        self.n_input_channels = n_input_channels
        self.n_features_backbone = n_features_backbone
        self.n_features_final_layer = n_features_final_layer
        self.n_classes = n_classes

        if self.n_classes == 2:
            self.nc = 1
        else:
            self.nc = self.n_classes

        self.init_final_layer_classification()
        self.init_bottleneck_classification()
        self.init_weighted_average()

    def init_final_layer_classification(self):
        d = self.n_features_final_layer
        input_d = d
        inter_d = self.n_features_final_layer * 2
        structure = [
            [input_d, inter_d, 3, 2],
            [d * 2, d * 2, 3, 2],
            [d * 4, d * 4, 3, 2],
        ]
        prediction_structure = [d * 4, d * 4, d * 4]
        self.resnet_backbone_args = {
            "spatial_dim": self.spatial_dim,
            "in_channels": input_d,
            "structure": structure,
            "maxpool_structure": [2 for _ in structure],
            "res_type": "resnet",
            "adn_fn": get_adn_fn(self.spatial_dim, "batch", "swish", 0.1),
        }
        self.resnet_prediction_args = {
            "in_channels": structure[-1][0],
            "structure": prediction_structure,
            "adn_fn": get_adn_fn(1, "batch", "swish", 0.1),
        }
        self.final_layer_classifier = torch.nn.Sequential(
            ResNet(self.resnet_backbone_args, self.resnet_prediction_args),
            torch.nn.Linear(d * 4, self.nc, bias=False),
        )

    def init_bottleneck_classification(self):
        d = self.n_features_backbone
        self.bottleneck_prediction_structure = [d, d * 2, d * 4, d * 2, d]

        self.bottleneck_classifier = torch.nn.Sequential(
            ProjectionHead(
                d,
                self.bottleneck_prediction_structure,
                adn_fn=get_adn_fn(1, "batch", "swish", 0.1),
            ),
            torch.nn.Linear(d, self.nc, bias=False),
        )

    def init_weighted_average(self):
        self.weighted_average = torch.nn.Linear(
            self.nc * 2, self.nc, bias=False
        )

    def forward(self, X, **kwargs):
        times = {}
        times["a"] = time.time()
        with torch.no_grad():
            pred, final_layer, bottleneck = self.u_net.forward(
                X, return_features=True, **kwargs
            )
        times["b"] = time.time()

        class_fl = self.final_layer_classifier(torch.cat([final_layer], axis=1))
        times["c"] = time.time()
        class_bn = self.bottleneck_classifier(bottleneck)
        times["d"] = time.time()
        features = torch.cat([class_fl, class_bn], axis=1)
        classification = self.weighted_average(features)
        times["e"] = time.time()

        return classification


class UNetEncoder(UNet):
    """
    U-Net encoder for classification.
    """

    def __init__(
        self,
        head_structure: List[int],
        n_classes: int,
        head_adn_fn: Callable = None,
        *args,
        **kwargs,
    ):
        """
        Args:
            head_structure (List[int]): structure for the prediction head.
            n_classes (int): number of classes.
            head_adn_fn (Callable, optional): activation-dropout-normalization
                function for the prediction head. Defaults to None (no
                function).
        """
        self.head_structure = head_structure
        self.n_classes = n_classes
        self.head_adn_fn = head_adn_fn

        kwargs["encoder_only"] = True
        super().__init__(*args, **kwargs)

        self.n_features = self.depth[-1]
        if self.head_structure is not None:
            self.prediction_head = MLP(
                self.n_features,
                self.n_classes,
                self.head_structure,
                self.head_adn_fn,
            )
        else:
            self.prediction_head = None

        self.initialize_head()
        self.output_features = self.n_features

    def initialize_head(self):
        if self.n_classes == 2:
            nc = 1
        else:
            nc = self.n_classes
        if self.head_structure is not None:
            self.prediction_head = MLP(
                self.n_features, nc, self.head_structure, self.head_adn_fn
            )
        else:
            self.prediction_head = None

    def forward_features(self, X: torch.Tensor) -> torch.Tensor:
        """
        Forward pass for features.

        Args:
            X (torch.Tensor): input tensor

        Returns:
            torch.Tensor
        """
        return self.forward_features(X, return_features=True)

    def forward(
        self, X: torch.Tensor, return_features: bool = False
    ) -> torch.Tensor:
        """
        Forward pass for this class.

        Args:
            X (torch.Tensor): input tensor
            return_features (bool, optional): returns the features rather than
                the classification_head output. Defaults to False.

        Returns:
            torch.Tensor
        """
        encoding_out = []
        curr = X
        for op, op_ds in self.encoding_operations:
            curr = op(curr)
            encoding_out.append(curr)
            curr = op_ds(curr)
        if self.prediction_head is None:
            return curr
        out = curr.flatten(start_dim=2).max(-1).values
        if return_features is True:
            return out
        out = self.prediction_head(out)
        return out


class ViTClassifier(ViT):
    """
    Implementation of the vision transformer (ViT) as a classifier.
    """

    def __init__(
        self, n_classes: int, use_class_token: bool = False, *args, **kwargs
    ):
        """
        Args:
            n_classes (int): number of classses.
            use_class_token (Union[str,bool], optional): whether a class token
                is being used. If "seqpool" uses the SeqPool method described
                in the compact convolutional transformers paper. Defaults to
                False.
            args, kwargs: args that are to be used to parametrize the ViT.
        """
        if use_class_token == "seqpool":
            kwargs["use_class_token"] = False
            self.use_seq_pool = True
        else:
            kwargs["use_class_token"] = use_class_token
            self.use_seq_pool = False
        super().__init__(*args, **kwargs)
        self.n_classes = n_classes

        if self.n_classes == 2:
            nc = 1
        else:
            nc = self.n_classes
        if self.use_seq_pool is True:
            self.seqpool = SeqPool(self.input_dim_primary)
        self.classification_layer = torch.nn.Sequential(
            MLP(
                self.input_dim_primary,
                nc,
                [self.input_dim_primary for _ in range(1)],
                adn_fn=get_adn_fn(1, "layer", "gelu"),
            )
        )
        self.output_features = self.input_dim_primary

    def forward_features(self, X: torch.Tensor) -> torch.Tensor:
        """
        Forward pass for features.

        Args:
            X (torch.Tensor): tensor of shape
                [-1,self.in_channels,*self.image_size]

        Returns:
            torch.Tensor: feature tensor.
        """
        return self.forward(X, return_features=True)

    def forward(
        self, X: torch.Tensor, return_features: bool = False
    ) -> torch.Tensor:
        """
        Forward pass.

        Args:
            X (torch.Tensor): tensor of shape
                [-1,self.in_channels,*self.image_size]
            return_features (bool, optional): returns the features rather than
                the classification_head output. Defaults to False.

        Returns:
            torch.Tensor: output tensor.
        """
        embeded_X, _ = super().forward(X)
        if self.n_registers > 0:
            embeded_X = embeded_X[:, self.n_registers :]
        if self.use_seq_pool is True:
            embeded_X = self.seqpool(embeded_X).squeeze(1)
        elif self.use_class_token is True:
            embeded_X = embeded_X[:, 0]
        else:
            embeded_X = embeded_X.mean(1)
        if return_features is True:
            return embeded_X
        classification = self.classification_layer(embeded_X)
        return classification


class FactorizedViTClassifier(FactorizedViT):
    """
    Implementation of a factorized ViT. The fundamental aspect of the
    factorized ViT is its separate processing within and between slice
    information. More concretely, for a given input tensor with shape
    [b,c,h,w,d]:

        1. The slices are linearly embeded as [b,d,(h/x*w/y),(x*y*c)]
        2. A transformer is applied to this (to the last two dimensions of the
            input)
        3. The information in the tensor is aggregated (using either a class
            token or MAP) and [b,d,(h/x*w/y),(x*y*c)] -> [b,(h/x*w/y),(x*y*c)]
        4. A transformer is applied to this in a more standard fashion.

    Since the arguments are similar between VitClassifier and
    FactorizedViTClassifier, the number of blocks (number_of_blocks) is split
    between the first transformer block (within slice) and second transformer
    block (between slices).
    """

    def __init__(
        self, n_classes: int, use_class_token: bool = False, *args, **kwargs
    ):
        """
        Args:
            n_classes (int): number of classses.
            use_class_token (Union[str,bool], optional): whether a class token
                is being used. If "seqpool" uses the SeqPool method described
                in the compact convolutional transformers paper. Defaults to
                False.
            args, kwargs: args that are to be used to parametrize the ViT.
        """
        if use_class_token == "seqpool":
            kwargs["use_class_token"] = False
            self.use_seq_pool = True
        else:
            kwargs["use_class_token"] = use_class_token
            self.use_seq_pool = False
        super().__init__(*args, **kwargs)
        self.n_classes = n_classes

        if self.n_classes == 2:
            nc = 1
        else:
            nc = self.n_classes
        if self.use_seq_pool is True:
            self.seqpool = SeqPool(self.input_dim_primary)
        self.classification_layer = torch.nn.Sequential(
            MLP(
                self.input_dim_primary,
                nc,
                [self.input_dim_primary for _ in range(1)],
                adn_fn=get_adn_fn(1, "layer", "gelu"),
            )
        )
        self.output_features = self.input_dim_primary

    def forward_features(self, X: torch.Tensor) -> torch.Tensor:
        """
        Forward pass for features.

        Args:
            X (torch.Tensor): tensor of shape
                [-1,self.in_channels,*self.image_size]

        Returns:
            torch.Tensor: feature tensor.
        """
        return self.forward(X, return_features=True)

    def forward(
        self, X: torch.Tensor, return_features: bool = False
    ) -> torch.Tensor:
        """
        Forward pass.

        Args:
            X (torch.Tensor): tensor of shape
                [-1,self.in_channels,*self.image_size]
            return_features (bool, optional): returns the features rather than
                the classification_head output. Defaults to False.

        Returns:
            torch.Tensor: output tensor.
        """
        embeded_X = super().forward(X)
        if self.n_registers > 0:
            embeded_X = embeded_X[:, self.n_registers :]
        if self.use_seq_pool is True:
            embeded_X = self.seqpool(embeded_X).squeeze(1)
        elif self.use_class_token is True:
            embeded_X = embeded_X[:, 0]
        else:
            embeded_X = embeded_X.mean(1)
        if return_features is True:
            return embeded_X
        classification = self.classification_layer(embeded_X)
        return classification


class MONAIViTClassifier(torch.nn.Module):
    """
    Small wraper around the MONAI ViT as a classifier with the same arguments
    as those used in ViT and FactorizedViT.
    """

    def __init__(
        self, n_classes: int, use_class_token: bool = False, *args, **kwargs
    ):
        import monai

        kwargs["use_class_token"] = use_class_token
        super().__init__()

        self.n_classes = n_classes
        if self.n_classes == 2:
            nc = 1
        else:
            nc = self.n_classes

        self.network = monai.networks.nets.vit.ViT(
            in_channels=kwargs["in_channels"],
            img_size=[int(x) for x in kwargs["image_size"]],
            patch_size=kwargs["patch_size"],
            hidden_size=kwargs["hidden_dim"],
            mlp_dim=kwargs["mlp_structure"][0],
            num_layers=kwargs["number_of_blocks"],
            num_heads=kwargs["n_heads"],
            pos_embed="conv",
            classification=True,
            num_classes=nc,
        )

    def forward(self, X: torch.Tensor) -> Tuple[torch.Tensor, TensorList]:
        """
        Forward pass.

        Args:
            X (torch.Tensor): tensor of shape
                [-1,self.in_channels,*self.image_size]

        Returns:
            torch.Tensor: tensor of shape [...,self.input_dim_primary]
        """
        return self.network(X)[0]


class TabularClassifier(torch.nn.Module):
    """
    Simple DL tabular classifier."""

    def __init__(
        self,
        n_features: int,
        mlp_structure: List[int],
        mlp_adn_fn: Callable,
        n_classes: int,
        feature_means: torch.Tensor = None,
        feature_stds: torch.Tensor = None,
    ):
        """
        Args:
            n_features (int): number of input features.
            mlp_structure (List[int]): structure of the MLP.
            mlp_adn_fn (Callable): ADN function for the MLP.
            n_classes (int): number of output classes/features.
            feature_means (torch.Tensor, optional): means of the features for
                normalization. Defaults to None.
            feature_stds (torch.Tensor, optional): standard deviation of the input
                features for normalization. Defaults to None.
        """
        super().__init__()
        self.n_features = n_features
        self.mlp_structure = mlp_structure
        self.mlp_adn_fn = mlp_adn_fn
        self.n_classes = n_classes
        self.feature_means = feature_means
        self.feature_stds = feature_stds

        self.init_mlp()
        self.init_normalization_parameters()
        self.output_features = 1 if self.n_classes == 2 else self.n_classes

    def init_mlp(self):
        nc = 1 if self.n_classes == 2 else self.n_classes
        self.mlp = MLP(
            input_dim=self.n_features,
            output_dim=nc,
            structure=self.mlp_structure,
            adn_fn=self.mlp_adn_fn,
        )

    def init_normalization_parameters(self):
        if self.feature_means is None:
            feature_means = torch.zeros([1, self.n_features])
        else:
            feature_means = torch.as_tensor(self.feature_means).reshape(1, -1)
        if self.feature_stds is None:
            feature_stds = torch.ones([1, self.n_features])
        else:
            feature_stds = torch.as_tensor(self.feature_stds).reshape(1, -1)

        self.mu = torch.nn.Parameter(feature_means.float(), requires_grad=False)
        self.sigma = torch.nn.Parameter(
            feature_stds.float(), requires_grad=False
        )

    def normalize(self, X: torch.Tensor) -> torch.Tensor:
        return (X - self.mu) / self.sigma

    def forward(self, X) -> torch.Tensor:
        return self.mlp(self.normalize(X))


class HybridClassifier(torch.nn.Module):
    def __init__(
        self,
        convolutional_module: torch.nn.Module,
        tabular_module: torch.nn.Module,
    ):
        super().__init__()
        self.convolutional_module = convolutional_module
        self.tabular_module = tabular_module

        self.set_n_classes()
        self.init_weight()

    def init_weight(self):
        self.raw_weight = torch.nn.Parameter(torch.ones([1]))

    def set_n_classes(self):
        n_classes_conv = None
        n_classes_tab = None
        if hasattr(self.convolutional_module, "n_classes"):
            n_classes_conv = self.convolutional_module.n_classes
        if hasattr(self.tabular_module, "n_classes"):
            n_classes_tab = self.tabular_module.n_classes

        if all([(n_classes_conv is not None), (n_classes_tab is not None)]):
            assert (
                n_classes_conv == n_classes_tab
            ), "convolutional_module.n_classes should be the same as \
                    tabular_module.n_classes"
        if n_classes_conv is not None:
            self.n_classes = n_classes_conv
        elif n_classes_tab is not None:
            self.n_classes = n_classes_tab
        else:
            err_message = "{} or {} should be defined".format(
                "convolutional_module.n_classes", "tabular_module.n_classes"
            )
            raise ValueError(err_message)

    def forward(
        self, X_conv: torch.Tensor, X_tab: torch.Tensor
    ) -> torch.Tensor:
        class_conv = self.convolutional_module(X_conv)
        class_tab = self.tabular_module(X_tab)

        weight = F.sigmoid(self.raw_weight)
        class_out = weight * class_conv + (1 - weight) * class_tab

        return class_out
