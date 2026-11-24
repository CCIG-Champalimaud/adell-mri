from typing import Sequence

import torch
from monai.networks.blocks import Convolution
from monai.networks.layers import Act
from monai.networks.nets.patchgan_discriminator import PatchDiscriminator
from monai.networks.utils import normal_init

from adell_mri.modules.classification.classification import VGGBackbone
from adell_mri.modules.layers.adn_fn import get_adn_fn
from adell_mri.modules.layers.conv_next import ConvNeXtBackbone
from adell_mri.modules.layers.linear_blocks import MLP
from adell_mri.modules.layers.res_net import ResNetBackbone
from adell_mri.modules.layers.vit import ViT


class Discriminator(torch.nn.Module):
    """
    Discriminator class. Supports auxiliary classifiers/regressors and
    returning features for feature matching losses or feature extraction.
    """

    def __init__(
        self,
        backbone: str | torch.nn.Module = "convnext",
        in_channels: int = 1,
        additional_features: int = 0,
        additional_classification_targets: list[int] = None,
        additional_regression_targets: int = None,
        noise_std: float = 0.0,
        *args,
        **kwargs,
    ):
        """
        Args:
            backbone (str | torch.nn.Module, optional): module or string
                specifying the type of discriminator. If a module is specified,
                it should have an ``output_features`` attribute with the number
                of output features. Defaults to "convnext".
            in_channels (int, optional): number of input channels. Defaults to
                1.
            additional_features (int, optional): number of additional features
                for classification. These are appended to the bottleneck.
                Defaults to 0.
            additional_classification_targets (list[int], optional): list with
                number of classes for additional classification targets.
                Defaults to None.
            additional_regression_targets (int, optional): number of additional
                regression targets. Defaults to None.
            noise_std (float, optional): standard deviation for noise to be
                added to the bottleneck output for regularization. Defaults to
                0.0.
        """
        super().__init__()
        self.net_type = backbone
        self.in_channels = in_channels
        self.additional_features = additional_features
        self.additional_classification_targets = (
            additional_classification_targets
        )
        self.additional_regression_targets = additional_regression_targets
        self.noise_std = noise_std
        self.network_args = args
        self.network_kwargs = kwargs

        self.init_backbone()
        self.init_classifier()
        self.init_additional_classifiers()
        self.init_additional_regressors()

    @property
    def backbone_conversion(self) -> dict[str, callable]:
        """
        Dictionary of backbones for str backbone specification.

        Returns:
            dict[str, callable]: dictionary with backbone callables which can
                be used to produce ``torch.nn.Module`` instances.
        """
        return {
            "convnext": ConvNeXtBackbone,
            "resnet": ResNetBackbone,
            "vgg": VGGBackbone,
            "vit": ViT,
        }

    def init_backbone(self):
        """
        Initialises backones if specified as a string in ``net_type`` or
        assigns ``net_type`` to ``self.backbone`` otherwise.

        Raises:
            NotImplementedError: if ``net_type`` is a string and the string is
                not present in ``self.backbone_conversion``.
        """
        self.network_kwargs["in_channels"] = self.in_channels
        spatial_dim = self.network_kwargs.pop(
            "spatial_dim", self.network_kwargs.pop("spatial_dimensions", 2)
        )
        if isinstance(self.net_type, str):
            if self.net_type not in self.backbone_conversion:
                raise NotImplementedError(
                    f"net_type should be one of {self.backbone_conversion}"
                )
            if self.net_type == "vgg":
                self.network_kwargs["spatial_dimensions"] = spatial_dim
            elif self.net_type in ["convnext", "resnet"]:
                self.network_kwargs["spatial_dim"] = spatial_dim
            self.backbone = self.backbone_conversion[self.net_type](
                *self.network_args, **self.network_kwargs
            )
        else:
            self.backbone = self.net_type

    def init_classifier(self):
        """
        Initialises classifier.
        """
        self.classifier = MLP(
            self.backbone.output_features + self.additional_features,
            1,
            structure=[
                self.backbone.output_features,
                self.backbone.output_features,
            ],
            adn_fn=get_adn_fn(1, "instance", "leaky_relu"),
        )

    def init_additional_classifiers(self):
        """
        Initialises additional classifiers if these are specified.
        """
        if self.additional_classification_targets is not None:
            self.additional_classifiers = torch.nn.ModuleList(
                [
                    MLP(
                        self.backbone.output_features
                        + self.additional_features,
                        n_classes,
                        structure=[
                            self.backbone.output_features,
                            self.backbone.output_features,
                        ],
                        adn_fn=get_adn_fn(1, "instance", "leaky_relu"),
                    )
                    for n_classes in self.additional_classification_targets
                ]
            )

    def init_additional_regressors(self):
        """
        Initialises additional regressors if these are specified.
        """
        if self.additional_regression_targets is not None:
            self.additional_regressors = MLP(
                self.backbone.output_features + self.additional_features,
                self.additional_regression_targets,
                structure=[
                    self.backbone.output_features,
                    self.backbone.output_features,
                ],
                adn_fn=get_adn_fn(1, "instance", "leaky_relu"),
            )

    def forward(
        self, X: torch.Tensor, X_features: torch.Tensor | None = None
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Forward method. Produces the main classification (0 or 1), additional
        classifications if specified, additional regressions if specified, and
        the backbone features.

        Args:
            X (torch.Tensor): input image tensor.
            X_features (torch.Tensor | None, optional): features to be appended
                to the backbone for classifiation. Defaults to None.

        Raises:
            ValueError: if ``additional_features`` is greater than 0 and no
                ``X_features`` is specified.

        Returns:
            tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        additional_classifications = None
        additional_regressions = None
        X = self.backbone(X)
        if isinstance(X, (tuple, list)):
            X = X[0]
        bottleneck = X
        if len(X.shape) > 2:
            X = X.flatten(start_dim=2).max(-1).values
        if self.additional_features > 0:
            if X_features is None:
                raise ValueError(
                    "additional_features > 0 but X_features is None"
                )
            X = torch.cat([X, X_features], dim=1)
        classification = self.classifier(
            X
            if self.noise_std == 0
            else X + torch.randn_like(X) * self.noise_std
        )
        if self.additional_classification_targets:
            additional_classifications = [
                cl(X) for cl in self.additional_classifiers
            ]
        if self.additional_regression_targets:
            additional_regressions = self.additional_regressors(X)

        return (
            classification,
            additional_classifications,
            additional_regressions,
            bottleneck,
        )


class StridedPatchDiscriminator(torch.nn.Module):
    """
    Extends ``PatchDiscriminator`` to include strides.
    """

    def __init__(
        self,
        spatial_dims: int,
        channels: int,
        in_channels: int,
        out_channels: int = 1,
        num_layers_d: int = 3,
        kernel_size: int = 4,
        activation: str | tuple = (Act.LEAKYRELU, {"negative_slope": 0.2}),
        norm: str | tuple = "BATCH",
        bias: bool = False,
        padding: int | Sequence[int] = 1,
        dropout: float | tuple = 0.0,
        last_conv_kernel_size: int | None = None,
        strides: Sequence[int] = None,
    ) -> None:
        super().__init__()
        self.num_layers_d = num_layers_d
        self.num_channels = channels
        if last_conv_kernel_size is None:
            last_conv_kernel_size = kernel_size

        if strides is None:
            strides = [2 for _ in range(self.num_layers_d)]
        strides.append(1)

        self.add_module(
            "initial_conv",
            Convolution(
                spatial_dims=spatial_dims,
                kernel_size=kernel_size,
                in_channels=in_channels,
                out_channels=channels,
                act=activation,
                bias=True,
                norm=None,
                dropout=dropout,
                padding=padding,
                strides=strides[0],
            ),
        )

        input_channels = channels
        output_channels = channels * 2

        # Initial Layer
        for l_ in range(self.num_layers_d):
            layer = Convolution(
                spatial_dims=spatial_dims,
                kernel_size=kernel_size,
                in_channels=input_channels,
                out_channels=output_channels,
                act=activation,
                bias=bias,
                norm=norm,
                dropout=dropout,
                padding=padding,
                strides=strides[l_ + 1],
            )
            self.add_module(f"{l_}", layer)
            input_channels = output_channels
            output_channels = output_channels * 2

        # Final layer
        self.add_module(
            "final_conv",
            Convolution(
                spatial_dims=spatial_dims,
                kernel_size=last_conv_kernel_size,
                in_channels=input_channels,
                out_channels=out_channels,
                bias=True,
                conv_only=True,
                padding=int((last_conv_kernel_size - 1) / 2),
                dropout=0.0,
                strides=1,
            ),
        )

        self.apply(normal_init)

    def forward(self, x: torch.Tensor) -> list[torch.Tensor]:
        """
        Args:
            x: input tensor

        Returns:
            list of intermediate features, with the last element being the output.
        """
        out = [x]
        for submodel in self.children():
            intermediate_output = submodel(out[-1])
            out.append(intermediate_output)

        return out[1:]
