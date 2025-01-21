import torch
import torch.nn.functional as F

from ..classification import VGGBackbone
from ..layers.adn_fn import get_adn_fn
from ..layers.conv_next import ConvNeXtBackbone
from ..layers.linear_blocks import MLP
from ..layers.res_net import ResNetBackbone
from ..layers.vit import ViT


class Discriminator(torch.nn.Module):
    """
    Discriminator class. Supports auxiliary classifiers/regressors and
    returning features for feature matching losses or feature extraction.
    """

    def __init__(
        self,
        backbone: str | torch.nn.Module = "convnext",
        n_channels: int = 1,
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
            n_channels (int, optional): number of input channels. Defaults to
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
        self.n_channels = n_channels
        self.additional_features = additional_features
        self.additional_classification_targets = additional_classification_targets
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
        self.network_kwargs["in_channels"] = self.n_channels
        if isinstance(self.net_type, str):
            if self.net_type not in self.backbone_conversion:
                raise NotImplementedError(
                    f"net_type should be one of {self.backbone_conversion}"
                )
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
                        self.backbone.output_features + self.additional_features,
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
                raise ValueError("additional_features > 0 but X_features is None")
            X = torch.cat([X, X_features], dim=1)
        classification = self.classifier(
            X if self.noise_std == 0 else X + torch.randn_like(X) * self.noise_std
        )
        if self.additional_classification_targets:
            additional_classifications = [cl(X) for cl in self.additional_classifiers]
        if self.additional_regression_targets:
            additional_regressions = self.additional_regressors(X)

        return (
            classification,
            additional_classifications,
            additional_regressions,
            bottleneck,
        )
