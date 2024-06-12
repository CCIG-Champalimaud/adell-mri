import torch
from ..layers.conv_next import ConvNeXtBackbone
from ..layers.res_net import ResNetBackbone
from ..layers.vit import ViT
from ..classification import VGGBackbone
from ..layers.linear_blocks import MLP
from ..layers.adn_fn import get_adn_fn


class Discriminator(torch.nn.Module):
    def __init__(
        self,
        backbone: str = "convnext",
        n_channels: int = 1,
        additional_features: int = 0,
        additional_classification_targets: list[int] = None,
        additional_regression_targets: int = None,
        *args,
        **kwargs,
    ):
        super().__init__()
        self.net_type = backbone
        self.n_channels = n_channels
        self.additional_features = additional_features
        self.additional_classification_targets = (
            additional_classification_targets
        )
        self.additional_regression_targets = additional_regression_targets
        self.network_args = args
        self.network_kwargs = kwargs

        self.init_backbone()
        self.init_classifier()
        self.init_additional_classifiers()
        self.init_additional_regressors()

    @property
    def backbone_conversion(self):
        return {
            "convnext": ConvNeXtBackbone,
            "resnet": ResNetBackbone,
            "vgg": VGGBackbone,
            "vit": ViT,
        }

    def init_backbone(self):
        self.network_kwargs["in_channels"] = self.n_channels
        if self.net_type not in self.backbone_conversion:
            raise NotImplementedError(
                f"net_type should be one of {self.backbone_conversion}"
            )
        self.backbone = self.backbone_conversion[self.net_type](
            *self.network_args, **self.network_kwargs
        )

    def init_classifier(self):
        self.classifier = MLP(
            self.backbone.output_features + self.additional_features,
            2,
            structure=[
                self.backbone.output_features,
                self.backbone.output_features,
            ],
            adn_fn=get_adn_fn(1, "instance", "leaky_relu"),
        )

    def init_additional_classifiers(self):
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
        self, X: torch.Tensor, X_features: torch.Tensor = None
    ) -> torch.Tensor:
        additional_classifications = None
        additional_regressions = None
        X = self.backbone(X)
        if len(X.shape) > 2:
            X = X.flatten(start_dim=2).max(-1).values
        if self.additional_features > 0:
            if X_features is None:
                raise ValueError(
                    "additional_features > 0 but X_features is None"
                )
            X = torch.cat([X, X_features], dim=1)
        classification = self.classifier(X)
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
        )
