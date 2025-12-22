from typing import List, Tuple

import torch

from adell_mri.modules.classification.classification import VGG, CatNet
from adell_mri.modules.layers.adn_fn import get_adn_fn
from adell_mri.modules.layers.linear_blocks import MLP
from adell_mri.modules.layers.standard_blocks import GlobalPooling


class DeconfoundedNet(VGG):
    """
    Feature deconfounder net. The base workings of this are relatively simple:
    a subset of the complete set of bottleneck features is used to
    classify/regress a given set of categorical or continuous confounders. With
    the adequate optimization (decorrelating confounder features + optimising
    confounder classification performance + optimising actual classsifier
    performance) should lead to bottleneck features which are decorrelated.
    """

    def __init__(
        self,
        n_features_deconfounder: int = None,
        n_cat_deconfounder: int | List[int] = None,
        n_cont_deconfounder: int = None,
        exclude_surrogate_variables: bool = False,
        deconfounder_structure: list[int] = None,
        *args,
        **kwargs,
    ):
        """
        Args:
            n_features_deconfounder (int, optional): number of features from
                the total that are calculated by the bottleneck. Defaults to
                None (no deconfounding).
            n_cat_deconfounder (int | List[int], optional): list containing
                the number of classes in each confounder. Defaults to None
                (no categorical deconfounding).
            n_cont_deconfounder (int, optional): number of continuous
                confounders. Defaults to None (no continuous deconfounding).
            exclude_surrogate_variables (bool, optional): whether to exclude
                surrogate variables
            deconfounder_structure (list[int], optional): structure of the
                deconfounder structure. Defaults to None (linear classifier).
        """
        self.n_features_deconfounder = n_features_deconfounder
        self.n_cat_deconfounder = n_cat_deconfounder
        self.n_cont_deconfounder = n_cont_deconfounder
        self.exclude_surrogate_variables = exclude_surrogate_variables
        self.deconfounder_structure = deconfounder_structure
        if self.exclude_surrogate_variables:
            kwargs["output_features"] = 512 - self.n_features_deconfounder
        super().__init__(*args, **kwargs)

        if self.n_features_deconfounder is None:
            self.n_features_deconfounder = 0
        if self.n_cat_deconfounder is None:
            self.n_cat_deconfounder = []
        if self.n_cont_deconfounder is None:
            self.n_cont_deconfounder = 0
        if self.deconfounder_structure is None:
            self.deconfounder_structure = []

        self.init_deconfounding_layers()
        self.gp = GlobalPooling()

    def init_deconfounding_layers(self):
        self.confound_classifiers = None
        self.confound_regressions = None
        if self.n_features_deconfounder > 0:
            self.confound_classifiers = torch.nn.ModuleList([])
            if len(self.n_cat_deconfounder) > 0:
                for n_class in self.n_cat_deconfounder:
                    self.confound_classifiers.append(
                        MLP(
                            self.n_features_deconfounder,
                            n_class,
                            self.deconfounder_structure,
                        )
                    )
            if self.n_cont_deconfounder > 0:
                self.confound_regressions = MLP(
                    self.n_features_deconfounder,
                    self.n_cont_deconfounder,
                    self.deconfounder_structure,
                )

    def forward(
        self, X: torch.Tensor, return_features: bool = False
    ) -> (
        Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]
        | torch.Tensor
    ):
        """
        Forward method.

        Args:
            X (torch.Tensor): input tensor
            return_features (bool, optional): returns the features rather than
                the classification_head output. Defaults to False.
            args, kwargs: features passed to the feature extraction module.

        Returns:
            Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor] | torch.Tensor:
                returns a tuple containing the classification, the confounder
                classifications (if self.n_cat_deconfounder is specified),
                the confounder regressior (if self.n_cont_deconfounder is
                specified) and the bottleneck features. If return_features is
                True, only the bottleneck features are returned.
        """
        features = self.conv3(self.conv2(self.conv1(X)))
        features = self.gp(features)
        if return_features is True:
            return features
        if self.exclude_surrogate_variables:
            classification = self.classification_layer(
                features[:, self.n_features_deconfounder :]
            )
        else:
            classification = self.classification_layer(features)
        confounder_classification = None
        confounder_regression = None
        if self.confound_classifiers is not None:
            confounder_classification = [
                confound_classifier(features[:, : self.n_features_deconfounder])
                for confound_classifier in self.confound_classifiers
            ]
        if self.confound_regressions is not None:
            confounder_regression = self.confound_regressions(
                features[:, : self.n_features_deconfounder]
            )
        output = (
            classification,
            confounder_classification,
            confounder_regression,
            features,
        )
        return output


class DeconfoundedNetGeneric(torch.nn.Module):
    """
    Feature deconfounder net. The base workings of this are relatively simple:
    a subset of the complete set of bottleneck features is used to
    classify/regress a given set of categorical or continuous confounders. With
    the adequate optimization (decorrelating confounder features + optimising
    confounder classification performance + optimising actual classsifier
    performance) should lead to bottleneck features which are decorrelated.

    This is similar to the previous version but takes as arguments a feature
    extraction module with a forward_features method which produces features.
    """

    def __init__(
        self,
        n_classes: int,
        in_channels: int = 1,
        feature_extraction_module: torch.nn.Module | str = "vgg",
        classification_structure: list[int] = [512, 512, 512],
        n_dim: int = 3,
        n_features_deconfounder: int = None,
        n_cat_deconfounder: int | List[int] = None,
        n_cont_deconfounder: int = None,
        exclude_surrogate_variables: bool = False,
        deconfounder_structure: list[int] = None,
        *args,
        **kwargs,
    ):
        """
        Args:
            n_classes (int): number of classes.
            in_channels (int): number of input channels. Defaults to 1.
            feature_extraction_module (torch.nn.Module): feature extraction
                module. Can be a torch module or a string corresponding to
                either "vgg" (for VGG) or "cat" (for CatNet). Defaults to
                "vgg".
            classification_structure (list[int], optional): list containing the
                number of neurons in each layer of the classification head.
                Defaults to [512, 512, 512].
            n_features_deconfounder (int, optional): number of features from
                the total that are calculated by the bottleneck. Defaults to
                None (no deconfounding).
            n_cat_deconfounder (int | List[int], optional): list containing
                the number of classes in each confounder. Defaults to None
                (no categorical deconfounding).
            n_cont_deconfounder (int, optional): number of continuous
                confounders. Defaults to None (no continuous deconfounding).
            exclude_surrogate_variables (bool, optional): whether to exclude
                surrogate variables
            deconfounder_structure (list[int], optional): structure of the
                deconfounder structure. Defaults to None (linear classifier).
        """
        super().__init__()
        self.n_classes = n_classes
        self.feature_extraction_module = feature_extraction_module
        self.in_channels = in_channels
        self.classification_structure = classification_structure
        self.n_dim = n_dim
        self.n_features_deconfounder = n_features_deconfounder
        self.n_cat_deconfounder = n_cat_deconfounder
        self.n_cont_deconfounder = n_cont_deconfounder
        self.exclude_surrogate_variables = exclude_surrogate_variables
        self.deconfounder_structure = deconfounder_structure
        self.args = args
        self.kwargs = kwargs

        if self.n_features_deconfounder is None:
            self.n_features_deconfounder = 0
        if self.n_cat_deconfounder is None:
            self.n_cat_deconfounder = []
        if self.n_cont_deconfounder is None:
            self.n_cont_deconfounder = 0
        if self.deconfounder_structure is None:
            self.deconfounder_structure = []

        self.init_feature_extractor_if_necessary()
        self.get_output_feature_size()
        self.init_deconfounding_layers()
        self.init_classification_layer()
        self.gp = GlobalPooling()

    def get_output_feature_size(self):
        if self.n_dim == 2:
            example = torch.rand(1, self.in_channels, 64, 64)
        elif self.n_dim == 3:
            example = torch.rand(1, self.in_channels, 64, 64, 16)
        self.n_output_features = (
            self.feature_extraction_module.forward_features(example).shape[1]
        )

    def init_feature_extractor_if_necessary(self):
        if isinstance(self.feature_extraction_module, str):
            if self.feature_extraction_module == "vgg":
                self.feature_extraction_module = VGG(
                    *self.args,
                    **self.kwargs,
                    in_channels=self.in_channels,
                    n_classes=self.n_classes,
                )
            elif self.feature_extraction_module == "cat":
                self.feature_extraction_module = CatNet(
                    *self.args,
                    **self.kwargs,
                    in_channels=self.in_channels,
                    n_classes=self.n_classes,
                )
            else:
                raise Exception(
                    "net_type '{}' not valid, has to be one of \
                    ['cat', 'vgg']".format(
                        self.net_type
                    )
                )

    def init_deconfounding_layers(self):
        self.confound_classifiers = None
        self.confound_regressions = None
        if self.n_features_deconfounder > 0:
            self.confound_classifiers = torch.nn.ModuleList([])
            if len(self.n_cat_deconfounder) > 0:
                for n_class in self.n_cat_deconfounder:
                    self.confound_classifiers.append(
                        MLP(
                            self.n_features_deconfounder,
                            n_class,
                            self.deconfounder_structure,
                        )
                    )
            if self.n_cont_deconfounder > 0:
                self.confound_regressions = MLP(
                    self.n_features_deconfounder,
                    self.n_cont_deconfounder,
                    self.deconfounder_structure,
                )

    def init_classification_layer(self):
        if self.exclude_surrogate_variables:
            n_out = self.n_output_features - self.n_features_deconfounder
        else:
            n_out = self.n_output_features
        n_classes = self.n_classes if self.n_classes > 2 else 1
        self.classification_layer = torch.nn.Sequential(
            MLP(
                n_out,
                n_classes,
                self.classification_structure,
                adn_fn=get_adn_fn(1, "batch", "gelu"),
            ),
        )

    def forward(
        self, X: torch.Tensor, return_features: bool = False
    ) -> (
        Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]
        | torch.Tensor
    ):
        """
        Forward method.

                Args:
                    X (torch.Tensor): input tensor
                    return_features (bool, optional): returns the features rather than
                        the classification_head output. Defaults to False.
                    args, kwargs: features passed to the feature extraction module.

                Returns:
                    Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor] | torch.Tensor:
                        returns a tuple containing the classification, the confounder
                        classifications (if self.n_cat_deconfounder is specified),
                        the confounder regressior (if self.n_cont_deconfounder is
                        specified) and the bottleneck features. If return_features is
                        True, only the bottleneck features are returned.
        """
        features = self.feature_extraction_module.forward_features(X)
        features = self.gp(features)
        if return_features is True:
            return features
        if self.exclude_surrogate_variables:
            classification = self.classification_layer(
                features[:, self.n_features_deconfounder :]
            )
        else:
            classification = self.classification_layer(features)
        confounder_classification = None
        confounder_regression = None
        if self.confound_classifiers is not None:
            confounder_classification = [
                confound_classifier(features[:, : self.n_features_deconfounder])
                for confound_classifier in self.confound_classifiers
            ]
        if self.confound_regressions is not None:
            confounder_regression = self.confound_regressions(
                features[:, : self.n_features_deconfounder]
            )
        output = (
            classification,
            confounder_classification,
            confounder_regression,
            features,
        )
        return output


class CategoricalConversion(torch.nn.Module):
    def __init__(self, key_lists: List[List[str]]):
        """
        Converts categorical variables into integer Tensors.

        Args:
            key_lists (List[List[str]]): nested list of lists, where each
            outter list element should be a list and each inner list element
            should be a categorical variable.
        """
        super().__init__()
        self.key_lists = key_lists

        self.init_conversion()

    def init_conversion(self):
        self.conversions = []
        for key_list in self.key_lists:
            self.conversions.append(
                {str(key): i for i, key in enumerate(key_list)}
            )

    def forward(self, X: List[str]) -> List[torch.Tensor]:
        assert len(X[0]) == len(self.key_lists)
        for i in range(len(X)):
            X[i] = [
                conversion[x] for x, conversion in zip(X[i], self.conversions)
            ]
        output = [
            torch.as_tensor([X[j][i] for j in range(len(X))]).long()
            for i in range(len(X[0]))
        ]
        return output
