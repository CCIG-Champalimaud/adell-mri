import torch
from .classification import VGG
from ...layers.linear_blocks import MLP
from ...layers.standard_blocks import GlobalPooling

from typing import List, Tuple


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
        *args,
        **kwargs
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
        """
        super().__init__(*args, **kwargs)
        self.n_features_deconfounder = n_features_deconfounder
        self.n_cat_deconfounder = n_cat_deconfounder
        self.n_cont_deconfounder = n_cont_deconfounder

        if self.n_features_deconfounder is None:
            self.n_features_deconfounder = 0
        if self.n_cat_deconfounder is None:
            self.n_cat_deconfounder = []
        if self.n_cont_deconfounder is None:
            self.n_cont_deconfounder = 0

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
                        MLP(self.n_features_deconfounder, n_class, [])
                    )
        if self.n_features_deconfounder > 0:
            if self.n_cont_deconfounder > 0:
                self.confound_regressions = MLP(
                    self.n_features_deconfounder, self.n_cont_deconfounder, []
                )

    def forward(
        self, X: torch.Tensor, return_features: bool = False
    ) -> (
        Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]
        | torch.Tensor
    ):
        """Forward method.

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
        classification = self.classification_layer(features)
        confounder_classification = None
        confounder_regression = None
        if self.confound_classifiers is not None:
            confounder_classification = [
                confound_classifier(
                    features[:, : self.n_features_deconfounder]
                )
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
        super().__init__()
        self.key_lists = key_lists

        self.init_conversion()

    def init_conversion(self):
        self.conversions = []
        for key_list in self.key_lists:
            self.conversions.append({key: i for i, key in enumerate(key_list)})

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
