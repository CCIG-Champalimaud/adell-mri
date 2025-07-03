from typing import Callable

import numpy as np
import torch

from adell_mri.modules.layers.gaussian_process import GaussianProcessLayer
from adell_mri.modules.layers.linear_blocks import MLP
from adell_mri.modules.layers.self_attention import (
    ConcurrentSqueezeAndExcite2d,
    ConcurrentSqueezeAndExcite3d,
)
from adell_mri.modules.classification.classification import CatNet


class GenericEnsemble(torch.nn.Module):
    """
    Generically combines multiple encoders to produce an ensemble model.
    """

    def __init__(
        self,
        spatial_dimensions: int,
        networks: list[torch.nn.Module],
        n_features: list[int] | int,
        head_structure: list[int],
        n_classes: int,
        head_adn_fn: Callable = None,
        sae: bool = False,
        gaussian_process: bool = False,
        split_input: bool = False,
    ):
        """
        Args:
            spatial_dimensions (int): spatial dimension of input.
            networks (list[torch.nn.Module]): list of Torch modules.
            n_features (list[int]): list of output sizes for networks.
            head_structure (list[int]): structure for the prediction head.
            n_classes (int): number of classes.
            head_adn_fn (Callable, optional): activation-dropout-normalization
                function for the prediction head. Defaults to None (no
                function).
            sae (bool, optional): applies a squeeze and excite layer to the
                output of each network. Defaults to False.
            gaussian_process (bool, optional): replaces the last layer with a
                gaussian process layer. Defaults to False.
            split_input (bool, optional): splits the input by channel and
                applies each network to each channel. Defaults to False.
        """
        super().__init__()
        self.spatial_dimensions = spatial_dimensions
        self.networks = torch.nn.ModuleList(networks)
        self.n_features = n_features
        self.head_structure = head_structure
        self.n_classes = n_classes
        self.head_adn_fn = head_adn_fn
        self.sae = sae
        self.gaussian_process = gaussian_process
        self.split_input = split_input

        if isinstance(self.n_features, int):
            self.n_features = [self.n_features for _ in self.networks]
        self.n_features_final = sum(self.n_features)
        self.initialize_sae_if_necessary()
        self.initialize_head()

    def initialize_head(self):
        if self.n_classes == 2:
            nc = 1
        else:
            nc = self.n_classes
        if self.gaussian_process is True:
            self.prediction_head = torch.nn.Sequential(
                self.head_adn_fn(self.n_features_final),
                MLP(
                    self.n_features_final,
                    self.head_structure[-1],
                    structure=self.head_structure[:-1],
                    adn_fn=self.head_adn_fn,
                ),
            )
            self.gaussian_process_head = GaussianProcessLayer(
                self.head_structure[-1], nc
            )
        else:
            self.prediction_head = torch.nn.Sequential(
                self.head_adn_fn(self.n_features_final),
                MLP(
                    self.n_features_final,
                    nc,
                    structure=self.head_structure,
                    adn_fn=self.head_adn_fn,
                ),
            )

    def initialize_sae_if_necessary(self):
        if self.sae is True:
            if self.spatial_dimensions == 2:
                self.preproc_method = torch.nn.ModuleList(
                    [ConcurrentSqueezeAndExcite2d(f) for f in self.n_features]
                )
            elif self.spatial_dimensions == 3:
                self.preproc_method = torch.nn.ModuleList(
                    [ConcurrentSqueezeAndExcite3d(f) for f in self.n_features]
                )
        else:
            self.preproc_method = torch.nn.ModuleList(
                [torch.nn.Identity() for _ in self.n_features]
            )

    def forward_features(
        self,
        X: torch.Tensor | list[torch.Tensor],
    ) -> torch.Tensor:
        return self.forward(X, return_features=True)

    def forward(
        self,
        X: torch.Tensor | list[torch.Tensor],
        return_features: bool = False,
    ) -> torch.Tensor:
        if isinstance(X, (torch.Tensor)):
            X = [X]
        if len(X) == 1:
            X = [X[0] for _ in self.networks]
        outputs = []
        for x, network, pp in zip(X, self.networks, self.preproc_method):
            if hasattr(network, "forward_features"):
                out = network.forward_features(x)
            else:
                out = network(x)
            out = pp(out)
            if len(out.shape) > 2:
                out = out.flatten(start_dim=2).max(-1).values
            outputs.append(out)
        outputs = torch.concat(outputs, 1)
        if return_features is True:
            return outputs
        output = self.prediction_head(outputs)
        if self.gaussian_process is True:
            output = self.gaussian_process_head(output)
        return output


class AveragingEnsemble(torch.nn.Module):
    """
    Generically combines the outputs of different models and returns the
    average.
    """

    def __init__(
        self,
        networks: list[torch.nn.Module],
        n_classes: int,
        idx: int = None,
    ):
        """
        Args:
            networks (list[torch.nn.Module]): list of Torch modules.
            n_classes (int): number of classes.
            idx (int, optional): index for the output in case this is a tuple.
                Defaults to None.
        """
        super().__init__()
        self.networks = torch.nn.ModuleList(networks)
        self.n_classes = n_classes
        self.idx = idx

    def forward(
        self,
        X: torch.Tensor | list[torch.Tensor],
    ) -> torch.Tensor:
        if isinstance(X, (torch.Tensor)):
            X = [X]
        if len(X) == 1:
            X = [X[0] for _ in self.networks]
        output = []
        for x, network in zip(X, self.networks):
            out = network(x)
            if self.idx is not None:
                out = out[0]
            output.append(out)
        output = sum(output) / len(output)
        if output.shape[1] == 1:
            output = torch.sigmoid(output)
        else:
            output = torch.softmax(output, 1)
        return output


class EnsembleNet(torch.nn.Module):
    """
    Creates an ensemble of networks which can be trained online. The
    input of each network can be different and the forward method supports
    predictions with missing data (as the average of all networks).
    """

    def __init__(self, cat_net_args: dict[str, int] | list[dict[str, int]]):
        """
        Args:
            cat_net_args (Union[Dict[str,int],list[Dict[str,int]]], optional):
                dictionary or list of dictionaries containing arguments for a
                CatNet models.
        """
        super().__init__()
        self.cat_net_args = cat_net_args

        self.coerce_cat_net_args_if_necessary()
        self.check_args()
        self.init_networks()
        self.define_final_activation()

    def check_args(self):
        n_classes = []
        for d in self.cat_net_args_:
            for k in d:
                if k == "n_classes":
                    n_classes.append(d[k])
        unique_classes = np.unique(n_classes)
        if len(unique_classes) != 1:
            raise Exception("Classes should be identical across CatNets")
        elif unique_classes[0] == 1:
            raise Exception(
                "n_classes == 1 not supported. If the problem is \
                binary set n_classes == 2"
            )
        else:
            self.n_classes_ = unique_classes[0]

    def coerce_cat_net_args_if_necessary(self):
        # coerce cat_net_args if necessary
        if isinstance(self.cat_net_args, dict):
            self.cat_net_args_ = [self.input_structure]
        elif isinstance(self.cat_net_args, list):
            self.cat_net_args_ = self.cat_net_args
        else:
            raise TypeError("cat_net_args must be dict or list of dicts")

    def init_networks(self):
        self.networks = torch.nn.Modulelist([])
        for c_n_a in zip(self.cat_net_args_):
            self.networks.append(CatNet(**c_n_a))

    def define_final_activation(self):
        if self.n_classes_ == 2:
            self.final_activation = torch.nn.Sigmoid()
        else:
            self.final_activation = torch.nn.Softmax(self.n_classes_, 1)

    def forward(self, X: list[torch.Tensor]):
        predictions = []
        for n, x in zip(self.networks, X):
            if x is not None:
                predictions.append(self.final_activation(n(x)))
        return sum(predictions) / len(predictions)
