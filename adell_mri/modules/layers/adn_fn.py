"""
Activation, dropout and normalisation builder.
"""

from typing import OrderedDict
from functools import partial

import torch

from adell_mri.modules.activations import activation_factory
from adell_mri.modules.layers.regularization import (
    LayerNormChannelsFirst,
    L2NormalizationLayer,
    LRN,
)

norm_fn_dict = {
    "batch": {
        1: torch.nn.BatchNorm1d,
        2: torch.nn.BatchNorm2d,
        3: torch.nn.BatchNorm3d,
    },
    "instance": {
        1: torch.nn.InstanceNorm1d,
        2: torch.nn.InstanceNorm2d,
        3: torch.nn.InstanceNorm3d,
    },
    "instance_affine": {
        1: partial(torch.nn.InstanceNorm1d, affine=True),
        2: partial(torch.nn.InstanceNorm2d, affine=True),
        3: partial(torch.nn.InstanceNorm3d, affine=True),
    },
    "layer": {
        1: torch.nn.LayerNorm,
        2: LayerNormChannelsFirst,
        3: LayerNormChannelsFirst,
    },
    "lrn": {
        1: LRN,
        2: LRN,
        3: LRN,
    },
    "identity": {
        1: torch.nn.Identity,
        2: torch.nn.Identity,
        3: torch.nn.Identity,
    },
    "l2": {
        1: L2NormalizationLayer,
        2: L2NormalizationLayer,
        3: L2NormalizationLayer,
    },
}


class ActDropNorm(torch.nn.Module):
    """
    Convenience function to combine activation, dropout and normalisation.
    Similar to ADN in MONAI.
    """

    def __init__(
        self,
        in_channels: int = None,
        ordering: str = "NDA",
        norm_fn: torch.nn.Module = torch.nn.BatchNorm2d,
        act_fn: torch.nn.Module = torch.nn.PReLU,
        dropout_fn: torch.nn.Module = torch.nn.Dropout,
        dropout_param: float = 0.0,
        inplace: bool = False,
    ):
        """
        Args:
            in_channels (int, optional): number of input channels. Defaults to
                None.
            ordering (str, optional): ordering of the N(ormalization),
                D(ropout) and A(ctivation) operations. Defaults to 'NDA'.
            norm_fn (torch.nn.Module, optional): torch module used for
                normalization. Defaults to torch.nn.BatchNorm2d.
            act_fn (torch.nn.Module, optional): activation function. Defaults
                to torch.nn.PReLU.
            dropout_fn (torch.nn.Module, optional): Function used for dropout.
                Defaults to torch.nn.Dropout.
            dropout_param (float, optional): parameter for dropout. Defaults
                to 0.
            inplace (bool, optional): inplace parameter for activation
                function. Defaults to True.
        """
        super().__init__()
        self.ordering = ordering
        self.norm_fn = norm_fn
        self.in_channels = in_channels
        self.act_fn = act_fn
        self.dropout_fn = dropout_fn
        self.dropout_param = dropout_param
        self.inplace = inplace

        self.name_dict = {
            "A": "activation",
            "D": "dropout",
            "N": "normalization",
        }
        self.init_layers()

    def init_layers(self):
        """
        Initiates the necessary layers.
        """
        if self.act_fn is None:
            self.act_fn = torch.nn.Identity
        if self.norm_fn is None:
            self.norm_fn = torch.nn.Identity
        if self.dropout_fn is None:
            self.dropout_fn = torch.nn.Identity

        self.op_dict = {
            "A": self.get_act_fn,
            "D": self.get_dropout_fn,
            "N": self.get_norm_fn,
        }

        self.op_list = OrderedDict()
        for k in self.ordering:
            self.op_list[self.name_dict[k]] = self.op_dict[k]()

        self.op = torch.nn.Sequential(self.op_list)

    def get_act_fn(self):
        try:
            return self.act_fn(inplace=self.inplace)
        except Exception:
            return self.act_fn()

    def get_dropout_fn(self):
        return self.dropout_fn(self.dropout_param)

    def get_norm_fn(self):
        return self.norm_fn(self.in_channels)

    def forward(self, X: torch.Tensor) -> torch.Tensor:
        """
        Forward function.

        Args:
            X (torch.Tensor)

        Returns:
            torch.Tensor
        """
        for op in self.op_list.values():
            X = op(X)
        return X


class ActDropNormBuilder:
    """
    Builder for ActDropNorm modules.
    """

    def __init__(
        self,
        ordering: str = "NDA",
        norm_fn: torch.nn.Module = torch.nn.BatchNorm2d,
        act_fn: torch.nn.Module = torch.nn.PReLU,
        dropout_fn: torch.nn.Module = torch.nn.Dropout,
        dropout_param: float = 0.0,
    ):
        """
        Args:
            ordering (str, optional): ordering of the N(ormalization),
                D(ropout) and A(ctivation) operations. Defaults to 'NDA'.
            norm_fn (torch.nn.Module, optional): torch module used for
                normalization. Defaults to torch.nn.BatchNorm2d.
            act_fn (torch.nn.Module, optional): activation function. Defaults
                to torch.nn.PReLU.
            dropout_fn (torch.nn.Module, optional): Function used for dropout.
                Defaults to torch.nn.Dropout.
            dropout_param (float, optional): parameter for dropout. Defaults
                to 0.
        """
        super().__init__()
        self.ordering = ordering
        self.norm_fn = norm_fn
        self.act_fn = act_fn
        self.dropout_fn = dropout_fn
        self.dropout_param = dropout_param

        self.name_dict = {
            "A": "activation",
            "D": "dropout",
            "N": "normalization",
        }

    def __call__(self, in_channels: int):
        return ActDropNorm(
            in_channels=in_channels,
            ordering=self.ordering,
            norm_fn=self.norm_fn,
            act_fn=self.act_fn,
            dropout_fn=self.dropout_fn,
            dropout_param=self.dropout_param,
        )


def get_adn_fn(
    spatial_dim: int,
    norm_fn: str = "batch",
    act_fn: str = "swish",
    dropout_param: float = 0.1,
) -> ActDropNormBuilder:
    """
    Returns a function that builds an ActDropNorm module.
    """
    if norm_fn not in norm_fn_dict:
        raise NotImplementedError(
            "norm_fn must be one of {}".format(norm_fn_dict.keys())
        )
    norm_fn = norm_fn_dict[norm_fn][spatial_dim]
    if isinstance(act_fn, str):
        if act_fn not in activation_factory:
            raise NotImplementedError(
                "act_fn must be function or one of {}".format(
                    activation_factory.keys()
                )
            )
        act_fn = activation_factory[act_fn]

    return ActDropNormBuilder(
        norm_fn=norm_fn, act_fn=act_fn, dropout_param=dropout_param
    )
