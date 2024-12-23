from typing import OrderedDict

import torch

from ...custom_types import List, ModuleList, Tuple, Union
from .batch_ensemble import BatchEnsembleWrapper
from .res_blocks import (
    ConvNeXtBlock2d,
    ConvNeXtBlock3d,
    ResidualBlock2d,
    ResidualBlock3d,
    ResNeXtBlock2d,
    ResNeXtBlock3d,
)
from .standard_blocks import ConvolutionalBlock2d, ConvolutionalBlock3d


def resnet_to_encoding_ops(res_net: List[torch.nn.Module]) -> ModuleList:
    """Convenience function generating UNet encoder from ResNet.

    Args:
        res_net (torch.nn.Module): a list of ResNet objects.

    Returns:
        encoding_operations: ModuleList of ModuleList objects containing
            pairs of convolutions and pooling operations.
    """
    backbone = [x.backbone for x in res_net]
    res_ops = [[x.input_layer, *x.operations] for x in backbone]
    res_pool_ops = [[x.first_pooling, *x.pooling_operations] for x in backbone]
    encoding_operations = [torch.nn.ModuleList([]) for _ in res_ops]
    for i in range(len(res_ops)):
        A = res_ops[i]
        B = res_pool_ops[i]
        for a, b in zip(A, B):
            encoding_operations[i].append(torch.nn.ModuleList([a, b]))
    encoding_operations = torch.nn.ModuleList(encoding_operations)
    return encoding_operations


class ResNetBackbone(torch.nn.Module):
    """
    Default ResNet backbone. Takes a `structure` and `maxpool_structure`
    to parameterize the entire network.
    """

    def __init__(
        self,
        spatial_dim: int,
        in_channels: int,
        structure: List[Tuple[int, int, int, int]],
        maxpool_structure: List[
            Union[Tuple[int, int], Tuple[int, int, int]]
        ] = None,
        padding=None,
        adn_fn: torch.nn.Module = torch.nn.Identity,
        res_type: str = "resnet",
        batch_ensemble: int = 0,
        skip_last_activation: bool = False,
    ):
        """
        Args:
            spatial_dim (int): number of dimensions.
            in_channels (int): number of input channels.
            structure (List[Tuple[int,int,int,int]]): Structure of the
                backbone. Each element of this list should contain 4 integers
                corresponding to the input channels, output channels, filter
                size and number of consecutive, identical blocks.
            maxpool_structure (List[Union[Tuple[int,int],Tuple[int,int,int]]],
                optional): The maxpooling structure used for the backbone.
                Defaults to size and stride 2 maxpooling.
            adn_fn (torch.nn.Module, optional): the
                activation-dropout-normalization module used. Defaults to
                torch.nn.Identity.
            res_type (str, optional): the type of residual operation. Can be
                either "resnet" (normal residual block) or "resnext" (ResNeXt
                block)
            batch_ensemble (int, optional): triggers batch-ensemble layers.
                Defines number of batch ensemble modules. Defaults to 0.
            skip_last_activation (bool, optional): skips the last activation in
                the ResNet backbone. Defaults to False.
        """
        super().__init__()
        self.spatial_dim = spatial_dim
        self.in_channels = in_channels
        self.structure = structure
        self.maxpool_structure = maxpool_structure
        if self.maxpool_structure is None:
            self.maxpool_structure = [2 for _ in self.structure]
        self.adn_fn = adn_fn
        self.res_type = res_type
        self.batch_ensemble = batch_ensemble
        self.skip_last_activation = skip_last_activation

        self.get_ops()
        self.init_layers()

        self.output_features = self.structure[-1][0]

    def get_ops(self):
        if self.spatial_dim == 2:
            if self.res_type == "resnet":
                self.res_op = ResidualBlock2d
            elif self.res_type == "resnext":
                self.res_op = ResNeXtBlock2d
            elif self.res_type == "convnext":
                self.res_op = ConvNeXtBlock2d
            elif self.res_type == "none":
                self.res_op = self.conv_wrapper_2d
            self.conv_op = torch.nn.Conv2d
            self.max_pool_op = torch.nn.MaxPool2d
        elif self.spatial_dim == 3:
            if self.res_type == "resnet":
                self.res_op = ResidualBlock3d
            elif self.res_type == "resnext":
                self.res_op = ResNeXtBlock3d
            elif self.res_type == "convnext":
                self.res_op = ConvNeXtBlock3d
            elif self.res_type == "none":
                self.res_op = self.conv_wrapper_3d
            self.conv_op = torch.nn.Conv3d
            self.max_pool_op = torch.nn.MaxPool3d

    def conv_wrapper_2d(
        self,
        in_channels: int,
        kernel_size: int,
        inter_channels: int,
        out_channels: int,
        adn_fn: callable,
        skip_activation: bool,
    ):
        return ConvolutionalBlock2d(
            in_channels=[in_channels],
            out_channels=[out_channels],
            kernel_size=[kernel_size],
            adn_fn=adn_fn,
            padding="same",
        )

    def conv_wrapper_3d(
        self,
        in_channels: int,
        kernel_size: int,
        inter_channels: int,
        out_channels: int,
        adn_fn: callable,
        skip_activation: bool,
    ):
        return ConvolutionalBlock3d(
            in_channels=[in_channels],
            out_channels=[out_channels],
            kernel_size=[kernel_size],
            adn_fn=adn_fn,
            padding="same",
        )

    def init_layers(self):
        f = self.structure[0][0]
        self.input_layer = torch.nn.Sequential(
            self.conv_op(self.in_channels, f, 7, padding="same"),
            self.adn_fn(f),
            self.conv_op(f, f, 3, padding="same"),
            self.adn_fn(f),
        )
        self.first_pooling = self.max_pool_op(2, 2)
        self.operations = torch.nn.ModuleList([])
        self.be_operations = torch.nn.ModuleList([])
        self.pooling_operations = torch.nn.ModuleList([])
        prev_inp = f
        for s, mp in zip(self.structure, self.maxpool_structure):
            op = torch.nn.ModuleList([])
            inp, inter, k, N = s
            op.append(self.res_op(prev_inp, k, inter, inp, self.adn_fn))
            for _ in range(1, N - 1):
                op.append(self.res_op(inp, k, inter, inp, self.adn_fn))
            if self.batch_ensemble > 0:
                op.append(self.res_op(inp, k, inter, inp, torch.nn.Identity))
                op = torch.nn.Sequential(*op)
                be_op = BatchEnsembleWrapper(
                    None, self.batch_ensemble, prev_inp, inp, self.adn_fn
                )
            else:
                op.append(self.res_op(inp, k, inter, inp, self.adn_fn))
                op = torch.nn.Sequential(*op)
                be_op = None

            prev_inp = inp
            self.operations.append(op)
            self.be_operations.append(be_op)
            self.pooling_operations.append(self.max_pool_op(mp, mp))

    def forward_with_intermediate(
        self, X: torch.Tensor, after_pool: bool = False, batch_idx: int = None
    ):
        X = self.input_layer(X)
        X = self.first_pooling(X)
        output_list = []
        for op, be_op, pool_op in zip(
            self.operations, self.be_operations, self.pooling_operations
        ):
            if self.batch_ensemble > 0:
                X = be_op(X, batch_idx, mod=op)
            else:
                X = op(X)
            pooled_X = pool_op(X)
            if after_pool is True:
                output_list.append(pooled_X)
            else:
                output_list.append(X)
            X = pooled_X
        return X, output_list

    def forward_intermediate(
        self, X: torch.Tensor, after_pool: bool = False, batch_idx: int = None
    ):
        output_list = []
        X = self.input_layer(X)
        if after_pool is False:
            output_list.append(X)
        X = self.first_pooling(X)
        if after_pool is True:
            output_list.append(X)
        for op, be_op, pooling_op in zip(
            self.operations, self.be_operations, self.pooling_operations
        ):
            if self.batch_ensemble > 0:
                X = be_op(X, batch_idx, mod=op)
            else:
                X = op(X)
            pooled_X = pooling_op(X)
            if after_pool is True:
                output_list.append(pooled_X)
            else:
                output_list.append(X)
            X = pooled_X
        return output_list

    def forward_regular(
        self, X: torch.Tensor, batch_idx: int = None
    ) -> torch.Tensor:
        X, _ = self.forward_with_intermediate(
            X, after_pool=False, batch_idx=batch_idx
        )
        return X

    def forward(
        self,
        X: torch.Tensor,
        return_intermediate: bool = False,
        after_pool: bool = False,
        batch_idx: int = None,
    ) -> torch.Tensor:
        if return_intermediate is True:
            return self.forward_with_intermediate(X, after_pool=after_pool)
        else:
            return self.forward_regular(X, batch_idx=batch_idx)


class ProjectionHead(torch.nn.Module):
    """
    Classification head. Takes a `structure` argument to parameterize
    the entire network. Takes in a [B,C,(H,W,D)] vector, flattens and
    performs convolution operations on it.
    """

    def __init__(
        self,
        in_channels: int,
        structure: List[int],
        adn_fn: torch.nn.Module = torch.nn.Identity,
    ):
        """
        Args:
            in_channels (int): number of input channels.
            structure (List[Tuple[int,int,int,int]]): Structure of the
                projection head.
            adn_fn (torch.nn.Module, optional): the
                activation-dropout-normalization module used. Defaults to
                Identity.
        """
        super().__init__()
        self.in_channels = in_channels
        self.structure = structure
        self.adn_fn = adn_fn

        self.init_head()

    def init_head(self):
        prev_d = self.in_channels
        ops = OrderedDict()
        for i, fd in enumerate(self.structure[:-1]):
            k = "linear_{}".format(i)
            ops[k] = torch.nn.Sequential(
                torch.nn.Linear(prev_d, fd), self.adn_fn(fd)
            )
            prev_d = fd
        fd = self.structure[-1]
        ops["linear_{}".format(i + 1)] = torch.nn.Linear(prev_d, fd)
        self.op = torch.nn.Sequential(ops)

    def forward(self, X):
        if len(X.shape) > 2:
            X = X.flatten(start_dim=2).max(-1).values
        o = self.op(X)
        return o


class ResNet(torch.nn.Module):
    """
    ResNet module.
    """

    def __init__(
        self,
        backbone_args: dict,
        projection_head_args: dict = None,
        prediction_head_args: dict = None,
    ):
        """
        Args:
            backbone_args (dict): parameter dict for ResNetBackbone.
            projection_head_args (dict): parameter dict for ProjectionHead.
            prediction_head_args (dict, optional): parameter dict for
                second ProjectionHead. Defaults to None.
        """
        super().__init__()
        self.backbone_args = backbone_args
        self.projection_head_args = projection_head_args
        self.prediction_head_args = prediction_head_args

        self.init_backbone()
        self.init_projection_head()
        self.init_prediction_head()

    def init_backbone(self):
        self.backbone = ResNetBackbone(**self.backbone_args)

    def init_projection_head(self):
        if self.projection_head_args is not None:
            try:
                d = self.projection_head_args["structure"][-1]
                norm_fn = self.projection_head_args["adn_fn"](d).norm_fn
            except Exception:
                norm_fn = torch.nn.LayerNorm
            self.projection_head = torch.nn.Sequential(
                ProjectionHead(**self.projection_head_args), norm_fn(d)
            )

    def init_prediction_head(self):
        if self.prediction_head_args is not None:
            self.prediction_head = ProjectionHead(**self.prediction_head_args)

    def forward_representation(self, X, *args, **kwargs):
        X = self.backbone(X, *args, **kwargs)
        return X

    def forward_representation_with_intermediate(self, X):
        X = self.backbone.forward_with_intermediate(X)
        return X

    def forward_intermediate(self, X):
        X = self.backbone.forward_intermediate(X)
        return X

    def forward(self, X, ret="projection"):
        X = self.backbone(X)
        if ret == "representation":
            return X
        X = self.projection_head(X)
        if ret == "projection":
            return X
        X = self.prediction_head(X)
        if ret == "prediction":
            return X


class ResNetSimSiam(torch.nn.Module):
    """
    Very similar to ResNet but with one peculiarity - no activation
    in the last layer of the projection head.
    """

    def __init__(
        self,
        backbone_args: dict,
        projection_head_args: dict,
        prediction_head_args: dict = None,
    ):
        """
        Args:
            backbone_args (dict): _description_
            projection_head_args (dict): _description_
            prediction_head_args (dict, optional): _description_. Defaults to None.
        """
        self.backbone_args = backbone_args
        self.projection_head_args = projection_head_args
        self.prediction_head_args = prediction_head_args

        self.init_backbone()
        self.init_projection_head()
        self.init_prediction_head()

    def init_backbone(self):
        self.backbone = ResNetBackbone(**self.backbone_args)

    def init_projection_head(self):
        self.projection_head = ProjectionHead(**self.projection_head_args)

    def init_prediction_head(self):
        if self.prediction_head_args is not None:
            self.prediction_head = ProjectionHead(**self.prediction_head_args)

    def forward(self, X, ret="projection"):
        X = self.backbone(X)
        if ret == "representation":
            return X
        X = self.projection_head(X)
        if ret == "projection":
            return X
        X = self.prediction_head(X)
        return X
