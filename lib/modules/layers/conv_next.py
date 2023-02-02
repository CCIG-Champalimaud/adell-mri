import torch

from .res_blocks import ConvNeXtBlock2d,ConvNeXtBlock3d
from .batch_ensemble import BatchEnsembleWrapper
from .regularization import LayerNorm
from .res_net import ProjectionHead

from typing import List,Tuple,Union

class ConvNeXtBackbone(torch.nn.Module):
    def __init__(
        self,
        spatial_dim:int,
        in_channels:int,
        structure:List[Tuple[int,int,int,int]],
        maxpool_structure:List[Union[Tuple[int,int],Tuple[int,int,int]]]=None,
        padding=None,
        adn_fn:torch.nn.Module=torch.nn.Identity,
        batch_ensemble:int=0):
        """Default ConvNeXt backbone. Takes a `structure` and `maxpool_structure`
        to parameterize the entire network.

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
            batch_ensemble (int, optional): triggers batch-ensemble layers. 
                Defines number of batch ensemble modules. Defaults to 0.
        """
        super().__init__()
        self.spatial_dim = spatial_dim
        self.in_channels = in_channels
        self.structure = structure
        self.maxpool_structure = maxpool_structure
        if self.maxpool_structure is None:
            self.maxpool_structure = [2 for _ in self.structure]
        self.adn_fn = adn_fn
        self.batch_ensemble = batch_ensemble
        
        self.get_ops()
        self.init_layers()

    def get_ops(self):
        if self.spatial_dim == 2:
            self.res_op = ConvNeXtBlock2d
            self.conv_op = torch.nn.Conv2d
            self.max_pool_op = torch.nn.MaxPool2d
        elif self.spatial_dim == 3:
            self.res_op = ConvNeXtBlock3d
            self.conv_op = torch.nn.Conv3d
            self.max_pool_op = torch.nn.MaxPool3d

    def init_layers(self):
        f = self.structure[0][0]
        self.input_layer = torch.nn.Sequential(
            self.conv_op(
                self.in_channels,f,4,stride=4),
            LayerNorm(f,data_format="channels_first"))
        if self.batch_ensemble > 0:
            self.input_layer = BatchEnsembleWrapper(
                self.input_layer,self.batch_ensemble,
                self.in_channels,f,self.adn_fn)
        self.operations = torch.nn.ModuleList([])
        self.pooling_operations = torch.nn.ModuleList([])
        prev_inp = f
        for s,mp in zip(self.structure,self.maxpool_structure):
            op = torch.nn.ModuleList([])
            inp,inter,k,N = s
            op.append(self.res_op(
                prev_inp,k,inter,inp,self.adn_fn))
            for _ in range(1,N-1):
                op.append(self.res_op(inp,k,inter,inp,self.adn_fn))
            if self.batch_ensemble > 0:
                op.append(self.res_op(inp,k,inter,inp,torch.nn.Identity))
                op = torch.nn.Sequential(*op)
                op = BatchEnsembleWrapper(
                    op,self.batch_ensemble,prev_inp,inp,self.adn_fn)
            else:
                op.append(self.res_op(inp,k,inter,inp,self.adn_fn))
                op = torch.nn.Sequential(*op)

            prev_inp = inp
            self.operations.append(op)
            self.pooling_operations.append(self.max_pool_op(mp,mp))
    
        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, (torch.nn.Conv2d, torch.nn.Linear)):
            torch.nn.init.trunc_normal_(m.weight, std=0.02)
            torch.nn.init.constant_(m.bias, 0)

    def forward_with_intermediate(self,X,after_pool=False):
        X = self.input_layer(X)
        output_list = []
        for op,pool_op in zip(self.operations,self.pooling_operations):
            if after_pool is False:
                X = op(X)
                output_list.append(X)
                X = pool_op(X)
            else:
                X = pool_op(op(X))
                output_list.append(X)
        return X,output_list

    def forward_regular(self,X,batch_idx=None):
        X = self.input_layer(X)
        for op,pool_op in zip(self.operations,self.pooling_operations):
            if self.batch_ensemble > 0:
                X = op(X,idx=batch_idx)
            else:
                X = op(X)
            X = pool_op(X)
        return X

    def forward(self,X,return_intermediate:bool=False,after_pool:bool=False,
                batch_idx:bool=None):
        if return_intermediate is True:
            return self.forward_with_intermediate(X,after_pool=after_pool)
        else:
            return self.forward_regular(X,batch_idx=batch_idx)

class ConvNeXt(torch.nn.Module):
    def __init__(self,
                 backbone_args:dict,
                 projection_head_args:dict,
                 prediction_head_args:dict=None):
        """Quick way of creating a ResNet.

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
        self.backbone = ConvNeXtBackbone(
            **self.backbone_args)
    
    def init_projection_head(self):
        try:
            d = self.projection_head_args["structure"][-1]
            norm_fn = self.projection_head_args["adn_fn"](d).norm_fn
        except:
            pass
        self.projection_head = torch.nn.Sequential(
            ProjectionHead(
                **self.projection_head_args),
            norm_fn(d))

    def init_prediction_head(self):
        if self.prediction_head_args is not None:
            self.prediction_head = ProjectionHead(
                **self.prediction_head_args)

    def forward(self,X,ret="projection"):
        X = self.backbone(X)
        if ret == "representation":
            return X
        X = self.projection_head(X)
        if ret == "projection":
            return X
        X = self.prediction_head(X)
        if ret == "prediction":
            return X
