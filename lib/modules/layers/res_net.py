import torch
from .res_blocks import ResidualBlock2d
from .res_blocks import ResidualBlock3d
from .res_blocks import ResNeXtBlock2d
from .res_blocks import ResNeXtBlock3d
from .res_blocks import ConvNeXtBlock2d
from .res_blocks import ConvNeXtBlock3d
from typing import OrderedDict
from .batch_ensemble import BatchEnsembleWrapper
from ...custom_types import List,ModuleList,Tuple,Union

def resnet_to_encoding_ops(res_net:List[torch.nn.Module])->ModuleList:
    """Convenience function generating UNet encoder from ResNet.

    Args:
        res_net (torch.nn.Module): a list of ResNet objects.

    Returns:
        encoding_operations: ModuleList of ModuleList objects containing 
            pairs of convolutions and pooling operations.
    """
    backbone = [x.backbone for x in res_net]
    res_ops = [[x.input_layer,*x.operations] for x in backbone]
    res_pool_ops = [[x.first_pooling,*x.pooling_operations]
                    for x in backbone]
    encoding_operations = [torch.nn.ModuleList([]) for _ in res_ops]
    for i in range(len(res_ops)):
        A = res_ops[i]
        B = res_pool_ops[i]
        for a,b in zip(A,B):
            encoding_operations[i].append(
                torch.nn.ModuleList([a,b]))
    encoding_operations = torch.nn.ModuleList(encoding_operations)
    return encoding_operations

class ResNetBackbone(torch.nn.Module):
    def __init__(
        self,
        spatial_dim:int,
        in_channels:int,
        structure:List[Tuple[int,int,int,int]],
        maxpool_structure:List[Union[Tuple[int,int],Tuple[int,int,int]]]=None,
        padding=None,
        adn_fn:torch.nn.Module=torch.nn.Identity,
        res_type:str="resnet",
        batch_ensemble:int=0):
        """Default ResNet backbone. Takes a `structure` and `maxpool_structure`
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
            res_type (str, optional): the type of residual operation. Can be 
                either "resnet" (normal residual block) or "resnext" (ResNeXt 
                block)
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
        self.res_type = res_type
        self.batch_ensemble = batch_ensemble
        
        self.get_ops()
        self.init_layers()

    def get_ops(self):
        if self.spatial_dim == 2:
            if self.res_type == "resnet":
                self.res_op = ResidualBlock2d
            elif self.res_type == "resnext":
                self.res_op = ResNeXtBlock2d
            self.conv_op = torch.nn.Conv2d
            self.max_pool_op = torch.nn.MaxPool2d
        elif self.spatial_dim == 3:
            if self.res_type == "resnet":
                self.res_op = ResidualBlock3d
            elif self.res_type == "resnext":
                self.res_op = ResNeXtBlock3d
            self.conv_op = torch.nn.Conv3d
            self.max_pool_op = torch.nn.MaxPool3d

    def batch_ensemble_op(self,
                          input_channels,
                          inter_channels,
                          output_channels,
                          kernel_size):
        if self.batch_ensemble > 0:
            tmp_op = self.res_op(
                input_channels,kernel_size,inter_channels,
                output_channels,torch.nn.Identity)
            tmp_op = BatchEnsembleWrapper(
                tmp_op,self.batch_ensemble,input_channels,
                output_channels,self.adn_fn)
        else:
            tmp_op = self.res_op(
                input_channels,kernel_size,inter_channels,
                output_channels,self.adn_fn)
        return tmp_op

    def init_layers(self):
        f = self.structure[0][0]
        self.input_layer = torch.nn.Sequential(
            self.conv_op(
                self.in_channels,f,7,padding="same"),
            self.adn_fn(f),
            self.conv_op(
                f,f,3,padding="same"),
            self.adn_fn(f))
        self.first_pooling = self.max_pool_op(2,2)
        self.operations = torch.nn.ModuleList([])
        self.pooling_operations = torch.nn.ModuleList([])
        prev_inp = f
        for s,mp in zip(self.structure,self.maxpool_structure):
            op = torch.nn.ModuleList([])
            inp,inter,k,N = s
            op.append(self.batch_ensemble_op(prev_inp,inter,inp,k))
            for _ in range(1,N):
                op.append(self.batch_ensemble_op(inp,inter,inp,k))
            prev_inp = inp
            op = torch.nn.Sequential(*op)
            self.operations.append(op)
            self.pooling_operations.append(self.max_pool_op(mp,mp))
    
    def forward_with_intermediate(self,X,after_pool=False):
        X = self.input_layer(X)
        X = self.first_pooling(X)
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
        X = self.first_pooling(X)
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

class ResNetBackboneAlt(torch.nn.Module):
    def __init__(
        self,
        spatial_dim:int,
        in_channels:int,
        structure:List[Tuple[int,int,int,int]],
        maxpool_structure:List[Union[Tuple[int,int],Tuple[int,int,int]]]=None,
        padding=None,
        adn_fn:torch.nn.Module=torch.nn.Identity,
        res_type:str="resnet",
        batch_ensemble:int=0):
        """Default ResNet backbone. Takes a `structure` and `maxpool_structure`
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
            res_type (str, optional): the type of residual operation. Can be 
                either "resnet" (normal residual block) or "resnext" (ResNeXt 
                block)
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
        self.res_type = res_type
        self.batch_ensemble = batch_ensemble
        
        self.get_ops()
        self.init_layers()

    def get_ops(self):
        if self.spatial_dim == 2:
            if self.res_type == "resnet":
                self.res_op = ResidualBlock2d
            elif self.res_type == "resnext":
                self.res_op = ResNeXtBlock2d
            elif self.res_type == "convnext":
                self.res_op = ConvNeXtBlock2d
            self.conv_op = torch.nn.Conv2d
            self.max_pool_op = torch.nn.MaxPool2d
        elif self.spatial_dim == 3:
            if self.res_type == "resnet":
                self.res_op = ResidualBlock3d
            elif self.res_type == "resnext":
                self.res_op = ResNeXtBlock3d
            elif self.res_type == "convnext":
                self.res_op = ConvNeXtBlock3d
            self.conv_op = torch.nn.Conv3d
            self.max_pool_op = torch.nn.MaxPool3d

    def init_layers(self):
        f = self.structure[0][0]
        self.input_layer = torch.nn.Sequential(
            self.conv_op(
                self.in_channels,f,7,padding="same"),
            self.adn_fn(f),
            self.conv_op(
                f,f,3,padding="same"))
        if self.batch_ensemble > 0:
            self.input_layer = BatchEnsembleWrapper(
                self.input_layer,self.batch_ensemble,
                self.in_channels,f,self.adn_fn)
        self.first_pooling = self.max_pool_op(2,2)
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
    
    def forward_with_intermediate(self,X,after_pool=False):
        X = self.input_layer(X)
        X = self.first_pooling(X)
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
        X = self.first_pooling(X)
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

class ProjectionHead(torch.nn.Module):
    def __init__(
        self,
        in_channels:int,
        structure:List[int],
        adn_fn:torch.nn.Module=torch.nn.Identity):
        """Classification head. Takes a `structure` argument to parameterize
        the entire network. Takes in a [B,C,(H,W,D)] vector, flattens and 
        performs convolution operations on it.

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
        for i,fd in enumerate(self.structure[:-1]):
            k = "linear_{}".format(i)
            ops[k] = torch.nn.Sequential(
                torch.nn.Linear(prev_d,fd),
                self.adn_fn(fd))
            prev_d = fd
        fd = self.structure[-1]
        ops["linear_{}".format(i+1)] = torch.nn.Linear(prev_d,fd)
        self.op = torch.nn.Sequential(ops)

    def forward(self,X):
        if len(X.shape) > 2:
            X = X.flatten(start_dim=2).max(-1).values
        o = self.op(X)
        return o

class ResNet(torch.nn.Module):
    def __init__(self,
                 backbone_args:dict,
                 projection_head_args:dict=None,
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
        self.backbone = ResNetBackbone(
            **self.backbone_args)
    
    def init_projection_head(self):
        if self.prediction_head_args is not None:
            try:
                d = self.projection_head_args["structure"][-1]
                norm_fn = self.projection_head_args["adn_fn"](d).norm_fn
            except:
                norm_fn = torch.nn.LayerNorm
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

class ResNetSimSiam(torch.nn.Module):
    def __init__(self,backbone_args:dict,projection_head_args:dict,
                 prediction_head_args:dict=None):
        """Very similar to ResNet but with a few pecularities: 1) no activation
        in the last layer of the projection head and 2)

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
        self.backbone = ResNetBackbone(
            **self.backbone_args)
    
    def init_projection_head(self):
        self.projection_head = ProjectionHead(
            **self.projection_head_args)

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
        return X