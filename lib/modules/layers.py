
from typing import OrderedDict
import numpy as np
import torch
import torch.nn.functional as F
from math import floor
from ..types import *

activation_factory = {
    "elu": torch.nn.ELU,
    "hard_shrink": torch.nn.Hardshrink,
    "hard_tanh": torch.nn.Hardtanh,
    "leaky_relu": torch.nn.LeakyReLU,
    "logsigmoid": torch.nn.LogSigmoid,
    "prelu": torch.nn.PReLU,
    "relu": torch.nn.ReLU,
    "relu6": torch.nn.ReLU6,
    "rrelu": torch.nn.RReLU,
    "selu": torch.nn.SELU,
    "celu": torch.nn.CELU,
    "sigmoid": torch.nn.Sigmoid,
    "softplus": torch.nn.Softplus,
    "softshrink": torch.nn.Softshrink,
    "softsign": torch.nn.Softsign,
    "tanh": torch.nn.Tanh,
    "tanhshrink": torch.nn.Tanhshrink,
    "threshold": torch.nn.Threshold,
    "softmin": torch.nn.Softmin,
    "softmax": torch.nn.Softmax,
    "logsoftmax": torch.nn.LogSoftmax,
    "swish": torch.nn.SiLU}

def split_int_into_n(i,n):
    r = i % n
    o = [floor(i/n) for _ in range(n)]
    idx = 0
    while r > 0:
        o[idx] += 1
        r -= 1
        idx += 1
    return o

def crop_to_size(X: torch.Tensor,output_size: list) -> torch.Tensor:
    """Crops a tensor to the size given by list. Assumes the first two 
    dimensions are the batch and channel dimensions.

    Args:
        X (torch.Tensor): torch Tensor to be cropped
        output_size (list): list with the output dimensions. Should be
        smaller or identical to the current dimensions and the list length
        should be len(X.shape)

    Returns:
        torch.Tensor: a resized torch Tensor
    """
    sh = list(X.shape)[2:]
    diff = [i-j for i,j in zip(sh,output_size)]
    a = [x//2 for x in diff]
    r = [i-j for i,j in zip(diff,a)]
    b = [i-j for i,j in zip(sh,r)]
    for i,(x,y) in enumerate(zip(a,b)):
        idx = torch.LongTensor(np.r_[x:y]).to(X.device)
        X = torch.index_select(X,i+2,idx)
    return X

def get_adn_fn(spatial_dim,norm_fn="batch",
               act_fn="swish",dropout_param=0.1):
    if norm_fn == "batch":
        if spatial_dim == 1:
            norm_fn = torch.nn.BatchNorm1d
        elif spatial_dim == 2:
            norm_fn = torch.nn.BatchNorm2d
        elif spatial_dim == 3:
            norm_fn = torch.nn.BatchNorm3d
    elif norm_fn == "instance":
        if spatial_dim == 1:
            norm_fn = torch.nn.InstanceNorm1d
        elif spatial_dim == 2:
            norm_fn = torch.nn.InstanceNorm2d
        elif spatial_dim == 3:
            norm_fn = torch.nn.InstanceNorm3d
    elif norm_fn == "identity":
        norm_fn = torch.nn.Identity
    act_fn = activation_factory[act_fn]
    def adn_fn(s):
        return ActDropNorm(
            s,norm_fn=norm_fn,act_fn=act_fn,
            dropout_param=dropout_param)

    return adn_fn

def unsqueeze_to_target(x:torch.Tensor,target:torch.Tensor):
    cur,tar = len(x.shape),len(target.shape)
    if cur < tar:
        for _ in range(tar-cur):
            x = x.unsqueeze(-1)
    return x

class ActDropNorm(torch.nn.Module):
    def __init__(self,in_channels:int=None,ordering:str='NDA',
                 norm_fn: torch.nn.Module=torch.nn.BatchNorm2d,
                 act_fn: torch.nn.Module=torch.nn.PReLU,
                 dropout_fn:torch.nn.Module=torch.nn.Dropout,
                 dropout_param: float=0.):
        """Convenience function to combine activation, dropout and 
        normalisation. Similar to ADN in MONAI.

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
        """
        super().__init__()
        self.ordering = ordering
        self.norm_fn = norm_fn
        self.in_channels = in_channels
        self.act_fn = act_fn
        self.dropout_fn = dropout_fn
        self.dropout_param = dropout_param

        self.name_dict = {"A":"activation","D":"dropout","N":"normalization"}
        self.init_layers()

    def init_layers(self):
        """Initiates the necessary layers.
        """
        if self.act_fn is None:
            self.act_fn = torch.nn.Identity
        if self.norm_fn is None:
            self.norm_fn = torch.nn.Identity
        if self.dropout_fn is None:
            self.dropout_fn = torch.nn.Identity

        op_dict = {
            "A":self.get_act_fn,
            "D":self.get_dropout_fn,
            "N":self.get_norm_fn}
        
        op_list = {}
        for k in self.ordering:
            op_list[self.name_dict[k]] = op_dict[k]()
        op_list = OrderedDict(op_list)

        self.op = torch.nn.Sequential(op_list)
        
    def get_act_fn(self):
        return self.act_fn()

    def get_dropout_fn(self):
        return self.dropout_fn(self.dropout_param)
    
    def get_norm_fn(self):
        return self.norm_fn(self.in_channels)
        
    def forward(self,X:torch.Tensor)->torch.Tensor:
        """Forward function.

        Args:
            X (torch.Tensor)

        Returns:
            torch.Tensor
        """
        return self.op(X)

class ResidualBlock2d(torch.nn.Module):
    def __init__(
        self,in_channels:int,kernel_size:int,
        inter_channels:int=None,out_channels:int=None,
        adn_fn:torch.nn.Module=torch.nn.Identity):
        """Default residual block in 2 dimensions. If `out_channels`
        is different from `in_channels` then a convolution is applied to
        the skip connection to match the number of `out_channels`.

        Args:
            in_channels (int): number of input channels.
            kernel_size (int): kernel size.
            inter_channels (int): number of intermediary channels. Defaults 
                to None.
            out_channels (int): number of output channels. Defaults to None.
            adn_fn (torch.nn.Module, optional): the activation-dropout-normalization
                module used. Defaults to torch.nn.Identity.
        """
        super().__init__()
        self.in_channels = in_channels
        self.kernel_size = kernel_size
        if inter_channels is not None:
            self.inter_channels = inter_channels
        else:
            self.inter_channels = self.in_channels
        if out_channels is not None:
            self.out_channels = out_channels
        else:
            self.out_channels = self.in_channels
        self.adn_fn = adn_fn

        self.init_layers()
    
    def init_layers(self):
        if self.inter_channels is not None:
            self.op = torch.nn.Sequential(
                torch.nn.Conv2d(
                    self.in_channels,self.inter_channels,1),
                self.adn_fn(self.inter_channels),
                torch.nn.Conv2d(
                    self.inter_channels,self.inter_channels,self.kernel_size,
                    padding="same"),
                self.adn_fn(self.inter_channels),
                torch.nn.Conv2d(
                    self.inter_channels,self.in_channels,1))
        else:
            self.op = torch.nn.Sequential(
                torch.nn.Conv2d(
                    self.in_channels,self.in_channels,self.kernel_size,
                    padding="same"),
                self.adn_fn(self.in_channels),
                torch.nn.Conv2d(
                    self.in_channels,self.in_channels,self.kernel_size,
                    padding="same"))

        # convolve residual connection to match possible difference in 
        # output channels
        if self.in_channels != self.out_channels:
            self.final_op = torch.nn.Conv2d(
                self.in_channels,self.out_channels,1)
        else:
            self.final_op = torch.nn.Identity()

        self.adn_op = self.adn_fn(self.out_channels)
    
    def forward(self,X):
        return self.adn_op(self.final_op(self.op(X) + X))

class ResidualBlock3d(torch.nn.Module):
    def __init__(
        self,in_channels:int,kernel_size:int,
        inter_channels:int=None,out_channels:int=None,
        adn_fn:torch.nn.Module=torch.nn.Identity):
        """Default residual block in 3 dimensions. If `out_channels`
        is different from `in_channels` then a convolution is applied to
        the skip connection to match the number of `out_channels`.

        Args:
            in_channels (int): number of input channels.
            kernel_size (int): kernel size.
            inter_channels (int): number of intermediary channels. Defaults 
                to None.
            out_channels (int): number of output channels. Defaults to None.
            adn_fn (torch.nn.Module, optional): the activation-dropout-normalization
                module used. Defaults to torch.nn.Identity.
        """
        super().__init__()
        self.in_channels = in_channels
        self.kernel_size = kernel_size
        if inter_channels is not None:
            self.inter_channels = inter_channels
        else:
            self.inter_channels = self.in_channels
        if out_channels is not None:
            self.out_channels = out_channels
        else:
            self.out_channels = self.in_channels
        self.adn_fn = adn_fn

        self.init_layers()
    
    def init_layers(self):
        if self.inter_channels is not None:
            self.op = torch.nn.Sequential(
                torch.nn.Conv3d(
                    self.in_channels,self.inter_channels,1),
                self.adn_fn(self.inter_channels),
                torch.nn.Conv3d(
                    self.inter_channels,self.inter_channels,self.kernel_size,
                    padding="same"),
                self.adn_fn(self.inter_channels),
                torch.nn.Conv3d(
                    self.inter_channels,self.in_channels,1))
        else:
            self.op = torch.nn.Sequential(
                torch.nn.Conv3d(
                    self.in_channels,self.in_channels,self.kernel_size,
                    padding="same"),
                self.adn_fn(self.in_channels),
                torch.nn.Conv3d(
                    self.in_channels,self.in_channels,self.kernel_size,
                    padding="same"))

        # convolve residual connection to match possible difference in 
        # output channels
        if self.in_channels != self.out_channels:
            self.final_op = torch.nn.Conv3d(
                self.in_channels,self.out_channels,1)
        else:
            self.final_op = torch.nn.Identity()

        self.adn_op = self.adn_fn(self.out_channels)
    
    def forward(self,X):
        return self.adn_op(self.final_op(self.op(X) + X))

class ParallelOperationsAndSum(torch.nn.Module):
    def __init__(self,
                 operation_list: ModuleList,
                 crop_to_smallest: bool=False) -> torch.nn.Module:
        """Module that uses a set of other modules on the original input
        and sums the output of this set of other modules as the output.

        Args:
            operation_list (ModuleList): list of PyTorch Modules (i.e. 
                [torch.nn.Conv2d(16,32,3),torch.nn.Conv2d(16,32,5)])
            crop_to_smallest (bool, optional): whether the output should be
                cropped to the size of the smallest operation output

        Returns:
            torch.nn.Module: a PyTorch Module
        """
        super().__init__()
        self.operation_list = operation_list
        self.crop_to_smallest = crop_to_smallest

    def forward(self,X: torch.Tensor) -> torch.Tensor:
        """Forward pass for this Module.

        Args:
            X (torch.Tensor): 5D tensor 

        Returns:
            torch.Tensor: 5D Tensor
        """
        outputs = []
        for operation in self.operation_list:
            outputs.append(operation(X))
        if self.crop_to_smallest == True:
            sh = []
            for output in outputs:
                sh.append(list(output.shape))
            crop_sizes = np.array(sh).min(axis=0)[2:]
            for i in range(len(outputs)):
                outputs[i] = crop_to_size(outputs[i],crop_sizes)
        output = outputs[0] + outputs[1]
        if len(outputs) > 2:
            for o in outputs[2:]:
                output = output + o
        return output

class ResNeXtBlock2d(torch.nn.Module):
    def __init__(
        self,in_channels:int,kernel_size:int,
        inter_channels:int=None,out_channels:int=None,
        adn_fn:torch.nn.Module=torch.nn.Identity,n_splits:int=16):
        """Default ResNeXt block in 2 dimensions. If `out_channels`
        is different from `in_channels` then a convolution is applied to
        the skip connection to match the number of `out_channels`.

        Args:
            in_channels (int): number of input channels.
            inter_channels (int): number of intermediary channels. Defaults 
                to None.
            out_channels (int): number of output channels. Defaults to None.
            kernel_size (int): kernel size.
            adn_fn (torch.nn.Module, optional): the activation-dropout-normalization
                module used. Defaults to torch.nn.Identity.
            n_splits (int, optional): number of branches in intermediate step 
                of the ResNeXt module. Defaults to 32.
        """
        super().__init__()
        self.in_channels = in_channels
        self.kernel_size = kernel_size
        if inter_channels is not None:
            self.inter_channels = inter_channels
        else:
            self.inter_channels = self.in_channels
        if out_channels is not None:
            self.out_channels = out_channels
        else:
            self.out_channels = self.in_channels
        self.adn_fn = adn_fn
        self.n_splits = n_splits

        self.init_layers()
    
    def init_layers(self):
        if self.inter_channels is None:
            self.inter_channels = self.output_channels
        self.n_channels_splits = split_int_into_n(
            self.inter_channels,n=self.n_splits)
        self.ops = torch.nn.ModuleList([])
        for n_channels in self.n_channels_splits:
            op = torch.nn.Sequential(
                torch.nn.Conv2d(self.in_channels,n_channels,1),
                self.adn_fn(n_channels),
                torch.nn.Conv2d(
                    n_channels,n_channels,self.kernel_size,padding="same"),
                self.adn_fn(n_channels),
                torch.nn.Conv2d(n_channels,self.out_channels,1))
            self.ops.append(op)
        
        self.op = ParallelOperationsAndSum(self.ops)

        # convolve residual connection to match possible difference in 
        # output channels
        if self.in_channels != self.out_channels:
            self.skip_op = torch.nn.Conv3d(
                self.in_channels,self.out_channels,1)
        else:
            self.skip_op = torch.nn.Identity()

        self.final_op = self.adn_fn(self.out_channels)

    def forward(self,X):
        return self.final_op(self.op(X) + self.skip_op(X))

class ResNeXtBlock3d(torch.nn.Module):
    def __init__(
        self,in_channels:int,kernel_size:int,
        inter_channels:int=None,out_channels:int=None,
        adn_fn:torch.nn.Module=torch.nn.Identity,n_splits:int=32):
        """Default ResNeXt block in 2 dimensions. If `out_channels`
        is different from `in_channels` then a convolution is applied to
        the skip connection to match the number of `out_channels`.

        Args:
            in_channels (int): number of input channels.
            inter_channels (int): number of intermediary channels. Defaults 
                to None.
            out_channels (int): number of output channels. Defaults to None.
            kernel_size (int): kernel size.
            adn_fn (torch.nn.Module, optional): the activation-dropout-normalization
                module used. Defaults to torch.nn.Identity.
            n_splits (int, optional): number of branches in intermediate step 
                of the ResNeXt module. Defaults to 32.
        """
        super().__init__()
        self.in_channels = in_channels
        self.kernel_size = kernel_size
        if inter_channels is not None:
            self.inter_channels = inter_channels
        else:
            self.inter_channels = self.in_channels
        if out_channels is not None:
            self.out_channels = out_channels
        else:
            self.out_channels = self.in_channels
        self.adn_fn = adn_fn
        self.n_splits = n_splits

        self.init_layers()
    
    def init_layers(self):
        if self.inter_channels is None:
            self.inter_channels = self.output_channels
        self.n_channels_splits = split_int_into_n(
            self.inter_channels,n=self.n_splits)
        self.ops = torch.nn.ModuleList([])
        for n_channels in self.n_channels_splits:
            op = torch.nn.Sequential(
                torch.nn.Conv3d(self.in_channels,n_channels,1),
                self.adn_fn(n_channels),
                torch.nn.Conv3d(
                    n_channels,n_channels,self.kernel_size,padding="same"),
                self.adn_fn(n_channels),
                torch.nn.Conv3d(n_channels,self.out_channels,1))
            self.ops.append(op)
        
        self.op = ParallelOperationsAndSum(self.ops)

        # convolve residual connection to match possible difference in 
        # output channels
        if self.in_channels != self.out_channels:
            self.skip_op = torch.nn.Conv3d(
                self.in_channels,self.out_channels,1)
        else:
            self.skip_op = torch.nn.Identity()

        self.final_op = self.adn_fn(self.out_channels)

    def forward(self,X):
        return self.final_op(self.op(X) + self.skip_op(X))

class ResNetBackbone(torch.nn.Module):
    def __init__(
        self,
        spatial_dim:int,
        in_channels:int,
        structure:List[Tuple[int,int,int,int]],
        maxpool_structure:List[Union[Tuple[int,int],Tuple[int,int,int]]]=None,
        padding=None,
        adn_fn:torch.nn.Module=torch.nn.Identity,
        res_type:str="resnet"):
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
                ActDropNorm.
            res_type (str, optional): the type of residual operation. Can be 
                either "resnet" (normal residual block) or "resnext" (ResNeXt 
                block)
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
            op.append(self.res_op(prev_inp,k,inter,inp,self.adn_fn))
            for _ in range(1,N):
                op.append(self.res_op(inp,k,inter,inp,self.adn_fn))
            prev_inp = inp
            op = torch.nn.Sequential(*op)
            self.operations.append(op)
            self.pooling_operations.append(self.max_pool_op(mp,mp))
    
    def forward_with_intermediate(self,X,after_pool=False):
        X = self.input_layer(X)
        X = self.first_pooling(X)
        output_list = []
        for op,pool_op in zip(self.operations,self.pooling_operations):
            if after_pool == False:
                X = op(X)
                output_list.append(X)
                X = pool_op(X)
            else:
                X = pool_op(op(X))
                output_list.append(X)
        return X,output_list

    def forward_regular(self,X):
        X = self.input_layer(X)
        X = self.first_pooling(X)
        for op,pool_op in zip(self.operations,self.pooling_operations):
            X = op(X)
            X = pool_op(X)
        return X

    def forward(self,X,return_intermediate=False,after_pool=False):
        if return_intermediate == True:
            return self.forward_with_intermediate(X,after_pool=after_pool)
        else:
            return self.forward_regular(X)

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
        self.backbone = ResNetBackbone(
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

class FeaturePyramidNetworkBackbone(torch.nn.Module):
    def __init__(
        self,
        backbone:torch.nn.Module,
        spatial_dim:int,
        structure:List[Tuple[int,int,int,int]],
        maxpool_structure:List[Union[Tuple[int,int],Tuple[int,int,int]]]=None,
        adn_fn:torch.nn.Module=torch.nn.Identity):
        """Feature pyramid network. Aggregates the intermediate features from a
        given backbone with `backbone.forward(X,return_intermediate=Trues)`.

        Args:
            backbone (torch.nn.Module): backbone module. Must have a `forward`
                method that takes a `return_intermediate=True` argument, returning
                the output and the features specified in the `structure` argument.
            spatial_dim (int): number of dimensions.
            structure (List[Tuple[int,int,int,int]]): Structure of the backbone.
                Only the first three integers of each element are used, corresponding
                to input features, output features and kernel size.
            maxpool_structure (List[Union[Tuple[int,int],Tuple[int,int,int]]], optional): 
                The maxpooling structure used for the backbone. Defaults to None.
            adn_fn (torch.nn.Module, optional): the activation-dropout-normalization
                module used. Defaults to torch.nn.Identity.
        """
        super().__init__()

        self.backbone = backbone
        self.spatial_dim = spatial_dim
        self.structure = structure
        self.maxpool_structure = maxpool_structure
        if self.maxpool_structure is None:
            self.maxpool_structure = [2 for _ in self.structure]
        self.adn_fn = adn_fn
        self.n_levels = len(structure)

        self.shape_check = False
        self.resize_ops = torch.nn.ModuleList([])

        self.get_upscale_ops()
        self.init_pyramid_layers()

    def get_upscale_ops(self):
        if self.spatial_dim == 2:
            self.res_op = ResidualBlock2d
            self.upscale_op = torch.nn.ConvTranspose2d
        if self.spatial_dim == 3:
            self.res_op = ResidualBlock3d
            self.upscale_op = torch.nn.ConvTranspose3d

    def init_pyramid_layers(self):
        self.pyramid_ops = torch.nn.ModuleList([])
        self.upscale_ops = torch.nn.ModuleList([])
        prev_inp = self.structure[-1][0]
        for s,mp in zip(self.structure[-1::-1],self.maxpool_structure[-1::-1]):
            i,inter,k,N = s
            self.pyramid_ops.append(
                self.res_op(prev_inp,k,inter,i,self.adn_fn))
            self.upscale_ops.append(
                self.upscale_op(
                    i,i,mp,stride=mp))
            prev_inp = i
    
    def forward(self, X):
        X,il = self.backbone.forward(X,return_intermediate=True)
        prev_x = X
        for pyr_op,up_op,x in zip(self.pyramid_ops,self.upscale_ops,il[-1::-1]):
            prev_x = pyr_op(prev_x)
            prev_x = up_op(prev_x)
            # shouldn't be necessary but cannot be easily avoided when working with
            # MRI scans with weird numbers of slices...
            if self.shape_check == False:
                if prev_x.shape != x.shape:
                    prev_x = F.interpolate(
                        prev_x,x.shape[2:],mode='nearest')
            prev_x = x + prev_x
        return prev_x

class ConvolutionalBlock3d(torch.nn.Module):
    def __init__(self,in_channels:List[int],out_channels:List[int],
                 kernel_size:List[int],adn_fn:torch.nn.Module=torch.nn.Identity,
                 adn_args:dict={},stride:int=1,
                 padding:str="valid"):
        """Assembles a set of blocks containing convolutions followed by 
        adn_fn operations. Used to quickly build convolutional neural
        networks.

        Args:
            in_channels (List[int]): list of input channels for convolutions.
            out_channels (List[int]): list of output channels for convolutions.
            kernel_size (List[int]): list of kernel sizes.
            adn_fn (torch.nn.Module, optional): module applied after 
            convolutions. Defaults to torch.nn.Identity.
            adn_args (dict, optional): args for the module applied after 
            convolutions. Defaults to {}.
            stride (int, optional): stride for the convolutions. Defaults to 1.
            padding (str, optional): padding for the convolutions. Defaults to
            "valid".
        """

        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding
        self.adn_fn = adn_fn
        self.adn_args = adn_args

        self.initialize_layers()
    
    def initialize_layers(self):
        """Initialize the layers for this Module.
        """
        self.mod_list = torch.nn.ModuleList()
        for i,o,k in zip(self.in_channels,
                         self.out_channels,
                         self.kernel_size):
            op = torch.nn.Sequential(
                torch.nn.Conv3d(
                    i,o,k,stride=self.stride,padding=self.padding),
                self.adn_fn(o,**self.adn_args))
            self.mod_list.append(op)
        self.op = torch.nn.Sequential(*self.mod_list)
    
    def forward(self,X: torch.Tensor) -> torch.Tensor:
        """Forward pass for this Module.

        Args:
            X (torch.Tensor)

        Returns:
            torch.Tensor
        """
        return self.op(X)

class GCN2d(torch.nn.Module):
    def __init__(self,in_channels:int,
                 out_channels:int,kernel_size:int,
                 adn_fn:torch.nn.Module=torch.nn.Identity,
                 adn_args:dict={}):
        """Global convolution network module. First introduced in [1]. Useful
        with very large kernel sizes to get information from very distant
        pixels. In essence, a n*n convolution is decomposed into two separate
        branches, where one is two convolutions (1*n->n*1) and the other is
        two convolutions (n*1->1*n). After this, the result from both branches
        is combined.

        [1] https://arxiv.org/pdf/1703.02719.pdf

        Args:
            in_channels (int): number of input channels.
            out_channels (int): number of output channels.
            kernel_size (int): kernel size.
            adn_fn (torch.nn.Module, optional): module applied after 
            convolutions. Defaults to torch.nn.Identity.
            adn_args (dict, optional): args for the module applied after 
            convolutions. Defaults to {}.
        """
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.adn_fn = adn_fn
        self.adn_args = adn_args

        self.init_layers()

    def init_layers(self):
        """Initializes layers.
        """
        self.op1 = torch.nn.Sequential(
            torch.nn.Conv2d(
                self.in_channels,self.out_channels,
                [self.kernel_size,1],padding="same"),
            self.adn_fn(self.out_channels,**self.adn_args),
            torch.nn.Conv2d(
                self.out_channels,self.out_channels,
                [1,self.kernel_size],padding="same"),
                self.adn_fn(self.out_channels,**self.adn_args))
        self.op2 = torch.nn.Sequential(
            torch.nn.Conv2d(
                self.in_channels,self.out_channels,
                [1,self.kernel_size],padding="same"),
            self.adn_fn(self.out_channels,**self.adn_args),
            torch.nn.Conv2d(
                self.out_channels,self.out_channels,
                [self.kernel_size,1],padding="same"),
            self.adn_fn(self.out_channels,**self.adn_args))
    
    def forward(self,X:torch.Tensor)->torch.Tensor:
        return self.op1(X) + self.op2(X)

class Refine2d(torch.nn.Module):
    def __init__(self,in_channels:int,kernel_size:int,
                 adn_fn:torch.nn.Module=torch.nn.Identity,
                 adn_args:dict={}):
        """Refinement module from the AHNet paper [1]. Essentially a residual
        module.

        [1] https://arxiv.org/pdf/1711.08580.pdf

        Args:
            in_channels (int): number of input channels.
            kernel_size (int): number of output channels.
            adn_fn (torch.nn.Module, optional): module applied after 
            convolutions. Defaults to torch.nn.Identity.
            adn_args (dict, optional): args for the module applied after 
            convolutions. Defaults to {}.
        """
        super().__init__()
        self.in_channels = in_channels
        self.kernel_size = kernel_size
        self.adn_fn = adn_fn
        self.adn_args = adn_args

        self.init_layers()

    def init_layers(self):
        """Initializes layers.
        """
        self.op = torch.nn.Sequential(
            torch.nn.Conv2d(
                self.in_channels,self.in_channels,
                self.kernel_size,padding="same"),
            self.adn_fn(self.in_channels,**self.adn_args),
            torch.nn.Conv2d(
                self.in_channels,self.in_channels,
                self.kernel_size,padding="same"),
            self.adn_fn(self.in_channels,**self.adn_args))
    
    def forward(self,X:torch.Tensor)->torch.Tensor:
        """Forward pass for this Module.

        Args:
            X (torch.Tensor)

        Returns:
            torch.Tensor
        """
        return X + self.op(X)

class AHNetDecoderUnit3d(torch.nn.Module):
    def __init__(self,in_channels:int,
                 adn_fn:torch.nn.Module=torch.nn.Identity,
                 adn_args:dict={}):
        """3D AHNet decoder unit from the AHNet paper [1]. Combines multiple, 
        branching and consecutive convolutions. Each unit is composed of a 
        residual-like operation followed by a concatenation with the original
        input.

        [1] https://arxiv.org/pdf/1711.08580.pdf

        Args:
            in_channels (int): number of input channels.
            adn_fn (torch.nn.Module, optional): module applied after 
            convolutions. Defaults to torch.nn.Identity.
            adn_args (dict, optional): args for the module applied after 
            convolutions. Defaults to {}.
        """
        super().__init__()
        self.in_channels = in_channels
        self.adn_fn = adn_fn
        self.adn_args = adn_args

        self.init_layers()

    def init_layers(self):
        ic = [self.in_channels for _ in range(3)]
        self.op1 = ConvolutionalBlock3d(
            ic,ic,[[1,1,1],[3,3,1],[1,1,1]],
            adn_fn=self.adn_fn,adn_args=self.adn_args,padding="same")
        self.op2 = ConvolutionalBlock3d(
            ic,ic,[[1,1,1],[1,1,3],[1,1,1]],
            adn_fn=self.adn_fn,adn_args=self.adn_args,padding="same")

    def forward(self,X:torch.Tensor)->torch.Tensor:
        X_1 = self.op1(X)
        X_2 = self.op2(X_1)
        X_3 = X_1 + X_2
        out = torch.cat([X,X_3],1)
        return out

class AHNetDecoder3d(torch.nn.Module):
    def __init__(self,in_channels:int,
                 adn_fn:torch.nn.Module=torch.nn.Identity,
                 adn_args:dict={"norm_fn":torch.nn.BatchNorm3d}):
        """Three consecutive AHNetDecoderUnit3d. Can be modified to include
        more but it is hard to know what concrete improvements this may lead
        to.

        Args:
            in_channels (int): number of input channels.
            adn_fn (torch.nn.Module, optional): module applied after 
            convolutions. Defaults to torch.nn.Identity.
            adn_args (dict, optional): args for the module applied after 
            convolutions. Defaults to {"norm_fn":torch.nn.BatchNorm3d}.
        """
        super().__init__()
        self.in_channels = in_channels
        self.adn_fn = adn_fn
        self.adn_args = adn_args
        
        self.init_layers()
    
    def init_layers(self):
        """Initializes layers.
        """
        self.op = torch.nn.Sequential(
            AHNetDecoderUnit3d(
                self.in_channels,self.adn_fn,self.adn_args),
            torch.nn.Conv3d(self.in_channels*2,self.in_channels,1),
            AHNetDecoderUnit3d(
                self.in_channels,self.adn_fn,self.adn_args),
            torch.nn.Conv3d(self.in_channels*2,self.in_channels,1),
            AHNetDecoderUnit3d(
                self.in_channels,self.adn_fn,self.adn_args),
            torch.nn.Conv3d(self.in_channels*2,self.in_channels,1))

    def forward(self,X:torch.Tensor)->torch.Tensor:
        """Forward pass for this class.

        Args:
            X (torch.Tensor)

        Returns:
            torch.Tensor
        """
        return self.op(X)

class AnysotropicHybridResidual(torch.nn.Module):
    def __init__(self,spatial_dim:int,in_channels:int,kernel_size:int,
                 adn_fn:torch.nn.Module=torch.nn.Identity,
                 adn_args:dict={}):
        """A 2D residual block that can be converted to a 3D residual block by
        increasing the number of spatial dimensions in the filters. Here I also
        transfer the parameters from `adn_fn`, particularly those belonging to
        activation/batch normalization layers.

        Args:
            spatial_dim (int): number of spatial dimensions.
            in_channels (int): number of input channels.
            kernel_size (int): kernel size.
            adn_fn (torch.nn.Module, optional): module applied after 
            convolutions. Defaults to torch.nn.Identity.
            adn_args (dict, optional): args for the module applied after 
            convolutions. Defaults to {}.
        """
        super().__init__()
        self.spatial_dim = spatial_dim
        self.in_channels = in_channels
        self.kernel_size = kernel_size
        self.adn_fn = adn_fn
        self.adn_args = adn_args
        self.dim = -1

        self.init_layers()
        if self.spatial_dim == 3:
            self.convert_to_3d()

    def init_layers(self):
        """Initialize layers.
        """
        self.op = self.get_op_2d()
        self.op_ds = self.get_downsample_op_2d()

    def get_op_2d(self):
        """Creates the 2D operation.
        """
        adn_args = self.adn_args.copy()
        adn_args["norm_fn"] = torch.nn.BatchNorm2d
        return torch.nn.Sequential(
            torch.nn.Conv2d(
                self.in_channels,self.in_channels,1),
            self.adn_fn(self.in_channels,**adn_args),
            torch.nn.Conv2d(
                self.in_channels,self.in_channels,
                self.kernel_size,padding="same"),
            self.adn_fn(self.in_channels,**adn_args),
            torch.nn.Conv2d(
                self.in_channels,self.in_channels,1),
            self.adn_fn(self.in_channels,**adn_args))

    def get_op_3d(self):
        """Creates the 3D operation.
        """
        adn_args = self.adn_args.copy()
        adn_args["norm_fn"] = torch.nn.BatchNorm3d
        K = [self.kernel_size for _ in range(3)]
        K[self.dim] = 1
        return torch.nn.Sequential(
            torch.nn.Conv3d(
                self.in_channels,self.in_channels,1),
                self.adn_fn(self.in_channels,**adn_args),
                torch.nn.Conv3d(
                    self.in_channels,self.in_channels,K,padding="same"),
                self.adn_fn(self.in_channels,**adn_args),
                torch.nn.Conv3d(
                    self.in_channels,self.in_channels,1),
                self.adn_fn(self.in_channels,**adn_args))
        
    def get_downsample_op_2d(self):
        """Creates the downsampling 2D operation.
        """
        return torch.nn.Conv2d(self.in_channels,self.in_channels,2,stride=2)

    def get_downsample_op_3d(self):
        """Creates the downsampling 3D operation.
        """
        return torch.nn.Sequential(
            torch.nn.Conv3d(
                self.in_channels,self.in_channels,[2,2,1],stride=[2,2,1]),
            torch.nn.MaxPool3d([1,1,2],stride=[1,1,2]))
    
    def forward(self,X:torch.Tensor)->torch.Tensor:
        """Forward pass for this class.

        Args:
            X (torch.Tensor)

        Returns:
            torch.Tensor
        """
        out = X + self.op(X)
        out = self.op_ds(out)
        return out
    
    def convert_to_3d(self)->None:
        """Converts the layer from 2D to 3D, handling all of the necessary
        weight transfers between layers.
        """
        if self.spatial_dim == 3:
            pass
        else:
            S = self.op.state_dict()
            for k in S:
                # adds an extra dim
                if 'weight' in k and len(S[k].shape) > 2:
                    S[k] = S[k].unsqueeze(self.dim)
            S_ds = self.op_ds.state_dict()
            S_ds = {"0."+k:S_ds[k] for k in S_ds}
            for k in S_ds:
                if 'weight' in k and len(S_ds[k].shape) > 2:
                    S_ds[k] = S_ds[k].unsqueeze(self.dim)

            adn_args = self.adn_args.copy()
            adn_args["norm_fn"] = torch.nn.BatchNorm3d
            self.op = self.get_op_3d()
            self.op.load_state_dict(S)
            self.op_ds = self.get_downsample_op_3d()
            self.op_ds.load_state_dict(S_ds)
            self.spatial_dim = 3

    def convert_to_2d(self)->None:
        """Converts the layer from 3D to 2D, handling all of the necessary
        weight transfers between layers.
        """
        if self.spatial_dim == 2:
            pass
        else:
            S = self.op.state_dict()
            for k in S:
                if 'weight' in k and len(S[k].shape) > 2:
                # removes a dim
                    S[k] = S[k].squeeze(self.dim)
            S_ds = self.op_ds.state_dict()
            S_ds = {k.replace('0.',''):S_ds[k] for k in S_ds}
            for k in S_ds:
                if 'weight' in k and len(S_ds[k].shape) > 2:
                    S_ds[k] = S_ds[k].squeeze(self.dim)

            self.op = self.get_op_2d()
            self.op.load_state_dict(S)
            self.op_ds = self.get_downsample_op_2d()
            self.op_ds.load_state_dict(S_ds)
            self.spatial_dim = 2

class AnysotropicHybridInput(torch.nn.Module):
    def __init__(self,spatial_dim:int,in_channels:int,out_channels:int,
                 kernel_size:int,
                 adn_fn:torch.nn.Module=torch.nn.Identity,adn_args:dict={}):
        """A 2D residual block that can be converted to a 3D residual block by
        increasing the number of spatial dimensions in the filters. Used as the 
        input layer for AHNet. Here I also transfer the parameters from 
        `adn_fn`, particularly those belonging to activation/batch 
        normalization layers. Unlike `AnysotropicHybridResidual`, this cannot 
        be converted from 3D to 2D.

        Args:
            spatial_dim (int): number of spatial dimensions.
            in_channels (int): number of input channels.
            kernel_size (int): kernel size.
            adn_fn (torch.nn.Module, optional): module applied after 
            convolutions. Defaults to torch.nn.Identity.
            adn_args (dict, optional): args for the module applied after 
            convolutions. Defaults to {}.
        """
        super().__init__()
        self.spatial_dim = spatial_dim
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.adn_fn = adn_fn
        self.adn_args = adn_args
        self.dim = -1

        self.init_layers()
        if self.spatial_dim == 3:
            self.convert_to_3d()

    def init_layers(self):
        """Initializes layers.
        """
        self.p = int(self.kernel_size//2)
        self.op = torch.nn.Sequential(
            torch.nn.Conv2d(
                self.in_channels,self.out_channels,
                self.kernel_size,stride=2,padding=self.p),
            self.adn_fn(self.out_channels,**self.adn_args))
    
    def forward(self,X:torch.Tensor)->torch.Tensor:
        """Forward pass for this class.

        Args:
            X (torch.Tensor)

        Returns:
            torch.Tensor
        """
        return self.op(X)
    
    def convert_to_3d(self)->None:
        """Converts the layer from 2D to 3D, handling all of the necessary
        weight transfers between layers.
        """
        if self.spatial_dim == 3:
            pass
        else:
            S = self.op.state_dict()
            for k in S:
                # adds an extra dim
                if 'weight' in k and len(S[k].shape) > 2:
                    S[k] = torch.stack([S[k],S[k],S[k]],dim=self.dim)
            K = [self.kernel_size for _ in range(3)]
            K[self.dim] = 3
            adn_args = self.adn_args.copy()
            adn_args["norm_fn"] = torch.nn.BatchNorm3d
            self.op = torch.nn.Sequential(
                torch.nn.Conv3d(
                    self.in_channels,self.out_channels,K,
                    padding=[self.p,self.p,1],stride=[2,2,1]),
                self.adn_fn(
                    self.out_channels,**adn_args))
            self.op.load_state_dict(S)
            self.spatial_dim = 3

class PyramidSpatialPooling3d(torch.nn.Module):
    def __init__(self,in_channels:int,levels:List[float]):
        """Pyramidal spatial pooling layer. In this operation, the image is 
        first downsample at different levels and convolutions are applied to 
        the downsampled image, retrieving features at different resoltuions [1].
        Quite similar to other, more recent developments encompassing atrous 
        convolutions.

        [1] https://arxiv.org/abs/1612.01105

        Args:
            in_channels (int): number of input channels.
            levels (List[float]): number of downsampling levels.
        """
        super().__init__()
        self.in_channels = in_channels
        self.levels = levels

        self.init_layers()

    def init_layers(self):
        self.pyramidal_ops = torch.nn.ModuleList([])
        for level in self.levels:
            op = torch.nn.Sequential(
                torch.nn.MaxPool3d(level,stride=level),
                torch.nn.Conv3d(self.in_channels,self.in_channels,3))
            self.pyramidal_ops.append(op)
    
    def forward(self,X):
        _,_,h,w,d = X.shape
        outs = [X]
        for op in self.pyramidal_ops:
            x = op(X)
            x = F.upsample(x,size=[h,w,d])
            outs.append(x)
        return torch.cat(outs,1)

class DepthWiseSeparableConvolution2d(torch.nn.Module):
    def __init__(self,in_channels:int,out_channels:int,
                 kernel_size:int=3,padding:int=1,
                 act_fn:torch.nn.Module=torch.nn.ReLU):
        """Depthwise separable convolution [1] for 2d inputs. Very lightweight
        version of a standard convolutional operation with a relatively small
        drop in performance. 

        [1] https://arxiv.org/abs/1902.00927v2

        Args:
            in_channels (int): number of input channels
            out_channels (int): number of output channels
            kernel_size (int, optional): kernel size. Defaults to 3.
            padding (int, optional): amount of padding. Defaults to 1.
            act_fn (torch.nn.Module, optional): activation function applied
            after convolution. Defaults to torch.nn.ReLU.
        """
        super().__init__()
        self.input_channels = in_channels
        self.output_channels = out_channels
        self.kernel_size = kernel_size
        self.padding = padding
        self.act_fn = act_fn

        self.init_layers()

    def init_layers(self):
        self.depthwise_op = torch.nn.Conv2d(
            self.input_channels,self.input_channels,
            kernel_size=self.kernel_size,padding=self.paddign,
            groups=self.input_channels)
        self.pointwise_op = torch.nn.Conv2d(
            self.input_channels,self.output_channels,
            kernel_size=1)
        self.act_op = self.act_fn()

    def forward(self,X:torch.Tensor)->torch.Tensor:
        X = self.depthwise_op(X)
        X = self.pointwise_op(X)
        X = self.act_op(X)
        return X

class DepthWiseSeparableConvolution3d(torch.nn.Module):
    def __init__(self,in_channels:int,out_channels:int,
                 kernel_size:int=3,padding:int=1,
                 act_fn:torch.nn.Module=torch.nn.ReLU):
        """Depthwise separable convolution [1] for 3d inputs. Very lightweight
        version of a standard convolutional operation with a relatively small
        drop in performance. 

        [1] https://arxiv.org/abs/1902.00927v2

        Args:
            in_channels (int): number of input channels
            out_channels (int): number of output channels
            kernel_size (int, optional): kernel size. Defaults to 3.
            padding (int, optional): amount of padding. Defaults to 1.
            act_fn (torch.nn.Module, optional): activation function applied
            after convolution. Defaults to torch.nn.ReLU.
        """
        super().__init__()
        self.input_channels = in_channels
        self.output_channels = out_channels
        self.kernel_size = kernel_size
        self.padding = padding
        self.act_fn = act_fn

        self.init_layers()

    def init_layers(self):
        self.depthwise_op = torch.nn.Conv3d(
            self.input_channels,self.input_channels,
            kernel_size=self.kernel_size,padding=self.padding,
            groups=self.input_channels)
        self.pointwise_op = torch.nn.Conv3d(
            self.input_channels,self.output_channels,
            kernel_size=1)
        self.act_op = self.act_fn()

    def forward(self,X:torch.Tensor)->torch.Tensor:
        X = self.depthwise_op(X)
        X = self.pointwise_op(X)
        X = self.act_op(X)
        return X

class SpatialPyramidPooling2d(torch.nn.Module):
    def __init__(self,in_channels:int,out_channels:int,
                 filter_sizes:List[int],
                 act_fn:torch.nn.Module=torch.nn.ReLU):
        """Spatial pyramid pooling for 2d inputs. Applies a set of differently
        sized filters to an input and then concatenates the output of each 
        filter.

        [1] https://arxiv.org/abs/1612.01105

        Args:
            in_channels (int): number of input channels
            out_channels (int): number of output channels
            filter_sizes (List[int], optional): list of kernel sizes. Defaults
            to 3.
            padding (int, optional): amount of padding. Defaults to 1.
            act_fn (torch.nn.Module, optional): activation function applied
            after convolution. Defaults to torch.nn.ReLU.
        """
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.filter_size = filter_sizes
        self.act_fn = act_fn

    def init_layers(self):
        self.layers = torch.nn.ModuleList([])
        for filter_size in self.filter_sizes:
            op = torch.nn.Sequential(
                torch.nn.Conv2d(
                    self.in_channels,self.out_channels,
                    kernel_size=filter_size,padding="same"),
                self.act_fn(),
                DepthWiseSeparableConvolution2d(
                    self.out_channels,self.out_channels,
                    kernel_size=filter_size,padding="same"),
                self.act_fn())
            self.layers.append(op)
    
    def forward(self,X:torch.Tensor)->torch.Tensor:
        outputs = []
        for layer in self.layers:
            outputs.append(layer(X))
        output = torch.cat(output,dim=1)
        return output

class SpatialPyramidPooling3d(torch.nn.Module):
    def __init__(self,in_channels:int,out_channels:int,
                 filter_sizes:List[int],
                 act_fn:torch.nn.Module=torch.nn.ReLU):
        """Spatial pyramid pooling for 3d inputs. Applies a set of differently
        sized filters to an input and then concatenates the output of each 
        filter.

        [1] https://arxiv.org/abs/1612.01105

        Args:
            in_channels (int): number of input channels
            out_channels (int): number of output channels
            filter_sizes (List[int], optional): list of kernel sizes. Defaults
            to 3.
            act_fn (torch.nn.Module, optional): activation function applied
            after convolution. Defaults to torch.nn.ReLU.
        """
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.filter_sizes = filter_sizes
        self.act_fn = act_fn

        self.init_layers()

    def init_layers(self):
        self.layers = torch.nn.ModuleList([])
        for filter_size in self.filter_sizes:
            op = torch.nn.Sequential(
                torch.nn.Conv3d(
                    self.in_channels,self.out_channels,
                    kernel_size=filter_size,padding="same"),
                self.act_fn(),
                DepthWiseSeparableConvolution3d(
                    self.out_channels,self.out_channels,
                    kernel_size=filter_size,padding="same"),
                self.act_fn())
            self.layers.append(op)
    
    def forward(self,X:torch.Tensor)->torch.Tensor:
        outputs = []
        for layer in self.layers:
            outputs.append(layer(X))
        output = torch.cat(outputs,dim=1)
        return output

class AtrousSpatialPyramidPooling2d(torch.nn.Module):
    def __init__(self,in_channels:int,out_channels:int,rates:List[int],
                 act_fn:torch.nn.Module=torch.nn.ReLU):
        """Atrous spatial pyramid pooling for 2d inputs. Applies a set of 
        differently sized dilated filters to an input and then concatenates
        the output of each  filter. Similar to SpatialPyramidPooling2d but 
        much less computationally demanding.

        [1] https://arxiv.org/abs/1612.01105

        Args:
            in_channels (int): number of input channels
            out_channels (int): number of output channels
            rates (List[int], optional): list dilation rates. 
            act_fn (torch.nn.Module, optional): activation function applied
            after convolution. Defaults to torch.nn.ReLU.
        """
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.rates = rates
        self.act_fn = act_fn

        self.init_layers()

    def init_layers(self):
        self.layers = torch.nn.ModuleList([])
        for rate in self.rates:
            op = torch.nn.Sequential(
                torch.nn.Conv2d(
                    self.in_channels,self.out_channels,
                    dilation=rate,padding="same"),
                self.act_fn(),
                DepthWiseSeparableConvolution2d(
                    self.out_channels,self.out_channels,
                    kernel_size=3,padding="same"),
                self.act_fn())
            self.layers.append(op)

    def forward(self,X:torch.Tensor)->torch.Tensor:
        outputs = []
        for layer in self.layers:
            outputs.append(layer(X))
        output = torch.cat(outputs,dim=1)
        return output

class AtrousSpatialPyramidPooling3d(torch.nn.Module):
    def __init__(self,in_channels:int,out_channels:int,rates:List[int],
                 act_fn:torch.nn.Module=torch.nn.ReLU):
        """Atrous spatial pyramid pooling for 3d inputs. Applies a set of 
        differently sized dilated filters to an input and then concatenates
        the output of each  filter. Similar to SpatialPyramidPooling3d but 
        much less computationally demanding.

        [1] https://arxiv.org/abs/1612.01105

        Args:
            in_channels (int): number of input channels
            out_channels (int): number of output channels
            rates (List[int], optional): list dilation rates. 
            act_fn (torch.nn.Module, optional): activation function applied
            after convolution. Defaults to torch.nn.ReLU.
        """
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.rates = rates
        self.act_fn = act_fn

        self.init_layers()

    def init_layers(self):
        self.layers = torch.nn.ModuleList([])
        for rate in self.rates:
            op = torch.nn.Sequential(
                torch.nn.Conv3d(
                    self.in_channels,self.out_channels,kernel_size=3,
                    dilation=rate,padding="same"),
                self.act_fn(),
                DepthWiseSeparableConvolution3d(
                    self.out_channels,self.out_channels,
                    kernel_size=3,padding="same"),
                self.act_fn())
            self.layers.append(op)

    def forward(self,X:torch.Tensor)->torch.Tensor:
        outputs = []
        for layer in self.layers:
            outputs.append(layer(X))
        output = torch.cat(outputs,dim=1)
        return output

class ReceptiveFieldBlock2d(torch.nn.Module):
    def __init__(self,in_channels:int,out_channels:int,
                 rates:List[int],
                 act_fn:torch.nn.Module=torch.nn.ReLU):
        """Receptive field block for 2d inputs [1]. A mid ground between a 
        residual operator and AtrousSpatialPyramidPooling2d - a series of 
        dilated convolutions is applied to the input, the output of these 
        dilated convolutions is concatenated and added to the input.

        [1] https://arxiv.org/pdf/1711.07767.pdf

        Args:
            in_channels (int): input channels.
            out_channels (int): output channels.
            rates (List[int]): dilation rates.
            act_fn (torch.nn.Module, optional): activation function. Defaults 
            to torch.nn.ReLU.
        """
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.rates = rates
        self.act_fn = act_fn

        self.out_c_list = split_int_into_n(
            self.out_channels,len(self.rates))
        
        self.init_layers()

    def init_layers(self):
        self.layers = torch.nn.ModuleList([])
        for rate,o in zip(self.rates,self.out_c_list):
            if rate == 1:
                op = torch.nn.Sequential(
                    torch.nn.Conv2d(
                        self.in_channels,o,kernel_size=1,padding="same"),
                    self.act_fn(),
                    torch.nn.Conv2d(o,o,kernel_size=3,padding="same"),
                    self.act_fn())
            else:
                op = torch.nn.Sequential(
                    torch.nn.Conv2d(
                        self.in_channels,o,kernel_size=1,padding="same"),
                    self.act_fn(),
                    torch.nn.Conv2d(o,o,kernel_size=rate,padding="same"),
                    self.act_fn(),
                    torch.nn.Conv2d(o,o,dilation=rate,kernel_size=3,
                                    padding="same"),
                    self.act_fn())
            self.layers.append(op)
        self.final_op = torch.nn.Conv2d(
            self.out_channels,self.out_channels,1)

    def forward(self,X:torch.Tensor)->torch.Tensor:
        outputs = []
        for layer in self.layers:
            outputs.append(layer(X))
        output = torch.cat(output,dim=1)
        output = self.final_op(output)
        output = X + output
        return output

class ReceptiveFieldBlock3d(torch.nn.Module):
    def __init__(self,in_channels:int,bottleneck_channels:int,
                 rates:List[int],
                 act_fn:torch.nn.Module=torch.nn.ReLU):
        """Receptive field block for 3d inputs [1]. A mid ground between a 
        residual operator and AtrousSpatialPyramidPooling3d - a series of 
        dilated convolutions is applied to the input, the output of these 
        dilated convolutions is concatenated and added to the input.

        [1] https://arxiv.org/pdf/1711.07767.pdf

        Args:
            in_channels (int): input channels.
            bottleneck_channels (int): number of channels in bottleneck.
            rates (List[int]): dilation rates.
            act_fn (torch.nn.Module, optional): activation function. Defaults 
            to torch.nn.ReLU.
        """
        super().__init__()
        self.in_channels = in_channels
        self.bottleneck_channels = bottleneck_channels
        self.rates = rates
        self.act_fn = act_fn

        self.out_c_list = split_int_into_n(
            self.bottleneck_channels,len(self.rates))
        
        self.init_layers()

    def init_layers(self):
        self.layers = torch.nn.ModuleList([])
        for rate,o in zip(self.rates,self.out_c_list):
            if rate == 1:
                op = torch.nn.Sequential(
                    torch.nn.Conv3d(
                        self.in_channels,o,kernel_size=1,padding="same"),
                    self.act_fn(),
                    torch.nn.Conv3d(o,o,kernel_size=3,padding="same"),
                    self.act_fn())
            else:
                op = torch.nn.Sequential(
                    torch.nn.Conv3d(
                        self.in_channels,o,kernel_size=1,padding="same"),
                    self.act_fn(),
                    torch.nn.Conv3d(o,o,kernel_size=rate,padding="same"),
                    self.act_fn(),
                    torch.nn.Conv3d(o,o,dilation=rate,kernel_size=3,
                                    padding="same"),
                    self.act_fn())
            self.layers.append(op)
        self.final_op = torch.nn.Conv3d(
            self.bottleneck_channels,self.in_channels,1)

    def forward(self,X:torch.Tensor)->torch.Tensor:
        outputs = []
        for layer in self.layers:
            outputs.append(layer(X))
        output = torch.cat(outputs,dim=1)
        output = self.final_op(output)
        output = X + output
        return output

class SpatialSqueezeAndExcite2d(torch.nn.Module):
    def __init__(self,input_channels:int):
        """Spatial squeeze and excite layer [1] for 2d inputs. Basically a 
        modular attention mechanism.

        [1] https://arxiv.org/abs/1803.02579

        Args:
            input_channels (int): number of input channels.
        """
        super().__init__()
        self.input_channels = input_channels
        
        self.init_layers()

    def init_layers(self):
        self.op = torch.nn.Sequential(
            torch.nn.Conv2d(self.input_channels,1,kernel_size=1),
            torch.nn.Sigmoid())
    
    def forward(self,X:torch.Tensor)->torch.Tensor:
        spatial_squeeze = self.op(X)
        X = X * spatial_squeeze
        return X

class SpatialSqueezeAndExcite3d(torch.nn.Module):
    def __init__(self,input_channels:int):
        """Spatial squeeze and excite layer [1] for 3d inputs. Basically a 
        modular attention mechanism.

        [1] https://arxiv.org/abs/1803.02579

        Args:
            input_channels (int): number of input channels.
        """
        super().__init__()
        self.input_channels = input_channels

        self.init_layers()
    
    def init_layers(self):
        self.op = torch.nn.Sequential(
            torch.nn.Conv3d(self.input_channels,1,kernel_size=1),
            torch.nn.Sigmoid())
    
    def forward(self,X:torch.Tensor)->torch.Tensor:
        spatial_squeeze = self.op(X)
        X = X * spatial_squeeze
        return X

class ChannelSqueezeAndExcite(torch.nn.Module):
    def __init__(self,input_channels:int):
        """Channel squeeze and excite. A self-attention mechanism at the 
        channel level.

        Args:
            input_channels (int): number of input channels.
        """
        super().__init__()
        self.input_channels = input_channels
    
        self.init_layers()

    def init_layers(self):
        I = self.input_channels
        self.op = torch.nn.Sequential(
            torch.nn.Linear(I,I),
            torch.nn.ReLU(),
            torch.nn.Linear(I,I),
            torch.nn.Sigmoid())
    
    def forward(self,X:torch.Tensor)->torch.Tensor:
        channel_average = torch.flatten(X,start_dim=2).mean(-1)
        channel_squeeze = self.op(channel_average)
        channel_squeeze = channel_squeeze.reshape(
            *channel_squeeze.shape,1,1,1)
        X = X * channel_squeeze
        return X

class ConcurrentSqueezeAndExcite2d(torch.nn.Module):
    def __init__(self,input_channels:int):
        """Concurrent squeeze and excite for 2d inputs. Combines channel and
        spatial squeeze and excite by adding the output of both.

        Args:
            input_channels (int): number of input channels.
        """
        super().__init__()
        self.input_channels = input_channels

        self.init_layers()

    def init_layers(self):
        self.spatial = SpatialSqueezeAndExcite2d(self.input_channels)
        self.channel = ChannelSqueezeAndExcite(self.input_channels)
    
    def forward(self,X):
        s = self.spatial(X)
        c = self.channel(X)
        output = s+c
        return output

class ConcurrentSqueezeAndExcite3d(torch.nn.Module):
    def __init__(self,input_channels:int):
        """Concurrent squeeze and excite for 3d inputs. Combines channel and
        spatial squeeze and excite by adding the output of both.

        Args:
            input_channels (int): number of input channels.
        """
        super().__init__()
        self.input_channels = input_channels
    
        self.init_layers()

    def init_layers(self):
        self.spatial = SpatialSqueezeAndExcite3d(self.input_channels)
        self.channel = ChannelSqueezeAndExcite(self.input_channels)
    
    def forward(self,X):
        s = self.spatial(X)
        c = self.channel(X)
        output = s+c
        return output

class GlobalPooling(torch.nn.Module):
    def __init__(self,mode:str="max"):
        """Wrapper for average and maximum pooling

        Args:
            mode (str, optional): pooling mode. Can be one of "average" or 
            "max". Defaults to "max".
        """
        super().__init__()
        self.mode = mode
        
        self.get_op()
    
    def get_op(self):
        if self.mode == "average":
            self.op = torch.mean
        elif self.mode == "max":
            self.op = torch.max
        else:
            raise "mode must be one of [average,max]"

    def forward(self,X):
        X = self.op(X.flatten(start_dim=2),-1)
        if self.mode == "max":
            X = X[0]
        return X

class DenseBlock(torch.nn.Module):
    def __init__(self,
                 spatial_dim:int,
                 structure:List[int],
                 kernel_size:int,
                 adn_fn:torch.nn.Module=torch.nn.PReLU,
                 structure_skip:List[int]=None,
                 return_all:bool=False):
        """Implementation of dense block with the possibility of including
        skip connections for architectures such as U-Net++.

        Args:
            spatial_dim (int): dimensionality of the input (2D or 3D).
            structure (List[int]): structure of the convolutions in the dense
                block. 
            kernel_size (int): kernel size for convolutions.
            adn_fn (torch.nn.Module, optional): function to be applied after 
                each convolution. Defaults to torch.nn.PReLU.
            structure_skip (List[int], optional): structure of the additional
                skip inputs. Skip inputs are optionally fed into the forward
                method and are appended to the outputs of each layer. For this
                reason, len(structure_skip) == len(structure) - 1. Defaults to
                None (normal dense block).
            return_all (bool, optional): Whether all outputs (intermediate 
                convolutions) from this layer should be returned. Defaults to 
                False.
        """
        super().__init__()
        self.spatial_dim = spatial_dim
        self.structure = structure
        self.kernel_size = kernel_size
        self.adn_fn = adn_fn
        self.structure_skip = structure_skip
        self.return_all = return_all

        if self.structure_skip is None or len(self.structure_skip) == 0:
            self.structure_skip = [0 for _ in range(len(self.structure)-1)]
        self.init_layers()
    
    def init_layers(self):
        if self.spatial_dim == 2:
            self.conv_op = torch.nn.Conv2d
        elif self.spatial_dim == 3:
            self.conv_op = torch.nn.Conv3d
        self.ops = torch.nn.ModuleList([])
        self.upscale_ops = torch.nn.ModuleList([])
        prev_d = self.structure[0]
        d = self.structure[1]
        k = self.kernel_size
        self.ops.append(
            torch.nn.Sequential(
                self.conv_op(prev_d,d,k,padding="same"),self.adn_fn(d)))
        for i in range(1,len(self.structure)-1):
            prev_d = sum(self.structure[:(i+1)]) + self.structure_skip[i-1]
            d = self.structure[i+1]
            self.ops.append(
                torch.nn.Sequential(
                    self.conv_op(prev_d,d,k,padding="same"),self.adn_fn(d)))

    def forward(self,X:torch.Tensor,X_skip:TensorList=None):
        """
        Args:
            X (torch.Tensor): input tensor.
            X_skip (TensorList, optional): list of tensors with an identical 
                size as X (except for the channel dimension). The length of
                this list should be len(self.structure). More details in 
                __init__. Defaults to None.

        Returns:
            torch.Tensor or TensorList
        """
        outputs = [X]
        out = X
        for i in range(len(self.ops)):
            if X_skip is not None and i > 0:
                xs = X_skip[i-1]
                xs = [F.interpolate(xs,out.shape[2:])]
            else:
                xs = []
            out = torch.cat([out,*outputs[:-1],*xs],1)
            out = self.ops[i](out)
            outputs.append(out)
        if self.return_all == True:
            return outputs
        else:
            return outputs[-1]

class UOut(torch.nn.Module):
    def __init__(self,beta: float=0.) -> torch.nn.Module:
        """Form of dropout suggested in [1]. Rather than dropping out 
        specific channels, each channel X is modified such that 
        $X' = X + rX$, where $x \sim U(-\beta,\beta)$. This guarantees
        a much smaller variance shift and allows for a dropout-like
        activation layer to be combined with batch-normalization without
        performance drops (more info on this in [1]). This operation is
        performed on the first dimension after the batch dimension (assumed
        to be the channel dimension).
        
        [1] https://ieeexplore.ieee.org/document/8953671

        Args:
            beta (float, optional): beta parameter for the uniform 
            distribution from which $r$ will be sampled for reference, the
            original authors use a value of 0.1. Defaults to 0.

        Returns:
            torch.nn.Module: a Torch Module
        """
        super().__init__()
        self.beta = beta
    
    def forward(self,X: torch.Tensor) -> torch.Tensor:
        """Forward pass for this Module.

        Args:
            X (torch.Tensor): Tensor 

        Returns:
            torch.Tensor: Tensor
        """
        if self.training == True:
            sh = list(X.shape)
            for i in range(2,len(sh)):
                sh[i] = 1
            r = torch.rand(sh).to(X.device)
            r = r*self.beta*2 - self.beta
            X = X + X*r
            return X
        else:
            return X

class BatchEnsemble(torch.nn.Module):
    def __init__(self,spatial_dim:int,n:int,
                 in_channels:int,out_channels:int,
                 adn_fn:Callable=torch.nn.Identity,
                 op_kwargs:dict=None):
        super().__init__()
        self.spatial_dim = spatial_dim
        self.n = n
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.adn_fn = adn_fn
        self.op_kwargs = op_kwargs
        
        self.correct_kwargs()
        self.initialize_layers()

    def correct_kwargs(self):
        if self.op_kwargs is None:
            if self.spatial_dim == 0:
                self.op_kwargs = {}
            else:
                self.op_kwargs = {"kernel_size":3}

    def initialize_layers(self):
        if self.spatial_dim == 0:
            self.mod = torch.nn.Linear(
                self.in_channels,self.out_channels,**self.op_kwargs)
        elif self.spatial_dim == 1:
            self.mod = torch.nn.Conv1d(
                self.in_channels,self.out_channels,**self.op_kwargs)
        elif self.spatial_dim == 2:
            self.mod = torch.nn.Conv2d(
                self.in_channels,self.out_channels,**self.op_kwargs)
        elif self.spatial_dim == 3:
            self.mod = torch.nn.Conv3d(
                self.in_channels,self.out_channels,**self.op_kwargs)
        self.all_weights = torch.nn.ParameterDict({
            "pre":torch.nn.Parameter(
                torch.as_tensor(np.random.normal(
                    1,0.1,size=[self.n,self.in_channels]),
                dtype=torch.float32)
            ),
            "post":torch.nn.Parameter(
                torch.as_tensor(np.random.normal(
                    1,0.1,size=[self.n,self.out_channels]),
                dtype=torch.float32)
            )})
        self.adn_op = self.adn_fn(self.out_channels)

    def forward(self,X:torch.Tensor,idx:int=None):
        b = X.shape[0]
        if idx is not None:
            pre = torch.unsqueeze(self.all_weights['pre'][idx],0)
            post = torch.unsqueeze(self.all_weights['post'][idx],0)
            X = torch.multiply(
                self.mod(X * unsqueeze_to_target(pre,X)),
                unsqueeze_to_target(post,X))
        elif self.training == True:
            idxs = np.random.randint(self.n,size=b)
            pre = torch.stack(
                [self.all_weights['pre'][idx] for idx in idxs])
            post = torch.stack(
                [self.all_weights['post'][idx] for idx in idxs])
            X = unsqueeze_to_target(pre,X) * X
            X = self.mod(X)
            X = unsqueeze_to_target(post,X) * X
        else:
            all_outputs = []
            for idx in range(self.n):
                pre = torch.unsqueeze(self.all_weights['pre'][idx],0)
                post = torch.unsqueeze(self.all_weights['post'][idx],0)
                o = torch.multiply(
                    self.mod(X * unsqueeze_to_target(pre,X)),
                    unsqueeze_to_target(post,X))
                all_outputs.append(o)
            X = torch.stack(all_outputs).mean(0)
        return self.adn_op(X)