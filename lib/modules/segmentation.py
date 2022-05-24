import torch
import torch.nn.functional as F
import torchmetrics
import pytorch_lightning as pl

from .zwei.lib.modules import *
from .drei.lib.modules import *

from typing import List,Dict,Callable

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

class ActDropNorm(torch.nn.Module):
    def __init__(self,in_channels=None,ordering='NDA',
                 norm_fn: torch.nn.Module=torch.nn.BatchNorm2d,
                 act_fn: torch.nn.Module=torch.nn.PReLU,
                 dropout_fn:torch.nn.Module=torch.nn.Dropout,
                 dropout_param: float=0.):
        super().__init__()
        self.ordering = ordering
        self.norm_fn = norm_fn
        self.in_channels = in_channels
        self.act_fn = act_fn
        self.dropout_fn = dropout_fn
        self.dropout_param = dropout_param

        self.init_layers()

    def init_layers(self):
        if self.act_fn is None:
            self.act_fn = torch.nn.Identity
        if self.norm_fn is None:
            self.norm_fn = torch.nn.Identity
        if self.dropout_fn is None:
            self.dropout_fn = torch.nn.Identity

        self.op_dict = {
            "A":lambda: self.act_fn(),
            "D":lambda: self.dropout_fn(self.dropout_param),
            "N":lambda: self.norm_fn(self.in_channels)}
        
        self.op_list = torch.nn.ModuleList([])
        for k in self.ordering:
            self.op_list.append(self.op_dict[k]())
        
        self.op = torch.nn.Sequential(*self.op_list)

    def forward(self,X:torch.Tensor)->torch.Tensor:
        return self.op(X)

class ConvolutionalBlock3d(torch.nn.Module):
    def __init__(self,in_channels: list,out_channels: list,kernel_size: list,
                 adn_fn:torch.nn.Module=ActDropNorm,adn_args:dict={},
                 stride: int=1,padding: str="valid") -> torch.nn.Module:
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
            X (torch.Tensor): 5D tensor.

        Returns:
            torch.Tensor: 5D tensor.
        """
        return self.op(X)

class GCN2d(torch.nn.Module):
    def __init__(self,in_channels:int,
                 out_channels:int,kernel_size:int,
                 adn_fn:torch.nn.Module=ActDropNorm,
                 adn_args:dict={}):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.adn_fn = adn_fn
        self.adn_args = adn_args

        self.init_layers()

    def init_layers(self):
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
                 adn_fn:torch.nn.Module=ActDropNorm,
                 adn_args:dict={}):
        super().__init__()
        self.in_channels = in_channels
        self.kernel_size = kernel_size
        self.adn_fn = adn_fn
        self.adn_args = adn_args

        self.init_layers()

    def init_layers(self):
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
        return X + self.op(X)

class AHNetDecoderUnit3d(torch.nn.Module):
    def __init__(self,in_channels:int,
                 adn_fn:torch.nn.Module=ActDropNorm,
                 adn_args:dict={}):
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
                 adn_fn:torch.nn.Module=ActDropNorm,
                 adn_args:dict={"norm_fn":torch.nn.BatchNorm3d}):
        super().__init__()
        self.in_channels = in_channels
        self.adn_fn = adn_fn
        self.adn_args = adn_args
        
        self.init_layers()
    
    def init_layers(self):
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
        return self.op(X)

class AnysotropicHybridResidual(torch.nn.Module):
    def __init__(self,spatial_dim:int,in_channels:int,kernel_size:int,
                 adn_fn:torch.nn.Module=ActDropNorm,
                 adn_args:dict={}):
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
        self.op = self.get_op_2d()
        self.op_ds = self.get_downsample_op_2d()

    def get_op_2d(self):
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
        return torch.nn.Conv2d(self.in_channels,self.in_channels,2,stride=2)

    def get_downsample_op_3d(self):
        return torch.nn.Sequential(
            torch.nn.Conv3d(
                self.in_channels,self.in_channels,[2,2,1],stride=[2,2,1]),
            torch.nn.MaxPool3d([1,1,2],stride=[1,1,2]))
    
    def forward(self,X:torch.Tensor)->torch.Tensor:
        out = X + self.op(X)
        out = self.op_ds(out)
        return out
    
    def convert_to_3d(self)->None:
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
                 adn_fn:torch.nn.Module=ActDropNorm,adn_args:dict={}):
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
        self.p = int(self.kernel_size//2)
        self.op = torch.nn.Sequential(
            torch.nn.Conv2d(
                self.in_channels,self.out_channels,
                self.kernel_size,stride=2,padding=self.p),
            self.adn_fn(self.out_channels,**self.adn_args))
    
    def forward(self,X:torch.Tensor)->torch.Tensor:
        return self.op(X)
    
    def convert_to_3d(self)->None:
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
        b,c,h,w,d = X.shape
        outs = [X]
        for op in self.pyramidal_ops:
            x = op(X)
            x = F.upsample(x,size=[h,w,d])
            outs.append(x)
        return torch.cat(outs,1)

class AHNet(torch.nn.Module):
    def __init__(self,in_channels:int,out_channels:int,spatial_dim=2,
                 n_classes:int=2,n_layers:int=5,
                 adn_fn:torch.nn.Module=ActDropNorm,adn_args:dict={}):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.spatial_dim = spatial_dim
        self.n_classes = n_classes
        self.n_layers = n_layers
        self.adn_fn = adn_fn
        self.adn_args = adn_args

        self.gcn_k_size = [63,31,15,9,7,5]
        self.psp_levels = [[2,2,1],[4,4,2],[8,8,4],[16,16,4]]

        self.init_layers_2d()
        self.init_layers_3d()

    def convert_to_3d(self):
        self.res_layer_1.convert_to_3d()
        for op in self.res_layers:
            op.convert_to_3d()
        self.spatial_dim = 3

    def init_layers_2d(self):
        O = self.out_channels
        self.res_layer_1 = AnysotropicHybridInput(
            2,self.in_channels,O,kernel_size=7,
            adn_fn=self.adn_fn,adn_args=self.adn_args)
        self.max_pool_1 = torch.nn.MaxPool2d(3,stride=2,padding=1)
        self.res_layers = torch.nn.ModuleList([
            AnysotropicHybridResidual(
                2,O,O,adn_fn=self.adn_fn,adn_args=self.adn_args)
            for _ in range(self.n_layers-1)])

        self.gcn_refine = torch.nn.ModuleList([
            torch.nn.Sequential(
                GCN2d(O,O,k,self.adn_fn,self.adn_args),
                Refine2d(O,3,self.adn_fn,self.adn_args))
            for k in self.gcn_k_size[:(self.n_layers+1)]])
        
        self.upsampling_ops = torch.nn.ModuleList([
            torch.nn.Upsample(scale_factor=2,mode="bilinear")])
        for _ in range(self.n_layers-1):
            op = torch.nn.Sequential(
                torch.nn.Upsample(scale_factor=2,mode="bilinear"),
                Refine2d(O,3,self.adn_fn,self.adn_args))
            self.upsampling_ops.append(op)

        if self.n_classes == 2:
            self.final_layer = torch.nn.Sequential(
                torch.nn.Conv2d(O,1,1),
                torch.nn.Sigmoid())
        else:
            self.final_layer = torch.nn.Sequential(
                torch.nn.Conv2d(O,self.n_classes,1),
                torch.nn.Softmax(1))

    def forward_2d(self,X:torch.Tensor)->torch.Tensor:
        out_tensors = []
        out_tensors.append(
            self.res_layer_1(X))
        out_tensors.append(self.max_pool_1(out_tensors[-1]))
        out = out_tensors[-1]
        for op in self.res_layers:
            out = op(out)
            out_tensors.append(out)

        for i in range(len(self.gcn_refine)):
            out_tensors[i] = self.gcn_refine[i](out_tensors[i])

        out = self.upsampling_ops[0](out_tensors[-1])
        for i in range(1,len(self.upsampling_ops)):
            out = self.upsampling_ops[i](out+out_tensors[-i-1])

        prediction = self.final_layer(out)

        return prediction

    def init_layers_3d(self):
        O = self.out_channels
        adn_args = self.adn_args.copy()
        adn_args["norm_fn"] = torch.nn.BatchNorm3d
        self.max_pool_1_3d = torch.nn.Sequential(
            torch.nn.MaxPool3d(
                [1,1,2],stride=[1,1,2],padding=[0,0,1]),
            torch.nn.MaxPool3d(
                [3,3,3],stride=[2,2,2],padding=[1,1,0]))
        
        self.upsampling_ops_3d = torch.nn.ModuleList([
            torch.nn.Upsample(scale_factor=2,mode="trilinear")
            for _ in range(self.n_layers)])

        self.decoder_ops_3d = [
            AHNetDecoder3d(O,self.adn_fn,adn_args)
            for _ in range(self.n_layers)]

        # this could perhaps be changed to an atrous operation
        self.psp_op = PyramidSpatialPooling3d(O,levels=self.psp_levels)
        
        if self.n_classes == 2:
            self.final_layer_3d = torch.nn.Sequential(
                torch.nn.Conv3d(O*len(self.psp_levels)+O,1,1),
                torch.nn.Sigmoid())
        else:
            self.final_layer_3d = torch.nn.Sequential(
                torch.nn.Conv3d(O*len(self.psp_levels)+O,self.n_classes,1),
                torch.nn.Softmax(1))

    def forward_3d(self,X:torch.Tensor)->torch.Tensor:
        out_tensors = []
        out_tensors.append(
            self.res_layer_1(X))
        out_tensors.append(self.max_pool_1_3d(out_tensors[-1]))
        out = out_tensors[-1]
        for op in self.res_layers:
            out = op(out)
            out_tensors.append(out)

        out = self.upsampling_ops_3d[0](out_tensors[-1])
        for i in range(1,len(self.upsampling_ops)):
            out = out + out_tensors[-i-1]
            out = self.decoder_ops_3d[i](out)
            out = self.upsampling_ops_3d[i](out)

        out = self.psp_op(out)
        prediction = self.final_layer_3d(out)

        return prediction

    def forward(self,X:torch.Tensor)->torch.Tensor:
        if self.spatial_dim == 2:
            return self.forward_2d(X)
        elif self.spatial_dim == 3:
            return self.forward_3d(X)

class UNet(torch.nn.Module):
    def __init__(
        self,
        spatial_dimensions: int=2,
        conv_type: str="regular",
        link_type: str="identity",
        upscale_type: str="upsample",
        interpolation: str="bilinear",
        norm_type: str="batch",
        dropout_type: str="dropout",
        padding: int=0,
        dropout_param: float=0.1,
        activation_fn: torch.nn.Module=torch.nn.PReLU,
        n_channels: int=1,
        n_classes: int=2,
        depth: list=[16,32,64],
        kernel_sizes: list=[3,3,3],
        strides: list=[2,2,2]) -> torch.nn.Module:
        """Standard U-Net [1] implementation. Features some useful additions 
        such as residual links, different upsampling types, normalizations 
        (batch or instance) and ropouts (dropout and U-out). This version of 
        the U-Net has been implemented in such a way that it can be easily 
        expanded.

        Args:
            spatial_dimensions (int, optional): number of dimensions for the 
            input (not counting batch or channels). Defaults to 2.
            conv_type (str, optional): types of base convolutional operations.
            For now it only supports convolutions ("regular"). Defaults to 
            "regular".
            link_type (str, optional): link type for the skip connections.
            Can be a regular convolution ("conv"), residual block ("residual) or
            the identity ("identity"). Defaults to "identity".
            upscale_type (str, optional): upscaling type for decoder. Can be 
            regular interpolate upsampling ("upsample") or transpose 
            convolutions ("transpose"). Defaults to "upsample".
            interpolation (str, optional): interpolation for the upsampling
            operation (if `upscale_type="upsample"`). Defaults to "bilinear".
            norm_type (str, optional): type of normalization. Can be batch
            normalization ("batch") or instance normalization ("instance"). 
            Defaults to "batch".
            dropout_type (str, optional): type of dropout. Can be either 
            regular dropout ("dropout") or U-out [2] ("uout"). Defaults to 
            "dropout".
            padding (int, optional): amount of padding for convolutions. 
            Defaults to 0.
            dropout_param (float, optional): parameter for dropout layers. 
            Sets the dropout rate for "dropout" and beta for "uout". Defaults 
            to 0.1.
            activation_fn (torch.nn.Module, optional): activation function to
            be applied after normalizing. Defaults to torch.nn.PReLU.
            n_channels (int, optional): number of channels in input. Defaults
            to 1.
            n_classes (int, optional): number of output classes. Defaults to 2.
            depth (list, optional): defines the depths of each layer of the 
            U-Net (the decoder will be the opposite). Defaults to [16,32,64].
            kernel_sizes (list, optional): defines the kernels of each layer 
            of the U-Net. Defaults to [3,3,3].
            strides (list, optional): defines the strides of each layer of the
            U-Net. Defaults to [2,2,2].

        [1] https://www.nature.com/articles/s41592-018-0261-2
        [2] https://openaccess.thecvf.com/content_CVPR_2019/papers/Li_Understanding_the_Disharmony_Between_Dropout_and_Batch_Normalization_by_Variance_CVPR_2019_paper.pdf

        Returns:
            torch.nn.Module: a U-Net module.
        """
        
        super().__init__()
        self.spatial_dimensions = spatial_dimensions
        self.conv_type = conv_type
        self.link_type = link_type
        self.upscale_type = upscale_type
        self.interpolation = interpolation
        self.norm_type = norm_type
        self.dropout_type = dropout_type
        self.padding = padding
        self.dropout_param = dropout_param
        self.activation_fn = activation_fn
        self.n_channels = n_channels
        self.n_classes = n_classes
        self.depth = depth
        self.kernel_sizes = kernel_sizes
        self.strides = strides
        
        # initialize all layers
        self.get_conv_op()
        self.get_norm_op()
        self.get_drop_op()
        self.init_encoder()
        self.init_upscale_ops()
        self.init_link_ops()
        self.init_decoder()
        self.init_final_layer()
                
        self.loss_accumulator = 0.
        self.loss_accumulator_d = 0.

    def get_norm_op(self):
        if self.norm_type is None:
            self.norm_op = torch.nn.Identity
            
        if self.spatial_dimensions == 2:
            if self.norm_type == "batch":
                self.norm_op = torch.nn.BatchNorm2d
            elif self.norm_type == "instance":
                self.norm_op = torch.nn.InstanceNorm2d

        if self.spatial_dimensions == 3:
            if self.norm_type == "batch":
                self.norm_op = torch.nn.BatchNorm3d
            elif self.norm_type == "instance":
                self.norm_op = torch.nn.InstanceNorm3d

    def get_drop_op(self):
        if self.dropout_type is None:
            self.drop_op = torch.nn.Identity
            
        elif self.dropout_type == "dropout":
            self.drop_op = torch.nn.Dropout
        elif self.dropout_type == "uout":
            self.drop_op = UOut
            
    def get_conv_op(self):
        if self.spatial_dimensions == 2:
            if self.conv_type == "regular":
                self.conv_op = torch.nn.Conv2d
        if self.spatial_dimensions == 3:
            if self.conv_type == "regular":
                self.conv_op = torch.nn.Conv3d
    
    def init_upscale_ops(self):
        depths_a = self.depth[:0:-1]
        depths_b = self.depth[-2::-1]
        if self.upscale_type == "upsample":
            if self.spatial_dimensions == 2:
                self.upscale_ops = [
                    torch.nn.Sequential(
                        torch.nn.Conv2d(d1,d2,1),
                        torch.nn.Upsample(scale_factor=s,mode=self.interpolation))
                    for d1,d2,s in zip(depths_a,depths_b,self.strides[::-1])]
            if self.spatial_dimensions == 3:
                self.upscale_ops = [
                    torch.nn.Sequential(
                        torch.nn.Conv3d(d1,d2,1),
                        torch.nn.Upsample(scale_factor=s,mode=self.interpolation))
                    for d1,d2,s in zip(depths_a,depths_b,self.strides[::-1])]
            self.upscale_ops = torch.nn.ModuleList(self.upscale_ops)
        elif self.upscale_type == "transpose":
            if self.spatial_dimensions == 2:
                self.upscale_ops = [
                    torch.nn.ConvTranspose2d(
                        d1,d2,2,stride=s,padding=0) 
                    for d1,d2,s in zip(depths_a,depths_b,self.strides[::-1])]
                self.upscale_ops = torch.nn.ModuleList(self.upscale_ops)
            if self.spatial_dimensions == 3:
                self.upscale_ops = [
                    torch.nn.ConvTranspose3d(
                        d1,d2,2,stride=s,padding=0) 
                    for d1,d2,s in zip(depths_a,depths_b,self.strides[::-1])]
                self.upscale_ops = torch.nn.ModuleList(self.upscale_ops)
    
    def init_link_ops(self):
        if self.link_type == "identity":
            self.link_ops = [
                torch.nn.Identity() for _ in self.depth[:-1]]
            self.link_ops = torch.nn.ModuleList(self.link_ops)
        elif self.link_type == "conv":
            if self.spatial_dimensions == 2:
                self.link_ops = [
                    ConvBatchAct2d(d,d,3,padding=self.padding)
                    for d in self.depth[-2::-1]]
                self.link_ops = torch.nn.ModuleList(self.link_ops)
            elif self.spatial_dimensions == 3:
                self.link_ops = [
                    ConvBatchAct3d(d,d,3,padding=self.padding) 
                    for d in self.depth[-2::-1]]
                self.link_ops = torch.nn.ModuleList(self.link_ops)
        elif self.link_type == "residual":
            if self.spatial_dimensions == 2:
                self.link_ops =[
                    ResidualBlock2d(d,3) for d in self.depth[-2::-1]]
                self.link_ops = torch.nn.ModuleList(self.link_ops)
            elif self.spatial_dimensions == 3:
                self.link_ops =[
                    ResidualBlock3d(d,3) for d in self.depth[-2::-1]]
                self.link_ops = torch.nn.ModuleList(self.link_ops)
    
    def interpolate_depths(self,a,b,n=3):
        return list(np.linspace(a,b,n,dtype=np.int32))

    def init_encoder(self):
        self.encoding_operations = torch.nn.ModuleList([])
        previous_d = self.n_channels
        for i in range(len(self.depth)-1):
            d,k,s = self.depth[i],self.kernel_sizes[i],self.strides[i]
            op = torch.nn.Sequential(
                self.conv_op(previous_d,d,kernel_size=k,stride=1,
                             padding=self.padding),
                self.norm_op(d),self.drop_op(self.dropout_param),
                self.activation_fn())
            op_downsample = torch.nn.Sequential(
                self.conv_op(d,d,kernel_size=k,stride=s,padding=self.padding),
                self.activation_fn())
            self.encoding_operations.append(
                torch.nn.ModuleList([op,op_downsample]))
            previous_d = d
        op = torch.nn.Sequential(
            self.conv_op(self.depth[-2],self.depth[-1],kernel_size=k,stride=1,
                         padding=self.padding),
            self.norm_op(self.depth[-1]),self.drop_op(self.dropout_param),
            self.activation_fn())
        op_downsample = torch.nn.Identity()        
        self.encoding_operations.append(
            torch.nn.ModuleList([op,op_downsample]))
    
    def init_decoder(self):
        self.decoding_operations = torch.nn.ModuleList([])
        depths = self.depth[-2::-1]
        kernel_sizes = self.kernel_sizes[-2::-1]
        previous_d = self.depth[-1]
        for i in range(len(depths)):
            d,k = depths[i],kernel_sizes[i]
            op = torch.nn.Sequential(
                self.conv_op(previous_d,d,kernel_size=k,stride=1,
                             padding=self.padding),
                self.activation_fn(),
                self.conv_op(d,d,kernel_size=k,stride=1,
                             padding=self.padding),
                self.norm_op(d),self.drop_op(self.dropout_param),
                self.activation_fn())
            self.decoding_operations.append(op)
            previous_d = d
            
    def init_final_layer(self):
        """Initializes the classification layer (simple linear layer).
        """
        o = self.depth[0]
        if self.n_classes > 2:
            self.final_layer = torch.nn.Sequential(
                self.conv_op(o,self.n_classes,1),
                torch.nn.Softmax(dim=1))
        else:
            # coherces to a binary classification problem rather than
            # to a multiclass problem with two classes
            self.final_layer = torch.nn.Sequential(
                self.conv_op(o,1,1),
                torch.nn.Sigmoid())
        
    def forward(self,X):
        encoding_out = []
        curr = X
        for op,op_ds in self.encoding_operations:
            curr = op(curr)
            encoding_out.append(curr)
            curr = op_ds(curr)
        bottleneck = curr
        for i in range(len(self.decoding_operations)):
            op = self.decoding_operations[i]
            link_op = self.link_ops[i]
            up = self.upscale_ops[i]
            encoded = link_op(encoding_out[-i-2])
            curr = up(curr)
            sh = list(curr.shape)[2:]
            sh2 = list(encoded.shape)[2:]
            if np.prod(sh) < np.prod(sh2):
                encoded = crop_to_size(encoded,sh)
            if np.prod(sh) > np.prod(sh2):
                curr = crop_to_size(curr,sh2)
            curr = torch.concat((curr,encoded),dim=1)
            curr = op(curr)
            
        curr = self.final_layer(curr)
        return curr
