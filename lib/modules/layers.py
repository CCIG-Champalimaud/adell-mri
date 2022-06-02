
import numpy as np
import torch
import torch.nn.functional as F
from math import floor
from typing import List,Dict

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

        self.op_dict = {
            "A":lambda: self.act_fn(),
            "D":lambda: self.dropout_fn(self.dropout_param),
            "N":lambda: self.norm_fn(self.in_channels)}
        
        self.op_list = torch.nn.ModuleList([])
        for k in self.ordering:
            self.op_list.append(self.op_dict[k]())
        
        self.op = torch.nn.Sequential(*self.op_list)

    def forward(self,X:torch.Tensor)->torch.Tensor:
        """Forward function.

        Args:
            X (torch.Tensor)

        Returns:
            torch.Tensor
        """
        return self.op(X)

class ConvolutionalBlock3d(torch.nn.Module):
    def __init__(self,in_channels:List[int],out_channels:List[int],
                 kernel_size:List[int],adn_fn:torch.nn.Module=ActDropNorm,
                 adn_args:dict={},stride:int=1,
                 padding:str="valid"):
        """Assembles a set of blocks containing convolutions followed by 
        ActDropNorm operations. Used to quickly build convolutional neural
        networks.

        Args:
            in_channels (List[int]): list of input channels for convolutions.
            out_channels (List[int]): list of output channels for convolutions.
            kernel_size (List[int]): list of kernel sizes.
            adn_fn (torch.nn.Module, optional): module applied after 
            convolutions. Defaults to ActDropNorm.
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
                 adn_fn:torch.nn.Module=ActDropNorm,
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
            convolutions. Defaults to ActDropNorm.
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
                 adn_fn:torch.nn.Module=ActDropNorm,
                 adn_args:dict={}):
        """Refinement module from the AHNet paper [1]. Essentially a residual
        module.

        [1] https://arxiv.org/pdf/1711.08580.pdf

        Args:
            in_channels (int): number of input channels.
            kernel_size (int): number of output channels.
            adn_fn (torch.nn.Module, optional): module applied after 
            convolutions. Defaults to ActDropNorm.
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
                 adn_fn:torch.nn.Module=ActDropNorm,
                 adn_args:dict={}):
        """3D AHNet decoder unit from the AHNet paper [1]. Combines multiple, 
        branching and consecutive convolutions. Each unit is composed of a 
        residual-like operation followed by a concatenation with the original
        input.

        [1] https://arxiv.org/pdf/1711.08580.pdf

        Args:
            in_channels (int): number of input channels.
            adn_fn (torch.nn.Module, optional): module applied after 
            convolutions. Defaults to ActDropNorm.
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
                 adn_fn:torch.nn.Module=ActDropNorm,
                 adn_args:dict={"norm_fn":torch.nn.BatchNorm3d}):
        """Three consecutive AHNetDecoderUnit3d. Can be modified to include
        more but it is hard to know what concrete improvements this may lead
        to.

        Args:
            in_channels (int): number of input channels.
            adn_fn (torch.nn.Module, optional): module applied after 
            convolutions. Defaults to ActDropNorm.
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
                 adn_fn:torch.nn.Module=ActDropNorm,
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
            convolutions. Defaults to ActDropNorm.
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
                 adn_fn:torch.nn.Module=ActDropNorm,adn_args:dict={}):
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
            convolutions. Defaults to ActDropNorm.
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
        self.act_op = self.act_fn(inplace=True)

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
        self.act_op = self.act_fn(inplace=True)

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
                self.act_fn(inplace=True),
                DepthWiseSeparableConvolution2d(
                    self.out_channels,self.out_channels,
                    kernel_size=filter_size,padding="same"),
                self.act_fn(inplace=True))
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
                torch.nn.Conv3d(
                    self.in_channels,self.out_channels,
                    kernel_size=filter_size,padding="same"),
                self.act_fn(inplace=True),
                DepthWiseSeparableConvolution3d(
                    self.out_channels,self.out_channels,
                    kernel_size=filter_size,padding="same"),
                self.act_fn(inplace=True))
            self.layers.append(op)
    
    def forward(self,X:torch.Tensor)->torch.Tensor:
        outputs = []
        for layer in self.layers:
            outputs.append(layer(X))
        output = torch.cat(output,dim=1)
        return output

class AtrousSpatialPyramidPooling2d(torch.nn.Module):
    def __init__(self,in_channels:int,out_channels:int,
                 rates:List[int],
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
                self.act_fn(inplace=True),
                DepthWiseSeparableConvolution2d(
                    self.out_channels,self.out_channels,
                    kernel_size=3,padding="same"),
                self.act_fn(inplace=True))
            self.layers.append(op)

    def forward(self,X:torch.Tensor)->torch.Tensor:
        outputs = []
        for layer in self.layers:
            outputs.append(layer(X))
        output = torch.cat(outputs,dim=1)
        return output

class AtrousSpatialPyramidPooling3d(torch.nn.Module):
    def __init__(self,in_channels:int,out_channels:int,
                 rates:List[int],
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
                self.act_fn(inplace=True),
                DepthWiseSeparableConvolution3d(
                    self.out_channels,self.out_channels,
                    kernel_size=3,padding="same"),
                self.act_fn(inplace=True))
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
                    self.act_fn(inplace=True),
                    torch.nn.Conv2d(o,o,kernel_size=3,padding="same"),
                    self.act_fn(inplace=True))
            else:
                op = torch.nn.Sequential(
                    torch.nn.Conv2d(
                        self.in_channels,o,kernel_size=1,padding="same"),
                    self.act_fn(inplace=True),
                    torch.nn.Conv2d(o,o,kernel_size=rate,padding="same"),
                    self.act_fn(inplace=True),
                    torch.nn.Conv2d(o,o,dilation=rate,kernel_size=3,
                                    padding="same"),
                    self.act_fn(inplace=True))
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
                    self.act_fn(inplace=True),
                    torch.nn.Conv3d(o,o,kernel_size=3,padding="same"),
                    self.act_fn(inplace=True))
            else:
                op = torch.nn.Sequential(
                    torch.nn.Conv3d(
                        self.in_channels,o,kernel_size=1,padding="same"),
                    self.act_fn(inplace=True),
                    torch.nn.Conv3d(o,o,kernel_size=rate,padding="same"),
                    self.act_fn(inplace=True),
                    torch.nn.Conv3d(o,o,dilation=rate,kernel_size=3,
                                    padding="same"),
                    self.act_fn(inplace=True))
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
        channel_squeeze = torch.unsqueeze(
            torch.unsqueeze(channel_squeeze,-1),-1)
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
