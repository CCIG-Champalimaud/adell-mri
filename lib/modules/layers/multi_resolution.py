import torch
import torch.nn.functional as F
from typing import List,Tuple,Union
from .utils import split_int_into_n
from .res_blocks import ResidualBlock2d
from .res_blocks import ResidualBlock3d
from .standard_blocks import DepthWiseSeparableConvolution2d
from .standard_blocks import DepthWiseSeparableConvolution3d

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
            if self.shape_check is False:
                if prev_x.shape != x.shape:
                    prev_x = F.interpolate(
                        prev_x,x.shape[2:],mode='nearest')
            prev_x = x + prev_x
        return prev_x

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

class SpatialPyramidPooling2d(torch.nn.Module):
    def __init__(self,in_channels:int,out_channels:int,
                 filter_sizes:List[int],
                 adn_fn:torch.nn.Module=torch.nn.Identity):
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
            adn_fn (torch.nn.Module, optional): ADN function applied after
                convolutions. Defaults to torch.nn.Identity.
        """
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.filter_size = filter_sizes
        self.adn_fn = adn_fn

    def init_layers(self):
        self.layers = torch.nn.ModuleList([])
        for filter_size in self.filter_sizes:
            op = torch.nn.Sequential(
                torch.nn.Conv2d(
                    self.in_channels,self.out_channels,
                    kernel_size=filter_size,padding="same"),
                self.adn_fn(self.out_channels),
                DepthWiseSeparableConvolution2d(
                    self.out_channels,self.out_channels,
                    kernel_size=filter_size,padding="same"),
                self.adn_fn(self.out_channels))
            self.layers.append(op)
    
    def forward(self,X:torch.Tensor)->torch.Tensor:
        outputs = []
        for layer in self.layers:
            outputs.append(layer(X))
        output = torch.cat(outputs,dim=1)
        return output

class SpatialPyramidPooling3d(torch.nn.Module):
    def __init__(self,in_channels:int,out_channels:int,
                 filter_sizes:List[int],
                 adn_fn:torch.nn.Module=torch.nn.Identity):
        """Spatial pyramid pooling for 3d inputs. Applies a set of differently
        sized filters to an input and then concatenates the output of each 
        filter.

        [1] https://arxiv.org/abs/1612.01105

        Args:
            in_channels (int): number of input channels
            out_channels (int): number of output channels
            filter_sizes (List[int], optional): list of kernel sizes. Defaults
            to 3.
            adn_fn (torch.nn.Module, optional): ADN function applied after
                convolutions. Defaults to torch.nn.Identity.
        """
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.filter_sizes = filter_sizes
        self.adn_fn = adn_fn

        self.init_layers()

    def init_layers(self):
        self.layers = torch.nn.ModuleList([])
        for filter_size in self.filter_sizes:
            op = torch.nn.Sequential(
                torch.nn.Conv3d(
                    self.in_channels,self.out_channels,
                    kernel_size=filter_size,padding="same"),
                self.adn_fn(self.out_channels),
                DepthWiseSeparableConvolution3d(
                    self.out_channels,self.out_channels,
                    kernel_size=filter_size,padding="same"),
                self.adn_fn(self.out_channels))
            self.layers.append(op)
    
    def forward(self,X:torch.Tensor)->torch.Tensor:
        outputs = []
        for layer in self.layers:
            outputs.append(layer(X))
        output = torch.cat(outputs,dim=1)
        return output

class AtrousSpatialPyramidPooling2d(torch.nn.Module):
    def __init__(self,in_channels:int,out_channels:int,rates:List[int],
                 adn_fn:torch.nn.Module=torch.nn.Identity):
        """Atrous spatial pyramid pooling for 2d inputs. Applies a set of 
        differently sized dilated filters to an input and then concatenates
        the output of each  filter. Similar to SpatialPyramidPooling2d but 
        much less computationally demanding.

        [1] https://arxiv.org/abs/1612.01105

        Args:
            in_channels (int): number of input channels
            out_channels (int): number of output channels
            rates (List[int], optional): list dilation rates. 
            adn_fn (torch.nn.Module, optional): ADN function applied after
                convolutions. Defaults to torch.nn.Identity.
        """
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.rates = rates
        self.adn_fn = adn_fn

        self.n_channels = split_int_into_n(self.out_channels,len(self.rates))
        
        self.init_layers()

    def init_layers(self):
        self.layers = torch.nn.ModuleList([])
        for rate,c in zip(self.rates,self.n_channels):
            op = torch.nn.Sequential(
                torch.nn.Conv2d(
                    self.in_channels,c,kernel_size=3,
                    dilation=rate,padding="same"),
                self.adn_fn(c),
                DepthWiseSeparableConvolution2d(
                    c,c,kernel_size=3,padding="same"),
                self.adn_fn(c))
            self.layers.append(op)

    def forward(self,X:torch.Tensor)->torch.Tensor:
        outputs = []
        for layer in self.layers:
            outputs.append(layer(X))
        output = torch.cat(outputs,dim=1)
        return output

class AtrousSpatialPyramidPooling3d(torch.nn.Module):
    def __init__(self,in_channels:int,out_channels:int,rates:List[int],
                 adn_fn:torch.nn.Module=torch.nn.Identity):
        """Atrous spatial pyramid pooling for 3d inputs. Applies a set of 
        differently sized dilated filters to an input and then concatenates
        the output of each  filter. Similar to SpatialPyramidPooling3d but 
        much less computationally demanding.

        [1] https://arxiv.org/abs/1612.01105

        Args:
            in_channels (int): number of input channels
            out_channels (int): number of output channels
            rates (List[int], optional): list dilation rates. 
            adn_fn (torch.nn.Module, optional): ADN function applied after
                convolutions. Defaults to torch.nn.Identity.
        """
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.rates = rates
        self.adn_fn = adn_fn

        self.n_channels = split_int_into_n(self.out_channels,len(self.rates))
        
        self.init_layers()

    def init_layers(self):
        self.layers = torch.nn.ModuleList([])
        for rate,c in zip(self.rates,self.n_channels):
            op = torch.nn.Sequential(
                torch.nn.Conv3d(
                    self.in_channels,c,kernel_size=3,
                    dilation=rate,padding="same"),
                self.adn_fn(c),
                DepthWiseSeparableConvolution3d(
                    c,c,kernel_size=3,padding="same"),
                self.adn_fn(c))
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
                 adn_fn:torch.nn.Module=torch.nn.Identity):
        """Receptive field block for 2d inputs [1]. A mid ground between a 
        residual operator and AtrousSpatialPyramidPooling2d - a series of 
        dilated convolutions is applied to the input, the output of these 
        dilated convolutions is concatenated and added to the input.

        [1] https://arxiv.org/pdf/1711.07767.pdf

        Args:
            in_channels (int): input channels.
            out_channels (int): output channels.
            rates (List[int]): dilation rates.
            adn_fn (torch.nn.Module, optional): ADN function applied after
                convolutions. Defaults to torch.nn.Identity.
        """
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.rates = rates
        self.adn_fn = adn_fn

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
                    self.adn_fn(o),
                    torch.nn.Conv2d(o,o,kernel_size=3,padding="same"),
                    self.adn_fn(o))
            else:
                op = torch.nn.Sequential(
                    torch.nn.Conv2d(
                        self.in_channels,o,kernel_size=1,padding="same"),
                    self.adn_fn(o),
                    torch.nn.Conv2d(o,o,kernel_size=rate,padding="same"),
                    self.adn_fn(o),
                    torch.nn.Conv2d(o,o,dilation=rate,kernel_size=3,
                                    padding="same"),
                    self.adn_fn(o))
            self.layers.append(op)
        self.final_op = torch.nn.Conv2d(
            self.out_channels,self.out_channels,1)

    def forward(self,X:torch.Tensor)->torch.Tensor:
        outputs = []
        for layer in self.layers:
            outputs.append(layer(X))
        output = torch.cat(outputs,dim=1)
        output = self.final_op(output)
        output = X + output
        return output

class ReceptiveFieldBlock3d(torch.nn.Module):
    def __init__(self,in_channels:int,bottleneck_channels:int,
                 rates:List[int],
                 adn_fn:torch.nn.Module=torch.nn.Identity):
        """Receptive field block for 3d inputs [1]. A mid ground between a 
        residual operator and AtrousSpatialPyramidPooling3d - a series of 
        dilated convolutions is applied to the input, the output of these 
        dilated convolutions is concatenated and added to the input.

        [1] https://arxiv.org/pdf/1711.07767.pdf

        Args:
            in_channels (int): input channels.
            bottleneck_channels (int): number of channels in bottleneck.
            rates (List[int]): dilation rates.
            adn_fn (torch.nn.Module, optional): ADN function applied after
                convolutions. Defaults to torch.nn.Identity.
        """
        super().__init__()
        self.in_channels = in_channels
        self.bottleneck_channels = bottleneck_channels
        self.rates = rates
        self.adn_fn = adn_fn

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
                    self.adn_fn(o),
                    torch.nn.Conv3d(o,o,kernel_size=3,padding="same"),
                    self.adn_fn(o))
            else:
                op = torch.nn.Sequential(
                    torch.nn.Conv3d(
                        self.in_channels,o,kernel_size=1,padding="same"),
                    self.adn_fn(o),
                    torch.nn.Conv3d(o,o,kernel_size=rate,padding="same"),
                    self.adn_fn(o),
                    torch.nn.Conv3d(o,o,dilation=rate,kernel_size=3,
                                    padding="same"),
                    self.adn_fn(o))
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
