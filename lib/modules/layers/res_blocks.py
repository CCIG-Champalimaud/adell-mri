import numpy as np
import torch
from .utils import crop_to_size
from .utils import split_int_into_n
from ...types import *

class ResidualBlock2d(torch.nn.Module):
    def __init__(
        self,
        in_channels:int,
        kernel_size:int,
        inter_channels:int=None,
        out_channels:int=None,
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
        out = self.adn_op(self.final_op(self.op(X) + X))
        return out

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
