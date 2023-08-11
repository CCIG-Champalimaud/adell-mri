import torch
import torch.nn.functional as F
import math 

from typing import Union

class EfficientConditioningAttentionBlock(torch.nn.Module):
    """
    Efficient conditioning attention block. Based on [1,2]. Works by linearly
    transforming the input vector and using it to derive a sigmoid gate for the 
    channels in a given input tensor. [1] suggests a way of automatically 
    calculating a kernel size for the 1D convolutional applied in before the 
    sigmoid gate. 
    
    This convolution can be switched (in this implementation) by an additional
    linear layer. 

    [1] https://arxiv.org/abs/1910.03151
    [2] https://github.com/gmongaras/Diffusion_models_from_scratch/blob/main/src/blocks/clsAttn.py
    """
    def __init__(self,
                 class_dimension:int,
                 input_channels:int, 
                 gamma:float=2, 
                 b:float=1,
                 op_type:str="conv"):
        """
        Args:
            class_dimension (int): number of classes.
            input_channels (int): number of input channels.
            gamma (float, optional): gamma for the calculation of kernel size. 
                Defaults to 2.
            b (float, optional): b for the calculation of kernel size. Defaults
                to 1.
            op_type (str, optional): either "conv" or "linear". Sets the 
                operation type that should be applied prior to the sigmoid 
                gating. If set to "linear", gamma and b are not used. Defaults
                to "conv".
        """
        super().__init__()
        self.class_dimension = class_dimension
        self.input_channels = input_channels
        self.gamma = gamma
        self.b = b
        self.op_type = op_type
        
        self.initialize_layers()

    def odd(self,i:Union[int,float])->int:
        i = int(i)
        if i % 2 == 0:
            i = i + 1
        return i

    def initialize_layers(self):
        self.class_to_channels = torch.nn.Linear(
            self.class_dimension, self.input_channels)

        if self.op_type == "conv":
            # calculate kernel size
            kernel_size = self.odd(
                math.log2(
                    self.input_channels)/self.gamma+(self.b/self.gamma))

            self.op = torch.nn.Sequential(
                torch.nn.SiLU(),
                torch.nn.Conv1d(
                1, 1, kernel_size, padding=kernel_size//2, bias=False))
        elif self.op_type == "linear":
            self.op = torch.nn.Sequential(
                torch.nn.SiLU(),
                torch.nn.Linear(self.input_channels,self.input_channels),
                torch.nn.LayerNorm(self.input_channels),
                torch.nn.SiLU(),
                torch.nn.Linear(self.input_channels,self.input_channels))

    def forward(self, X:torch.Tensor, cls:torch.Tensor)->torch.Tensor:
        n_channels = len(X.shape)
        cls = self.class_to_channels(cls)
        if self.op_type == "conv":
            cls = F.sigmoid(self.op(cls.unsqueeze(1))).permute(0, 2, 1)
        elif self.op_type == "linear":
            cls = F.sigmoid(self.op(cls).unsqueeze(-1))
        cls = cls.reshape(-1,self.input_channels,1,
                          *[1 for _ in range(n_channels-3)])
        return X * cls