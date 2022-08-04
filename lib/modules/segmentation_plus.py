import torch
import torch.nn.functional as F

from .layers import *
from ..types import *

from .segmentation import UNet
from typing import List

class UNetPlusPlus(UNet):
    def __init__(
        self,
        spatial_dimensions: int=2,
        encoding_operations: List[ModuleList]=None,
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
        strides: list=[2,2,2],
        bottleneck_classification: bool=False) -> torch.nn.Module:
        """Standard U-Net++ [1] implementation. Features some useful additions 
        such as residual links, different upsampling types, normalizations 
        (batch or instance) and ropouts (dropout and U-out). This version of 
        the U-Net has been implemented in such a way that it can be easily 
        expanded.

        Args:
            spatial_dimensions (int, optional): number of dimensions for the 
                input (not counting batch or channels). Defaults to 2.
            encoding_operations (List[ModuleList], optional): encoding
                operations (uses these rather than a standard U-Net encoder). 
                Must be a list where each element is of length and composed of
                an encoding operation and a downsampling operation.
            conv_type (str, optional): types of base convolutional operations.
                For now it only supports convolutions ("regular"). Defaults to 
                "regular".
            link_type (str, optional): redundant, for compatibility purposes.
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
            bottleneck_classification (bool, optional): sets up a 
                classification task using the channel-wise maximum of the 
                bottleneck layer. Defaults to False.

        [1] https://www.nature.com/articles/s41592-018-0261-2
        [2] https://openaccess.thecvf.com/content_CVPR_2019/papers/Li_Understanding_the_Disharmony_Between_Dropout_and_Batch_Normalization_by_Variance_CVPR_2019_paper.pdf

        Returns:
            torch.nn.Module: a U-Net module.
        """
        super().__init__(
            spatial_dimensions=spatial_dimensions,
            encoding_operations=encoding_operations,
            conv_type=conv_type,
            upscale_type=upscale_type,
            interpolation=interpolation,
            norm_type=norm_type,
            dropout_type=dropout_type,
            padding=padding,
            dropout_param=dropout_param,
            activation_fn=activation_fn,
            n_channels=n_channels,
            n_classes=n_classes,
            depth=depth,
            kernel_sizes=kernel_sizes,
            strides=strides,
            bottleneck_classification=bottleneck_classification)
                    
        # initialize all layers
        self.get_norm_op()
        self.get_drop_op()

        self.get_conv_op()
        if self.backbone is None:
            self.init_encoder()
        else:
            self.init_encoder_backbone()
        self.init_upscale_ops()
        self.init_link_ops()
        self.init_decoder()
        self.init_final_layer()
        if self.bottleneck_classification == True:
            self.init_bottleneck_classifier()
                
        self.loss_accumulator = 0.
        self.loss_accumulator_d = 0.
          
    def init_link_ops(self):
        """Initializes linking (skip) operations.
        """
        self.link_ops = torch.nn.ModuleList([])
        for i,idx in enumerate(range(len(self.depth)-2,-1,-1)):
            d = self.depth[idx]
            next_d = self.depth[idx+1]
            structure = [d for _ in range(i+2)]
            structure_skip = [next_d for _ in range(i)]
            op = DenseBlock(
                self.spatial_dimensions,structure,3,self.adn_fn,
                structure_skip,True)
            self.link_ops.append(op)
          
    def init_final_layer(self):
        """Initializes the classification layer (simple linear layer).
        """
        if self.n_classes > 2:
            self.final_act = torch.nn.Softmax(dim=1)
            nc = self.n_classes
        else:
            # coherces to a binary classification problem rather than
            # to a multiclass problem with two classes
            self.final_act = torch.nn.Sigmoid()
            nc = 1
        o = self.depth[0]
        self.final_layer = self.conv_op_dec(o,nc,1)
        self.final_layer_aux = torch.nn.ModuleList(
            [self.conv_op_dec(o,nc,1)
             for _ in range(len(self.depth)-1)])

    def forward(self,X:torch.Tensor,return_aux=False)->torch.Tensor:
        """Forward pass for this class.

        Args:
            X (torch.Tensor)

        Returns:
            torch.Tensor
        """
        encoding_out = []
        curr = X
        for op,op_ds in self.encoding_operations:
            curr = op(curr)
            encoding_out.append(curr)
            curr = op_ds(curr)
        bottleneck = curr
        link_outputs = []
        for i in range(len(self.decoding_operations)):
            op = self.decoding_operations[i]
            link_op = self.link_ops[i]
            up = self.upscale_ops[i]
            if len(link_outputs) > 0:
                lo = link_outputs[-1][:-1]
            else:
                lo = None
            encoded = link_op(encoding_out[-i-2],lo)
            link_outputs.append(encoded)
            encoded = encoded[-1]
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
        curr = self.final_act(curr)
        # return auxiliary classification layers
        if return_aux == True:
            curr_aux = []
            for op,x in zip(self.final_layer_aux,link_outputs[-1][1:-1]):
                curr_aux.append(self.final_act(op(x)))
        else:
            curr_aux = None
        # return bottleneck classification
        if self.bottleneck_classification == True:
            bottleneck = bottleneck.flatten(start_dim=2).max(-1).values
            bn_out = self.bottleneck_classifier(bottleneck)
        else:
            bn_out = None
        return curr,curr_aux,bn_out
