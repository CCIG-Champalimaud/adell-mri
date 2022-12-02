import torch
import torch.nn.functional as F

from ..layers.standard_blocks import DenseBlock
from ..layers.utils import crop_to_size
from ...custom_types import *

from .unet import UNet
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
        bottleneck_classification: bool=False,
        skip_conditioning: int=None,
        feature_conditioning: int=None,
        feature_conditioning_params: Dict[str,torch.Tensor]=None) -> torch.nn.Module:
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
            skip_conditioning (int, optional): conditions skip connections
                with a given image with skip_conditioning channels during
                forward passes.
            feature_conditioning (int,optional): linearly transforms tabular 
                features and adds them to each channel of the skip connections.
                Useful to include tabular features in the prediction algorithm.
                Defaults to None.
            feature_conditioning_params (Dict[str,torch.Tensor], optional): 
                dictionary with keys "mean" and "std" to normalize the tabular 
                features. Must be present if feature conditioning is used. 
                Defaults to None.

        [1] https://www.nature.com/articles/s41592-018-0261-2
        [2] https://openaccess.thecvf.com/content_CVPR_2019/papers/Li_Understanding_the_Disharmony_Between_Dropout_and_Batch_Normalization_by_Variance_CVPR_2019_paper.pdf

        Returns:
            torch.nn.Module: a U-Net++ module.
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
            bottleneck_classification=bottleneck_classification,
            skip_conditioning=skip_conditioning,
            feature_conditioning=feature_conditioning,
            feature_conditioning_params=feature_conditioning_params)

        # initialize all layers
        self.get_norm_op()
        self.get_drop_op()

        self.get_conv_op()
        if self.encoding_operations is None:
            self.init_encoder()
        else:
            self.init_encoder_backbone()
        self.init_upscale_ops()
        self.init_link_ops()
        self.init_decoder()
        self.init_final_layer()
        if self.bottleneck_classification == True:
            self.init_bottleneck_classifier()
        if self.feature_conditioning is not None:
            self.init_feature_conditioning_operations()
          
    def init_link_ops(self):
        """Initializes linking (skip) operations.
        """
        if self.skip_conditioning is not None:
            ex = self.skip_conditioning
        else:
            ex = 0
        self.link_ops = torch.nn.ModuleList([])
        for i,idx in enumerate(range(len(self.depth)-2,-1,-1)):
            d = self.depth[idx]
            next_d = self.depth[idx+1]
            structure = [d for _ in range(i+2)]
            structure_skip = [next_d for _ in range(i)]
            structure[0] += ex
            if len(structure_skip) > 0:
                structure_skip[0] += ex
            op = DenseBlock(
                self.spatial_dimensions,structure,3,self.adn_fn,
                structure_skip,True)
            self.link_ops.append(op)

    def init_final_layer(self):
        """Initializes the classification layer (simple linear layer).
        """
        if self.skip_conditioning is not None:
            ex = self.skip_conditioning
        else:
            ex = 0
        if self.n_classes > 2:
            self.final_act = torch.nn.Softmax(dim=1)
            nc = self.n_classes
        else:
            # coherces to a binary classification problem rather than
            # to a multiclass problem with two classes
            self.final_act = torch.nn.Sigmoid()
            nc = 1
        if self.spatial_dimensions == 2:
            op = torch.nn.Conv2d 
        elif self.spatial_dimensions == 3:
            op = torch.nn.Conv3d
        o = self.depth[0]
        self.final_layer = torch.nn.Sequential(
            op(o,o,3,padding="same"),
            self.adn_fn(o),
            op(o,o,1,padding="same"),
            self.adn_fn(o),
            op(o,nc,1))
        S = [o+ex for _ in self.depth[:-1]]
        S[-1] = S[-1] - ex
        self.final_layer_aux = torch.nn.ModuleList(
            [torch.nn.Sequential(
                op(s,s-ex,3,padding="same"),
                self.adn_fn(s-ex),
                op(s-ex,s-ex,1,padding="same"),
                self.adn_fn(s-ex),
                op(s-ex,nc,1))
             for s in S])

    def forward(self,X:torch.Tensor,
                return_aux=False,
                X_skip_layer:torch.Tensor=None,
                X_feature_conditioning:torch.Tensor=None,
                return_features=False)->torch.Tensor:
        """Forward pass for this class.

        Args:
            X (torch.Tensor)

        Returns:
            torch.Tensor
        """
        # check if channel dim is available and if not include it 
        if X_skip_layer is not None:
            if len(X_skip_layer.shape) < len(X.shape):
                X_skip_layer = X_skip_layer.unsqueeze(1)

        # normalise features
        if X_feature_conditioning is not None:
            X_feature_conditioning = X_feature_conditioning - self.f_mean
            X_feature_conditioning = X_feature_conditioning / self.f_std

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
            if X_skip_layer is not None:
                S = encoding_out[-i-2].shape[2:]
                xfl = F.interpolate(X_skip_layer,S,mode='nearest')
                link_op_input = torch.cat([encoding_out[-i-2],xfl],axis=1)
            else:
                link_op_input = encoding_out[-i-2]
            encoded = link_op(link_op_input,lo)
            if X_feature_conditioning is not None:
                op = self.feature_conditioning_ops[i]
                transformed_features = op(X_feature_conditioning)
                encoded = torch.multiply(
                    encoded,
                    self.unsqueeze_to_dim(transformed_features,encoded))
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
        
        final_features = curr

        curr = self.final_layer(curr)
        curr = self.final_act(curr)
        if return_features == True:
            return curr,final_features,bottleneck

        # return auxiliary classification layers
        if return_aux == True:
            curr_aux = []
            for op,x in zip(self.final_layer_aux,link_outputs[-1][1:-1]):
                if X_skip_layer is not None:
                    x = torch.cat([x,X_skip_layer],axis=1)
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
