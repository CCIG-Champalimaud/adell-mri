import torch
import torch.nn.functional as F

from .layers import *
from ..types import *

from typing import List

class AHNet(torch.nn.Module):
    def __init__(self,in_channels:int,out_channels:int,spatial_dim=2,
                 n_classes:int=2,n_layers:int=5,
                 adn_fn:torch.nn.Module=ActDropNorm,adn_args:dict={}):
        """Implementation of the AHNet (anysotropic hybrid network), which is 
        capable of learning segmentation features in 2D and then learn how to
        use in 3D images. More details in [1].

        [1] https://arxiv.org/abs/1711.08580 

        Args:
            in_channels (int): number of input channels.
            out_channels (int): number of output channels.
            spatial_dim (int, optional): number of initial spatial dimensions.
            Defaults to 2.
            n_classes (int, optional): number of classes. Defaults to 2.
            n_layers (int, optional): number of layers. In the 2D case this 
            changes how many AH residual/GCN/Refine modules there are, in the
            3D case this changes how many AH decoders are instantiated. 
            Defaults to 5.
            adn_fn (torch.nn.Module, optional): module applied after 
            convolutions. Defaults to ActDropNorm.
            adn_args (dict, optional): args for the module applied after 
            convolutions. Defaults to {}.
        """
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
        """Converts the relevant operations to 3D.
        """
        self.res_layer_1.convert_to_3d()
        for op in self.res_layers:
            op.convert_to_3d()
        self.spatial_dim = 3

    def init_layers_2d(self):
        """Initializes the 2D layers.
        """
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
        """Forward pass for this class for 2D images.

        Args:
            X (torch.Tensor)

        Returns:
            torch.Tensor
        """
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
        """Initializes the 3D layers.
        """
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
        """Forward pass for this class for 3D images.

        Args:
            X (torch.Tensor)

        Returns:
            torch.Tensor
        """
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
        """Forward pass for this class. Uses `self.spatial_dim` to decide 
        between 2D and 3D operations.
        
        Args:
            X (torch.Tensor)

        Returns:
            torch.Tensor
        """
        if self.spatial_dim == 2:
            return self.forward_2d(X)
        elif self.spatial_dim == 3:
            return self.forward_3d(X)

class UNet(torch.nn.Module):
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
        feature_conditioning_params: Dict[str,torch.Tensor]=None,
        deep_supervision: bool=False) -> torch.nn.Module:
        """Standard U-Net [1] implementation. Features some useful additions 
        such as residual links, different upsampling types, normalizations 
        (batch or instance) and ropouts (dropout and U-out). This version of 
        the U-Net has been implemented in such a way that it can be easily 
        expanded.

        Args:
            spatial_dimensions (int, optional): number of dimensions for the 
                input (not counting batch or channels). Defaults to 2.
            encoding_operations (List[ModuleList], optional): backbone operations 
                (uses these rather than a standard U-Net encoder). 
                Must be a list where each element is a list containing a 
                convolutional operation and a downsampling operation.
            conv_type (str, optional): types of base convolutional operations.
                For now it supports regular convolutions ("regular"), residual
                convolutions ("resnet") and convolutions followed by squeeze
                and excite modules ("sae"). Defaults to "regular".
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
            bottleneck_classification (bool, optional): sets up a 
                classification task using the channel-wise maximum of the 
                bottleneck layer. Defaults to False.
            skip_conditioning (int, optional): assumes that the skip 
                layers will be conditioned by an image provided as the 
                second argument of forward. This parameter specifies the 
                number of channels in that image. Useful if any priors 
                (complementary segmentation masks) are available.
            feature_conditioning (int,optional): linearly transforms tabular 
                features and adds them to each channel of the skip connections.
                Useful to include tabular features in the prediction algorithm.
                Defaults to None.
            feature_conditioning_params (Dict[str,torch.Tensor], optional): 
                dictionary with keys "mean" and "std" to normalize the tabular 
                features. Must be present if feature conditioning is used. 
                Defaults to None.
            deep_supervision (bool, optional): forward method returns 
                segmentation predictions obtained from each decoder block.
        [1] https://www.nature.com/articles/s41592-018-0261-2
        [2] https://openaccess.thecvf.com/content_CVPR_2019/papers/Li_Understanding_the_Disharmony_Between_Dropout_and_Batch_Normalization_by_Variance_CVPR_2019_paper.pdf

        Returns:
            torch.nn.Module: a U-Net module.
        """
        
        super().__init__()
        self.spatial_dimensions = spatial_dimensions
        self.encoding_operations = encoding_operations
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
        self.bottleneck_classification = bottleneck_classification
        self.skip_conditioning = skip_conditioning
        self.feature_conditioning = feature_conditioning
        self.feature_conditioning_params = feature_conditioning_params
        self.deep_supervision = deep_supervision
        
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

        self.loss_accumulator = 0.
        self.loss_accumulator_d = 0.

    def get_norm_op(self):
        """Retrieves the normalization operation using `self.norm_type`.
        """
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
        """Retrieves the dropout operations using `self.dropout_type`.
        """
        if self.dropout_type is None:
            self.drop_op = torch.nn.Identity

        elif self.dropout_type == "dropout":
            self.drop_op = torch.nn.Dropout
        elif self.dropout_type == "uout":
            self.drop_op = UOut
            
    def get_conv_op(self):
        """Retrieves the convolutional operations using `self.conv_type`.
        """
        if self.spatial_dimensions == 2:
            if self.conv_type == "regular":
                self.conv_op_enc = torch.nn.Conv2d
                self.conv_op_dec = torch.nn.Conv2d
            elif self.conv_type == "resnet":
                self.conv_op_enc = self.res_block_conv_2d
                self.conv_op_dec = torch.nn.Conv2d
            elif self.conv_type == "sae":
                self.conv_op_enc = torch.nn.Conv2d
                self.conv_op_dec = self.sae_2d
            elif self.conv_type == "asp":
                self.conv_op_enc = self.asp_2d
                self.conv_op_dec = self.sae_2d
        if self.spatial_dimensions == 3:
            if self.conv_type == "regular":
                self.conv_op_enc = torch.nn.Conv3d
                self.conv_op_dec = torch.nn.Conv3d
            elif self.conv_type == "resnet":
                self.conv_op_enc = self.res_block_conv_3d
                self.conv_op_dec = torch.nn.Conv3d
            elif self.conv_type == "sae":
                self.conv_op_enc = torch.nn.Conv3d
                self.conv_op_dec = self.sae_3d
            elif self.conv_type == "asp":
                self.conv_op_enc = self.asp_3d
                self.conv_op_dec = self.sae_3d
    
    def res_block_conv_2d(self,in_d,out_d,kernel_size,
                          stride=None,padding=None):
        """Convenience wrapper for ResidualBlock2d.
        """
        if in_d > 32:
            inter_d = int(in_d//2)
        else:
            inter_d = None
        return ResidualBlock2d(
            in_d,kernel_size,inter_d,out_d,adn_fn=self.adn_fn)

    def res_block_conv_3d(self,in_d,out_d,kernel_size,
                          stride=None,padding=None):
        """Convenience wrapper for ResidualBlock3d.
        """
        if in_d > 32:
            inter_d = int(in_d//2)
        else:
            inter_d = None
        return ResidualBlock3d(
            in_d,kernel_size,inter_d,out_d,adn_fn=self.adn_fn)

    def sae_2d(self,in_d,out_d,kernel_size,stride=1,padding=0):
        return torch.nn.Sequential(
            torch.nn.Conv2d(in_d,out_d,kernel_size=kernel_size,
                            stride=stride,padding=padding),
            ConcurrentSqueezeAndExcite2d(out_d))

    def sae_3d(self,in_d,out_d,kernel_size,stride=None,padding=None):
        return torch.nn.Sequential(
            torch.nn.Conv3d(in_d,out_d,kernel_size=kernel_size,
                            stride=stride,padding=padding),
            ConcurrentSqueezeAndExcite3d(out_d))

    def asp_2d(self,in_d,out_d,kernel_size,stride=1,padding=0):
        return AtrousSpatialPyramidPooling2d(
            in_d,out_d,[1,2],get_adn_fn(2,"instance",
            self.activation_fn,self.dropout_param))

    def asp_3d(self,in_d,out_d,kernel_size,stride=None,padding=None):
        return AtrousSpatialPyramidPooling3d(
            in_d,out_d,[1,2],get_adn_fn(3,"instance",
            self.activation_fn,self.dropout_param))

    def init_upscale_ops(self):
        """Initializes upscaling operations.
        """
        depths_a = self.depth[:0:-1]
        depths_b = self.depth[-2::-1]
        if self.upscale_type == "upsample":
            if self.spatial_dimensions == 2:
                upscale_ops = [
                    torch.nn.Sequential(
                        torch.nn.Conv2d(d1,d2,1),
                        torch.nn.Upsample(scale_factor=s,mode=self.interpolation))
                    for d1,d2,s in zip(depths_a,depths_b,self.strides[::-1])]
            if self.spatial_dimensions == 3:
                upscale_ops = [
                    torch.nn.Sequential(
                        torch.nn.Conv3d(d1,d2,1),
                        torch.nn.Upsample(scale_factor=s,mode=self.interpolation))
                    for d1,d2,s in zip(depths_a,depths_b,self.strides[::-1])]
        elif self.upscale_type == "transpose":
            if self.spatial_dimensions == 2:
                upscale_ops = [
                    torch.nn.ConvTranspose2d(
                        d1,d2,2,stride=s,padding=0) 
                    for d1,d2,s in zip(depths_a,depths_b,self.strides[::-1])]
            if self.spatial_dimensions == 3:
                upscale_ops = [
                    torch.nn.ConvTranspose3d(
                        d1,d2,2,stride=s,padding=0) 
                    for d1,d2,s in zip(depths_a,depths_b,self.strides[::-1])]
            self.upscale_ops = torch.nn.ModuleList(upscale_ops)
    
    def init_link_ops(self):
        """Initializes linking (skip) operations.
        """
        if self.skip_conditioning is not None:
            ex = self.skip_conditioning
        else:
            ex = 0
        if self.link_type == "identity":
            self.link_ops = [
                torch.nn.Identity() for _ in self.depth[:-1]]
            self.link_ops = torch.nn.ModuleList(self.link_ops)
        elif self.link_type == "conv":
            if self.spatial_dimensions == 2:
                self.link_ops = [
                    torch.nn.Conv2d(d+ex,d,3,padding=self.padding)
                    for d in self.depth[-2::-1]]
                self.link_ops = torch.nn.ModuleList(self.link_ops)
            elif self.spatial_dimensions == 3:
                self.link_ops = [
                    torch.nn.Conv3d(d+ex,d,3,padding=self.padding)
                    for d in self.depth[-2::-1]]
                self.link_ops = torch.nn.ModuleList(self.link_ops)
        elif self.link_type == "residual":
            if self.spatial_dimensions == 2:
                self.link_ops =[
                    ResidualBlock2d(d+ex,3,out_channels=d,adn_fn=self.adn_fn) 
                    for d in self.depth[-2::-1]]
                self.link_ops = torch.nn.ModuleList(self.link_ops)
            elif self.spatial_dimensions == 3:
                self.link_ops =[
                    ResidualBlock3d(d+ex,3,out_channels=d,adn_fn=self.adn_fn) 
                    for d in self.depth[-2::-1]]
                self.link_ops = torch.nn.ModuleList(self.link_ops)
    
    def interpolate_depths(self,a:int,b:int,n=3)->List[int]:
        """Interpolates between two whole numbers. Not really used.

        Args:
            a (int): start integer
            b (int): final integer
            n (int, optional): number of points. Defaults to 3.

        Returns:
            (List[int]): list of `n` integers sampled between `a` and `b`.
        """
        return list(np.linspace(a,b,n,dtype=np.int32))

    def init_encoder(self):
        """Initializes the encoder operations.
        """
        self.encoding_operations = torch.nn.ModuleList([])
        previous_d = self.n_channels
        for i in range(len(self.depth)-1):
            d,k,s = self.depth[i],self.kernel_sizes[i],self.strides[i]
            op = torch.nn.Sequential(
                self.conv_op_enc(previous_d,d,kernel_size=k,stride=1,
                                 padding=self.padding),
                self.adn_fn(d))
            op_downsample = torch.nn.Sequential(
                self.conv_op_enc(d,d,kernel_size=k,stride=s,
                                 padding=self.padding),
                self.adn_fn(d))
            self.encoding_operations.append(
                torch.nn.ModuleList([op,op_downsample]))
            previous_d = d
        op = torch.nn.Sequential(
            self.conv_op_enc(self.depth[-2],self.depth[-1],kernel_size=k,
                             stride=1,padding=self.padding),
            self.adn_fn(self.depth[-1]))
        op_downsample = torch.nn.Identity()        
        self.encoding_operations.append(
            torch.nn.ModuleList([op,op_downsample]))

    def init_encoder_backbone(self):
        """Initializes the encoder operations.
        """
        if self.spatial_dimensions == 2:
            mp = torch.nn.MaxPool2d
        elif self.spatial_dimensions == 3:
            mp = torch.nn.MaxPool3d
        for i in range(len(self.encoding_operations)):
            self.encoding_operations[i][1] = mp(
                self.kernel_sizes[i],2,self.kernel_sizes[i]//2)
        self.encoding_operations[-1][1] = torch.nn.Identity()

    def init_decoder(self):
        """Initializes the decoder operations.
        """
        self.decoding_operations = torch.nn.ModuleList([])
        depths = self.depth[-2::-1]
        kernel_sizes = self.kernel_sizes[-2::-1]
        self.deep_supervision_ops = torch.nn.ModuleList([])
        for i in range(len(depths)):
            d,k = depths[i],kernel_sizes[i]
            op = torch.nn.Sequential(
                self.conv_op_dec(d*2,d,kernel_size=k,stride=1,
                                 padding=self.padding),
                self.adn_fn(d),
                self.conv_op_dec(d,d,kernel_size=k,stride=1,
                                 padding=self.padding),
                self.adn_fn(d))
            self.decoding_operations.append(op)
            if self.deep_supervision == True:
                self.deep_supervision_ops.append(self.get_final_layer(d))

    def get_final_layer(self,d:int)->torch.nn.Module:
        """Returns the final layer.

        Args:
            d (int): depth.

        Returns:
            torch.nn.Module: final classification layer.
        """
        if self.spatial_dimensions == 2:
            op = torch.nn.Conv2d
        elif self.spatial_dimensions == 3:
            op = torch.nn.Conv3d
        if self.n_classes > 2:
            return torch.nn.Sequential(
                op(d,d,3,padding=1),self.adn_fn(d),
                op(d,d,1),self.adn_fn(d),
                op(d,self.n_classes,1),
                torch.nn.Softmax(dim=1))
        else:
            # coherces to a binary classification problem rather than
            # to a multiclass problem with two classes
            return torch.nn.Sequential(
                op(d,d,3,padding=1),self.adn_fn(d),
                op(d,d,1),self.adn_fn(d),
                op(d,1,1),
                torch.nn.Sigmoid())

    def init_final_layer(self):
        """Initializes the classification layer (simple linear layer).
        """
        o = self.depth[0]
        self.final_layer = self.get_final_layer(o)

    def init_bottleneck_classifier(self):
        """Initiates the layers for bottleneck classification.
        """
        nc = self.n_classes if self.n_classes > 2 else 1
        self.bottleneck_classifier = torch.nn.Linear(
            self.depth[-1],nc)

    def adn_fn(self,s:int)->torch.Tensor:
        """Convenience wrapper for ADN function.

        Args:
            s (int): number of layers.

        Returns:
            torch.Tensor: ActDropNorm module
        """
        return ActDropNorm(
            in_channels=s,ordering='NDA',
            norm_fn=self.norm_op,
            act_fn=self.activation_fn,
            dropout_fn=self.drop_op,
            dropout_param=self.dropout_param)

    def init_feature_conditioning_operations(self):
        depths = self.depth[-2::-1]
        self.feature_conditioning_ops = torch.nn.ModuleList([])
        if self.feature_conditioning_params is not None:
            self.f_mean = torch.nn.parameter.Parameter(
                self.feature_conditioning_params["mean"],requires_grad=False)
            self.f_std = torch.nn.parameter.Parameter(
                self.feature_conditioning_params["std"],requires_grad=False)
        else:
            self.f_mean = torch.nn.parameter.Parameter(
                torch.zeros([self.feature_conditioning],requires_grad=False))
            self.f_std = torch.nn.parameter.Parameter(
                torch.ones([self.feature_conditioning],requires_grad=False))
        for d in depths:
            op = torch.nn.Sequential(
                torch.nn.Linear(self.feature_conditioning,d),
                get_adn_fn(1,"batch","swish",self.dropout_param)(d),
                torch.nn.Linear(d,d),
                get_adn_fn(1,"batch","sigmoid",self.dropout_param)(d))
            self.feature_conditioning_ops.append(op)

    def unsqueeze_to_dim(self,X:torch.Tensor,
                         target:torch.Tensor)->torch.Tensor:
        n = len(target.shape) - len(X.shape)
        if n > 0:
            for _ in range(n):
                X = X.unsqueeze(-1)
        return X

    def forward(self,X:torch.Tensor,
                X_skip_layer:torch.Tensor=None,
                X_feature_conditioning:torch.Tensor=None,
                return_features=False,
                return_bottleneck=False)->torch.Tensor:
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
        if return_bottleneck == True:
            return None,None,bottleneck
        
        deep_outputs = []
        for i in range(len(self.decoding_operations)):
            op = self.decoding_operations[i]
            link_op = self.link_ops[i]
            up = self.upscale_ops[i]
            if X_skip_layer is not None:
                S = encoding_out[-i-2].shape[2:]
                xfl = F.interpolate(X_skip_layer,S,mode='nearest')
                link_op_input = torch.cat([encoding_out[-i-2],xfl],axis=1)
            else:
                link_op_input = encoding_out[-i-2]
            encoded = link_op(link_op_input)
            if X_feature_conditioning is not None:
                feat_op = self.feature_conditioning_ops[i]
                transformed_features = feat_op(X_feature_conditioning)
                transformed_features = self.unsqueeze_to_dim(
                    transformed_features,encoded)
                encoded = torch.multiply(
                    encoded,transformed_features)
            curr = up(curr)
            sh = list(curr.shape)[2:]
            sh2 = list(encoded.shape)[2:]
            if np.prod(sh) < np.prod(sh2):
                encoded = crop_to_size(encoded,sh)
            if np.prod(sh) > np.prod(sh2):
                curr = crop_to_size(curr,sh2)
            curr = torch.concat((curr,encoded),dim=1)
            curr = op(curr)
            deep_outputs.append(curr)

        final_features = curr

        curr = self.final_layer(curr)
        if return_features == True:
            return curr,final_features,bottleneck

        if self.bottleneck_classification == True:
            bottleneck = bottleneck.flatten(start_dim=2).max(-1).values
            bn_out = self.bottleneck_classifier(bottleneck)
        else:
            bn_out = None
        
        if self.deep_supervision == True:
            for i in range(len(deep_outputs)):
                o = deep_outputs[i]
                op = self.deep_supervision_ops[i]
                deep_outputs[i] = op(o)
            return curr,bn_out,deep_outputs

        return curr,bn_out
