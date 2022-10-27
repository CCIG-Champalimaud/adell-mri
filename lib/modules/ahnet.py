import torch
from .layers import *

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
