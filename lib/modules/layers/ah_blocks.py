import torch
from .standard_blocks import ConvolutionalBlock2d
from .standard_blocks import ConvolutionalBlock3d

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
