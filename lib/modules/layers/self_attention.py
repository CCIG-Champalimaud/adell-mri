import torch

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
        n = X.dim()
        channel_average = torch.flatten(X,start_dim=2).mean(-1)
        channel_squeeze = self.op(channel_average)
        channel_squeeze = channel_squeeze.reshape(
            *channel_squeeze.shape,*[1 for _ in range(n-2)])
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
