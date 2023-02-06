import torch
import torch.nn.functional as F

class UOut(torch.nn.Module):
    def __init__(self,beta: float=0.) -> torch.nn.Module:
        """Form of dropout suggested in [1]. Rather than dropping out 
        specific channels, each channel X is modified such that 
        $X' = X + rX$, where $x \sim U(-\beta,\beta)$. This guarantees
        a much smaller variance shift and allows for a dropout-like
        activation layer to be combined with batch-normalization without
        performance drops (more info on this in [1]). This operation is
        performed on the first dimension after the batch dimension (assumed
        to be the channel dimension).
        
        [1] https://ieeexplore.ieee.org/document/8953671

        Args:
            beta (float, optional): beta parameter for the uniform 
            distribution from which $r$ will be sampled for reference, the
            original authors use a value of 0.1. Defaults to 0.

        Returns:
            torch.nn.Module: a Torch Module
        """
        super().__init__()
        self.beta = beta
    
    def forward(self,X: torch.Tensor) -> torch.Tensor:
        """Forward pass for this Module.

        Args:
            X (torch.Tensor): Tensor 

        Returns:
            torch.Tensor: Tensor
        """
        if self.training is True:
            sh = list(X.shape)
            for i in range(2,len(sh)):
                sh[i] = 1
            r = torch.rand(sh).to(X.device)
            r = r*self.beta*2 - self.beta
            X = X + X*r
            return X
        else:
            return X

class LayerNorm(torch.nn.Module):
    # from: https://github.com/facebookresearch/VICRegL/blob/main/convnext.py
    r""" LayerNorm that supports two data formats: channels_last (default) or channels_first. 
    The ordering of the dimensions in the inputs. channels_last corresponds to inputs with 
    shape (batch_size, height, width, channels) while channels_first corresponds to inputs 
    with shape (batch_size, channels, height, width).
    """

    def __init__(self, normalized_shape, eps=1e-6, data_format="channels_last"):
        super().__init__()
        self.weight = torch.nn.Parameter(torch.ones(normalized_shape))
        self.bias = torch.nn.Parameter(torch.zeros(normalized_shape))
        self.eps = eps
        self.data_format = data_format
        if self.data_format not in ["channels_last", "channels_first"]:
            raise NotImplementedError
        self.normalized_shape = (normalized_shape,)

    def forward(self, x):
        if self.data_format == "channels_last":
            return F.layer_norm(
                x, self.normalized_shape, self.weight, self.bias, self.eps
            )
        elif self.data_format == "channels_first":
            u = x.mean(1, keepdim=True)
            s = (x - u).pow(2).mean(1, keepdim=True)
            x = (x - u) / torch.sqrt(s + self.eps)
            x = self.weight[:, None, None] * x + self.bias[:, None, None]
            return x

class LayerNormChannelsFirst(torch.nn.Module):
    # adapted from: https://github.com/facebookresearch/VICRegL/blob/main/convnext.py
    r""" LayerNorm that supports channels_first. 
    """

    def __init__(self, normalized_shape, eps=1e-6):
        super().__init__()
        self.weight = torch.nn.Parameter(torch.ones(normalized_shape))
        self.bias = torch.nn.Parameter(torch.zeros(normalized_shape))
        self.eps = eps
        self.normalized_shape = (normalized_shape,)

    def forward(self, x):
        u = x.mean(1, keepdim=True)
        s = (x - u).pow(2).mean(1, keepdim=True)
        x = (x - u) / torch.sqrt(s + self.eps)
        x = self.weight[:, None, None] * x + self.bias[:, None, None]
        return x
