import torch

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
        if self.training == True:
            sh = list(X.shape)
            for i in range(2,len(sh)):
                sh[i] = 1
            r = torch.rand(sh).to(X.device)
            r = r*self.beta*2 - self.beta
            X = X + X*r
            return X
        else:
            return X