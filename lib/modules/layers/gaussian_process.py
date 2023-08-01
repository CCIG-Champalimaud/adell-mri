import torch
from torch.distributions.multivariate_normal import MultivariateNormal
from .utils import unsqueeze_to_target

from typing import Tuple

class GaussianProcessLayer(torch.nn.Module):
    """Gaussian process layer as implemented in [1]. The model is first trained
    and only `self.weights` is updated. `self.W` and `self.b` are randomly sampled
    parameters for random Fourier features (this is generally used to map the
    input feature space to a lower dimension space, making the GP more 
    efficient) and are not updated. 

    The inverse covariance can be updated at the end of the training cheaply
    by running `self.update_inv_cov` over the several batches that comprise a 
    training epoch. 

    The covariance can then then be calculated (and stored in `self.cov`) by
    running `self.get_cov`. This enables sampling from the inferred distribution
    using `self.rsample(X,n)`, where X is the input tensor and n is the number
    of samples.
    
    [1] https://arxiv.org/pdf/2006.10108.pdf
    """
    def __init__(self,in_channels:int,out_channels:int,m:float=0.999):
        """
        Args:
            in_channels (int): input channels.
            out_channels (int): output channels (after RFF transform)
            m (float, optional): momentum for updating inv covariance matrix. 
                Defaults to 0.999.
        """
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.m = m

        self.initialize_params()

    def initialize_params(self):
        i = self.in_channels
        o = self.out_channels
        self.scaling_term = torch.sqrt(torch.as_tensor(2./i))
        self.W = torch.nn.Parameter(
            torch.normal(torch.zeros([1,o,i]),torch.ones([1,o,i])).float(),
            requires_grad=False)
        self.b = torch.nn.Parameter(
            torch.rand([1,o]).float()*torch.pi,
            requires_grad=False)
        self.weights = torch.nn.Parameter(
            torch.normal(torch.zeros([o,o]),torch.ones([o,o])).float(),
            requires_grad=True)
        self.inv_conv = torch.nn.Parameter(
            torch.eye(o,o).unsqueeze(0).float(),
            requires_grad=False)
    
    def update_inv_cov(self,X:torch.Tensor,y:torch.Tensor):
        y = y.unsqueeze(-1)
        phi = self.calculate_phi(X)
        phi,phi_t = phi.unsqueeze(-2),phi.unsqueeze(-1)
        K = torch.matmul(phi_t,phi)
        if len(K.shape) > 3:
            K = K.flatten(start_dim=1,end_dim=-3)
            K = K.mean(1)
        update_term = y * (1-y) * K
        self.inv_conv.data = torch.add(
            self.inv_conv * self.m,
            (1-self.m) * update_term.sum(0))

    def get_cov(self):
        self.cov = torch.linalg.inv(self.inv_conv)

    def calculate_phi(self,X:torch.Tensor):
        """
        Calculates phi (low rank matrix used to calculate the kernel matrix)

        Args:
            X (torch.Tensor): input tensor.

        Returns:
            phi
        """
        X = X.swapaxes(1,-1)
        X = X.unsqueeze(-1)
        W = unsqueeze_to_target(self.W,X,1)
        mm = torch.matmul(-W,X).squeeze(-1)
        return self.scaling_term*torch.cos(mm+self.b)
    
    def forward(self,X:torch.Tensor):
        """
        Uses phi to calculate the mean (phi * self.weights)

        Args:
            X (torch.Tensor): input tensor.

        Returns:
            mean of the Gaussian process
        """
        phi = self.calculate_phi(X)
        output = phi @ self.weights.T
        if len(output.shape) > 2:
            output = output.swapaxes(1,-1)
        return output
    
    def get_parameters(self,X:torch.Tensor)->Tuple[torch.Tensor,torch.Tensor]:
        """
        Returns the mean and covariance for a set of input samples.

        Args:
            X (torch.Tensor): input tensor.

        Raises:
            Exception: if self.get_cov() has not been called.

        Returns:
            Tuple[torch.Tensor,torch.Tensor]: mean and covariance for Gaussian 
                process.
        """
        if hasattr(self,"cov") == False:
            raise Exception(
                "self.get_cov() must be called before getting parameters")
        phi = self.calculate_phi(X)
        mean = phi @ self.weights
        cov = phi @ self.cov
        return mean,cov

    def rsample(self,X:torch.Tensor,n_samples:int)->torch.Tensor:
        """
        Samples from fitted Gaussian process conditioned on input samples X.

        Args:
            X (torch.Tensor): input tensor with shape [b,in_channels].
            n_samples (int): number of samples.

        Returns:
            torch.Tensor: tensor with shape [n_samples,b,out_channels].
        """
        mean,cov = self.get_parameters(X)
        mvn = MultivariateNormal(mean,cov)
        return mvn.rsample(n_samples)