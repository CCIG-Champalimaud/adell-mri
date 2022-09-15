from re import M
import pytorch_lightning as pl
import torch.nn.functional as F

from copy import deepcopy
from ..types import *

def reshape_weight_to_matrix(weight: torch.Tensor,dim:int=0) -> torch.Tensor:
    """Reshapes an n-dimensional tensor into a matrix.
    
    From https://pytorch.org/docs/stable/_modules/torch/nn/utils/spectral_norm.html

    Args:
        weight (torch.Tensor): weight matrix.
        dim (int, optional): dimension corresponing to first matrix dimension.
            Defaults to 0.

    Returns:
        torch.Tensor: reshaped tensor.
    """
    weight_mat = weight
    if dim != 0:
        weight_mat = weight_mat.permute(
            dim,
            *[d for d in range(weight_mat.dim()) if d != dim])
    height = weight_mat.size(0)
    return weight_mat.reshape(height, -1)

class SpectralNorm(pl.Callback):
    def __init__(self,power_iterations,eps=1e-8,name="weight"):
        """Callback that performs spectral normalization before each training
        batch. It uses the same power iteration implementation as specified in
        [1] and is largely based in the PyTorch implementation [2].

        Importantly, this stores u and v as a parameter dict within the 
        callback rather than as a part of 

        [1] https://arxiv.org/abs/1802.05957
        [2] https://pytorch.org/docs/stable/_modules/torch/nn/utils/spectral_norm.html

        Args:
            power_iterations (_type_): _description_
            eps (_type_, optional): _description_. Defaults to 1e-8.
            name (str, optional): _description_. Defaults to "weight".
        """
        self.power_iterations = power_iterations
        self.eps = eps
        self.name = name
        
        self.u_dict = torch.nn.ParameterDict({})
        self.v_dict = torch.nn.ParameterDict({})

    def on_train_batch_start(self,
                             trainer: pl.Trainer,
                             pl_module: pl.LightningModule,
                             batch: Sequence,
                             batch_idx: int):
        return self(pl_module)
    
    def __call__(self,module):
        for k,param in module.named_parameters():
            if self.name in k:
                weight = deepcopy(param.data)
                sh = weight.shape
                weight_mat = reshape_weight_to_matrix(weight)
                h, w = weight_mat.size()
                if k not in self.u_dict:
                    u = weight.new_empty(h).normal_(0,1)
                    v = weight.new_empty(w).normal_(0,1)
                    self.u_dict[k.replace(".","_")] = torch.nn.Parameter(u)
                    self.v_dict[k.replace(".","_")] = torch.nn.Parameter(v)
                
                u = self.u_dict[k.replace(".","_")].data
                v = self.v_dict[k.replace(".","_")].data

                with torch.no_grad():
                    for p in range(self.power_iterations):
                        v = F.normalize(
                            torch.mv(weight_mat.t(),u),
                            dim=0,eps=self.eps,out=v)
                        u = F.normalize(
                            torch.mv(weight_mat,v),
                            dim=0,eps=self.eps, out=u)

                self.u_dict[k.replace(".","_")].data = u
                self.v_dict[k.replace(".","_")].data = v

                sigma = torch.dot(u,torch.mv(weight_mat, v))
                weight = weight/sigma

                param.data = weight.reshape(sh)
