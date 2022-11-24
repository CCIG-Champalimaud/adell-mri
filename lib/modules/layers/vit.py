import numpy as np
import torch
import einops

from .linear_blocks import MultiHeadAttention
from .linear_blocks import MLP
from .adn_fn import get_adn_fn
from ...types import Size2dOr3d,List

class LinearEmbedding(torch.nn.Module):
    def __init__(self,
                 image_size:Size2dOr3d,
                 patch_size:Size2dOr3d,
                 n_channels:int):
        super().__init__()
        self.image_size = image_size
        self.patch_size = patch_size
        self.n_channels = n_channels
        
        assert len(self.image_size) == len(self.patch_size),\
            "image_size and patch_size should have the same length"
        assert len(self.image_size) in [2,3],\
            "len(image_size) has to be 2 or 3"
        assert all(
            [x % y == 0 for x,y in zip(self.image_size,self.patch_size)]),\
            "the elements of image_size should be divisible by those in patch_size"
        
        self.calculate_parameters()
        self.initialize_positional_embeddings()
        self.get_einop_params()
        
    def calculate_parameters(self):
        """Calculates a few handy parameters for the linear embedding.
        """
        self.n_dims = len(self.image_size)
        self.n_patches_split = [
            x // y for x,y in zip(self.image_size,self.patch_size)]
        self.n_patches = np.prod(self.n_patches_split)
        self.n_tokens = np.prod(self.patch_size) * self.n_channels
        self.linearized_dim = [-1,self.n_patches,self.n_tokens]
        
    def initialize_positional_embeddings(self):
        """Initilizes the positional embedding.
        """
        self.positional_embedding = torch.nn.Parameter(
            1,self.n_patches,self.n_tokens)
        
    def get_einop_params(self):
        """Defines all necessary einops constants. This reduces the amount of 
        inference that einops.rearrange has to do internally and ensurest that
        this operation is a bit easier to inspect.
        """
        if self.n_dims == 2:
            self.einop_str = "b c (h x) (w y) -> b (h w) (c x y)"
            self.einop_inv_str = "b (h w) (c x y) -> b c (h x) (w y)"
        elif self.n_dims:
            self.einop_str = "b c (h x) (w y) (d z) -> b (h w d) (c x y z)"
            self.einop_inv_str = "b (h w d) (c x y z) -> b c (h x) (w y) (d z)"
        self.einop_dict = {
            k:s for s,k in zip(self.patch_size,["x","y","z"])}
        self.einop_dict["c"] = self.n_channels
        self.einop_dict.update(
            {k:s for s,k in zip(self.n_patches_split,["h","w","d"])})
    
    def rearrange(self,x:torch.Tensor)->torch.Tensor:
        """Applies einops.rearrange given the parameters inferred in 
        self.get_einop_params.

        Args:
            x (torch.Tensor): a tensor of size (b,c,h,w,(d))

        Returns:
            torch.Tensor: a tensor of size (b,h*x,w*y,(d*z))
        """
        return einops.rearrange(x,self.einop_str,**self.einop_dict)

    def rearrange_inverse(self,x:torch.Tensor)->torch.Tensor:
        """Reverses the self.rearrange operation using the parameters inferred
        in self.get_einop_params.

        Args:
            x (torch.Tensor): a tensor of size (b,h*x,w*y,(d*z))

        Returns:
            x torch.Tensor: a tensor of size (b,c,h,w,(d))
        """
        return einops.rearrange(x,self.einop_inv_str,**self.einop_dict)
    
    def forward(self,X):
        # output should always be [X.shape[0],self.n_patches,self.n_tokens]
        return self.rearrange(X) + self.positional_embedding
    
class TransformerBlock(torch.nn.Module):
    def __init__(self,
                 input_dim_primary:int,
                 attention_dim:int,
                 hidden_dim:int,
                 n_heads=4,
                 mlp_structure:List[int]=[32,32],
                 adn_fn=torch.nn.Identity):
        super().__init__()
        self.input_dim_primary = input_dim_primary
        self.attention_dim = attention_dim
        self.hidden_dim = hidden_dim
        self.n_heads = n_heads
        self.mlp_structure = mlp_structure
        self.adn_fn = adn_fn
        
        self.mha = MultiHeadAttention(
            self.input_dim_primary,
            self.attention_dim,
            self.hidden_dim,
            self.input_dim_primary,
            n_heads=self.n_heads)
        
        self.init_layer_norm_ops()
        self.init_mlp()
    
    def init_layer_norm_ops(self):
        """Initializes the MLP in the last step of the transformer.
        """
        self.norm_op_1 = torch.nn.LayerNorm(self.input_dim_primary)
        self.norm_op_2 = torch.nn.LayerNorm(self.input_dim_primary)

    def init_mlp(self):
        """Initializes the MLP in the last step of the transformer.
        """
        self.mlp = MLP(self.input_dim_primary,self.input_dim_primary,
                       self.mlp_structure,self.adn_fn)
    
    def forward(self,X):
        X = X + self.mha(self.norm_op_1(X))
        X = X + self.mlp(self.norm_op_2(X))
        return X