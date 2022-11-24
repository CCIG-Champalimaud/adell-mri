import numpy as np
import torch
import einops
from copy import deepcopy

from .linear_blocks import MultiHeadAttention
from .linear_blocks import MLP
from .adn_fn import get_adn_fn
from ...types import Size2dOr3d,List,Union

class LinearEmbedding(torch.nn.Module):
    def __init__(self,
                 image_size:Size2dOr3d,
                 patch_size:Size2dOr3d,
                 n_channels:int,
                 dropout_rate:float=0.0,
                 embed_method:str="linear"):
        super().__init__()
        self.image_size = image_size
        self.patch_size = patch_size
        self.n_channels = n_channels
        self.dropout_rate = dropout_rate
        self.embed_method = embed_method
        
        assert self.embed_method in ["linear","convolutional"],\
            "embed_method must be one of 'linear' or 'convolutional'"
        assert len(self.image_size) == len(self.patch_size),\
            "image_size and patch_size should have the same length"
        assert len(self.image_size) in [2,3],\
            "len(image_size) has to be 2 or 3"
        assert all(
            [x % y == 0 for x,y in zip(self.image_size,self.patch_size)]),\
            "the elements of image_size should be divisible by those in patch_size"
        
        self.calculate_parameters()
        self.init_conv_if_necessary()
        self.init_dropout()
        self.initialize_positional_embeddings()
        self.get_einop_params()
        self.linearized_dim = [-1,self.n_patches,self.n_tokens]
        
    def init_conv_if_necessary(self):
        """Initializes a convolutional if embed_method == "convolutional"
        """
        if self.embed_method == "convolutional":
            self.n_tokens = self.n_tokens // self.n_channels
            if self.n_dims == 2:
                self.conv = torch.nn.Conv2d(
                    self.n_channels,self.n_tokens,
                    self.patch_size,stride=self.patch_size)
            elif self.n_dims == 3:
                self.conv = torch.nn.Conv3d(
                    self.n_channels,self.n_tokens,
                    self.patch_size,stride=self.patch_size)

    def init_dropout(self):
        self.drop_op = torch.nn.Dropout(self.dropout_rate)

    def calculate_parameters(self):
        """Calculates a few handy parameters for the linear embedding.
        """
        self.n_dims = len(self.image_size)
        self.n_patches_split = [
            x // y for x,y in zip(self.image_size,self.patch_size)]
        self.n_patches = np.prod(self.n_patches_split)
        self.n_tokens = np.prod(self.patch_size) * self.n_channels
        
    def initialize_positional_embeddings(self):
        """Initilizes the positional embedding.
        """
        self.positional_embedding = torch.nn.Parameter(
            torch.rand([1,self.n_patches,self.n_tokens]))
        
    def get_einop_params(self):
        """Defines all necessary einops constants. This reduces the amount of 
        inference that einops.rearrange has to do internally and ensurest that
        this operation is a bit easier to inspect.
        """
        if self.embed_method == "linear":
            if self.n_dims == 2:
                self.einop_str = "b c (h x) (w y) -> b (h w) (c x y)"
                self.einop_inv_str = "b (h w) (c x y) -> b c (h x) (w y)"
            elif self.n_dims == 3:
                self.einop_str = "b c (h x) (w y) (d z) -> b (h w d) (c x y z)"
                self.einop_inv_str = "b (h w d) (c x y z) -> b c (h x) (w y) (d z)"
            self.einop_dict = {
                k:s for s,k in zip(self.patch_size,["x","y","z"])}
            self.einop_dict["c"] = self.n_channels
            self.einop_dict.update(
                {k:s for s,k in zip(self.n_patches_split,["h","w","d"])})

        elif self.embed_method == "convolutional":
            if self.n_dims == 2:
                self.einop_str = "b c h w -> b (h w) c"
                self.einop_inv_str = "b (h w) c -> b c h w"
            elif self.n_dims == 3:
                self.einop_str = "b c h w d -> b (h w d) c"
                self.einop_inv_str = "b (h w d) c -> b c h w d"
            self.einop_dict = {}
            self.einop_dict["c"] = self.n_tokens
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

    def rearrange_inverse(self,x:torch.Tensor,**kwargs)->torch.Tensor:
        """Reverses the self.rearrange operation using the parameters inferred
        in self.get_einop_params.

        Args:
            x (torch.Tensor): a tensor of size (b,h*x,w*y,(d*z))
            kwargs: arguments that will be appended to self.einop_dict

        Returns:
            x torch.Tensor: a tensor of size (b,c,h,w,(d))
        """
        einop_dict = deepcopy(self.einop_dict)
        for k in kwargs:
            einop_dict[k] = kwargs[k]
        return einops.rearrange(x,self.einop_inv_str,**self.einop_dict)
    
    def forward(self,X):
        # output should always be [X.shape[0],self.n_patches,self.n_tokens]
        if self.embed_method == "convolutional":
            X = self.conv(X)
        return self.drop_op(self.rearrange(X) + self.positional_embedding)
    
class TransformerBlock(torch.nn.Module):
    def __init__(self,
                 input_dim_primary:int,
                 attention_dim:int,
                 hidden_dim:int,
                 n_heads:int=4,
                 mlp_structure:List[int]=[32,32],
                 dropout_rate:float=0.0,
                 adn_fn=torch.nn.Identity):
        super().__init__()
        self.input_dim_primary = input_dim_primary
        self.attention_dim = attention_dim
        self.hidden_dim = hidden_dim
        self.n_heads = n_heads
        self.mlp_structure = mlp_structure
        self.dropout_rate = dropout_rate
        self.adn_fn = adn_fn
        
        self.mha = MultiHeadAttention(
            self.input_dim_primary,
            self.attention_dim,
            self.hidden_dim,
            self.input_dim_primary,
            n_heads=self.n_heads)
        
        self.init_layer_norm_ops()
        self.init_mlp()
    
    def init_drop_ops(self):
        """Initializes the dropout operations.
        """
        self.drop_op_1 = torch.nn.Dropout(self.dropout_rate)
        self.drop_op_2 = torch.nn.Dropout(self.dropout_rate)

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

class TransformerBlockStack(torch.nn.Module):
    def __init__(self,
                 number_of_blocks:int,
                 input_dim_primary:int,
                 attention_dim:int,
                 hidden_dim:int,
                 n_heads:int=4,
                 mlp_structure:List[int]=[32,32],
                 dropout_rate:float=0.0,
                 adn_fn=torch.nn.Identity):
        super().__init__()
        self.number_of_blocks = number_of_blocks
        self.input_dim_primary = input_dim_primary
        self.attention_dim = attention_dim
        self.hidden_dim = hidden_dim
        self.n_heads = n_heads
        self.mlp_structure = mlp_structure
        self.dropout_rate = dropout_rate
        self.adn_fn = adn_fn

        self.init_transformer_blocks()
        
    def check_mlp_structure(self,x):
        if isinstance(x[0],list) == False:
            return [x for _ in range(self.number_of_blocks)]
        else:
            return x

    def convert_to_list_if_necessary(self,x):
        if isinstance(x,list) == False:
            return [x for _ in range(self.number_of_blocks)]
        else:
            return self.check_if_consistent(x)
    
    def check_if_consistent(self,x):
        assert len(x) == self.number_of_blocks

    def init_transformer_blocks(self):
        input_dim_primary = self.convert_to_list_if_necessary(
            self.input_dim_primary)
        attention_dim = self.convert_to_list_if_necessary(
            self.attention_dim)
        hidden_dim = self.convert_to_list_if_necessary(
            self.hidden_dim)
        n_heads = self.convert_to_list_if_necessary(
            self.n_heads)
        mlp_structure = self.check_mlp_structure(
            self.mlp_structure)
        self.transformer_blocks = torch.nn.ModuleList([])
        for i,a,h,n,m in zip(input_dim_primary,
                             attention_dim,
                             hidden_dim,
                             n_heads,
                             mlp_structure):
            self.transformer_blocks.append(
                TransformerBlock(i,a,h,n,m,self.dropout_rate,self.adn_fn))

    def forward(self,X,return_at:Union[str,List[int]]="end"):
        if return_at == "end" or return_at == None:
            return_at = []
        outputs = []
        for i,block in enumerate(self.transformer_blocks):
            X = block(X)
            if i in return_at:
                outputs.append(X)
        return X,outputs
    
class ViT(torch.nn.Module):
    def __init__(self,
                 image_size:Size2dOr3d,
                 patch_size:Size2dOr3d,
                 n_channels:int,
                 number_of_blocks:int,
                 attention_dim:int,
                 hidden_dim:int,
                 n_heads:int=4,
                 dropout_rate:float=0.0,
                 embed_method:str="linear",
                 mlp_structure:List[int]=[32,32],
                 adn_fn=torch.nn.Identity):
        super().__init__()
        self.image_size = image_size
        self.patch_size = patch_size
        self.n_channels = n_channels
        self.number_of_blocks = number_of_blocks
        self.attention_dim = attention_dim
        self.hidden_dim = hidden_dim
        self.n_heads = n_heads
        self.dropout_rate = dropout_rate
        self.embed_method = embed_method
        self.mlp_structure = mlp_structure
        self.adn_fn = adn_fn
        
        self.embedding = LinearEmbedding(
            image_size=self.image_size,
            patch_size=self.patch_size,
            n_channels=self.n_channels,
            dropout_rate=self.dropout_rate
        )
        
        self.input_dim_primary = self.embedding.n_tokens
        
        self.tbs = TransformerBlockStack(
            number_of_blocks=self.number_of_blocks,
            input_dim_primary=self.input_dim_primary,
            attention_dim=self.attention_dim,
            hidden_dim=self.hidden_dim,
            n_heads=self.n_heads,
            mlp_structure=self.mlp_structure,
            dropout_rate=self.dropout_rate,
            adn_fn=self.adn_fn
        )
    
    def forward(self,X,return_at:Union[str,List[int]]="end"):
        embeded_X = self.embedding(X)
        if return_at == "end" or return_at == None:
            return_at = []
        outputs = []
        for i,block in enumerate(self.tbs.transformer_blocks):
            embeded_X = block(embeded_X)
            if i in return_at:
                outputs.append(embeded_X)
        return embeded_X,outputs
