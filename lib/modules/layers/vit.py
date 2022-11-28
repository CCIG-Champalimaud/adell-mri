import time
import numpy as np
import torch
import torch.nn.functional as F
import einops
from itertools import product
from copy import deepcopy

from .linear_blocks import MultiHeadSelfAttention
from .linear_blocks import MLP
from ...types import *

def cyclic_shift_batch(X:torch.Tensor,shift:List[int]):
    """Applies what the authors from SWIN call a "cyclic shift".

    Args:
        X (:torch.Tensors): input tensor.
        shift (List[int]): size of shift along all dimensions.

    Returns:
        torch.Tensor: rolled input.
    """
    dims = [i for i in range(2,len(shift)+2)]
    return torch.roll(X,shifts=shift,dims=dims)

def downsample_ein_op_dict(ein_op_dict:Dict[str,int],
                           scale:int=1)->Dict[str,int]:
    """Downsamples a einops constant dictionary given a downsampling scale 
    factor. Useful for UNETR/SWIN applications. This can then be used to 
    parametrize the LinearEmbedding.rearrange_inverse operation in a way that
    a new tensor with downsampled resolution and more feature maps is produced.

    Args:
        ein_op_dict (Dict[str,int]): a dictionary containing x,y,z,h,w,d,c
            keys (z and d are optional), as produced by 
            LinearEmbedding.get_einop_params.
        scale (int, optional): downsampling scale. Defaults to 1.

    Returns:
        Dict[str,int]: dictionary where the x,y(,z) values are rescaled.
    """
    ein_op_dict = deepcopy(ein_op_dict)
    pairs = (("h","x"),("w","y"),("d","z"))
    p = 0
    for dim,coord in pairs:
        if dim in ein_op_dict:
            p += 1
            ein_op_dict[coord] = ein_op_dict[coord] // scale
    ein_op_dict["c"] = ein_op_dict["c"] * scale**p
    return ein_op_dict

def window_partition(x:torch.Tensor,window_size:Size2dOr3d)->torch.Tensor:
    """
    Reshapes an image/volume batch tensor into smaller image/volumes of
    window_size. Generalizes the implementation in [1] to both images and 
    volumes.
    
    [1] https://github.com/microsoft/Swin-Transformer/blob/main/models/swin_transformer.py
    
    Args:
        x (torch.Tensor): tensor with shape (b,h,w,(d),c)
        window_size (int): window size
    Returns:
        windows: (num_windows*b,
                  window_size[0],
                  window_size[1],
                  (window_size[2]),c)
    """
    sh = x.shape
    B,in_shape,C = sh[0],sh[1:-1],sh[-1]
    view_sh = [B]
    for s,w in zip(in_shape,window_size):
        view_sh.append(s//w)
        view_sh.append(w)
    view_sh.append(C)
    permute_dims = [0,
                    *[1+i*2 for i in range(len(in_shape))],
                    *[2+i*2 for i in range(len(in_shape))]]
    permute_dims.append(len(permute_dims))
    x = x.view(*view_sh)
    windows = x.permute(*permute_dims).contiguous().view(-1,*window_size,C)
    return windows

def generate_mask(image_size:Size2dOr3d,
                  window_size:Size2dOr3d,
                  shift_size:Size2dOr3d)->torch.Tensor:
    """Masks the attention in a self-attention module in shifted inputs. A 
    generalization of the mask generation in [1] for 2 and 3 dimensions.
    
    [1] https://github.com/microsoft/Swin-Transformer/blob/main/models/swin_transformer.py

    Args:
        image_size (Size2dOr3d): size of image.
        window_size (Size2dOr3d): size of window.
        shift_size (Size2dOr3d): size of the shift.

    Returns:
        torch.Tensor: tensor used to mask attention outputs.
    """
    if isinstance(window_size,list) == False:
        window_size = [window_size for _ in image_size]
    if isinstance(shift_size,list) == False:
        shift_size = [shift_size for _ in image_size]
    attn_mask = None
    if any([x > 0 for x in shift_size]):
        # calculate attention mask for SW-MSA
        img_mask = torch.zeros((1, *image_size, 1))  # 1,*image_size,1
        slices = [(slice(0, -w),slice(-w, -s),slice(-s, None))
                  for w,s in zip(window_size,shift_size)]
        cnt = 0
        for x in product(*slices):
            if len(x) == 2:
                img_mask[:, x[0], x[1], :] = cnt
            if len(x) == 3:
                img_mask[:, x[0], x[1], x[2], :] = cnt
            cnt += 1

        mask_windows = window_partition(img_mask, window_size)  # nW, window_size, window_size, 1
        mask_windows = mask_windows.view(-1,np.prod(window_size))
        attn_mask = mask_windows.unsqueeze(1) - mask_windows.unsqueeze(2)
        attn_mask = attn_mask.masked_fill(
            attn_mask != 0, 
            float(-100.0)).masked_fill(attn_mask == 0, float(0.0))

    return attn_mask

class LinearEmbedding(torch.nn.Module):
    """Linearly embeds images as described in the vision transformer paper
    [1]. Essentially, it rearranges a given image of size [b,c,h,w] with a 
    given patch size [p1,p2] such that the output is 
    [b,c*(h//p1)*(w//p2),p1*p2]. This class also features the operations 
    necessary to i) reverse this operation and ii) "reverse" this operation
    considering that the output has to be a downscaled version of the input
    as described in the UNETR paper [2]. The convolutional embeding method
    was inspired by the PatchEmbeddingBlock in MONAI [3] but I reworked it
    so that it *always* uses einops, making operations more easily 
    reversible.
    
    Here, I also included an argument that enables the support of windows as 
    in SWIN.
    
    [1] https://arxiv.org/pdf/2010.11929.pdf
    [2] https://arxiv.org/pdf/2103.10504.pdf
    [3] https://docs.monai.io/en/stable/_modules/monai/networks/blocks/patchembedding.html
    [4] https://arxiv.org/pdf/2103.14030.pdf
    """
    def __init__(self,
                 image_size:Size2dOr3d,
                 patch_size:Size2dOr3d,
                 n_channels:int,
                 window_size:Size2dOr3d=None,
                 dropout_rate:float=0.0,
                 embed_method:str="linear"):
        """    
        Args:
            image_size (Size2dOr3d): size of the input image.
            patch_size (Size2dOr3d): size of the patch size.
            n_channels (int): number of channels in the input image.
            window_size (Size2dOr3d, optional): window size for windowed
                multi-head attention. Defaults to None (no windowing)
            dropout_rate (float, optional): dropout rate of the dropout 
                operation that is applied after the sum of the linear 
                embeddings with the positional embeddings. Defaults to 0.0.
            embed_method (str, optional): . Defaults to "linear".
        """
        super().__init__()
        self.image_size = image_size
        self.patch_size = patch_size
        self.n_channels = n_channels
        self.window_size = window_size
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
        self.linearized_dim = [-1,self.n_patches,self.n_features]
        
    def init_conv_if_necessary(self):
        """Initializes a convolutional if embed_method == "convolutional"
        """
        if self.embed_method == "convolutional":
            self.n_features = self.n_features // self.n_channels
            if self.n_dims == 2:
                self.conv = torch.nn.Conv2d(
                    self.n_channels,self.n_features,
                    self.patch_size,stride=self.patch_size)
            elif self.n_dims == 3:
                self.conv = torch.nn.Conv3d(
                    self.n_channels,self.n_features,
                    self.patch_size,stride=self.patch_size)

    def init_dropout(self):
        """Initialises the dropout operation that is applied after adding the
        positional embeddings to the sequence embeddings.
        """
        self.drop_op = torch.nn.Dropout(self.dropout_rate)

    def calculate_parameters(self):
        """Calculates a few handy parameters for the linear embedding.
        """
        self.n_dims = len(self.image_size)
        if self.window_size is None:
            self.n_patches_split = [
                x // y for x,y in zip(self.image_size,self.patch_size)]
        else:
            # number of patches will be smaller but the number of features
            # remains the same
            self.n_windows = [x // y for x,y in zip(self.image_size,
                                                    self.window_size)]
            self.n_patches_split = [
                x // z // y for x,y,z in zip(self.image_size,
                                             self.patch_size,
                                             self.n_windows)]
        self.n_patches = np.prod(self.n_patches_split)
        self.n_features = np.prod(self.patch_size) * self.n_channels
        
    def initialize_positional_embeddings(self):
        """Initilizes the positional embedding.
        """
        self.positional_embedding = torch.nn.Parameter(
            torch.zeros([1,self.n_patches,self.n_features]))
        
    def get_einop_params(self):
        """Defines all necessary einops constants. This reduces the amount of 
        inference that einops.rearrange has to do internally and ensurest that
        this operation is a bit easier to inspect.
        """
        def einops_tuple(l):
            if isinstance(l,list):
                if len(l) > 1:
                    return "({})".format(" ".join(l))
                else:
                    return l[0]
            else:
                return l
        if self.embed_method == "linear":
            self.lh = ["b","c",["h","x"],["w","y"]]
            self.rh = ["b",["h","w"],["c","x","y"]]
            self.einop_dict = {
                k:s for s,k in zip(self.patch_size,["x","y","z"])}
            self.einop_dict["c"] = self.n_channels
            self.einop_dict.update(
                {k:s for s,k in zip(self.n_patches_split,["h","w","d"])})
            if self.n_dims == 3:
                self.lh.append(["d","z"])
                self.rh[-2].append("d")
                self.rh[-1].append("z")
            if self.window_size is not None:
                windowing_vars = ["w{}".format(i+1) 
                                  for i in range(self.n_dims)]
                for i,w in enumerate(windowing_vars):
                    self.lh[i+2].insert(0,w)
                self.rh.insert(1,einops_tuple(windowing_vars))
                self.einop_dict.update(
                    {k:w for w,k in zip(self.n_windows,["w1","w2","w3"])})

        elif self.embed_method == "convolutional":
            self.lh = ["b","c",["h"],["w"]]
            self.rh = ["b",["h","w"],"c"]
            self.einop_dict = {}
            self.einop_dict["c"] = self.n_features
            self.einop_dict.update(
                {k:s for s,k in zip(self.n_patches_split,["h","w","d"])})
            if self.window_size is not None:
                windowing_vars = ["w{}".format(i+1) 
                                  for i in range(self.n_dims)]
                for i,w in enumerate(windowing_vars):
                    self.lh[i+2].insert(0,w)
                self.rh.insert(1,einops_tuple(windowing_vars))
            if self.n_dims == 3:
                self.lh.append("d")
                self.rh[-2].append("d")

        self.einop_str = "{lh} -> {rh}".format(
            lh=" ".join([einops_tuple(x) for x in self.lh]),
            rh=" ".join([einops_tuple(x) for x in self.rh]))
        self.einop_inv_str = "{lh} -> {rh}".format(
            lh=" ".join([einops_tuple(x) for x in self.rh]),
            rh=" ".join([einops_tuple(x) for x in self.lh]))
                    
    def rearrange(self,X:torch.Tensor)->torch.Tensor:
        """Applies einops.rearrange given the parameters inferred in 
        self.get_einop_params.

        Args:
            X (torch.Tensor): a tensor of size (b,c,h,w,(d))

        Returns:
            torch.Tensor: a tensor of size (b,h*x,w*y,(d*z))
        """
        return einops.rearrange(X,self.einop_str,**self.einop_dict)

    def rearrange_inverse(self,X:torch.Tensor,**kwargs)->torch.Tensor:
        """Reverses the self.rearrange operation using the parameters inferred
        in self.get_einop_params.

        Args:
            X (torch.Tensor): a tensor of size (b,h*x,w*y,(d*z))
            kwargs: arguments that will be appended to self.einop_dict

        Returns:
            x torch.Tensor: a tensor of size (b,c,h,w,(d))
        """
        einop_dict = deepcopy(self.einop_dict)
        for k in kwargs:
            einop_dict[k] = kwargs[k]
        return einops.rearrange(X,self.einop_inv_str,**einop_dict)

    def rearrange_rescale(self,X:torch.Tensor,scale:int)->torch.Tensor:
        """Reverses the self.rearrange operation using the parameters inferred
        in self.get_einop_params but rescales the resolution so that the "extra"
        features are stacked on the channel dimension (for UNETR, etc.).

        Args:
            X (torch.Tensor): a tensor of size (b,h*x,w*y,(d*z))
            scale: factor by which resolution will be downsampled

        Returns:
            x torch.Tensor: a tensor of size (b,c,h,w,(d))
        """
        einop_dict = downsample_ein_op_dict(deepcopy(self.einop_dict),scale)
        return einops.rearrange(X,self.einop_inv_str,**einop_dict)
    
    def forward(self,X:torch.Tensor)->torch.Tensor:
        """Forward pass.

        Args:
            X (torch.Tensor): a tensor with shape 
                [-1,self.n_channels,*self.image_size]

        Returns:
            torch.Tensor: the embedding of X, with size 
                [X.shape[0],self.n_patches,self.n_features].
        """
        # output should always be [X.shape[0],self.n_patches,self.n_features]
        if self.embed_method == "convolutional":
            X = self.conv(X)
        return self.drop_op(self.rearrange(X) + self.positional_embedding)
    
class TransformerBlock(torch.nn.Module):
    """
    Transformer block. Built on top of a multi-head attention (MHA), it 
    can be summarised as:
    
                                |---------------|
    input -> MHA -> Add -> LayerNorm -> MLP -> Add -> LayerNorm -> Output
        |--------------|
        
    First introduced, to the best of my knowledge, in [1].
    
    [1] https://arxiv.org/abs/1706.03762
    """
    def __init__(self,
                 input_dim_primary:int,
                 attention_dim:int,
                 hidden_dim:int,
                 n_heads:int=4,
                 mlp_structure:List[int]=[128,128],
                 dropout_rate:float=0.0,
                 window_size:Size2dOr3d=None,
                 adn_fn:Callable=torch.nn.Identity):
        """
        Args:
            input_dim_primary (int): size of input.
            attention_dim (int): size of attention.
            hidden_dim (int): size of hidden dimension (output of attention
                modules).
            n_heads (int, optional): number of attention heads. Defaults to 4.
            mlp_structure (List[int], optional): hidden layer structure. 
                Should be a list of ints. Defaults to [32,32].
            dropout_rate (float, optional): dropout rate, applied to the output
                of each sub-layer. Defaults to 0.0.
            window_size (bool, optional): window_size for windowed W-MSA.
                Defaults to None (regular block of transformers).
            adn_fn (Callable, optional): function that returns a 
                torch.nn.Module that does activation/dropout/normalization,
                used in the MLP sub-layer. Should take as arguments the number
                of channels in a given layer. Defaults to torch.nn.Identity.
        """
        super().__init__()
        self.input_dim_primary = input_dim_primary
        self.attention_dim = attention_dim
        self.hidden_dim = hidden_dim
        self.n_heads = n_heads
        self.mlp_structure = mlp_structure
        self.dropout_rate = dropout_rate
        self.window_size = window_size
        self.adn_fn = adn_fn
        
        self.mha = MultiHeadSelfAttention(
            self.input_dim_primary,
            self.attention_dim,
            self.hidden_dim,
            self.input_dim_primary,
            window_size=self.window_size,
            n_heads=self.n_heads)
        
        self.init_drop_ops()
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
    
    def forward(self,X:torch.Tensor,mask=None)->torch.Tensor:
        """Forward pass.

        Args:
            X (torch.Tensor): tensor of shape [...,self.input_dim_primary]
            mask (torch.Tensor): attention masking tensor. Should have shape
                [].

        Returns:
            torch.Tensor: tensor of shape [...,self.input_dim_primary]
        """
        X = self.norm_op_1(X + self.drop_op_1(self.mha(X,mask=mask)))
        X = self.norm_op_2(X + self.drop_op_2(self.mlp(X)))
        return X

class TransformerBlockStack(torch.nn.Module):
    """Convenience function that stacks a series of transformer blocks.
    """
    def __init__(self,
                 number_of_blocks:int,
                 input_dim_primary:int,
                 attention_dim:int,
                 hidden_dim:int,
                 n_heads:int=4,
                 mlp_structure:List[int]=[128,128],
                 dropout_rate:float=0.0,
                 adn_fn:Callable=torch.nn.Identity,
                 window_size:Size2dOr3d=None,
                 post_transformer_act:Callable=F.gelu):
        """
        Args:
            number_of_blocks (int): number of blocks to be stacked.
            input_dim_primary (int): size of input.
            attention_dim (int): size of attention.
            hidden_dim (int): size of hidden dimension (output of attention
                modules).
            n_heads (int, optional): number of attention heads. Defaults to 4.
            mlp_structure (List[int], optional): hidden layer structure. 
                Should be a list of ints. Defaults to [32,32].
            dropout_rate (float, optional): dropout rate, applied to the output
                of each sub-layer. Defaults to 0.0.
            adn_fn (Callable, optional): function that returns a 
                torch.nn.Module that does activation/dropout/normalization,
                used in the MLP sub-layer. Should take as arguments the number
                of channels in a given layer. Defaults to torch.nn.Identity.
            window_size (bool, optional): window_size for windowed W-MSA.
                Defaults to None (regular block of transformers).
            post_transformer_act (Callable, optional): 
                activation applied to the output of each transformer.
        """
        super().__init__()
        self.number_of_blocks = number_of_blocks
        self.input_dim_primary = input_dim_primary
        self.attention_dim = attention_dim
        self.hidden_dim = hidden_dim
        self.n_heads = n_heads
        self.mlp_structure = mlp_structure
        self.dropout_rate = dropout_rate
        self.adn_fn = adn_fn
        self.window_size = window_size
        self.post_transformer_act = post_transformer_act

        self.init_transformer_blocks()
        
    def check_mlp_structure(self,
                            x:Union[List[List[int]],
                                    List[int]])->List[List[int]]:
        """Checks and corrects the MLP structure to see that everything is 
        as expected.

        Args:
            x (Union[List[List[int]], List[int]]): MLP structure or list of
                MLP structures.

        Returns:
            List[List[int]]: list of MLP structures.
        """
        if isinstance(x[0],list) == False:
            return [x for _ in range(self.number_of_blocks)]
        else:
            return x

    def convert_to_list_if_necessary(self,
                                     x:Union[List[int],int])->List[int]:
        """Checks and corrects inputs if necessary.

        Args:
            x (Union[List[int],int]): integer or list of integers

        Returns:
            List[int]: list of integers.
        """
        if isinstance(x,list) == False:
            return [x for _ in range(self.number_of_blocks)]
        else:
            return self.check_if_consistent(x)
    
    def check_if_consistent(self,x:Sequence):
        """Checks that the size of x is self.number_of_blocks

        Args:
            x (_type_): _description_
        """
        assert len(x) == self.number_of_blocks

    def init_transformer_blocks(self):
        """Initialises the transformer blocks.
        """
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
                TransformerBlock(i,a,h,n,m,self.dropout_rate,
                                 window_size=self.window_size,
                                 adn_fn=self.adn_fn))

    def forward(
        self,
        X:torch.Tensor,
        return_at:Union[str,List[int]]="end")->Tuple[torch.Tensor,TensorList]:
        """Forward pass.

        Args:
            X (torch.Tensor): tensor of shape [...,self.input_dim_primary]
            return_at (Union[str,List[int]], optional): sets the intermediary 
                outputs that will be returned together with the final output.

        Returns:
            torch.Tensor: tensor of shape [...,self.input_dim_primary]
            List[torch.Tensor]: list of intermediary tensors corresponding to
                the ith transformer outputs, where i is contained in return_at.
                Same shape as the final output.
        """
        if isinstance(return_at,list):
            assert max(return_at) < self.number_of_blocks,\
                "max(return_at) should be smaller than self.number_of_blocks"
        if return_at == "end" or return_at == None:
            return_at = []
        outputs = []
        for i,block in enumerate(self.transformer_blocks):
            X = self.post_transformer_act(block(X))
            if i in return_at:
                outputs.append(X)
        return X,outputs

class SWINTransformerBlock(torch.nn.Module):
    """Shifted window transformer module.
    """
    def __init__(self,
                 image_size:Size2dOr3d,
                 patch_size:Size2dOr3d,
                 window_size:Size2dOr3d,
                 n_channels:int,
                 attention_dim:int,
                 hidden_dim:int,
                 shift_size:int=0,
                 n_heads:int=4,
                 dropout_rate:float=0.0,
                 embed_method:str="linear",
                 mlp_structure:List[int]=[32,32],
                 adn_fn=torch.nn.Identity):
        """
        Args:
            image_size (Size2dOr3d): size of the input image.
            patch_size (Size2dOr3d): size of the patch size.
            window_size (Size2dOr3d, optional): window size for windowed
                multi-head attention. Defaults to None (no windowing)
            n_channels (int): number of channels in the input image.
            input_dim_primary (int): size of input.
            attention_dim (int): size of attention.
            hidden_dim (int): size of hidden dimension (output of attention
                modules).
            shift_size (int, optional): size of shift in patches (will be 
                multiplied by patch_size to get the actual patch_size).
                Defaults to 0 (no shift).
            dropout_rate (float, optional): dropout rate of the dropout 
                operations applied throughout this module. Defaults to 0.0.
            embed_method (str, optional): . Defaults to "linear".
            n_heads (int, optional): number of attention heads. Defaults to 4.
            mlp_structure (List[int], optional): hidden layer structure. 
                Should be a list of ints. Defaults to [32,32].
            adn_fn (Callable, optional): function that returns a 
                torch.nn.Module that does activation/dropout/normalization,
                used in the MLP sub-layer. Should take as arguments the number
                of channels in a given layer. Defaults to torch.nn.Identity.
        """
        super().__init__()
        self.image_size = image_size
        self.patch_size = patch_size
        self.window_size = window_size
        self.n_channels = n_channels
        self.attention_dim = attention_dim
        self.hidden_dim = hidden_dim
        self.shift_size = shift_size
        self.n_heads = n_heads
        self.dropout_rate = dropout_rate
        self.embed_method = embed_method
        self.mlp_structure = mlp_structure
        self.adn_fn = adn_fn
        
        self.embedding = LinearEmbedding(
            image_size=self.image_size,
            patch_size=self.patch_size,
            n_channels=self.n_channels,
            window_size=self.window_size,
            dropout_rate=self.dropout_rate
        )
        
        self.input_dim_primary = self.embedding.n_features
        
        self.tb = TransformerBlock(
            input_dim_primary=self.input_dim_primary,
            attention_dim=self.attention_dim,
            hidden_dim=self.hidden_dim,
            n_heads=self.n_heads,
            mlp_structure=self.mlp_structure,
            dropout_rate=self.dropout_rate,
            adn_fn=self.adn_fn,
            window_size=window_size,
        )
        
        # window_size here has to be given in *number of patches*
        self.attention_mask = generate_mask(
            image_size=[x//y for x,y in zip(image_size,patch_size)],
            window_size=[x//y for x,y in zip(window_size,patch_size)],
            shift_size=shift_size)
    
    def forward(self,
                X:torch.Tensor,
                scale:int=None)->Tuple[torch.Tensor,TensorList]:
        """Forward pass.

        Args:
            X (torch.Tensor): tensor of shape 
                [-1,self.n_channels,*self.image_size]
            scale (int): downsampling scale for output. Defaults to None
                (returns the non-rearranged output).

        Returns:
            torch.Tensor: tensor of shape [...,self.input_dim_primary]
            List[torch.Tensor]: list of intermediary tensors corresponding to
                the ith transformer outputs, where i is contained in return_at.
                Same shape as the final output.
        """
        if self.shift_size > 0:
            X = cyclic_shift_batch(X,[s * self.shift_size 
                                      for s in self.patch_size])
        embeded_X = self.embedding(X)
        output = self.tb(embeded_X,mask=self.attention_mask)
        output = self.embedding.rearrange_rescale(output,scale)
        if scale is not None:
            if self.shift_size > 0:
                output = cyclic_shift_batch(output,[-s * self.shift_size 
                                                    for s in self.patch_size])
        return output

class SWINTransformerBlockStack(torch.nn.Module):
    """Shifted window transformer module.
    """
    def __init__(self,
                 image_size:Size2dOr3d,
                 patch_size:Size2dOr3d,
                 window_size:Size2dOr3d,
                 shift_sizes:List[int],
                 n_channels:int,
                 attention_dim:int,
                 hidden_dim:int,
                 n_heads:int=4,
                 dropout_rate:float=0.0,
                 embed_method:str="linear",
                 mlp_structure:List[int]=[32,32],
                 adn_fn=torch.nn.Identity,
                 post_transformer_act:Callable=F.gelu):
        """
        Args:
            image_size (Size2dOr3d): size of the input image.
            patch_size (Size2dOr3d): size of the patch size.
            window_size (Size2dOr3d, optional): window size for windowed
                multi-head attention. Defaults to None (no windowing)
            shift_sizes (List[int], optional): list of shift sizes in 
                patches.
            n_channels (int): number of channels in the input image.
            input_dim_primary (int): size of input.
            attention_dim (int): size of attention.
            hidden_dim (int): size of hidden dimension (output of attention
                modules).
            dropout_rate (float, optional): dropout rate of the dropout 
                operations applied throughout this module. Defaults to 0.0.
            embed_method (str, optional): . Defaults to "linear".
            n_heads (int, optional): number of attention heads. Defaults to 4.
            mlp_structure (List[int], optional): hidden layer structure. 
                Should be a list of ints. Defaults to [32,32].
            adn_fn (Callable, optional): function that returns a 
                torch.nn.Module that does activation/dropout/normalization,
                used in the MLP sub-layer. Should take as arguments the number
                of channels in a given layer. Defaults to torch.nn.Identity.
            post_transformer_act (Callable, optional): 
                activation applied to the output of each transformer.
        """
        super().__init__()
        self.image_size = image_size
        self.patch_size = patch_size
        self.window_size = window_size
        self.shift_sizes = shift_sizes
        self.n_channels = n_channels
        self.attention_dim = attention_dim
        self.hidden_dim = hidden_dim
        self.n_heads = n_heads
        self.dropout_rate = dropout_rate
        self.embed_method = embed_method
        self.mlp_structure = mlp_structure
        self.adn_fn = adn_fn
        self.post_transformer_act = post_transformer_act
        
        self.init_swin_transformers()
                    
    def convert_to_list_of_lists_if_necessary(
        self,
        x:Union[List[List[int]],
                List[int]])->List[List[int]]:
        """Checks and corrects list structure to see that everything is 
        as expected.

        Args:
            x (Union[List[List[int]], List[int]]): list structure or list of
                lists structure.

        Returns:
            List[List[int]]: list of lists structures.
        """
        if isinstance(x[0],list) == False:
            return [x for _ in self.shift_sizes]
        else:
            return x

    def convert_to_list_if_necessary(self,
                                     x:Union[List[int],int])->List[int]:
        """Checks and corrects inputs if necessary.

        Args:
            x (Union[List[int],int]): integer or list of integers

        Returns:
            List[int]: list of integers.
        """
        if isinstance(x,list) == False:
            return [x for _ in self.shift_sizes]
        else:
            return self.check_if_consistent(x)
    
    def check_if_consistent(self,x:Sequence):
        """Checks that the size of x is self.number_of_blocks

        Args:
            x (_type_): _description_
        """
        assert len(x) == len(self.shift_sizes)

    def init_swin_transformers(self):
        attention_dim = self.convert_to_list_if_necessary(
            self.attention_dim)
        hidden_dim = self.convert_to_list_if_necessary(
            self.hidden_dim)
        n_heads = self.convert_to_list_if_necessary(
            self.n_heads)
        dropout_rate = self.convert_to_list_if_necessary(
            self.dropout_rate)
        mlp_structure = self.convert_to_list_of_lists_if_necessary(
            self.mlp_structure)
        self.stbs = torch.nn.ModuleList([])
        for ss,ad,hd,nh,dr,mlp_s in zip(self.shift_sizes,
                                        attention_dim,
                                        hidden_dim,
                                        n_heads,
                                        dropout_rate,
                                        mlp_structure):
            stb = SWINTransformerBlock(
                image_size=self.image_size,
                patch_size=self.patch_size,
                window_size=self.window_size,
                n_channels=self.n_channels,
                attention_dim=ad,
                hidden_dim=hd,
                shift_size=ss,
                n_heads=nh,
                dropout_rate=dr,
                embed_method=self.embed_method,
                mlp_structure=mlp_s,
                adn_fn=self.adn_fn
            )
            self.stbs.append(stb)

    def forward(self,
                X:torch.Tensor,
                scale:int=1)->Tuple[torch.Tensor,TensorList]:
        """Forward pass.

        Args:
            X (torch.Tensor): tensor of shape 
                [-1,self.n_channels,*self.image_size]

        Returns:
            torch.Tensor: tensor of shape [...,self.input_dim_primary]
            List[torch.Tensor]: list of intermediary tensors corresponding to
                the ith transformer outputs, where i is contained in return_at.
                Same shape as the final output.
        """
        for block in self.stbs[:-1]:
            X = self.post_transformer_act(block(X,scale=1))
        X = self.stbs[-1](X,scale=scale)
        return X

class ViT(torch.nn.Module):
    """Vision transformer module. Put more simply, it is the 
    concatenation of a LinearEmbedding and a TransformberBlockStack [1].
    
    [1] https://arxiv.org/abs/2010.11929
    """
    def __init__(self,
                 image_size:Size2dOr3d,
                 patch_size:Size2dOr3d,
                 n_channels:int,
                 number_of_blocks:int,
                 attention_dim:int,
                 hidden_dim:int,
                 window_size:Size2dOr3d=None,
                 n_heads:int=4,
                 dropout_rate:float=0.0,
                 embed_method:str="linear",
                 mlp_structure:List[int]=[32,32],
                 adn_fn=torch.nn.Identity,
                 post_transformer_act:Callable=F.gelu):
        """
        Args:
            image_size (Size2dOr3d): size of the input image.
            patch_size (Size2dOr3d): size of the patch size.
            n_channels (int): number of channels in the input image.
            number_of_blocks (int): number of blocks to be stacked.
            input_dim_primary (int): size of input.
            attention_dim (int): size of attention.
            hidden_dim (int): size of hidden dimension (output of attention
                modules).
            dropout_rate (float, optional): dropout rate of the dropout 
                operations applied throughout this module. Defaults to 0.0.
            embed_method (str, optional): . Defaults to "linear".
            window_size (Size2dOr3d, optional): window size for windowed MSA.
                Defaults to None (regular ViT).
            n_heads (int, optional): number of attention heads. Defaults to 4.
            mlp_structure (List[int], optional): hidden layer structure. 
                Should be a list of ints. Defaults to [32,32].
            adn_fn (Callable, optional): function that returns a 
                torch.nn.Module that does activation/dropout/normalization,
                used in the MLP sub-layer. Should take as arguments the number
                of channels in a given layer. Defaults to torch.nn.Identity.
            post_transformer_act (Callable, optional): 
                activation applied to the output of each transformer.
        """
        super().__init__()
        self.image_size = image_size
        self.patch_size = patch_size
        self.n_channels = n_channels
        self.number_of_blocks = number_of_blocks
        self.attention_dim = attention_dim
        self.hidden_dim = hidden_dim
        self.window_size = window_size
        self.n_heads = n_heads
        self.dropout_rate = dropout_rate
        self.embed_method = embed_method
        self.mlp_structure = mlp_structure
        self.adn_fn = adn_fn
        self.post_transformer_act = post_transformer_act
        
        self.embedding = LinearEmbedding(
            image_size=self.image_size,
            patch_size=self.patch_size,
            n_channels=self.n_channels,
            window_size=self.window_size,
            dropout_rate=self.dropout_rate
        )
        
        self.input_dim_primary = self.embedding.n_features
        
        self.tbs = TransformerBlockStack(
            number_of_blocks=self.number_of_blocks,
            input_dim_primary=self.input_dim_primary,
            attention_dim=self.attention_dim,
            hidden_dim=self.hidden_dim,
            n_heads=self.n_heads,
            mlp_structure=self.mlp_structure,
            dropout_rate=self.dropout_rate,
            adn_fn=self.adn_fn,
            window_size=window_size,
            post_transformer_act=self.post_transformer_act
        )
    
    def forward(
        self,
        X:torch.Tensor,
        return_at:Union[str,List[int]]="end")->Tuple[torch.Tensor,TensorList]:
        """Forward pass.

        Args:
            X (torch.Tensor): tensor of shape 
                [-1,self.n_channels,*self.image_size]
            return_at (Union[str,List[int]], optional): sets the intermediary 
                outputs that will be returned together with the final output.

        Returns:
            torch.Tensor: tensor of shape [...,self.input_dim_primary]
            List[torch.Tensor]: list of intermediary tensors corresponding to
                the ith transformer outputs, where i is contained in return_at.
                Same shape as the final output.
        """
        if isinstance(return_at,list):
            assert max(return_at) < self.number_of_blocks,\
                "max(return_at) should be smaller than self.number_of_blocks"
        embeded_X = self.embedding(X)
        if return_at == "end" or return_at == None:
            return_at = []
        outputs = []
        for i,block in enumerate(self.tbs.transformer_blocks):
            embeded_X = block(embeded_X)
            if i in return_at:
                outputs.append(embeded_X)
        return embeded_X,outputs
