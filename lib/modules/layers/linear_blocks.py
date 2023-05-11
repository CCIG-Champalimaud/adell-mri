"""
Linear blocks and self-attention modules. For the attention modules, I have 
tried to keep these as close as possible to the algorithms presented in [1].

[1] https://arxiv.org/abs/2207.09238
"""

import torch
import numpy as np
from typing import List

from ...custom_types import Size2dOr3d

def get_relative_position_indices(window_size:Size2dOr3d)->torch.Tensor:
    """Relative position indices generalized to n dimensions. The original
    version is in [1].
    
    [1] https://github.com/microsoft/Swin-Transformer/blob/main/models/swin_transformer.py

    Args:
        window_size (Size2dOr3d): size of window.

    Returns:
        torch.Tensor: indices used to index a relative position embedding 
            table.
    """
    n = len(window_size)
    coords = torch.stack(
        torch.meshgrid(
            [torch.arange(ws) for ws in window_size]))  # n, Wh, Ww
    coords_flatten = torch.flatten(coords, 1)  # n, Wh*Ww
    relative_coords = torch.subtract(
        coords_flatten[:, :, None],coords_flatten[:, None, :]) # n, Wh*Ww, Wh*Ww
    relative_coords = relative_coords.permute(1, 2, 0).contiguous()  # Wh*Ww, Wh*Ww, n
    for i in range(n):
        # shift to start from 0
        relative_coords[:,:,i] = relative_coords[:,:,i] + window_size[i]-1
        l = [2*w-1 for w in window_size[(i+1):]]
        if len(l) > 0:
            relative_coords[:,:,i] = relative_coords[:,:,i] * float(np.prod(l))
    relative_position_index = relative_coords.sum(-1)
    return relative_position_index

class MLP(torch.nn.Module):
    """Standard multilayer perceptron.
    """
    def __init__(self,
                 input_dim:int,
                 output_dim:int,
                 structure:List[int]=[],
                 adn_fn:torch.nn.Module=torch.nn.Identity):
        """
        Args:
            input_dim (int): input dimension.
            output_dim (int): output dimension.
            structure (List[int], optional): hidden layer structure. Should 
                be a list of ints. Defaults to [].
            adn_fn (torch.nn.Module, optional): function that returns a 
                torch.nn.Module that does activation/dropout/normalization.
                Should take as arguments the number of channels in a given
                layer. Defaults to torch.nn.Identity.
        """
        super().__init__()
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.structure = structure
        self.adn_fn = adn_fn

        self.init_layers()

    def init_layers(self):
        """Initialises layers.
        """
        curr_in = self.input_dim
        ops = torch.nn.ModuleList([])
        if len(self.structure) > 0:
            curr_out = self.structure[0]
            for i in range(1,len(self.structure)):
                ops.append(torch.nn.Linear(curr_in,curr_out))
                ops.append(self.adn_fn(curr_out))
                curr_in = curr_out
                curr_out = self.structure[i]
            ops.append(torch.nn.Linear(curr_in,curr_out))
        else:
            curr_out = curr_in
        ops.append(self.adn_fn(curr_out))
        ops.append(torch.nn.Linear(curr_out,self.output_dim))
        self.op = torch.nn.Sequential(*ops)
    
    def forward(self,X:torch.Tensor)->torch.Tensor:
        """Forward pass. Expects the input to have two or more dimensions.

        Args:
            X (torch.Tensor): tensor with shape [...,self.input_dim]

        Returns:
            torch.Tensor: tensor with shape [...,self.output_dim]
        """
        return self.op(X)
    
class Attention(torch.nn.Module):
    """Attention module [1].
    
    [1] https://arxiv.org/abs/2207.09238
    """
    def __init__(self,
                 input_dim_primary:int,
                 input_dim_context:int,
                 attention_dim:int,
                 output_dim:int):
        """        
        Args:
            input_dim_primary (int): size of primary input.
            input_dim_context (int): size of context input (used to calculate
                the attention).
            attention_dim (int): size of attention.
            output_dim (int): size of output.
        """
        super().__init__()
        self.input_dim_primary = input_dim_primary
        self.input_dim_context = input_dim_context
        self.attention_dim = attention_dim
        self.output_dim = output_dim
        
        self.init_layers()
    
    def init_layers(self):
        """Initialises layers.
        """
        self.q = MLP(self.input_dim_primary,self.attention_dim)
        self.k = MLP(self.input_dim_context,self.attention_dim)
        self.v = MLP(self.input_dim_context,self.output_dim)
        self.sm = torch.nn.Softmax(1)
        self.reg_const = torch.sqrt(torch.as_tensor(self.attention_dim))
    
    def forward(self,
                X_primary:torch.Tensor,
                X_context:torch.Tensor):
        """Forward pass. Expects the input to have two or more dimensions.

        Args:
            X_primary (torch.Tensor): tensor with shape 
                [...,self.input_dim_primary]
            X_context (torch.Tensor): tensor with shape 
                [...,self.input_dim_context]

        Returns:
            torch.Tensor: tensor with shape [...,self.output_dim]
        """
        Q = self.q(X_primary)
        K = self.k(X_context)
        V = self.v(X_context)
        S = Q @ torch.transpose(K,-1,-2)
        S = self.sm(S / self.reg_const)
        V_tilde = V * S
        return V_tilde

class SelfAttention(torch.nn.Module):
    """Self-attention module. Same as the attention module but the primary and
    context sequences are the same [1].
    
    [1] https://arxiv.org/abs/2207.09238
    """
    def __init__(self,
                 input_dim:int,
                 attention_dim:int,
                 output_dim:int):
        """
        Args:
            input_dim (int): size of input.
            attention_dim (int): size of attention.
            output_dim (int): size of output.
        """
        super().__init__()
        self.input_dim = input_dim
        self.attention_dim = attention_dim
        self.output_dim = output_dim
        
        self.init_layers()

    def init_layers(self):
        """Initialises layers.
        """
        self.qkv_dim = self.attention_dim*2+self.output_dim
        self.qkv = torch.nn.Linear(
            self.input_dim,self.qkv_dim,bias=False)
        self.q_idx = torch.arange(self.attention_dim).long()
        self.k_idx = torch.arange(self.attention_dim,
                                  self.attention_dim*2).long()
        self.v_idx = torch.arange(self.attention_dim*2,
                                  self.qkv_dim).long()
        self.sm = torch.nn.Softmax(1)
        self.reg_const = torch.sqrt(torch.as_tensor(self.attention_dim))

    def forward(self,X:torch.Tensor)->torch.Tensor:
        """Forward pass. Expects the input to have two or more dimensions.

        Args:
            X_primary (torch.Tensor): tensor with shape 
                [...,self.input_dim]

        Returns:
            torch.Tensor: tensor with shape [...,self.output_dim]
        """
        QKV = self.qkv(X)
        Q,K,V = (QKV[:,...,self.q_idx],
                 QKV[:,...,self.k_idx],
                 QKV[:,...,self.v_idx])
        S = Q @ torch.transpose(K,-1,-2)
        S = self.sm(S / self.reg_const)
        V_tilde = S @ V
        return V_tilde

class SeqPool(torch.nn.Module):
    def __init__(self,n_features):
        super().__init__()
        self.n_features = n_features
        self.g = torch.nn.Linear(self.n_features,1)
    
    def forward(self,X:torch.Tensor)->torch.Tensor:
        attn =  torch.softmax(self.g(X).swapaxes(1,2),-1)
        return attn @ X

class MultiHeadSelfAttention(torch.nn.Module):
    """Module composed of several self-attention modules which calculate
    a set of self-attention outputs, concatenates them, and applies a 
    linear operation (matrix mul and addition of bias) on the output [1].
    This implementation is greatly inspired by the SWIN implementation [2]. 
    Here we skip the bias term in the QKV projection and add a layer norm
    operation to the Q and K terms as suggested in [3] for increased training 
    stability.
    
    [1] https://arxiv.org/abs/2207.09238
    [2] https://github.com/microsoft/Swin-Transformer/blob/main/models/swin_transformer.py
    [3] https://arxiv.org/pdf/2302.05442.pdf
    """
    def __init__(self,
                 input_dim:int,
                 attention_dim:int,
                 hidden_dim:int,
                 output_dim:int,
                 n_heads:int=4,
                 dropout_rate:float=0.0,
                 window_size:bool=False):
        """        
        Args:
            input_dim (int): size of primary input.
            attention_dim (int): size of attention.
            hidden_dim (int): size of self-attention outputs.
            output_dim (int): size of last linear operation output.
            n_heads (int, optional): number of concurrent self-attention 
                heads. Defaults to 4.
            dropout_rate (float, optional): rate for the dropout applied to 
                the attention scores.
            window_size (bool, optional): window_size for windowed W-MSA.
                Defaults to None (regular MSA).
        """
        super().__init__()
        self.input_dim = input_dim
        self.attention_dim = attention_dim
        self.hidden_dim = hidden_dim
        self.output_dim = output_dim
        self.n_heads = n_heads
        self.dropout_rate = dropout_rate
        self.window_size = window_size
        
        assert (attention_dim % n_heads) == 0, \
            "attention_dim must be divisible by n_heads"
        assert (hidden_dim % n_heads) == 0, \
            "hidden_dim must be divisible by n_heads"

        self.init_layers()
        self.init_output_layer()
        self.init_weights()

    def init_layers(self):
        """Initialises all attention heads as a single set of Linear models,
        makes computation more efficient and is equivalent to initialising 
        multiple attention heads.
        """
        real_attention_dim = self.attention_dim // self.n_heads
        real_hidden_dim = self.hidden_dim // self.n_heads
        self.qkv_dim = self.attention_dim*2+self.hidden_dim
        self.qkv = torch.nn.Linear(
            self.input_dim,self.qkv_dim,bias=False)
        a,b,c = (real_attention_dim,
                 real_attention_dim*2,
                 real_attention_dim*2 + real_hidden_dim)
        self.q_idx = torch.arange(a).long()
        self.k_idx = torch.arange(a,b).long()
        self.v_idx = torch.arange(b,c).long()
        self.sm = torch.nn.Softmax(-1)
        self.reg_const = torch.sqrt(torch.as_tensor(
            self.attention_dim / self.n_heads))
        self.drop_op = torch.nn.Dropout(self.dropout_rate)
        self.q_norm = torch.nn.LayerNorm(real_attention_dim)
        self.k_norm = torch.nn.LayerNorm(real_attention_dim)
        if self.window_size:
            self.relative_position_bias_table = torch.nn.Parameter(
                torch.zeros(
                    int(np.prod([2*ws-1 for ws in self.window_size])),
                    self.n_heads))
            torch.nn.init.trunc_normal_(self.relative_position_bias_table)
            self.relative_position_index = get_relative_position_indices(
                self.window_size)
    
    def init_output_layer(self):
        """Initialises the last (linear) output layer.
        """
        self.output_layer = torch.nn.Linear(
            self.hidden_dim,self.output_dim)
    
    def init_weights(self):
        """Initialize weights with Xavier uniform (got this from the original
        transformer code and from the Annotated Transformer).
        """
        torch.nn.init.xavier_uniform_(self.qkv.weight)
        torch.nn.init.xavier_uniform_(self.output_layer.weight)

    def forward(self,X:torch.Tensor,mask=None)->torch.Tensor:
        """Forward pass. Expects the input to have two or more dimensions.

        Args:
            X (torch.Tensor): tensor with shape 
                [...,self.input_dim]
            mask (torch.Tensor): attention masking tensor.

        Returns:
            torch.Tensor: tensor with shape [...,self.output_dim]
        """
        sh = X.shape
        b,t,f = sh[:-2],sh[-2],sh[-1]
        QKV = self.qkv(X)
        permute_dims = [*[i for i in range(len(b))],
                        len(b)+1,len(b),len(b)+2]
        QKV = QKV.reshape(*b,t,self.n_heads,
                          self.qkv_dim // self.n_heads).permute(*permute_dims)
        Q,K,V = QKV[...,self.q_idx],QKV[...,self.k_idx],QKV[...,self.v_idx]
        Q = self.q_norm(Q)
        K = self.k_norm(K)
        S = Q @ torch.transpose(K,-1,-2)
        S = S / self.reg_const
        if self.window_size:
            relative_position_bias = self.relative_position_bias_table[
                self.relative_position_index.clone()[:t,:t].reshape(-1)
            ].reshape(-1,t,t)
            S = S + relative_position_bias
        if mask is not None:
            S = S + mask.unsqueeze(1).unsqueeze(0)
        S = self.drop_op(self.sm(S))
        V_tilde = S @ V
        V_tilde = V_tilde.transpose(1,2).reshape(*b,t,self.hidden_dim)
        output = self.output_layer(V_tilde)
        return output
