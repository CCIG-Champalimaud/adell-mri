"""
Linear blocks and self-attention modules. For the attention modules, I have 
tried to keep these as close as possible to the algorithms presented in [1].

[1] https://arxiv.org/abs/2207.09238
"""

import torch
from typing import List

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
        self.qkv = MLP(self.input_dim,
                       self.qkv_dim)
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
    
class MultiHeadAttention(torch.nn.Module):
    """Module composed of several (self-)attention modules which calculate
    a set of (self-)attention outputs, concatenates them, and applies a 
    linear operation (matrix mul and addition of bias) on the output [1].
    
    [1] https://arxiv.org/abs/2207.09238
    """
    def __init__(self,
                 input_dim_primary:int,
                 attention_dim:int,
                 hidden_dim:int,
                 output_dim:int,
                 input_dim_context:int=None,
                 n_heads:int=4):
        """        
        Args:
            input_dim_primary (int): size of primary input.
            attention_dim (int): size of attention.
            hidden_dim (int): size of (self-)attention outputs.
            output_dim (int): size of last linear operation output.
            input_dim_context (int): size of context input (used to calculate
                the attention). Defaults to None (triggers self-attention).
            n_heads (int, optional): number of concurrent (self-)attention 
                heads. Defaults to 4.
        """
        super().__init__()
        self.input_dim_primary = input_dim_primary
        self.attention_dim = attention_dim
        self.hidden_dim = hidden_dim
        self.output_dim = output_dim
        self.input_dim_context = input_dim_context
        self.n_heads = n_heads

        self.init_attention_heads()
        self.init_output_layer()

    def init_attention_heads(self):
        """Initialises all attention heads.
        """
        self.heads = torch.nn.ModuleList([])
        for _ in range(self.n_heads):
            if self.input_dim_context is not None:
                head = Attention(
                    self.input_dim_primary,
                    self.input_dim_context,
                    self.attention_dim,
                    self.hidden_dim)
            else:
                head = SelfAttention(
                    self.input_dim_primary,
                    self.attention_dim,
                    self.hidden_dim)
            self.heads.append(head)
    
    def init_output_layer(self):
        """Initialises the last (linear) output layer.
        """
        self.output_layer = torch.nn.Linear(
            self.hidden_dim * self.n_heads,self.output_dim)
        
    def forward(self,
                X_primary:torch.Tensor,
                X_context:torch.Tensor=None)->torch.Tensor:
        """Forward pass. Expects the input to have two or more dimensions.

        Args:
            X_primary (torch.Tensor): tensor with shape 
                [...,self.input_dim_primary]
            X_context (torch.Tensor, optional): tensor with shape 
                [...,self.input_dim_context]. Defaults to None (only used if
                input_dim_context is specified in init)

        Returns:
            torch.Tensor: tensor with shape [...,self.output_dim]
        """
        outputs = []
        for head in self.heads:
            if self.input_dim_context is not None:
                outputs.append(head(X_primary,X_context))
            else:
                outputs.append(head(X_primary))
        output = torch.cat(outputs,-1)
        return self.output_layer(output)