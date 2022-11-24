import torch
from typing import List

class MLP(torch.nn.Module):
    def __init__(self,
                 input_dim:int,
                 output_dim:int,
                 structure:List[int]=[],
                 adn_fn:torch.nn.Module=torch.nn.Identity):
        super().__init__()
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.structure = structure
        self.adn_fn = adn_fn

        self.init_layers()

    def init_layers(self):
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
    
    def forward(self,X):
        return self.op(X)
    
class Attention(torch.nn.Module):
    def __init__(self,
                 input_dim_primary:int,
                 input_dim_context:int,
                 attention_dim:int,
                 output_dim:int):
        super().__init__()
        self.input_dim_primary = input_dim_primary
        self.input_dim_context = input_dim_context
        self.attention_dim = attention_dim
        self.output_dim = output_dim
        
        self.init_layers()
    
    def init_layers(self):
        self.q = MLP(self.input_dim_primary,self.attention_dim)
        self.k = MLP(self.input_dim_context,self.attention_dim)
        self.v = MLP(self.input_dim_context,self.output_dim)
        self.sm = torch.nn.Softmax(1)
        self.reg_const = torch.sqrt(torch.as_tensor(self.attention_dim))
    
    def forward(self,
                X_primary:torch.Tensor,
                X_context:torch.Tensor):
        Q = self.q(X_primary)
        K = self.k(X_context)
        V = self.v(X_context)
        S = Q @ torch.transpose(K,-1,-2)
        S = self.sm(S / self.reg_const)
        V_tilde = V * S
        return V_tilde
    
class SelfAttention(Attention):
    def __init__(self,
                 input_dim:int,
                 attention_dim:int,
                 output_dim:int):
        super().__init__(
            input_dim,
            input_dim,
            attention_dim,
            output_dim
        )
        self.input_dim = input_dim
        
        self.init_layers()
        
    def forward(self,
                X_primary:torch.Tensor):
        Q = self.q(X_primary)
        K = self.k(X_primary)
        V = self.v(X_primary)
        S = Q @ torch.transpose(K,-1,-2)
        S = self.sm(S / self.reg_const)
        V_tilde = S @ V
        return V_tilde
    
class MultiHeadAttention(torch.nn.Module):
    def __init__(self,
                 input_dim_primary:int,
                 attention_dim:int,
                 hidden_dim:int,
                 output_dim:int,
                 input_dim_context:int=None,
                 n_heads=4):
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
        self.output_layer = torch.nn.Linear(
            self.hidden_dim * self.n_heads,self.output_dim)

    def forward(self,X_primary:torch.Tensor,X_context:torch.Tensor=None):
        outputs = []
        for head in self.heads:
            if X_context is not None:
                outputs.append(head(X_primary,X_context))
            else:
                outputs.append(head(X_primary))
        output = torch.cat(outputs,-1)
        return self.output_layer(output)