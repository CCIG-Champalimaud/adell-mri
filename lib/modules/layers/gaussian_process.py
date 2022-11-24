import torch
from .utils import unsqueeze_to_target

class GaussianProcessLayer(torch.nn.Module):
    def __init__(self,in_channels:int,out_channels:int,m:float=0.999):
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
            torch.normal(torch.zeros([1,o]),torch.ones([1,o])).float(),
            requires_grad=True)
        self.inv_conv = torch.nn.Parameter(
            torch.eye(o,o).unsqueeze(0).float(),
            requires_grad=False)
    
    def update_inv_cov(self,X,y):
        y = y.unsqueeze(-1)
        phi = self.calculate_phi(X)
        phi,phi_t = phi.unsqueeze(-2),phi.unsqueeze(-1)
        K = torch.matmul(phi_t,phi)
        if len(K.shape) > 3:
            K = K.flatten(start_dim=1,end_dim=-3)
            K = K.mean(1)
        update_term = y*(1-y) * K
        self.inv_conv.data = torch.add(
            self.inv_conv * self.m,
            (1-self.m) * update_term.sum(0))

    def get_cov(self):
        self.cov = torch.linalg.inv(self.inv_conv)

    def calculate_phi(self,X):
        X = X.swapaxes(1,-1)
        X = X.unsqueeze(-1)
        W = unsqueeze_to_target(self.W,X,1)
        mm = torch.matmul(-W,X).squeeze(-1)
        return self.scaling_term*torch.cos(mm+self.b)
    
    def forward(self,X):
        phi = self.calculate_phi(X)
        output = phi * self.weights
        if len(output.shape) > 2:
            output = output.swapaxes(1,-1)
        return output
    