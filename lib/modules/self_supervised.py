import torch
import torch.functional as F
from collections import OrderedDict
from copy import deepcopy

from ..types import *

def cos_sim(x:torch.Tensor,y:torch.Tensor)->torch.Tensor:
    x,y = x.flatten(start_dim=1),y.flatten(start_dim=1)
    x,y = x.unsqueeze(1),y.unsqueeze(0)
    n = torch.sum(x*y,axis=-1)
    d = torch.multiply(torch.norm(x,2,-1),torch.norm(y,2,-1))
    return n/d

def standardize(x:torch.Tensor)->torch.Tensor:
    return torch.divide(
        x - torch.mean(x,0,keepdim=True),
        torch.std(x,0,keepdim=True))

def pearson_corr(x:torch.Tensor,y:torch.Tensor)->torch.Tensor:
    x,y = x.flatten(start_dim=1),y.flatten(start_dim=1)
    x,y = standardize(x),standardize(y)
    x,y = x.unsqueeze(1),y.unsqueeze(0)
    n = torch.sum(x*y,axis=-1)
    d = torch.multiply(torch.norm(x,2,-1),torch.norm(y,2,-1))
    return n/d

def cos_dist(x:torch.Tensor,y:torch.Tensor,center)->torch.Tensor:
    return 1 - cos_sim(x,y,center)

def barlow_twins_loss(x:torch.Tensor,y:torch.Tensor,l:float=0.02)->torch.Tensor:
    diag_idx = torch.arange(0,x.shape)
    n = x.shape[0]
    C = pearson_corr(x,y)
    inv_term = torch.diagonal(1 - C)[diag_idx,diag_idx]
    red_term = torch.square(C)
    red_term[diag_idx,diag_idx] = 0
    loss = torch.add(
        inv_term.sum()/x.shape[0],
        red_term.sum()/(n*(n-1))*l)
    return loss

def simsiam_loss(x1:torch.Tensor,x2:torch.Tensor)->torch.Tensor:
    x1 = x1/F.norm(x1,2,-1).unsqueeze(1)
    x2 = x2/F.norm(x2,2,-1).unsqueeze(1)
    return -torch.sum(x1*x2,1).mean()

def byol_loss(x1:torch.Tensor,x2:torch.Tensor)->torch.Tensor:
    return 2*simsiam_loss(x1,x2)+2

class ExponentialMovingAverage(torch.nn.Module):
    def __init__(self,decay:float,final_decay:float=None,n_steps=None):
        """Exponential moving average for model weights. The weight-averaged
        model is kept as `self.shadow` and each iteration of self.update leads
        to weight updating. This implementation is heavily based on that 
        available in https://www.zijianhu.com/post/pytorch/ema/.

        Essentially, self.update(model) is called, a shadow version of the 
        model (i.e. self.shadow) is updated using the exponential moving 
        average formula such that $v'=(1-decay)*(v_{shadow}-v)$, where
        $v$ is the new parameter value, $v'$ is the updated value and 
        $v_{shadow}$ is the exponential moving average value (i.e. the shadow).

        Args:
            decay (float): decay for the exponential moving average.
            final_decay (float, optional): final value for decay. Defaults to
                None (same as initial decay).
            n_steps (float, optional): number of updates until `decay` becomes
                `final_decay` with linear scaling. Defaults to None.
        """
        super().__init__()
        self.decay = decay
        self.final_decay = final_decay
        self.n_steps = n_steps
        self.shadow = None
        self.step = 0

        if self.final_decay is None:
            self.final_decay = self.decay
        self.slope = (self.final_decay - self.decay)/self.n_steps
        self.intercept = self.decay

    def set_requires_grad_false(self,model):
        for k,p in model.named_parameters():
            if p.requires_grad == True:
                p.requires_grad = False

    @torch.no_grad()
    def update(self,model:torch.nn.Module):
        if self.shadow is None:
            # this effectively skips the first epoch
            self.shadow = deepcopy(model)
            self.shadow.training = False
            self.set_requires_grad_false(self.shadow)
        else:
            model_params = OrderedDict(model.named_parameters())
            shadow_params = OrderedDict(self.shadow.named_parameters())

            shadow_params_keys = list(shadow_params.keys())

            different_params = set.difference(
                set(shadow_params_keys),
                set(model_params.keys()))
            assert len(different_params) == 0

            for name, param in model_params.items():
                if 'shadow' not in name:
                    shadow_params[name].sub_(
                       (1.-self.decay) * (shadow_params[name]-param))
            
            self.decay = self.step*self.slope + self.intercept
            self.step += 1

class ContrastiveDistanceLoss(torch.nn.Module):
    def __init__(self,dist_p=2,random_sample=False,margin=1,
                 dev="cpu",loss_type="pairwise",dist_type="euclidean"):
        super().__init__()
        self.dist_p = dist_p
        self.random_sample = random_sample
        self.margin = margin
        self.dev = dev
        self.loss_type = loss_type
        self.dist_type = dist_type
        
        self.loss_options = ["pairwise","triplet"]
        self.dist_options = ["euclidean","cosine"]
        self.torch_margin = torch.as_tensor(
            [self.margin],dtype=torch.float32,device=self.dev)

        if self.loss_type not in self.loss_options:
            raise Exception("Loss `{}` not in `{}`".format(
                self.loss_type,self.loss_options))
        
        if self.dist_type not in self.dist_options:
            raise Exception("dist_type `{}` not in `{}`".format(
                self.loss_type,self.dist_options))

    def dist(self,x:torch.Tensor,y:torch.Tensor):
        if self.dist_type == "euclidean":
            return torch.cdist(x,y,self.dist_p)
        elif self.dist_type == "cosine":
            return cos_dist(x,y)

    def pairwise_distance(self,X1,X2,is_same):
        X1 = X1.flatten(start_dim=1)
        X2 = X2.flatten(start_dim=1)
        dist = self.dist(X1,X2)
        dist = torch.add(
            is_same*dist,
            (1-is_same.float())*torch.maximum(
                torch.zeros_like(dist),
                self.torch_margin - dist))
        if self.random_sample == True:
            # randomly samples one entry for each element
            n = dist.shape[0]
            x_idx = torch.arange(0,n,1,dtype=torch.int32)
            y_idx = torch.randint(0,n,size=[n])
            dist = dist[x_idx,y_idx]
        else:
            dist = dist.sum(-1)/(dist.shape[-1]-1)
        return dist
    
    def triplet_distance(self,X1,X2,is_same):
        X1 = X1.flatten(start_dim=1)
        X2 = X2.flatten(start_dim=1)
        dist = self.dist(X1,X2)
        # retrieve negative examples with the lowest distance to 
        # each anchor
        hard_negatives = torch.where(
            is_same,
            torch.ones_like(dist)*torch.inf,
            dist).min(1).values
        # retrieve positive examples with the highest distance to
        # each anchor
        hard_positives = torch.where(
            torch.logical_not(is_same),
            -torch.ones_like(dist)*torch.inf,
            dist).max(1).values
        # calculates loss given both hard negatives and positives
        triplet_loss = torch.maximum(
            torch.zeros_like(hard_negatives),
            self.margin + hard_positives - hard_negatives)
        return triplet_loss

    def forward(self,X:torch.Tensor,y:torch.Tensor):
        if isinstance(X,list):
            X1,X2 = X
        else:
            X1,X2 = X,X
        y1,y2 = y.unsqueeze(0),y.unsqueeze(1)
        is_same = y1 == y2
        if self.loss_type == "pairwise":
            loss = self.pairwise_distance(X1,X2,is_same)
        elif self.loss_type == "triplet":
            loss = self.triplet_distance(X1,X2,is_same)
        return loss.mean()

class BarlowTwinsLoss(torch.nn.Module):
    def __init__(self,moving:bool=False,lam=0.2):
        super().__init__()
        self.moving = moving
        self.lam = lam
        
        self.count = 0.
        self.sum = None
        self.sum_of_squares = None
        self.average = None
        self.std = None

    def standardize(self,x:torch.Tensor)->torch.Tensor:
        if self.moving == False and self.sum is None:
            o = torch.divide(
                x - torch.mean(x,0,keepdim=True),
                torch.std(x,0,keepdim=True))
        else:
            o = torch.divide(x-self.average,self.std)
        return o

    def pearson_corr(self,x:torch.Tensor,y:torch.Tensor)->torch.Tensor:
        x,y = x.flatten(start_dim=1),y.flatten(start_dim=1)
        x,y = self.standardize(x),self.standardize(y)
        x,y = x.unsqueeze(1),y.unsqueeze(0)
        n = torch.sum(x*y,axis=-1)
        d = torch.multiply(torch.norm(x,2,-1),torch.norm(y,2,-1))
        return n/d

    def calculate_loss(self,x,y,update=True):
        if update == True:
            n = x.shape[0]
            f = x.shape[1]
            if self.sum is None:
                self.sum = torch.zeros([1,f],device=x.device)
                self.sum_of_squares = torch.zeros([1,f],device=x.device)
            self.sum = torch.add(
                self.sum,
                torch.sum(x+y,0,keepdim=True))
            self.sum_of_squares = torch.add(
                self.sum_of_squares,
                torch.sum(torch.square(x)+torch.square(y),0,keepdim=True))
            self.count += 2*n
        return self.barlow_twins_loss(x,y)

    def barlow_twins_loss(self,x:torch.Tensor,y:torch.Tensor)->torch.Tensor:
        diag_idx = torch.arange(0,x.shape[0])
        n = x.shape[0]
        C = self.pearson_corr(x,y)
        inv_term = torch.diagonal(1 - C)
        red_term = torch.square(C)
        red_term[diag_idx,diag_idx] = 0
        loss = torch.add(inv_term.sum()/n,red_term.sum()/n*self.lam)
        return loss

    def calculate_average_std(self):
        self.average = self.sum / self.count
        self.std = self.sum_of_squares - torch.square(self.sum)/self.count
    
    def reset(self):
        self.count = 0.
        self.sum[()] = 0
        self.sum_of_squares[()] = 0

    def forward(self,X1:torch.Tensor,X2:torch.Tensor,update:bool=True):
        loss = self.calculate_loss(X1,X2,update)
        return loss.sum()
