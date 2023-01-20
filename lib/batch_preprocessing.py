"""Contains methods that are to be applied to whole batches.
"""

import torch
import numpy as np

def label_smoothing(y,smooth_factor):
    return torch.where(y < 0.5,y + smooth_factor,y - smooth_factor)

def mixup(x,y,mixup_alpha,g=None):
    batch_size = y.shape[0]
    if g is None:
        g = np.random.default_rng()
    mixup_factor = torch.as_tensor(
        g.beta(mixup_alpha,mixup_alpha,batch_size),dtype=x.dtype,
        device=x.device)
    mixup_factor_x = mixup_factor.reshape(
        [-1]+[1 for _ in range(1,len(x.shape))])
    mixup_perm = g.permutation(batch_size)
    x = x * mixup_factor_x + x[mixup_perm] * (1.0-mixup_factor_x)
    y = y * mixup_factor + y[mixup_perm] * (1.0-mixup_factor)
    return x,y

def partial_mixup(x,y,mixup_alpha,mixup_fraction=0.5,g=None):
    batch_size = y.shape[0]
    if g is None:
        g = np.random.default_rng()
    mxu_i = g.binomial(1,mixup_fraction,batch_size).astype(bool)
    mixup_factor = torch.as_tensor(
        g.beta(mixup_alpha,mixup_alpha,mxu_i.sum()),dtype=x.dtype,
        device=x.device)
    mixup_factor_x = mixup_factor.reshape(
        [-1]+[1 for _ in range(1,len(x.shape))])
    mixup_perm = g.permutation(batch_size)
    x[mxu_i] = torch.add(
        x[mxu_i] * mixup_factor_x,
        x[mixup_perm][mxu_i] * (1-mixup_factor_x))
    y[mxu_i] = torch.add(
        y[mxu_i] * mixup_factor,
        y[mixup_perm][mxu_i] * (1-mixup_factor))
    return x,y
    
class BatchPreprocessing:
    def __init__(self,
                 label_smoothing:float=None,
                 mixup_alpha:float=None,
                 partial_mixup:float=None,
                 seed:int=42):
        self.label_smoothing = label_smoothing
        self.mixup_alpha = mixup_alpha
        self.partial_mixup = partial_mixup
        self.seed = seed
        
        if self.mixup_alpha is not None:
            self.g = np.random.default_rng(seed)
    
    def __call__(self,X,y):
        if self.label_smoothing is not None:
            y = label_smoothing(y,self.label_smoothing)
        if self.mixup_alpha is not None:
            initial_y_dtype = y.dtype
            y = y.float()
            if self.partial_mixup is not None:
                X,y = partial_mixup(X,y,self.mixup_alpha,self.partial_mixup,
                                    self.g)
            else:
                X,y = mixup(X,y,self.mixup_alpha,self.g)
            y = y.to(initial_y_dtype)
        return X,y