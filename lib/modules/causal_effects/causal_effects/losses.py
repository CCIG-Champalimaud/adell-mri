"""
Implementations of different loss functions for segmentation tasks.
"""

import torch
import torch.nn.functional as F
from typing import Union,Tuple
from itertools import combinations

eps = 1e-6

def pt(pred:torch.Tensor,target:torch.Tensor,
       threshold:float=0.5)->torch.Tensor:
    """Convenience function to convert probabilities of predicting
    the positive class to probability of predicting the corresponding
    target.

    Args:
        pred (torch.Tensor): prediction probabilities.
        target (torch.Tensor): target class.
        threshold (float, optional): threshold for the positive class in the
        focal loss. Helpful in cases where one is trying to model the 
            probability explictly. Defaults to 0.5.

    Returns:
        torch.Tensor: prediction of element i in `pred` predicting class
        i in `target.`
    """
    return torch.where(target > threshold,pred,1-pred)

def binary_cross_entropy(pred:torch.Tensor,
                         target:torch.Tensor,
                         weight:float=1.,
                         scale:float=1.,
                         eps:float=eps)->torch.Tensor:
    """Standard implementation of binary cross entropy.

    Args:
        pred (torch.Tensor): prediction probabilities.
        target (torch.Tensor): target class.
        weight (float, optional): weight for the positive
            class. Defaults to 1.
        scale (float, optional): factor to scale loss before reducing.
        eps (float, optional): epsilon factor to avoid floating-point
            imprecisions.

    Returns:
        torch.Tensor: a tensor with size equal to the batch size (first 
        dimension of `pred`).
    """
    pred = torch.flatten(pred,start_dim=1)
    target = torch.flatten(target,start_dim=1)
    a = weight*target*torch.log(pred+eps)
    b = (1-target)*torch.log(1-pred+eps)
    return -torch.mean((a+b)*scale,dim=1)

    
def mean_squared_error(pred:torch.Tensor,
                       target:torch.Tensor)->torch.Tensor:
    """Standard implementation of mean squared error.

    Args:
        pred (torch.Tensor): prediction probabilities.
        target (torch.Tensor): target class.

    Returns:
        torch.Tensor: a tensor with size equal to the batch size (first 
        dimension of `pred`).
    """
    pred = torch.flatten(pred,start_dim=1)
    target = torch.flatten(target,start_dim=1)
    return torch.mean((target-pred)**2,dim=1)

def root_mean_squared_error(pred:torch.Tensor,
                            target:torch.Tensor)->torch.Tensor:
    """Standard implementation of root mean squared error.

    Args:
        pred (torch.Tensor): prediction probabilities.
        target (torch.Tensor): target class.

    Returns:
        torch.Tensor: a tensor with size equal to the batch size (first 
        dimension of `pred`).
    """
    pred = torch.flatten(pred,start_dim=1)
    target = torch.flatten(target,start_dim=1)
    return torch.sqrt(torch.mean(torch.abs(target-pred)**2,dim=1))

# TODO
def decorrelation_loss(pred:torch.Tensor,
                       target_ce:torch.Tensor,
                       target_ae:torch.Tensor,
                       beta:float=1,gamma:float=1,
                       weight:float=1.,scale:float=1.,
                       eps:float=eps)->torch.Tensor:
    """Implementation fo the decorrelation loss from https://arxiv.org/abs/2008.09858

    Args:
        pred (torch.Tensor): prediction probabilities.
        target (torch.Tensor): target class.
        weight (float, optional): weight for the positive
            class. Defaults to 1.
        scale (float, optional): factor to scale loss before reducing.
        eps (float, optional): epsilon factor to avoid floating-point
            imprecisions.

    Returns:
        torch.Tensor: a tensor with size equal to the batch size (first 
        dimension of `pred`).
    """
    pred = torch.flatten(pred,start_dim=1)
    target_ce = torch.flatten(target_ce,start_dim=1)
    target_ae = torch.flatten(target_ae,start_dim=1)
    ce_loss = binary_cross_entropy(pred, target_ce, weight, scale, eps)
    ae_loss = mean_squared_error(pred, target_ae)
    reg = 0
    return ce_loss + beta*ae_loss + gamma*reg