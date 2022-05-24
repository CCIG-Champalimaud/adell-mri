"""
Implementations of different loss functions for segmentation tasks.
"""

import torch
from typing import Union

EPS = 1e-8
FOCAL_DEFAULT = {"alpha":None,"gamma":1}
TVERSKY_DEFAULT = {"alpha":1,"beta":1,"gamma":1}

def pt(pred:torch.Tensor,target:torch.Tensor)->torch.Tensor:
    """Convenience function to convert probabilities of predicting
    the positive class to probability of predicting the corresponding
    target.

    Args:
        pred (torch.Tensor): prediction probabilities.
        target (torch.Tensor): target class.

    Returns:
        torch.Tensor: prediction of element i in `pred` predicting class
        i in `target.`
    """
    return torch.where(target == 1,pred,1-pred)

def binary_cross_entropy(pred:torch.Tensor,
                         target:torch.Tensor,
                         weight:float=1.)->torch.Tensor:
    """Standard implementation of binary cross entropy.

    Args:
        pred (torch.Tensor): prediction probabilities.
        target (torch.Tensor): target class.
        weight (float, optional): weight for the positive
        class. Defaults to 1.

    Returns:
        torch.Tensor: a tensor with size equal to the batch size (first 
        dimension of `pred`).
    """
    pred = torch.flatten(pred,start_dim=1)
    target = torch.flatten(target,start_dim=1)
    a = -weight*target*torch.log(pred+EPS)
    b = -(1-weight)*(1-target)*torch.log(1-pred+EPS)
    return torch.mean(a+b,dim=1)

def binary_focal_loss(pred:torch.Tensor,
                      target:torch.Tensor,
                      alpha:float,
                      gamma:float)->torch.Tensor: 
    """Binary focal loss. Uses `alpha` to weight the positive class and 
    `lambda` to suppress examples which are easy to classify (given that 
    `lambda`>1). `lambda` is also known as the focusing parameter. In essence,
    `focal_loss = (1-pt(pred,target))**gamma*bce`, where `bce` is the binary
    cross entropy [1].

    [1] https://arxiv.org/abs/1708.02002

    Args:
        pred (torch.Tensor): prediction probabilities.
        target (torch.Tensor): target class.
        alpha (float): positive class weight.
        gamma (float): focusing parameter.

    Returns:
         torch.Tensor: a tensor with size equal to the batch size (first 
        dimension of `pred`).
    """
    p = pt(pred,target)
    p = torch.flatten(p,start_dim=1)
    bce = -torch.log(p+EPS)
    
    return torch.mean(alpha*((1-p+EPS)**gamma)*bce,dim=-1)
    
def generalized_dice_loss(pred:torch.Tensor,
                          target:torch.Tensor,
                          weight:float=1)->torch.Tensor:
    """Dice loss adapted to cases of very high class imbalance. In essence
    it adds class weights to the calculation of the Dice loss [1]. If 
    `weights=1` it defaults to the regular Dice loss. This implementation 
    works for both the binary and categorical cases.

    [1] https://arxiv.org/pdf/1707.03237.pdf

    Args:
        pred (torch.Tensor): prediction probabilities.
        target (torch.Tensor): target class.
        weight (float, optional): class weights. Defaults 
        to 1.

    Returns:
         torch.Tensor: a tensor with size equal to the batch size (first 
        dimension of `pred`).
    """
    if pred.shape != target.shape:
        target = classes_to_one_hot(target)
    I = torch.flatten(pred*target,start_dim=2)
    U = torch.flatten(pred+target,start_dim=2)
    if isinstance(weight,torch.Tensor) == True:
        weight = unsqueeze_to_shape(weight,I.shape,1)
    I = torch.sum(I*weight,dim=-1).sum(-1)
    U = torch.sum(U*weight,dim=-1).sum(-1)
    return 1-2*I/U

def binary_focal_tversky_loss(pred:torch.Tensor,
                              target:torch.Tensor,
                              alpha:float,
                              beta:float,
                              gamma:float=1)->torch.Tensor:
    """Binary focal Tversky loss. Very similar to the original Tversky loss
    but features a `gamma` term similar to the focal loss to focus the 
    learning on harder/rarer examples [1]. The Tversky loss itself is very 
    similar to the Dice score loss but weighs false positives and false 
    negatives differently (set as the `alpha` and `beta` hyperparameter,
    respectively).

    [1] https://arxiv.org/abs/1810.07842

    Args:
        pred (torch.Tensor): prediction probabilities.
        target (torch.Tensor): target class.
        alpha (float): weight for false positives.
        beta (float): weight for false negatives.
        gamma (float): focusing parameter.

    Returns:
         torch.Tensor: a tensor with size equal to the batch size (first 
        dimension of `pred`).
    """
    p_fore = torch.flatten(pred,start_dim=1)
    p_back = 1-p_fore
    t_fore = torch.flatten(target,start_dim=1)
    t_back = 1-t_fore
    n = torch.sum(p_fore*t_fore,dim=1) + 1
    d_1 = alpha*torch.sum(p_fore*t_back,dim=1)
    d_2 = beta*torch.sum(p_back*t_fore,dim=1)
    d = n + d_1 + d_2 + 1
    return 1-(n/d)**gamma

def combo_loss(pred:torch.Tensor,
               target:torch.Tensor,
               alpha:float=0.5,
               beta:float=1)->torch.Tensor:
    """Combo loss. Simply put, it is a weighted combination of the Dice loss
    and the weighted cross entropy [1]. `alpha` is the weight for each loss 
    and beta is the weight for the binary cross entropy.

    [1] https://arxiv.org/abs/1805.02798

    Args:
        pred (torch.Tensor): prediction probabilities.
        target (torch.Tensor): target class.
        alpha (float, optional): weight term for both losses. Binary cross 
        entropy is scaled by `alpha` and the Dice score is scaled by 
        `1-alpha`. Defaults to 0.5.
        beta (float, optional): weight for the cross entropy. Defaults to 1.

    Returns:
         torch.Tensor: a tensor with size equal to the batch size (first 
        dimension of `pred`).
    """
    bdl = generalized_dice_loss(pred,target)
    bce = binary_cross_entropy(pred,target,beta)
    return (alpha)*bce + (1-alpha)*bdl

def hybrid_focal_loss(pred:torch.Tensor,
                      target:torch.Tensor,
                      lam:float=0.5,
                      focal_params:dict={},
                      tversky_params:dict={})->torch.Tensor:
    """Hybrid focal loss. A combination of the focal loss and the focal 
    Tversky loss. In essence, a weighted sum of both, where `lam` defines
    how both losses are combined.

    Args:
        pred (torch.Tensor): prediction probabilities.
        target (torch.Tensor): target class.
        lam (float, optional): weight term for both losses. Focal loss is 
        scaled by `lam` and the Tversky focal loss is scaled by `1-lam`.
        Defaults to 0.5.
        focal_params (dict, optional): dictionary with parameters for the 
        focal loss. Defaults to {}.
        tversky_params (dict, optional): dictionary with parameters for the
        Tversky focal loss. Defaults to {}.

    Returns:
        torch.Tensor: a tensor with size equal to the batch size (first 
        dimension of `pred`).
    """
    a = focal_params["alpha"]
    if a is None or isinstance(a,int)==True or isinstance(a,float)==True:
        focal_params["alpha"] = torch.ones([1])
    bfl = binary_focal_loss(pred,target,**focal_params)
    bftl = binary_focal_tversky_loss(pred,target,**tversky_params)
    return lam*bfl + (1-lam)*bftl    

def unified_focal_loss(pred:torch.Tensor,
                       target:torch.Tensor,
                       delta:float,
                       gamma:float,
                       lam:float=0.5)->torch.Tensor:
    """Unified focal loss. A combination of the focal loss and the focal 
    Tversky loss. In essence, a weighted sum of both, where `lam` defines
    how both losses are combined but with fewer parameters than the hybrid
    focal loss (`gamma` and `delta` parametrize different aspects of each
    loss).

    Args:
        pred (torch.Tensor): prediction probabilities.
        target (torch.Tensor): target class.
        delta (float): equivalent to `alpha` in focal loss
        and `alpha` and `1-beta` in Tversky focal loss
        gamma (float): equivalent to `gamma` in both
        losses
        lam (float, optional): weight term for both losses. Focal loss is 
        scaled by `lam` and the Tversky focal loss is scaled by `1-lam`.
        Defaults to 0.5.

    Returns:
        torch.Tensor: a tensor with size equal to the batch size (first 
        dimension of `pred`).
    """
    bfl = binary_focal_loss(pred,target,delta,1-gamma)
    bftl = binary_focal_tversky_loss(pred,target,delta,1-delta,gamma)
    return lam*bfl + (1-lam)*bftl

def mc_pt(pred:torch.Tensor,target:torch.Tensor)->torch.Tensor:
    """Convenience function to convert probabilities of each class
    to probability of predicting the corresponding target. Also works with
    one hot-encoded classes.

    Args:
        pred (torch.Tensor): prediction probabilities.
        target (torch.Tensor): target class.

    Returns:
        torch.Tensor: prediction of element i in `pred` predicting class
        i in `target.`
    """
    return torch.where(target > 0.5,pred,1-pred).to(pred.device)
    
def classes_to_one_hot(X:torch.Tensor)->torch.Tensor:
    """Converts classes to one-hot and permutes the dimension so that the
    channels (classes) are in second.

    Args:
        X (torch.Tensor): an indicator tensor.

    Returns:
        torch.Tensor: one-hot encoded tensor.
    """
    n_dim = len(list(X.shape))
    out_dim = [0,n_dim]
    out_dim.extend([i for i in range(1,n_dim)])
    return torch.nn.functional.one_hot(
        X.long(), num_classes=3).permute(out_dim).to(X.device)

def unsqueeze_to_shape(X:torch.Tensor,
                       target_shape:Union[list,tuple],
                       dim:int=1)->torch.Tensor:
    """Unsqueezes a vector `X` into a shape that is castable to a given
    target_shape.

    Args:
        X (torch.Tensor): a one-dimensional tensor
        target_shape (Union[list,tuple]): shape to which `X` will be castable.
        dim (int, optional): dimension corresponding to the vector in the 
        output. Defaults to 1.

    Returns:
        torch.Tensor: a tensor.
    """
    X = torch.flatten(X)
    output_shape = []
    for i,x in enumerate(target_shape):
        if i == dim:
            output_shape.append(int(X.shape[0]))
        else:
            output_shape.append(1)
    return X.reshape(output_shape)

def cat_cross_entropy(pred:torch.Tensor,
                      target:torch.Tensor,
                      weight:Union[float,torch.Tensor]=1.)->torch.Tensor:
    """Standard implementation of categorical cross entropy.

    Args:
        pred (torch.Tensor): prediction probabilities.
        target (torch.Tensor): target class.
        weight (Union[float,torch.Tensor], optional): class weights.
        Should be the same shape as the number of classes. Defaults to 1.

    Returns:
        torch.Tensor: a tensor with size equal to the batch size (first 
        dimension of `pred`).
    """
    if pred.shape != target.shape:
        target = classes_to_one_hot(target)
    if isinstance(weight,torch.Tensor) == True:
        weight = unsqueeze_to_shape(weight,pred.shape,1)
    out = -target*torch.log(pred+EPS)
    out = torch.flatten(out*weight,start_dim=1)
    return torch.mean(out,dim=1)

def mc_focal_loss(pred:torch.Tensor,
                  target:torch.Tensor,
                  alpha:torch.Tensor,
                  gamma:Union[float,torch.Tensor])->torch.Tensor:
    """Categorical focal loss. Uses `alpha` to weight classes and `lambda` to
    suppress examples which are easy to classify (given that `lambda`>1). 
    `lambda` is also known as the focusing parameter. In essence, 
    `focal_loss = (1-mc_pt(pred,target))**gamma*ce`, where `ce` is the 
    categorical cross entropy [1].

    [1] https://arxiv.org/abs/1708.02002

    Args:
        pred (torch.Tensor): prediction probabilities.
        target (torch.Tensor): target class.
        alpha (torch.Tensor): class weights.
        gamma (Union[float,torch.Tensor]): focusing parameter.

    Returns:
         torch.Tensor: a tensor with size equal to the batch size (first 
        dimension of `pred`).
    """
    alpha = unsqueeze_to_shape(alpha,pred.shape,dim=1)
    if pred.shape != target.shape:
        target = classes_to_one_hot(target)
    p = mc_pt(pred,target)
    ce = -target*torch.log(pred+EPS)
    out = torch.flatten(alpha*((1-p+EPS)**gamma)*ce,start_dim=1)
    return torch.mean(out,dim=1)

def mc_focal_tversky_loss(pred:torch.Tensor,
                          target:torch.Tensor,
                          alpha:torch.Tensor,
                          beta:torch.Tensor,
                          gamma:Union[torch.Tensor,float]=1.)->torch.Tensor:
    """Categorical focal Tversky loss. Very similar to the original Tversky
    loss but features a `gamma` term similar to the focal loss to focus the 
    learning on harder/rarer examples [1]. The Tversky loss itself is very 
    similar to the Dice score loss but weighs false positives and false 
    negatives differently (set as the `alpha` and `beta` hyperparameter,
    respectively).

    [1] https://arxiv.org/abs/1810.07842

    Args:
        pred (torch.Tensor): prediction probabilities.
        target (torch.Tensor): target class.
        alpha (torch.Tensor): weight for false positives.
        beta (torch.Tensor): weight for false negatives.
        gamma (Union[torch.Tensor,float]): focusing parameter.

    Returns:
         torch.Tensor: a tensor with size equal to the batch size (first 
        dimension of `pred`).
    """
    if pred.shape != target.shape:
        target = classes_to_one_hot(target)
    p_fore = torch.flatten(pred,start_dim=2)
    p_back = 1-p_fore
    t_fore = torch.flatten(target,start_dim=2)
    t_back = 1-t_fore
    n = torch.sum(p_fore*t_fore,dim=-1) + 1
    d_1 = alpha*torch.sum(p_fore*t_back,dim=-1)
    d_2 = beta*torch.sum(p_back*t_fore,dim=-1)
    d = n + d_1 + d_2 + 1
    return torch.sum(1-torch.pow(n/d,gamma),dim=-1)

def mc_combo_loss(pred:torch.Tensor,
                  target:torch.Tensor,
                  alpha:float=0.5,
                  beta:Union[float,torch.Tensor]=1)->torch.Tensor:
    """Combo loss. Simply put, it is a weighted combination of the Dice loss
    and the weighted cross entropy [1]. `alpha` is the weight for each loss 
    and beta is the weight for the binary cross entropy.

    [1] https://arxiv.org/abs/1805.02798

    Args:
        pred (torch.Tensor): prediction probabilities.
        target (torch.Tensor): target class.
        alpha (float, optional): weight term for both losses. Cross
        entropy is scaled by `alpha` and the Dice score is scaled by 
        `1-alpha`. Defaults to 0.5.
        beta (Union[float,torch.Tensor], optional): weight for the cross
        entropy. Defaults to 1.

    Returns:
         torch.Tensor: a tensor with size equal to the batch size (first 
        dimension of `pred`).
    """
    bdl = generalized_dice_loss(pred,target)
    bce = cat_cross_entropy(pred,target,beta)
    return (alpha)*bce + (1-alpha)*bdl

def mc_hybrid_focal_loss(pred:torch.Tensor,
                         target:torch.Tensor,
                         lam:float=1.,
                         focal_params:dict={},
                         tversky_params:dict={})->torch.Tensor:
    """Hybrid focal loss. A combination of the focal loss and the focal 
    Tversky loss. In essence, a weighted sum of both, where `lam` defines
    how both losses are combined.

    Args:
        pred (torch.Tensor): prediction probabilities.
        target (torch.Tensor): target class.
        lam (float, optional): weight term for both losses. Focal loss is 
        scaled by `lam` and the Tversky focal loss is scaled by `1-lam`.
        Defaults to 0.5.
        focal_params (dict, optional): dictionary with parameters for the 
        focal loss. Defaults to {}.
        tversky_params (dict, optional): dictionary with parameters for the
        Tversky focal loss. Defaults to {}.

    Returns:
        torch.Tensor: a tensor with size equal to the batch size (first 
        dimension of `pred`).
    """
    a = focal_params["alpha"]
    if a is None or isinstance(a,int)==True or isinstance(a,float)==True:
        focal_params["alpha"] = torch.ones(pred.shape[1])
    fl = mc_focal_loss(pred,target,**focal_params)
    ftl = mc_focal_tversky_loss(pred,target,**tversky_params)
    return lam*fl + (1-lam)*ftl    

def mc_unified_focal_loss(pred:torch.Tensor,
                          target:torch.Tensor,
                          delta:torch.Tensor,
                          gamma:Union[torch.Tensor,float],
                          lam:float)->torch.Tensor:
    """Unified focal loss. A combination of the focal loss and the focal 
    Tversky loss. In essence, a weighted sum of both, where `lam` defines
    how both losses are combined but with fewer parameters than the hybrid
    focal loss (`gamma` and `delta` parametrize different aspects of each
    loss).

    Args:
        pred (torch.Tensor): prediction probabilities.
        target (torch.Tensor): target class.
        delta (torch.Tensor): equivalent to `alpha` in focal loss
        and `alpha` and `1-beta` in Tversky focal loss
        gamma (Union[torch.Tensor,float]): equivalent to `gamma` in both
        losses
        lam (float, optional): weight term for both losses. Focal loss is 
        scaled by `lam` and the Tversky focal loss is scaled by `1-lam`.
        Defaults to 0.5.

    Returns:
        torch.Tensor: a tensor with size equal to the batch size (first 
        dimension of `pred`).
    """
    fl = mc_focal_loss(pred,target,delta,1-gamma)
    ftl = mc_focal_tversky_loss(pred,target,delta,1-delta,gamma)
    return lam*fl + (1-lam)*ftl