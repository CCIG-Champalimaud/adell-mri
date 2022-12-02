"""
Implementations of different loss functions for segmentation tasks.
"""

import torch
import torch.nn.functional as F
from typing import Union,Tuple
from itertools import combinations

eps = 1e-6
FOCAL_DEFAULT = {"alpha":None,"gamma":1}
TVERSKY_DEFAULT = {"alpha":1,"beta":1,"gamma":1}

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

def binary_focal_loss(pred:torch.Tensor,
                      target:torch.Tensor,
                      alpha:float,
                      gamma:float,
                      threshold:float=0.5,
                      scale:float=1.0,
                      eps:float=eps)->torch.Tensor: 
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
        threshold (float, optional): threshold for the positive class in the
            focal loss. Helpful in cases where one is trying to model the 
            probability explictly. Defaults to 0.5.
        scale (float, optional): factor to scale loss before reducing.
        eps (float, optional): epsilon factor to avoid floating-point
            imprecisions.

    Returns:
        torch.Tensor: a tensor with size equal to the batch size (first 
            dimension of `pred`).
    """
    alpha = torch.as_tensor(alpha).type_as(pred)
    gamma = torch.as_tensor(gamma).type_as(pred)

    p = pt(pred,target,threshold)
    w = torch.where(target>0,alpha*torch.ones_like(p),torch.ones_like(p))
    p = torch.flatten(p,start_dim=1)
    w = torch.flatten(w,start_dim=1)
    bce = -torch.log(p+eps)
    
    x = w * ((1-p+eps)**gamma)
    print(scale,x,bce)
    x = scale*x*bce
    loss = torch.mean(x,dim=-1)
    return loss

def binary_focal_loss_(pred:torch.Tensor,
                       target:torch.Tensor,
                       alpha:float,
                       gamma:float,
                       scale:float=1.0,
                       eps:float=eps)->torch.Tensor: 
    """Implementation of binary focal loss. Uses `alpha` to weight
    the positive class and `lambda` to suppress examples which are easy to 
    classify (given that `lambda`>1). `lambda` is also known as the focusing 
    parameter. In essence, `focal_loss = (1-pt(pred,target))**gamma*bce`, where 
    `bce` is the binary cross entropy [1].

    Inspired in [2].

    [1] https://arxiv.org/abs/1708.02002
    [2] https://github.com/ultralytics/yolov5/blob/c23a441c9df7ca9b1f275e8c8719c949269160d1/utils/loss.py#L35

    Args:
        pred (torch.Tensor): prediction probabilities.
        target (torch.Tensor): target class.
        alpha (float): positive class weight.
        gamma (float): focusing parameter.
        scale (float, optional): factor to scale loss before reducing.
        eps (float, optional): epsilon factor to avoid floating-point
            imprecisions.

    Returns:
         torch.Tensor: a tensor with size equal to the batch size (first 
        dimension of `pred`).
    """
    alpha = torch.as_tensor(alpha).type_as(pred)
    gamma = torch.as_tensor(gamma).type_as(pred)

    if len(pred.shape) > 2:
        pred = torch.flatten(pred,start_dim=1)
    
    target = target.reshape(pred.shape)
    loss = -(target*torch.log(pred+eps)+(1-target)*torch.log(1-pred+eps))
    target_bin = (target>0).float()
    alpha_factor = target_bin*alpha + (1-target_bin)*(1-alpha)
    modulating_factor = torch.pow(torch.abs(target - pred)+eps,gamma)
    loss *= alpha_factor * modulating_factor
    return torch.mean(loss*scale,1)

def weighted_mse(pred:torch.Tensor,
                 target:torch.Tensor,
                 alpha:float,
                 threshold:float=0.5)->torch.Tensor: 
    """Weighted MSE. Useful for object detection tasks.

    Args:
        pred (torch.Tensor): prediction probabilities.
        target (torch.Tensor): target class.
        alpha (float): positive class weight.
        gamma (float): focusing parameter.
        threshold (float, optional): threshold for the positive class in the
        focal loss. Helpful in cases where one is trying to model the 
        probability explictly. Defaults to 0.5.

    Returns:
         torch.Tensor: a tensor with size equal to the batch size (first 
        dimension of `pred`).
    """
    alpha = torch.as_tensor(alpha).type_as(pred)

    pred = torch.flatten(pred,start_dim=1)
    target = torch.flatten(target,start_dim=1)
    mse = torch.square(pred-target)
    positive_mse = mse[target >= threshold].mean(-1)
    negative_mse = mse[target < threshold].mean(-1)/alpha
    
    return positive_mse + negative_mse

def generalized_dice_loss(pred:torch.Tensor,
                          target:torch.Tensor,
                          weight:float=1.,
                          smooth:float=1.)->torch.Tensor:
    """Dice loss adapted to cases of very high class imbalance. In essence
    it adds class weights to the calculation of the Dice loss [1]. If 
    `weights=1` it defaults to the regular Dice loss. This implementation 
    works for both the binary and categorical cases.

    [1] https://arxiv.org/pdf/1707.03237.pdf

    Args:
        pred (torch.Tensor): prediction probabilities.
        target (torch.Tensor): target class.
        weight (float, optional): class weights. Defaults to 1.
        smooth (float, optional): term preventing the loss from being
            undefined.

    Returns:
         torch.Tensor: a tensor with size equal to the batch size (first 
        dimension of `pred`).
    """
    weight = torch.as_tensor(weight).type_as(pred)
    scaling_term = 0.0001
    
    if pred.shape != target.shape:
        target = classes_to_one_hot(target)
        weight = unsqueeze_to_shape(weight,[1,1],1)
    
    target = torch.flatten(target,start_dim=2)
    pred = torch.flatten(pred,start_dim=2)
    smooth = smooth * scaling_term
    # adjusting the values of target and pred to avoid fp16 issues
    tp = torch.sum(target * pred * scaling_term,2)
    fp = torch.sum((1.-target) * pred * scaling_term,2)
    fn = torch.sum(target * (1.-pred) * scaling_term,2)
        
    cl_dice = torch.divide(2.*(tp*weight) + smooth,
                           2.*(tp*weight) + fp + fn + smooth)
    return cl_dice

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
    alpha = torch.as_tensor(alpha).type_as(pred)
    beta = torch.as_tensor(beta).type_as(pred)
    gamma = torch.as_tensor(gamma).type_as(pred)

    p_fore = torch.flatten(pred,start_dim=1)
    p_back = 1-p_fore
    t_fore = torch.flatten(target,start_dim=1)
    t_back = 1-t_fore
    tp = torch.sum(p_fore*t_fore,dim=1)
    fn = torch.sum(p_fore*t_back,dim=1)
    fp = torch.sum(p_back*t_fore,dim=1)
    nd = (tp+1)/(tp + alpha*fn + beta*fp+1)

    return 1-(nd)**gamma

def combo_loss(pred:torch.Tensor,
               target:torch.Tensor,
               alpha:float=0.5,
               beta:float=1,
               gamma:float=1.,
               scale:float=1.0)->torch.Tensor:
    """Combo loss. Simply put, it is a weighted combination of the Dice loss
    and the weighted focal loss [1]. `alpha` is the weight for each loss 
    and beta is the weight for the binary cross entropy.

    [1] https://arxiv.org/abs/1805.02798

    Args:
        pred (torch.Tensor): prediction probabilities.
        target (torch.Tensor): target class.
        alpha (float, optional): weight term for both losses. Binary cross 
            entropy is scaled by `alpha` and the Dice score is scaled by 
            `1-alpha`. Defaults to 0.5.
        beta (float, optional): weight for the cross entropy. Defaults to 1.
        gamma (float, optional): focusing parameter. Setting this to 0 leads
            this loss to be the weighted sum of generalized Dice loss and
            BCE. Defaults to 1.0.
        scale (float, optional): factor to scale focal loss before reducing.
            Defaults to 1.0.

    Returns:
        torch.Tensor: a tensor with size equal to the batch size (first 
            dimension of `pred`).
    """
    alpha = torch.as_tensor(alpha).type_as(pred)
    beta = torch.as_tensor(beta).type_as(pred)

    bdl = generalized_dice_loss(pred,target,beta) * scale
    bce = binary_focal_loss(pred,target,beta,gamma,scale=scale)
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
                       lam:float=0.5,
                       threshold:float=0.5,
                       scale:float=1.0)->torch.Tensor:
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
        threshold (float, optional): threshold for the positive class in the
            focal loss. Helpful in cases where one is trying to model the 
            probability explictly. Defaults to 0.5.
        scale (float, optional): factor to scale focal loss before reducing.

    Returns:
        torch.Tensor: a tensor with size equal to the batch size (first 
        dimension of `pred`).
    """
    delta = torch.as_tensor(delta).type_as(pred)
    gamma = torch.as_tensor(gamma).type_as(pred)

    bfl = binary_focal_loss(pred,target,delta,1-gamma,threshold,scale)
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
                      weight:Union[float,torch.Tensor]=1.,
                      scale:float=1.0,
                      eps:float=eps)->torch.Tensor:
    """Standard implementation of categorical cross entropy.

    Args:
        pred (torch.Tensor): prediction probabilities.
        target (torch.Tensor): target class.
        weight (Union[float,torch.Tensor], optional): class weights.
            Should be the same shape as the number of classes. Defaults to 1.
        scale (float, optional): factor to scale loss before reducing.
        eps (float, optional): epsilon factor to avoid floating-point
            imprecisions.

    Returns:
        torch.Tensor: a tensor with size equal to the batch size (first 
        dimension of `pred`).
    """
    weight = torch.as_tensor(weight).type_as(pred)

    if pred.shape != target.shape:
        target = classes_to_one_hot(target)
    if isinstance(weight,torch.Tensor) == True:
        weight = unsqueeze_to_shape(weight,pred.shape,1)
    out = -target*torch.log(pred+eps)
    out = torch.flatten(out*weight,start_dim=1)
    return torch.mean(out*scale,dim=1)

def mc_focal_loss(pred:torch.Tensor,
                  target:torch.Tensor,
                  alpha:torch.Tensor,
                  gamma:Union[float,torch.Tensor],
                  scale:float=1.0,
                  eps:float=eps)->torch.Tensor:
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
        scale (float, optional): factor to scale loss before reducing.
        eps (float, optional): epsilon factor to avoid floating-point
            imprecisions.

    Returns:
         torch.Tensor: a tensor with size equal to the batch size (first 
        dimension of `pred`).
    """
    alpha = torch.as_tensor(alpha).type_as(pred)
    gamma = torch.as_tensor(gamma).type_as(pred)

    alpha = unsqueeze_to_shape(alpha,pred.shape,dim=1)
    if pred.shape != target.shape:
        target = classes_to_one_hot(target)
    p = mc_pt(pred,target)
    ce = -target*torch.log(pred+eps)
    out = torch.flatten(alpha*((1-p+eps)**gamma)*ce,start_dim=1)
    return torch.mean(out*scale,dim=1)

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
    alpha = torch.as_tensor(alpha).type_as(pred)
    beta = torch.as_tensor(beta).type_as(pred)
    gamma = torch.as_tensor(gamma).type_as(pred)

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
    return torch.mean(1-torch.pow(n/d,gamma),dim=-1)

def mc_combo_loss(pred:torch.Tensor,
                  target:torch.Tensor,
                  alpha:float=0.5,
                  beta:Union[float,torch.Tensor]=1,
                  scale:float=1.0)->torch.Tensor:
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
        scale (float, optional): factor to scale CE loss before reducing.

    Returns:
         torch.Tensor: a tensor with size equal to the batch size (first 
        dimension of `pred`).
    """
    alpha = torch.as_tensor(alpha).type_as(pred)
    beta = torch.as_tensor(beta).type_as(pred)

    bdl = generalized_dice_loss(pred,target)
    bce = cat_cross_entropy(pred,target,beta,scale)
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
                          lam:float,
                          scale:float=1.0)->torch.Tensor:
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
        scale (float, optional): factor to scale focal loss before reducing.

    Returns:
        torch.Tensor: a tensor with size equal to the batch size (first 
        dimension of `pred`).
    """
    delta = torch.as_tensor(delta).type_as(pred)
    gamma = torch.as_tensor(gamma).type_as(pred)

    fl = mc_focal_loss(pred,target,delta,1-gamma,scale)
    ftl = mc_focal_tversky_loss(pred,target,delta,1-delta,gamma)
    return lam*fl + (1-lam)*ftl

def complete_iou_loss(
    a:torch.Tensor,
    b:torch.Tensor,ndim:int=3)->Tuple[torch.Tensor,torch.Tensor,torch.Tensor]:
    """Calculates the complete IoU loss as proposed by Zheng et al. [1] for 
    any given number of spatial dimensions. 
    Combines three components - the IoU loss, the minimization of the distance
    between centers and the minimization of the difference in aspect ratios.
    To calculate the aspect ratios for n-dimensions, the aspect ratio between
    each dimension pair are averaged.

    [1] https://arxiv.org/abs/1911.08287

    Args:
        a (torch.Tensor): set of n bounding boxes.
        b (torch.Tensor): set of n bounding boxes.
        ndim (int, optional): Number of spatial dimensions. Defaults to 3.

    Returns:
        iou: the IoU loss
        cpd_component: the distance between centers component of the loss
        ar_component: the aspect ratio component of the loss
    """
    a_tl,b_tl = a[:,:ndim],b[:,:ndim]
    a_br,b_br = a[:,ndim:],b[:,ndim:]
    inter_tl = torch.maximum(a_tl,b_tl)
    inter_br = torch.minimum(a_br,b_br)
    a_size = a_br - a_tl + 1
    b_size = b_br - b_tl + 1
    inter_size = inter_br - inter_tl + 1
    a_center = (a_tl + a_br)/2
    b_center = (b_tl + b_br)/2
    diag_tl = torch.minimum(a_tl,b_tl)
    diag_br = torch.maximum(a_br,b_br)
    # calculate IoU
    inter_area = torch.prod(inter_size,axis=-1)
    union_area = torch.subtract(
        torch.prod(a_size,axis=-1) + torch.prod(b_size,axis=-1),
        inter_area)
    iou = inter_area / union_area
    # distance between centers and between corners of external bounding box 
    # for distance IoU loss
    center_distance = torch.square(a_center-b_center).sum(-1)
    bb_distance = torch.square(diag_br-diag_tl).sum(-1)
    cpd_component = center_distance/bb_distance
    # aspect ratio component
    pis = torch.pi**2
    ar_list = []
    for i,j in combinations(range(ndim),2):
        ar_list.append(
            4/pis*(torch.subtract(
                torch.arctan(a_size[:,i]/a_size[:,j]),
                torch.arctan(b_size[:,i]/b_size[:,j])))**2)
    v = sum(ar_list)/len(ar_list)
    alpha = v/((1-iou)+v)
    ar_component = v * alpha

    return iou,cpd_component,ar_component

def ordinal_sigmoidal_loss(pred:torch.Tensor,
                           target:torch.Tensor,
                           n_classes:int,
                           weight:torch.Tensor=None):
    def label_to_ordinal(label,n_classes,ignore_0=True):
        one_hot = F.one_hot(label,n_classes)
        one_hot = one_hot.unsqueeze(1).swapaxes(1,-1).squeeze(-1)
        one_hot = torch.clamp(one_hot,max=1)
        one_hot_cumsum = torch.cumsum(one_hot,axis=1) - one_hot
        output = torch.ones_like(one_hot_cumsum,device=one_hot_cumsum.device)
        output = output - one_hot_cumsum
        if ignore_0 == True:
            output = output[:,1:]
        return output

    weight = torch.as_tensor(weight).type_as(pred)

    target_ordinal = label_to_ordinal(target,n_classes)
    loss = F.binary_cross_entropy_with_logits(
        pred,target_ordinal.float(),reduction="none")
    loss = loss.flatten(start_dim=1).sum(1)
    if weight is not None:
        weight_sample = weight[target]
        loss = loss * weight_sample
    
    return loss