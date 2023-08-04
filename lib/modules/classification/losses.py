import torch
import torch.nn.functional as F

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
        if ignore_0 is True:
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

class OrdinalSigmoidalLoss(torch.nn.Module):
    def __init__(self,weight:torch.Tensor,n_classes:int):
        self.n_classes = n_classes
        self.weight = torch.as_tensor(weight)
    
    def __call__(self,pred,target):
        return ordinal_sigmoidal_loss(
            pred,target,self.n_classes,self.weight)
    