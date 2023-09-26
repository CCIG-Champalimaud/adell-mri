import torch
import einops
from math import prod

class AnatomicalContrastiveLoss(torch.nn.Module):
    """
    Implementation of the anatomical loss method suggested in "Bootstrapping
    Semi-supervised Medical Image Segmentation with Anatomical-aware Contrastive
    Distillation". Allow for both local and global KL divergence calculation.

    This was altered to extract a fixed number of hard examples (unlike what was
    specified in the original paper, which uses a threshold to define hard 
    examples).
    """
    def __init__(self,
                 n_classes: int,
                 n_features: int,
                 batch_size: int,
                 top_k: int=100,
                 ema_theta: float=0.9,
                 tau: float=0.1):
        super().__init__()
        self.n_classes = n_classes
        self.n_features = n_features
        self.batch_size = batch_size
        self.top_k = top_k
        self.ema_theta = ema_theta
        self.tau = tau

        self.average_representations = torch.zeros(
            [1,self.n_classes,self.n_features])
        
        self.hard_examples = torch.zeros(
            [batch_size, self.top_k, n_features])
        self.hard_example_class = torch.zeros(
            [batch_size, self.top_k, 1])
    
    def update_average_class_representation(self,
                                            pred: torch.Tensor, 
                                            b: torch.LongTensor,
                                            c: torch.LongTensor,
                                            v: torch.LongTensor):
        for i in range(self.n_classes):
            rep = pred[b[c == i],:,v[c == i]]
            if prod(rep.shape) > 0:
                rep = einops.rearrange(rep,"b c v -> c (b v)")
                self.average_representations[:,i,:] = torch.add(
                    self.average_representations[:,i,:] * (1 - self.ema_theta),
                    rep.mean(1) * self.ema_theta)

    def update_hard_examples(self, 
                             proba: torch.Tensor, 
                             embeddings: torch.Tensor,
                             c: torch.LongTensor):
        weights = proba.prod(1)
        for i in range(self.batch_size):
            top_k = weights[i].topk(self.top_k)
            self.hard_examples[i] = embeddings[i,:,top_k]
            self.hard_example_class[i] = c[top_k]

    def delete(self, X, idx):
        return torch.cat([X[:,:idx], X[:,(idx+1):]],1)

    def l_anco(self):
        for i in range(self.n_classes):
            he = self.hard_examples[self.hard_example_class == i]
            par = self.average_representations[:,i,:]
            nar = self.delete(self.average_representations,i)
            num = torch.exp(he * par / self.tau)
            neg_den = torch.sum(
                torch.exp(he[:,None] * nar[:,:,None] / self.tau),
                1, keepdim=True)
            out = -torch.log(num / (num / neg_den))
            return torch.sum(out.flatten(start_dim=1),1)

    def forward(self, 
                proba: torch.Tensor, 
                y: torch.Tensor, 
                embeddings: torch.Tensor):
        # expects y to be one hot encoded
        proba = proba.flatten(start_dim=2)
        y = y.flatten(start_dim=2)
        embeddings = embeddings.flatten(start_dim=2)

        # get indices for y
        b,c,v = torch.where(y > 0)

        # update average class representation
        self.update_average_class_representation(embeddings, b, c, v)

        # mine hard examples 
        self.update_hard_examples(proba=proba, embeddings=embeddings)

        # calculate the anatomical contrastive loss
        return self.l_anco() 
