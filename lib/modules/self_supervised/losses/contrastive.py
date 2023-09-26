import torch
import torch.nn.functional as F
import einops
from math import prod
from .functional import cos_dist

from copy import deepcopy

class KLDivergence(torch.nn.Module):
    """
    Implementation of the KL divergence method suggested in "Bootstrapping 
    Semi-supervised Medical Image Segmentation with Anatomical-aware Contrastive
    Distillation". Allow for both local and global KL divergence calculation.
    """
    def __init__(self,
                 mode: str="global"):
        """
        Args:
            mode (str, optional): whether the input feature maps should be 
            reduced to their global average ("global") or simply flattened to
            calculate local differences ("local"). Defaults to "global".
        """
        super().__init__()
        self.mode = mode

        assert mode in ["global","local"], \
            f"mode {mode} not supported. available options: 'global', 'local'"

    def average_pooling(self,X: torch.Tensor):
        if len(X.shape) > 2:
            return X.flatten(start_dim=2).mean(-1)
        return X

    def forward(self,
                X_1: torch.Tensor,
                X_2: torch.Tensor,
                anchors: torch.Tensor):
        if self.mode == "global":
            X_1 = self.average_pooling(X_1)
            X_2 = self.average_pooling(X_2)
            anchors = self.average_pooling(anchors)
        elif self.mode == "local":
            X_1 = X_1.flatten(start_dim=2)
            X_2 = X_2.flatten(start_dim=2)
            anchors = anchors.flatten(start_dim=2)

        p_1 = F.softmax(F.cosine_similarity(X_1[:,None],anchors),dim=1)
        p_2 = F.softmax(F.cosine_similarity(X_2[None,:],anchors),dim=1)
        kl_div = torch.sum(p_1 * (torch.log(p_1) - torch.log(p_2)))
        return kl_div

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
        if self.random_sample is True:
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

    def forward(self,X1:torch.Tensor,X2:torch.Tensor=None,y:torch.Tensor=None):
        if isinstance(X1,list):
            X1,X2 = X1
        if X2 is None:
            X2 = deepcopy(X1)
        if y is None:
            y = torch.ones([X1.shape[0]])
        y1,y2 = y.unsqueeze(0),y.unsqueeze(1)
        is_same = y1 == y2
        if self.loss_type == "pairwise":
            loss = self.pairwise_distance(X1,X2,is_same)
        elif self.loss_type == "triplet":
            loss = self.triplet_distance(X1,X2,is_same)
        return loss.mean()
