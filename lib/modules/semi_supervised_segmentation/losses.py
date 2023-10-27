import numpy as np
import torch
import torch.nn.functional as F
from queue import Queue
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
                rep = rep.permute(1,0)
                self.average_representations[:,i,:] = torch.add(
                    self.average_representations[:,i,:] * (1 - self.ema_theta),
                    rep.mean(1) * self.ema_theta)

    def update_hard_examples(self, 
                             proba: torch.Tensor, 
                             embeddings: torch.Tensor,
                             labels: torch.LongTensor):
        weights = proba.prod(1)
        for i in range(self.batch_size):
            top_k = weights[i].topk(self.top_k)
            self.hard_examples[i] = embeddings[i,:,top_k.indices].permute(1,0)
            self.hard_example_class[i] = labels[i,top_k.indices].unsqueeze(-1)

    def delete(self, X, idx):
        return torch.cat([X[:,:idx], X[:,(idx+1):]],1)

    def l_anco(self):
        output = torch.zeros([self.batch_size,self.n_classes]).to(
            self.average_representations)
        for batch in range(self.batch_size):
            for nc in range(self.n_classes):
                idx,_ = torch.where(self.hard_example_class[batch] == nc)
                he = self.hard_examples[batch,idx]
                par = self.average_representations[:,nc,:]
                nar = self.delete(self.average_representations,nc)
                num = torch.exp(he * par / self.tau)
                neg_den = torch.sum(
                    torch.exp(he[:,None] * nar / self.tau),
                    1, keepdim=True)
                out = -torch.log(num / (num + neg_den))
                if prod(out.shape) > 0:
                    output[batch,nc] = torch.mean(out.flatten(start_dim=1))
        return - output.sum() / self.batch_size

    def forward(self, 
                proba: torch.Tensor, 
                y: torch.Tensor, 
                embeddings: torch.Tensor):
        # expects y to be one hot encoded
        proba = proba.flatten(start_dim=2)
        y = y.flatten(start_dim=2)
        embeddings = embeddings.flatten(start_dim=2)
        labels = torch.argmax(y, dim=1)

        # get indices for y
        b,c,v = torch.where(y > 0)

        # update average class representation
        self.update_average_class_representation(embeddings, b, c, v)

        # mine hard examples 
        self.update_hard_examples(proba=proba, 
                                  embeddings=embeddings, 
                                  labels=labels)

        # calculate the anatomical contrastive loss
        return self.l_anco()

class NearestNeighbourLoss(torch.nn.Module):
    """
    Nearest neighbour loss:
    
    1. For a given FIFO queue of past elements and a new sample, use the oldest
        elements from the queue to calculate the distances between the new sample
        and the old elements
    2. Maximise the cosine similarity between queue elements and elements from
        the new sample belonging to the same class.

    Based on Frosst 2019 [1].

    [1] https://proceedings.mlr.press/v97/frosst19a.html
    """
    def __init__(self,
                 maxsize: int,
                 n_classes: int,
                 max_elements_per_batch: int,
                 n_samples_per_class: int,
                 temperature: float=0.1,
                 seed: int=42):
        super().__init__()
        self.maxsize = maxsize
        self.n_classes = n_classes
        self.max_elements_per_batch = max_elements_per_batch
        self.n_samples_per_class = n_samples_per_class
        self.temperature = temperature
        self.seed = seed

        self.q = [Queue(maxsize=self.maxsize) for _ in range(self.n_classes)]
        self.rng = np.random.default_rng(seed)
    
    def put(self, 
            X: torch.Tensor,
            y: torch.Tensor):
        X = X.flatten(start_dim=2)
        y = y.flatten(start_dim=2)
        b, c, v = torch.where(y > 0)
        for cl in range(self.n_classes):
            idx = c == cl
            elements = X[b[idx], :, v[idx]]
            n_elements = elements.shape[0]
            if n_elements > self.max_elements_per_batch:
                elements = elements[
                    self.rng.choice(n_elements,self.max_elements_per_batch)]
            if n_elements > 0:
                self.q[cl].put(elements.detach())

    def get_from_class(self, n: int, cl: int):
        q = self.q[cl]
        n_elements = q.qsize()
        return [q.get() for _ in self.rng.choice(n_elements,n)]

    def get(self, n: int, cl: int=None):
        if cl is not None:
            output = self.get_from_class(n,cl)
        else:
            output = []
            sample = self.rng.choice(self.n_classes,size=n)
            un,count = np.unique(sample,return_counts=True)
            for cl, n in zip(un, count):
                output.append(self.get_from_class(n, cl))
        return torch.cat(output,0)

    def __len__(self):
        return sum([q.qsize() for q in self.q])

    def get_past_samples(self, device="cuda"):
        n_samples = [np.minimum(self.n_samples_per_class,self.q[cl].qsize())
                     for cl in range(self.n_classes)]
        past_samples = [
            self.get(n,cl)
            for cl,n in zip(range(self.n_classes),n_samples)]
        past_sample_labels = torch.as_tensor(np.concatenate(
            [np.repeat(cl,past_sample.shape[0]) 
             for cl,past_sample in zip(range(self.n_classes),past_samples)],0),
            device=device)
        past_sample_labels = F.one_hot(past_sample_labels,self.n_classes)
        past_samples = torch.cat(past_samples)
        return past_samples, past_sample_labels

    def forward(self,
                X: torch.Tensor,
                y: torch.Tensor):
        X = X.flatten(start_dim=2).permute(0, 2, 1)
        y = y.flatten(start_dim=2).permute(0, 2, 1)
        b,c,v = torch.where(y > 0)
        past_samples, past_sample_labels = self.get_past_samples(X.device)
        distances = 1 - F.cosine_similarity(
            X[:,:,None], past_samples[None,None,:],-1)
        is_same = torch.sum(y[:,:,None] * past_sample_labels[None,None,:],-1)
        same_class_distances = torch.exp(
            - distances * is_same / self.temperature)
        other_class_distances = torch.exp(
            - distances * (1 - is_same) / self.temperature)
        return torch.nanmean(
            same_class_distances.nansum(-1) / other_class_distances.nansum(-1))

class PseudoLabelCrossEntropy(torch.nn.Module):
    """
    Calculates cross-entropy between probability map p_1 and pseudo-labels 
    calculated from probability map p_2 given a probability threshold.

    Useful for distillation, semi-supervised learning, etc.
    """
    def __init__(self, 
                 threshold: float,
                 *args,
                 **kwargs):
        super().__init__()
        self.threshold = threshold

        self.ce = torch.nn.CrossEntropyLoss(*args, **kwargs)

    def forward(self, pred: torch.Tensor, proba: torch.Tensor):
        pseudo_y = proba > self.threshold
        return self.ce(pred, pseudo_y.float())
