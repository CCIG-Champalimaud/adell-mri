import numpy as np
import torch
from itertools import product

from typing import List, Tuple, Union

Coords = Union[
    Tuple[int,int,int,int],
    Tuple[int,int,int,int,int,int]]

class TransformerMasker(torch.nn.Module):
    def __init__(self,
                 image_dimensions:List[int],
                 min_patch_size:List[int],
                 max_patch_size:List[int],
                 n_features:int=None,
                 n_patches:int=4,
                 seed:int=42):
        super().__init__()
        self.image_dimensions = image_dimensions
        self.min_patch_size = min_patch_size
        self.max_patch_size = max_patch_size
        self.n_features = n_features
        self.n_patches = n_patches
        self.seed = seed

        self.rng = np.random.default_rng(seed)
        self.n_dim = len(self.image_dimensions)

        assert self.n_dim in [2,3]
        self.initialize_positional_embedding_if_necessary()

    def sample_patch(self)->Coords:
        upper_sampling_bound = [
            size-patch_size 
            for size,patch_size in zip(self.image_dimensions,
                                       self.max_patch_size)]
        lower_bound = np.array(
            [self.rng.integers(0,i) for i in upper_sampling_bound])
        patch_size = np.array(
            [self.rng.integers(m,M) 
             for m,M in zip(self.min_patch_size,self.max_patch_size)])
        upper_bound = lower_bound + patch_size
        return [*lower_bound,*upper_bound]
    
    def initialize_positional_embedding_if_necessary(self):
        if self.n_features is not None:
            self.positional_embedding = torch.nn.Parameter(
                torch.as_tensor(
                    np.zeros(
                        [np.prod(self.image_dimensions),
                         self.n_features]),
                     dtype=torch.float32))
            torch.nn.init.trunc_normal_(
                self.positional_embedding,mean=0.0,std=0.02,a=-2.0,b=2.0)
        else:
            self.positional_embedding = None

    def sample_patches(self,n_patches:int)->List[Coords]:
        return [self.sample_patch() for _ in range(n_patches)]

    def retrieve_patch(self,
                       X:torch.Tensor,
                       patch_coords:List[int],
                       mask_vector:torch.Tensor=None)->Tuple[torch.Tensor,List[int]]:
        upper_bound, lower_bound = (patch_coords[:self.n_dim],
                                    patch_coords[self.n_dim:])
        patch_coords = [
            c for c in product(*[range(upper_bound[i],lower_bound[i])
                                 for i in range(self.n_dim)])]
        patch_coords = np.array(patch_coords)
        if self.n_dim == 2:
            h,w = self.image_dimensions
            expression = np.array([w,1])[None,:]
            patch_idx = np.sum(patch_coords * expression,1)
        elif self.n_dim == 3:
            h,w,d = self.image_dimensions
            expression = np.array([w,1,w*h])[None,:]
        patch_idx = np.sum(patch_coords * expression,1)
        X_patches = X[...,patch_idx,:]
        long_coords = patch_idx

        return X_patches,long_coords

    def __call__(self,
                 X:torch.Tensor,
                 mask_vector:torch.Tensor=None,
                 n_patches:int=None,
                 patch_coords:List[Coords]=None)->Tuple[torch.Tensor,List[torch.Tensor]]:
        if n_patches is None:
            n_patches = self.n_patches
        if patch_coords is None:
            patch_coords = self.sample_patches(n_patches)
        all_patches = []
        full_coords = []
        for coords in patch_coords:
            patch,long_coords = self.retrieve_patch(X,coords)
            all_patches.append(patch)
            full_coords.append(long_coords)
        if mask_vector is not None:
            full_coords = np.unique(np.concatenate(full_coords))
            X[:,full_coords,:] = mask_vector.to(X)
            if self.positional_embedding is not None:
                p = self.positional_embedding[None,full_coords,:]
                X[:,full_coords,:] = X[:,full_coords,:] + p.to(X)
        return X,all_patches,patch_coords

class ConvolutionalMasker:
    # TODO: add positional embedding
    def __init__(self,
                 image_dimensions:List[int],
                 min_patch_size:List[int],
                 max_patch_size:List[int],
                 n_patches:int=4,
                 seed:int=42):
        self.image_dimensions = image_dimensions
        self.min_patch_size = min_patch_size
        self.max_patch_size = max_patch_size
        self.seed = seed
        self.n_patches = n_patches

        self.rng = np.random.default_rng(seed)
        self.n_dim = len(self.image_dimensions)

        assert self.n_dim in [2,3]

    def sample_patch(self)->Coords:
        upper_sampling_bound = [
            size-patch_size 
            for size,patch_size in zip(self.image_dimensions,
                                       self.max_patch_size)]
        lower_bound = np.array(
            [self.rng.integers(0,i) for i in upper_sampling_bound])
        patch_size = np.array(
            [self.rng.integers(m,M) 
             for m,M in zip(self.min_patch_size,self.max_patch_size)])
        upper_bound = lower_bound + patch_size
        return [*lower_bound,*upper_bound]
    
    def sample_patches(self,n_patches:int)->List[Coords]:
        return [self.sample_patch() for _ in range(n_patches)]

    def retrieve_patch(self,
                       X:torch.Tensor,
                       patch_coords:List[int])->Tuple[torch.Tensor,List[int]]:
        upper_bound, lower_bound = (patch_coords[:self.n_dim],
                                    patch_coords[self.n_dim:])

        if self.n_dim == 2:
            x1,y1 = upper_bound
            x2,y2 = lower_bound
            X_patches = X[:,:,x1:x2,y1:y2]
            long_coords = list(product(range(x1,x2),range(y1,y2)))
        elif self.n_dim == 3:
            x1,y1,z1 = upper_bound
            x2,y2,z2 = lower_bound
            X_patches = X[:,:,x1:x2,y1:y2,z1:z2]
            long_coords = list(product(range(x1,x2),range(y1,y2),range(z1,z2)))
            
        return X_patches,long_coords

    def __call__(self,
                 X:torch.Tensor,
                 mask_vector:torch.Tensor=None,
                 patch_coords:List[Coords]=None)->Tuple[torch.Tensor,List[torch.Tensor]]:
        if patch_coords is None:
            patch_coords = self.sample_patches(self.n_patches)
        all_patches = []
        full_coords = [[] for _ in range(self.n_dim)]
        for coords in patch_coords:
            patch,long_coords = self.retrieve_patch(X,coords)
            all_patches.append(patch)
            for i in range(len(full_coords)):
                full_coords[i].extend(long_coords[i])
        if mask_vector is not None:
            mask_c = np.unique(np.array(full_coords),axis=0).T
            mask_vector = mask_vector[None,:,None]
            if self.n_dim == 2:
                X[:,:,mask_c[0],mask_c[1]] = mask_vector
            if self.n_dim == 3:
                X[:,:,mask_c[0],mask_c[1],mask_c[2]] = mask_vector
        return X,all_patches,patch_coords

def get_masker(model_type:str,*args,**kwargs):
    if model_type == "transformer":
        return TransformerMasker(*args,**kwargs)
    elif model_type == "convolutional":
        return ConvolutionalMasker(*args,**kwargs)
    else:
        raise NotImplemented(
            "model_type must be either 'transformer' or 'convolutional'")
