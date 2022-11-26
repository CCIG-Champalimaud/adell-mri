import os
import sys
sys.path.append(os.path.join(os.path.dirname(os.path.abspath(__file__)),'..'))
import pytest

import torch
from lib.modules.segmentation.unetr import UNETR
from copy import deepcopy
h,w,d,c = 32,32,16,1

depths = [[16,32,64],[16,32,64,128]]
spatial_dims = [2,3]

def get_vit_params():
    return {
        "image_size":[h,w,d],
        "patch_size":[8,8,8],
        "number_of_blocks":8,
        "attention_dim":64,
        "hidden_dim":64,
        "return_at":[2,4,5],
        "n_heads":8,
        "dropout_rate":0.1,
        "embed_method":"linear",
    }

param_list = []
for dim in [2,3]:
    for D in depths:
        for conv_type in ["regular","resnet"]:
            param_list.append((D,dim,conv_type))

@pytest.mark.parametrize("D,sd,conv_type",param_list)
def test_unetr_base(D,sd,conv_type):
    K = [3 for _ in D]
    if sd == 2:
        i = torch.rand(size=[1,c,h,w])
        output_size = [1,1,h,w]
    elif sd == 3:
        i = torch.rand(size=[1,c,h,w,d])
        output_size = [1,1,h,w,d]
        
    vit_params = get_vit_params()
    vit_params["image_size"] = vit_params["image_size"][:sd]
    vit_params["patch_size"] = vit_params["patch_size"][:sd]
    vit_params["return_at"] = vit_params["return_at"][:(len(D)-1)]
    
    a = UNETR(**vit_params,spatial_dimensions=sd,
              depth=D,upscale_type="transpose",padding=1,
              kernel_sizes=K,conv_type=conv_type,link_type="identity")
    o,bb = a(i)
    assert list(o.shape) == output_size

param_list = []
for dim in [2,3]:
    for D in depths:
        for conv_type in ["regular","resnet"]:
            param_list.append((D,dim,conv_type))

@pytest.mark.parametrize("D,sd,conv_type",param_list)
def test_unetr_skip(D,sd,conv_type):
    K = [3 for _ in D]
    S = [2 for _ in D]
    if sd == 2:
        i = torch.rand(size=[1,c,h,w])
        i_skip = torch.rand(size=[1,c,h,w])
        output_size = [1,1,h,w]
    elif sd == 3:
        i = torch.rand(size=[1,c,h,w,d])
        i_skip = torch.rand(size=[1,c,h,w,d])
        output_size = [1,1,h,w,d]
        
    vit_params = get_vit_params()
    vit_params["image_size"] = vit_params["image_size"][:sd]
    vit_params["patch_size"] = vit_params["patch_size"][:sd]
    vit_params["return_at"] = vit_params["return_at"][:(len(D)-1)]

    a = UNETR(**vit_params,spatial_dimensions=sd,
              depth=D,upscale_type="transpose",padding=1,
              kernel_sizes=K,skip_conditioning=1,
              link_type="conv",conv_type=conv_type)
    o,bb = a(i,X_skip_layer=i_skip)
    assert list(o.shape) == output_size

param_list = []
for dim in [2,3]:
    for D in depths:
        param_list.append((D,dim))

@pytest.mark.parametrize("D,sd",param_list)
def test_unetr_skip_feature(D,sd):
    n_features = 4
    K = [3 for _ in D]
    S = [2 for _ in D]
    if sd == 2:
        i = torch.rand(size=[2,c,h,w])
        i_skip = torch.rand(size=[2,1,h,w])
        i_feat = torch.rand(size=[2,n_features])
        output_size = [2,1,h,w]
    elif sd == 3:
        i = torch.rand(size=[2,c,h,w,d])
        i_skip = torch.rand(size=[2,1,h,w,d])
        i_feat = torch.rand(size=[2,n_features])
        output_size = [2,1,h,w,d]

    vit_params = get_vit_params()
    vit_params["image_size"] = vit_params["image_size"][:sd]
    vit_params["patch_size"] = vit_params["patch_size"][:sd]
    vit_params["return_at"] = vit_params["return_at"][:(len(D)-1)]

    a = UNETR(**vit_params,spatial_dimensions=sd,
              depth=D,upscale_type="transpose",padding=1,
              kernel_sizes=K,feature_conditioning=n_features,
              skip_conditioning=1,
              feature_conditioning_params={
              "mean":torch.zeros_like(i_feat),
              "std":torch.ones_like(i_feat)},
              link_type="conv")
    o,bb = a(i,X_feature_conditioning=i_feat,X_skip_layer=i_skip)
    assert list(o.shape) == output_size
