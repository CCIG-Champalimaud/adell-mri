import os
import sys
sys.path.append(os.path.join(os.path.dirname(os.path.abspath(__file__)),'..'))
import pytest

import torch
from lib.modules.diffusion.unet import DiffusionUNet

h,w,d,c = 32,32,20,1

depths = [[16,32,64],[16,32,64,128]]
spatial_dims = [2,3]

param_list = []
for dim in spatial_dims:
    for D in depths:
        for conv_type in ["regular","resnet"]:
            for strides in ["regular","irregular"]:
                param_list.append((D,dim,conv_type,strides))

@pytest.mark.parametrize("D,sd,conv_type,strides",param_list)
def test_unet_base(D,sd,conv_type,strides):
    K = [3 for _ in D]
    if strides == "irregular":
        if sd == 2:
            S = [[2,2],*[[2,1] for _ in D[1:]]]
        elif sd == 3:
            S = [[2,2,2],*[[2,2,1] for _ in D[1:]]]
    else:
        S = [2 for _ in D]
    if sd == 2:
        i = torch.rand(size=[1,c,h,w])
        output_size = [1,c,h,w]
    elif sd == 3:
        i = torch.rand(size=[1,c,h,w,d])
        output_size = [1,c,h,w,d]
    a = DiffusionUNet(
        spatial_dimensions=sd,depth=D,upscale_type="transpose",padding=1,n_channels=c,
        strides=S,kernel_sizes=K,conv_type=conv_type,link_type="identity")
    o = a(i)
    assert list(o.shape) == output_size

@pytest.mark.parametrize("D,sd,conv_type,strides",param_list)
def test_unet_class_base(D,sd,conv_type,strides):
    K = [3 for _ in D]
    if strides == "irregular":
        if sd == 2:
            S = [[2,2],*[[2,1] for _ in D[1:]]]
        elif sd == 3:
            S = [[2,2,2],*[[2,2,1] for _ in D[1:]]]
    else:
        S = [2 for _ in D]
    
    cls = torch.rand(size=[1,2])
    if sd == 2:
        i = torch.rand(size=[1,c,h,w])
        output_size = [1,c,h,w]
    elif sd == 3:
        i = torch.rand(size=[1,c,h,w,d])
        output_size = [1,c,h,w,d]
    a = DiffusionUNet(
        spatial_dimensions=sd,depth=D,upscale_type="transpose",padding=1,n_channels=c,
        strides=S,kernel_sizes=K,conv_type=conv_type,link_type="identity",
        classifier_free_guidance=True)
    o = a(i,cls)
    assert list(o.shape) == output_size
