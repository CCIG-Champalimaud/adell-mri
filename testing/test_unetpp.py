import os
import sys
sys.path.append(os.path.join(os.path.dirname(os.path.abspath(__file__)),'..'))
import pytest

import torch
from lib.modules.segmentation.unetpp import *

h,w,d,c = 32,32,16,1

depths = [[16,32,64],[16,32,64,128]]
spatial_dims = [2,3]
conv_types = ["regular","resnet"]

param_list = []
for dim in [2,3]:
    for D in depths:
        for conv_type in ["regular","resnet"]:
            param_list.append((D,dim,conv_type))
@pytest.mark.parametrize("D,sd,conv_type",param_list)
def test_unetpp_base(D,sd,conv_type):
    K = [3 for _ in D]
    S = [2 for _ in D]
    if sd == 2:
        i = torch.rand(size=[1,c,h,w])
        out_shape = [1,1,h,w]
    elif sd == 3:
        i = torch.rand(size=[1,c,h,w,d])
        out_shape = [1,1,h,w,d]
    a = UNetPlusPlus(
        sd,depth=D,upscale_type="transpose",padding=1,
        strides=S,kernel_sizes=K,conv_type=conv_type,
        bottleneck_classification=True)
    o,o_aux,bb = a(i,return_aux=True)
    assert list(o.shape) == out_shape
    for x in o_aux:
        assert list(x.shape) == out_shape

param_list = []
for dim in [2,3]:
    for D in depths:
        for conv_type in ["regular","resnet"]:
            param_list.append((D,dim,conv_type))
@pytest.mark.parametrize("D,sd,conv_type",param_list)
def test_unetpp_base_skip(D,sd,conv_type):
    K = [3 for _ in D]
    S = [2 for _ in D]
    if sd == 2:
        i = torch.rand(size=[1,c,h,w])
        i_skip = torch.rand(size=[1,c,h,w])
        out_shape = [1,1,h,w]
    elif sd == 3:
        i = torch.rand(size=[1,c,h,w,d])
        i_skip = torch.rand(size=[1,c,h,w,d])
        out_shape = [1,1,h,w,d]
    a = UNetPlusPlus(
        sd,depth=D,upscale_type="transpose",padding=1,
        strides=S,kernel_sizes=K,conv_type=conv_type,
        bottleneck_classification=True,
        skip_conditioning=1)
    o,o_aux,bb = a(i,return_aux=True,X_skip_layer=i_skip)
    assert list(o.shape) == out_shape
    for x in o_aux:
        assert list(x.shape) == out_shape
