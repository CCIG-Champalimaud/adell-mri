import os, sys
sys.path.append(os.path.join(os.path.dirname(os.path.abspath(__file__)), '..'))

import torch
from lib.modules.segmentation_plus import *

h,w,d,c = 128,128,32,1

depths = [[16,32,64,128],[16,32,64,128,256]]
spatial_dims = [2,3]
conv_types = ["regular","resnet"]

def unetpp_base(D,sd,conv_type):
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

def unetpp_base_skip(D,sd,conv_type):
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

def test_unet_2d():
    for d in depths:
        for c in conv_types:
            unetpp_base(d,2,c)

def test_unet_3d():
    for d in depths:
        for c in conv_types:
            unetpp_base(d,3,c)

def test_unet_2d_skip():
    for d in depths:
        for c in conv_types:
            unetpp_base_skip(d,2,c)

def test_unet_3d_skip():
    for d in depths:
        for c in conv_types:
            unetpp_base_skip(d,3,c)
