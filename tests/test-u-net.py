import os, sys
sys.path.append(os.path.join(os.path.dirname(os.path.abspath(__file__)), '..'))

import torch
from lib.modules.segmentation import *

h,w,d,c = 128,128,32,1

depths = [[16,32,64],[16,32,64,128]]
spatial_dims = [2,3]
for D in depths:
    K = [3 for _ in D]
    S = [2 for _ in D]
    for sd in spatial_dims:
        print("Testing {}D with depth={}".format(sd,D))
        if sd == 2:
            i = torch.rand(size=[1,c,h,w])
        elif sd == 3:
            i = torch.rand(size=[1,c,h,w,d])
        a = UNet(sd,depth=D,upscale_type="transpose",padding=1,
                 strides=S,kernel_sizes=K)
        o = a(i)

for D in depths:
    K = [3 for _ in D]
    S = [2 for _ in D]
    for sd in spatial_dims:
        for conv_type in ["regular","resnet"]:
            print("Testing {}D with depth={} and conv_type={} and skip_layer_conditioning".format(
                sd,D,conv_type))
            if sd == 2:
                i = torch.rand(size=[1,c,h,w])
                i_skip = torch.rand(size=[1,c,h,w])
            elif sd == 3:
                i = torch.rand(size=[1,c,h,w,d])
                i_skip = torch.rand(size=[1,c,h,w,d])
            a = UNet(sd,depth=D,upscale_type="transpose",padding=1,
                 strides=S,kernel_sizes=K,skip_conditioning=1,
                 link_type="conv")
            o,bb = a(i,X_skip_layer=i_skip)
            print("\tOutput shape:")
            print("\t\t{}".format(o.shape))

n_features = 4
for D in depths:
    K = [3 for _ in D]
    S = [2 for _ in D]
    for sd in spatial_dims:
        for conv_type in ["regular","resnet"]:
            print("Testing {}D with depth={} and conv_type={} and and skip layer and feature conditioning".format(
                sd,D,conv_type))
            if sd == 2:
                i = torch.rand(size=[2,c,h,w])
                i_skip = torch.rand(size=[2,1,h,w])
                i_feat = torch.rand(size=[2,n_features])
            elif sd == 3:
                i = torch.rand(size=[2,c,h,w,d])
                i_skip = torch.rand(size=[2,1,h,w,d])
                i_feat = torch.rand(size=[2,n_features])
            a = UNet(sd,depth=D,upscale_type="transpose",padding=1,
                 strides=S,kernel_sizes=K,feature_conditioning=n_features,
                 skip_conditioning=1,
                 feature_conditioning_params={
                    "mean":torch.zeros_like(i_feat),
                    "std":torch.ones_like(i_feat)},
                 link_type="conv")
            o,bb = a(i,X_feature_conditioning=i_feat,X_skip_layer=i_skip)
            print("\tOutput shape:")
            print("\t\t{}".format(o.shape))
