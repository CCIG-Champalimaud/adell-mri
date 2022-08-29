import os, sys
sys.path.append(os.path.join(os.path.dirname(os.path.abspath(__file__)), '..'))

import torch
from lib.modules.segmentation_plus import *

h,w,d,c = 128,128,32,1

depths = [[16,32,64],[16,32,64,128],[16,32,64,128,256]]
spatial_dims = [2,3]
for D in depths:
    K = [3 for _ in D]
    S = [2 for _ in D]
    for sd in spatial_dims:
        for conv_type in ["regular","resnet"]:
            print("Testing {}D with depth={} and conv_type={}".format(sd,D,conv_type))
            if sd == 2:
                i = torch.rand(size=[1,c,h,w])
            elif sd == 3:
                i = torch.rand(size=[1,c,h,w,d])
            a = UNetPlusPlus(
                sd,depth=D,upscale_type="transpose",padding=1,
                strides=S,kernel_sizes=K,conv_type=conv_type,
                bottleneck_classification=True)
            o,o_aux,bb = a(i,return_aux=True)
            print("\tOutput shape:")
            print("\t\t{}".format(o.shape))
            print("\tAux. output shape:")
            for x in o_aux:
                print("\t\t{}".format(x.shape))

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
            a = UNetPlusPlus(
                sd,depth=D,upscale_type="transpose",padding=1,
                strides=S,kernel_sizes=K,conv_type=conv_type,
                bottleneck_classification=True,
                skip_conditioning=1)
            o,o_aux,bb = a(i,return_aux=True,X_skip_layer=i_skip)
            print("\tOutput shape:")
            print("\t\t{}".format(o.shape))
            print("\tAux. output shape:")
            for x in o_aux:
                print("\t\t{}".format(x.shape))