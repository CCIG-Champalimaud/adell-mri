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
