import sys
sys.path.append("..")

import torch
from modules.segmentation import *

h,w,d = 128,128,32
i = torch.rand(size=[1,1,h,w])
a = AHNet(1,64,2,2,n_layers=3)
o = a(i)

print("Shape ([1,1,{},{}]) is correct for 2D operation:".format(h//2,w//2),
      list(o.shape) == [1,1,h//2,w//2])

a.convert_to_3d()

print("Conversion to 3D successful")

i_3d = torch.rand(size=[1,1,h,w,d])
o_3d = a(i_3d)

print("Shape ([1,1,{},{},{}]) is correct for 3D operation:".format(h//2,w//2,d//2),
      list(o_3d.shape) == [1,1,h//2,w//2,d//2])
