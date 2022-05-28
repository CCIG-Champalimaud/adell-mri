import sys
sys.path.append("..")

import torch
from modules.segmentation import *

h,w,d,c = 128,128,32,16
i = torch.rand(size=[1,c,h,w,d])
a = AHNetDecoder3d(c)
o = a(i)

print("Shape is [1,{},{},{},{}]:".format(c,h,w,d),
      list(o.shape) == [1,c,h,w,d])