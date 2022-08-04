import sys
sys.path.append("..")

import torch
from ..lib.modules.layers import *

h,w,d,c = 128,128,32,16
i = torch.rand(size=[1,c,h,w,d])
levels = [[4,4,2],[8,8,4],[16,16,8]]
p = PyramidSpatialPooling3d(16,levels)
o = p(i)

print("Shape is [1,{},{},{},{}]:".format(c*(len(levels)+1),h,w,d),
      list(o.shape) == [1,c*(len(levels)+1),h,w,d])