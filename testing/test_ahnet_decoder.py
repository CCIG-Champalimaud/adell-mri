import sys
import os
sys.path.append(os.path.join(os.path.dirname(os.path.abspath(__file__)), '..'))

import torch
from lib.modules.layers import AHNetDecoder3d

def test_ah_net_decoder():
      h,w,d,c = 128,128,32,16
      i = torch.rand(size=[1,c,h,w,d])
      a = AHNetDecoder3d(c)
      o = a(i)

      assert (list(o.shape) == [1,c,h,w,d],
            "output shape is not {}".format([1,c,h,w,d]))