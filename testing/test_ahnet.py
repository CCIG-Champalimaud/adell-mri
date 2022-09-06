import sys
import os
sys.path.append(os.path.join(os.path.dirname(os.path.abspath(__file__)), '..'))

import torch
from lib.modules.segmentation import AHNet

h,w,d = 128,128,32

def test_ahnet_2d():
      i = torch.rand(size=[1,1,h,w])
      a = AHNet(1,64,2,2,n_layers=3)
      o = a(i)

      assert (list(o.shape) == [1,1,h//2,w//2],
              "ahnet 2d output shape not correct")

def test_ahnet_3d():
      a = AHNet(1,64,2,2,n_layers=3)

      a.convert_to_3d()

      i_3d = torch.rand(size=[1,1,h,w,d])
      o_3d = a(i_3d)

      assert (list(o_3d.shape) == [1,1,h//2,w//2,d//2],
              "ahnet 3d output shape not correct")
