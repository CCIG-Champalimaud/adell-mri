import sys
import os
sys.path.append(os.path.join(os.path.dirname(os.path.abspath(__file__)), '..'))

import torch
from lib.modules.layers import DenseBlock,ActDropNorm

c,h,w,d = [1,32,32,16]
adn_fn_2d = ActDropNorm
adn_fn_3d = lambda s: ActDropNorm(s,norm_fn=torch.nn.BatchNorm3d)

print("Testing case of a single convolution (no dense connections)")
structure = [1,16]
D = DenseBlock(2,structure,3,adn_fn_2d)
input_tensor = torch.rand(size=[1,c,h,w])
D(input_tensor)
print("\tSuccess")

print("Testing 2D case")
structure = [1,16,16,16]
D = DenseBlock(2,structure,3,adn_fn_2d)
input_tensor = torch.rand(size=[1,c,h,w])
D(input_tensor)
print("\tSuccess")

print("Testing 2D case with increasing structure")
structure = [1,16,32,48]
D = DenseBlock(2,structure,3,adn_fn_2d)
input_tensor = torch.rand(size=[1,c,h,w])
D(input_tensor)
print("\tSuccess")

print("Testing 3D case with increasing structure")
structure = [1,16,32,48]
D = DenseBlock(3,structure,3,adn_fn_3d)
input_tensor = torch.rand(size=[1,c,h,w,d])
D(input_tensor)
print("\tSuccess")

print("Testing skip connections")
structure = [1,16,32,16]
structure_skip = [16,16]
D = DenseBlock(3,structure,3,adn_fn_3d,structure_skip=structure_skip)
input_tensor = torch.rand(size=[1,c,h,w,d])
input_skips = [torch.rand(size=[1,s,h,w,d]) for s in structure_skip]
o = D(input_tensor,input_skips)
print("\tSuccess")
print("\tOutput shape = {}".format(o.shape))