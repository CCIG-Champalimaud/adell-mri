import sys
import os
sys.path.append(os.path.join(os.path.dirname(os.path.abspath(__file__)), '..'))

import torch
from lib.modules.layers import ResidualBlock2d,ResidualBlock3d,ActDropNorm,BatchEnsemble

c,h,w,d = [1,32,32,16]
adn_fn_2d = ActDropNorm
adn_fn_3d = lambda s: ActDropNorm(s,norm_fn=torch.nn.BatchNorm3d)

print("Testing 2D case")
D = BatchEnsemble(5,ResidualBlock2d(1,3,32,32,adn_fn=adn_fn_2d),c,32)
input_tensor = torch.rand(size=[1,c,h,w])
D(input_tensor)
D(input_tensor,1)
D.training = False
D(input_tensor)
print("\tSuccess")

print("Testing 3D case")
D = BatchEnsemble(5,ResidualBlock3d(1,3,32,32,adn_fn=adn_fn_3d),c,32)
input_tensor = torch.rand(size=[1,c,h,w,d])
D(input_tensor)
D(input_tensor,1)
D.training = False
D(input_tensor)
print("\tSuccess")
