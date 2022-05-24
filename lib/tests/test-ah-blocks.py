import sys
sys.path.append("..")

import torch
from modules.segmentation import *

import numpy as np

i = torch.ones([1,1,32,32])
i_3d = torch.stack([i,i,i],-1)
c = AnysotropicHybridResidual(2,1,3)
output_2d = c.op(i)
c.convert_to_3d()
output_3d = c.op(i_3d)
c.convert_to_2d()
output_2d_rec = c.op(i)

print("Results for AH residual block:")
print("\tSame result in 3D:",torch.all(torch.isclose(output_2d,output_3d[:,:,:,:,1],atol=1e-6)))
print("\tSame result in 2D:",torch.all(torch.isclose(output_2d,output_2d_rec,atol=1e-6)))

i = torch.ones([1,1,32,32])
i_3d = torch.stack([i],-1)
c = AnysotropicHybridInput(2,1,16,7)
output_2d = c(i)
c.convert_to_3d()
output_3d = c(i_3d)

print("Results for AH input:")
print("\tSame result in 3D:",torch.all(output_2d.unsqueeze(-1) == output_3d))