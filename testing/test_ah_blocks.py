import sys
import os
sys.path.append(os.path.join(os.path.dirname(os.path.abspath(__file__)), '..'))

import torch
from lib.modules.layers import AnysotropicHybridResidual,AnysotropicHybridInput

import numpy as np

def test_ah_res():
    i = torch.ones([1,1,32,32])
    i_3d = torch.stack([i,i,i],-1)
    c = AnysotropicHybridResidual(2,1,3)
    output_2d = c.op(i)
    c.convert_to_3d()
    output_3d = c.op(i_3d)
    c.convert_to_2d()
    output_2d_rec = c.op(i)

    assert (torch.all(torch.isclose(output_2d,output_3d[:,:,:,:,1],atol=1e-6)),
            "ah_res 2d and 3d output not similar")
    assert (torch.all(torch.isclose(output_2d,output_2d_rec,atol=1e-6)),
            "ah_res 2d and rec 2d output not similar")

def test_ah_input():
    i = torch.ones([1,1,32,32])
    i_3d = torch.stack([i],-1)
    c = AnysotropicHybridInput(2,1,16,7)
    output_2d = c(i)
    c.convert_to_3d()
    output_3d = c(i_3d)

    assert (torch.all(output_2d.unsqueeze(-1) == output_3d),
            "ah_input 2d and 3d output not similar")