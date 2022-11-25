import sys
import os
sys.path.append(os.path.join(os.path.dirname(os.path.abspath(__file__)),'..'))

import torch
from lib.modules.layers.self_attention import (ConcurrentSqueezeAndExcite2d,
                                               ConcurrentSqueezeAndExcite3d)

b,c,h,w,d = 1,16,64,64,16
input_tensor_2d = torch.ones([b,c,h,w])
input_tensor_3d = torch.ones([b,c,h,w,d])

def test_sae_2d():
    sae = ConcurrentSqueezeAndExcite2d(c)
    output = sae(input_tensor_2d)
    assert list(output.shape) == list(input_tensor_2d.shape)

def test_sae_3d():
    sae = ConcurrentSqueezeAndExcite3d(c)
    output = sae(input_tensor_3d)
    assert list(output.shape) == list(input_tensor_3d.shape)

test_sae_2d()
test_sae_3d()