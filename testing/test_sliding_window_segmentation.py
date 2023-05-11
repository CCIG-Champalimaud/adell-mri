import sys
import os
sys.path.append(os.path.join(os.path.dirname(os.path.abspath(__file__)),'..'))

import torch
from lib.modules.segmentation.pl import UNetPL
from lib.utils.inference import SlidingWindowSegmentation

h,w,d,c = 32,32,32,1

depth = [16,32,64]
kernel_sizes = [3 for _ in depth]
strides = [2 for _ in depth]
spatial_dims = [2,3]

def test_sliding_window_inference():
    unet = UNetPL(spatial_dimensions=3,
                  depth=depth,
                  kernel_sizes=kernel_sizes,
                  strides=strides,
                  interpolation="trilinear",
                  padding=1)
    input_tensor = torch.ones([1,c,h,w,d])
    sli = SlidingWindowSegmentation(
        [8,8,8],
        lambda x: unet.forward(x)[0],n_classes=1)
    output = sli(input_tensor)
    assert list(output.shape) == [1,1,h,w,d]
    assert output.max() <= 1
    assert output.min() > 0

def test_sliding_window_inference_pl():
    unet = UNetPL(spatial_dimensions=3,
                  depth=depth,
                  kernel_sizes=kernel_sizes,
                  strides=strides,
                  interpolation="trilinear",
                  padding=1)
    input_dictionary = {"image":torch.ones([1,c,h,w,d])}
    sli = SlidingWindowSegmentation(
        [8,8,8],
        lambda x: unet.predict_step(x)[0],n_classes=1)
    output = sli(input_dictionary)
    assert list(output.shape) == [1,1,h,w,d]
    assert output.max() <= 1
    assert output.min() > 0