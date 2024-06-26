import sys
import os

sys.path.append(os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))

import torch
from adell_mri.modules.segmentation.pl import UNetPL
from adell_mri.utils.inference import (
    SlidingWindowSegmentation,
    SegmentationInference,
)

h, w, d, c = 32, 32, 32, 1

depth = [16, 32, 64]
kernel_sizes = [3 for _ in depth]
strides = [2 for _ in depth]
spatial_dims = [2, 3]


def test_sliding_window_inference():
    unet = torch.nn.Identity()
    input_tensor = torch.rand([1, c, h, w, d])
    sli = SlidingWindowSegmentation(
        [8, 8, 8], lambda x: unet.forward(x), n_classes=1
    )
    output = sli(input_tensor)
    assert list(output.shape) == [1, 1, h, w, d]
    assert output.max() <= 1
    assert output.min() >= 0
    assert torch.all(torch.isclose(output, input_tensor))


def test_segmentation_inference():
    unet = torch.nn.Identity()
    input_tensor = torch.rand([1, c, h, w, d])
    sli = SegmentationInference(
        base_inference_function=lambda x: unet.forward(x),
        sliding_window_size=[8, 8, 8],
        n_classes=1,
        flip=True,
    )
    output = sli(input_tensor)
    assert list(output.shape) == [1, 1, h, w, d]
    assert output.max() <= 1
    assert output.min() >= 0
    assert torch.all(torch.isclose(output, input_tensor))


def test_sliding_window_inference_pl():
    unet = UNetPL(
        spatial_dimensions=3,
        depth=depth,
        kernel_sizes=kernel_sizes,
        strides=strides,
        interpolation="trilinear",
        padding=1,
    )
    input_dictionary = {"image": torch.ones([1, c, h, w, d])}
    sli = SlidingWindowSegmentation(
        [8, 8, 8], lambda x: unet.predict_step(x)[0], n_classes=1
    )
    output = sli(input_dictionary)
    assert list(output.shape) == [1, 1, h, w, d]
    assert output.max() <= 1
    assert output.min() >= 0


def test_segmentation_inference_pl():
    unet = UNetPL(
        spatial_dimensions=3,
        depth=depth,
        kernel_sizes=kernel_sizes,
        strides=strides,
        interpolation="trilinear",
        padding=1,
    )
    input_dictionary = {"image": torch.ones([1, c, h, w, d])}
    sli = SegmentationInference(
        base_inference_function=lambda x: unet.predict_step(x)[0],
        sliding_window_size=[8, 8, 8],
        n_classes=1,
        flip=True,
    )
    output = sli(input_dictionary)
    assert list(output.shape) == [1, 1, h, w, d]
    assert output.max() <= 1
    assert output.min() >= 0


def test_segmentation_inference_dropout_pl():
    unet = UNetPL(
        spatial_dimensions=3,
        depth=depth,
        kernel_sizes=kernel_sizes,
        strides=strides,
        interpolation="trilinear",
        padding=1,
    )
    input_dictionary = {"image": torch.ones([1, c, h, w, d])}
    sli = SegmentationInference(
        base_inference_function=lambda x: unet.predict_step(x)[0],
        sliding_window_size=[8, 8, 8],
        n_classes=1,
        flip=True,
        mc_iterations=2,
    )
    output = sli(input_dictionary)
    assert list(output.shape) == [1, 2, h, w, d]
    assert output.max() <= 1
    assert output.min() >= 0
