import os
import sys

sys.path.append(os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
import time
from typing import List, Tuple, Union

import monai
import numpy as np
import torch


class RandomAffined(monai.transforms.RandomizableTransform):
    def __init__(
        self,
        keys: List[str],
        spatial_sizes: List[Union[Tuple[int, int, int], Tuple[int, int]]],
        mode: List[str],
        prob: float = 0.1,
        rotate_range: Union[Tuple[int, int, int], Tuple[int, int]] = [0, 0, 0],
        shear_range: Union[Tuple[int, int, int], Tuple[int, int]] = [0, 0, 0],
        translate_range: Union[Tuple[int, int, int], Tuple[int, int]] = [
            0,
            0,
            0,
        ],
        scale_range: Union[Tuple[int, int, int], Tuple[int, int]] = [0, 0, 0],
        device: "str" = "cpu",
    ):

        self.keys = keys
        self.spatial_sizes = [
            np.array(s, dtype=np.int32) for s in spatial_sizes
        ]
        self.mode = mode
        self.prob = prob
        self.rotate_range = np.array(rotate_range)
        self.shear_range = np.array(shear_range)
        self.translate_range = np.array(translate_range)
        self.scale_range = np.array(scale_range)
        self.device = device

        self.affine_trans = {
            k: monai.transforms.Affine(
                spatial_size=s, mode=m, device=self.device
            )
            for k, s, m in zip(self.keys, self.spatial_sizes, self.mode)
        }

        self.get_translation_adjustment()

    def get_random_parameters(self):
        angle = self.R.uniform(-self.rotate_range, self.rotate_range)
        shear = self.R.uniform(-self.shear_range, self.shear_range)
        trans = self.R.uniform(-self.translate_range, self.translate_range)
        scale = self.R.uniform(1 - self.scale_range, 1 + self.scale_range)

        return angle, shear, trans, scale

    def get_translation_adjustment(self):
        # we have to adjust the translation to ensure that all inputs
        # do not become misaligned. to do this I assume that the first image
        # is the reference
        ref_size = self.spatial_sizes[0]
        self.trans_adj = {
            k: s / ref_size for k, s in zip(self.keys, self.spatial_sizes)
        }

    def randomize(self):
        angle, shear, trans, scale = self.get_random_parameters()
        for k in self.affine_trans:
            # we only need to update the affine grid
            self.affine_trans[k].affine_grid = monai.transforms.AffineGrid(
                rotate_params=list(angle),
                shear_params=list(shear),
                translate_params=list(trans),
                scale_params=list(np.float32(scale * self.trans_adj[k])),
                device=self.device,
            )

    def __call__(self, data):
        self.randomize()
        for k in self.keys:
            if self.R.uniform() < self.prob:
                transform = self.affine_trans[k]
                data[k], data[k + "_affine"] = transform(data[k])
        return data


input_shape = [1, 128, 128, 16]
input_shape_ds = [1, 64, 64, 8]

data = {"a": torch.rand(input_shape), "b": torch.rand(input_shape_ds)}

t = monai.transforms.RandAffined(
    keys=["a", "b"],
    mode=["bilinear", "nearest"],
    prob=1.0,
    rotate_range=[np.pi / 6, np.pi / 6, np.pi / 6],
    shear_range=[0, 0, 0],
    translate_range=[10, 10, 3],
    scale_range=[0.1, 0.1, 0.1],
)

t_1 = time.time()
o = t(data)
t_2 = time.time()

print("Result from MONAI:", o["a"].shape, o["b"].shape)
print("\tTime elapsed:", t_2 - t_1)

t = RandomAffined(
    keys=["a", "b"],
    spatial_sizes=[input_shape[1:], input_shape_ds[1:]],
    mode=["bilinear", "nearest"],
    prob=1.0,
    rotate_range=[np.pi / 6, np.pi / 6, np.pi / 6],
    shear_range=[0, 0, 0],
    translate_range=[10, 10, 3],
    scale_range=[0.1, 0.1, 0.1],
)

t_1 = time.time()
o = t(data)
t_2 = time.time()

print("Result from own implementaion", o["a"].shape, o["b"].shape)
print("\tTime elapsed:", t_2 - t_1)
