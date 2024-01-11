import sys
import os

sys.path.append(os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))

import numpy as np
import torch
from adell_mri.utils.monai_transforms import GetAllCrops, GetAllCropsd
from adell_mri.utils.utils import safe_collate_crops

input_tensor_size = np.array([1, 128, 128, 16])
crop_size = np.array([32, 32, 8])
n_crops = np.prod(input_tensor_size[1:] / crop_size)


def test_gac():
    input_tensor = torch.zeros(input_tensor_size.tolist())
    gac = GetAllCrops(size=crop_size)
    assert len(gac(input_tensor)) == n_crops


def test_gacd():
    input_tensor = torch.zeros(input_tensor_size.tolist())
    gac = GetAllCropsd(keys=["image"], size=crop_size)
    assert len(gac({"image": input_tensor})) == n_crops


def test_gacd_and_collate():
    input_tensor = torch.zeros(input_tensor_size.tolist())
    gac = GetAllCropsd(keys=["image"], size=crop_size)
    out = safe_collate_crops([gac({"image": input_tensor})])
    assert out["image"].shape[0] == n_crops
