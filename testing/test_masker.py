import sys
import os

sys.path.append(os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
import pytest
from itertools import product

import numpy as np
import torch
from adell_mri.utils.masking import get_masker

in_channels = 16
mask_vector = torch.zeros(in_channels)
params = list([x for x in product([2, 3], [False, True])])


@pytest.mark.parametrize("n_dim,with_mask", params)
def test_masker_transformer(n_dim, with_mask):
    if n_dim == 2:
        image_dimensions = [32, 32]
        min_patch_size = [1, 1]
        max_patch_size = [4, 4]
    elif n_dim == 3:
        image_dimensions = [32, 32, 16]
        min_patch_size = [1, 1, 1]
        max_patch_size = [4, 4, 4]
    masker = get_masker(
        "transformer",
        image_dimensions=image_dimensions,
        min_patch_size=min_patch_size,
        max_patch_size=max_patch_size,
    )
    n_tokens = int(np.prod(image_dimensions))
    if with_mask == True:
        masker(torch.ones(1, n_tokens, in_channels), mask_vector)
    else:
        masker(torch.ones(1, n_tokens, in_channels))


@pytest.mark.parametrize("n_dim,with_mask", params)
def test_masker_convolutional(n_dim, with_mask):
    if n_dim == 2:
        image_dimensions = [32, 32]
        min_patch_size = [1, 1]
        max_patch_size = [4, 4]
    elif n_dim == 3:
        image_dimensions = [32, 32, 16]
        min_patch_size = [1, 1, 1]
        max_patch_size = [4, 4, 4]
    masker = get_masker(
        "convolutional",
        image_dimensions=image_dimensions,
        min_patch_size=min_patch_size,
        max_patch_size=max_patch_size,
    )
    if with_mask == True:
        masker(torch.ones(1, in_channels, *image_dimensions), mask_vector)
    else:
        masker(torch.ones(1, in_channels, *image_dimensions))
