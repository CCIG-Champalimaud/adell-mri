import sys
import os

sys.path.append(os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
import time
import numpy as np
import torch
import argparse
from tqdm import trange

from adell_mri.modules.augmentations import AugmentationWorkhorsed
from adell_mri.modules.augmentations import (
    generic_augments,
    mri_specific_augments,
    spatial_augments,
)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument("--transforms", default=None, nargs="+")
    parser.add_argument("--n_iter", default=100, type=int)

    args = parser.parse_args()

    if args.transforms is None:
        all_augments = (
            generic_augments + mri_specific_augments + spatial_augments
        )
    else:
        all_augments = args.transforms

    print(all_augments)
    ah = AugmentationWorkhorsed(all_augments, ["image"], N=2)

    I = {"image": np.random.rand(1, 256, 256, 16)}

    N = args.n_iter

    for k in ah.transforms:
        print(f"Benchmarking {k}")
        a = time.time()
        for _ in trange(N):
            ah.transforms[k](I)
        b = time.time()
        print((b - a) / N)
