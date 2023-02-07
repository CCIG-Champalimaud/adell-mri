import sys
import os
sys.path.append(os.path.join(os.path.dirname(os.path.abspath(__file__)),'..'))
import time
import numpy as np
import argparse
import torch
import monai
from tqdm import tqdm

from lib.modules.augmentations import AugmentationWorkhorsed
from lib.modules.augmentations import (
    generic_augments,mri_specific_augments,spatial_augments)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    
    parser.add_argument("--transforms",default=None,nargs="+")
    parser.add_argument("--n_iter",default=100,type=int)
    
    args = parser.parse_args()
    
    if args.transforms is None:
        all_augments = generic_augments + mri_specific_augments + spatial_augments
    else:
        all_augments = args.transforms

    all_augments = [k for k in all_augments if "_z" not in k]
    ah = AugmentationWorkhorsed(all_augments,["image"],N=2,
                                dropout_size=(32,32))

    N = args.n_iter
    
    time_init = time.time()
    for k in ah.transforms:
        I = {"image":np.random.rand(1,256,256)}
        a = time.time()
        with tqdm(range(N)) as pbar:
            for i in pbar:
                pbar.set_description("transform={}; iteration={}".format(
                    k,i
                ))
                out = ah.transforms[k](I)
                print(out)
        b = time.time()
        print(
            "Transform: {}; average time per transform: {:.5f}s".format(
                k,(b-a)/N))
    time_end = time.time()
    
    time_elapsed = time_end-time_init
    print("Total elapsed time:",time_elapsed)
    print("Total number of transforms:",len(ah.transforms)*N)
    print("Average time per transform:",time_elapsed/(len(ah.transforms)*N))