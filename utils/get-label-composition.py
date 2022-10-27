import argparse
import os
import nibabel as nib
import SimpleITK as sitk
import re
import numpy as np
from glob import glob
from tqdm import tqdm

desc = """
Counts which classes are present in a folder containing segmentation maps.
"""

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=desc)

    parser.add_argument(
        "--input_path",dest="input_path",type=str,
        help="Path to directory containing masks.",required=True)
    parser.add_argument(
        "--pattern",dest="pattern",type=str,
        default='*',
        help="Pattern to match masks")

    args = parser.parse_args()

    total_voxels = {}
    for path in tqdm(glob(os.path.join(args.input_path,args.pattern))):
        path_sub = path.split(os.sep)[-1]
        image = sitk.GetArrayFromImage(sitk.ReadImage(path))
        un,co = np.unique(image,return_counts=True)
        for u,c in zip(un,co):
            if u not in total_voxels:
                total_voxels[u] = []
            total_voxels[u].append(c)

    total_im = sum([len(total_voxels[u]) for u in total_voxels])
    total_vo = sum([np.sum(total_voxels[u]) for u in total_voxels])
    for u in total_voxels:
        print(u,len(total_voxels[u]),np.sum(total_voxels[u]),
              len(total_voxels[u])/total_im,np.sum(total_voxels[u])/total_vo)