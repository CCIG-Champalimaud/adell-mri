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

    for path in tqdm(glob(os.path.join(args.input_path,args.pattern))):
        path_sub = path.split(os.sep)[-1]
        image = sitk.GetArrayFromImage(sitk.ReadImage(path))
        print(path_sub+','+':'.join([str(x) for x in np.unique(image)]))