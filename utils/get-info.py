import os
import argparse
import numpy as np
import SimpleITK as sitk
from glob import glob
from tqdm import tqdm

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Returns the minimum pixel spacing of a folder")

    parser.add_argument('--input_dir',dest='input_dir')
    parser.add_argument('--pattern',dest='pattern',default="*nii.gz")
    parser.add_argument('--parameter',dest='parameter',default="spacing",
                        choices=["spacing","size"])
    parser.add_argument('--quantile',dest='quantile',default=0.5,type=float)
    args = parser.parse_args()

    all_info = []
    for p in tqdm(glob(os.path.join(args.input_dir,args.pattern))):
        x = sitk.ReadImage(p)
        if args.parameter == "spacing":
            info = x.GetSpacing()
        if args.parameter == "size":
            info = x.GetSize()
        all_info.append(info)
    
    print(
        ','.join([str(x) for x in np.quantile(all_info,args.quantile,axis=0)]))