import os
import argparse
import random
import yaml
import numpy as np
import torch
import monai
import SimpleITK as sitk
from tqdm import tqdm

from lib.utils import (
    get_prostatex_path_dictionary,
    get_size_spacing_dict)

def mkdir(path):
    try: os.makedirs(path)
    except Exception as e: pass

def resample_image(img,output_spacing=[1,1,1],is_label=False):
    # change orientation
    img = sitk.DICOMOrient(img,'RAS')
    # set up resampling operation
    resample = sitk.ResampleImageFilter()
    if is_label == True:
        resample.SetInterpolator(sitk.sitkNearestNeighbor)
    else:
        resample.SetInterpolator(sitk.sitkBSpline)
    resample.SetOutputDirection(img.GetDirection())
    resample.SetOutputOrigin(img.GetOrigin())
    resample.SetOutputSpacing(output_spacing)

    orig_size = np.array(img.GetSize(), dtype=int)
    orig_spacing = np.array(img.GetSpacing())
    new_size = orig_size*(orig_spacing/np.array(output_spacing))
    new_size = np.ceil(new_size).astype(int)
    new_size = [int(s) for s in new_size]
    resample.SetSize(new_size)

    output_img = resample.Execute(img)

    return output_img

def get_crop_coords(V,crop_dim):
    cx,cy,cz = crop_dim
    h,w,d = V.GetWidth(),V.GetHeight(),V.GetDepth()
    c = np.maximum([h-cx,w-cy,d-cz],0)
    c1 = c[0]//2,c[1]//2,c[2]//2
    c2 = c[0]-c1[0],c[1]-c1[1],c[2]-c1[2]

    return c1,c2

def get_pad_coords(V,pad_dim):
    # assumes shape of V is equal or smaller than pad_dim
    cx,cy,cz = pad_dim
    h,w,d = V.GetWidth(),V.GetHeight(),V.GetDepth()
    p = cx-h,cy-w,cz-d
    p1 = p[0]//2,p[1]//2,p[2]//2
    p2 = p[0]-p1[0],p[1]-p1[1],p[2]-p1[2]

    return p1,p2

def crop(V,lower,upper):
    lower = tuple([int(x) for x in lower])
    upper = tuple([int(x) for x in upper])
    C = sitk.CropImageFilter()
    C.SetLowerBoundaryCropSize(lower)
    C.SetUpperBoundaryCropSize(upper)
    cropped_image = C.Execute(V)

    return cropped_image

def pad(V,lower,upper):
    lower = tuple([int(x) for x in lower])
    upper = tuple([int(x) for x in upper])
    P = sitk.ConstantPadImageFilter()
    P.SetPadLowerBound(lower)
    P.SetPadUpperBound(upper)
    cropped_image = P.Execute(V)

    return cropped_image

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="MONAI-based utility to standardize scan spacing and size.")

    # data
    parser.add_argument(
        '--base_path',dest='base_path',type=str,
        help="Path to ProstateX dataset",required=True)
    parser.add_argument(
        '--output_path',dest='output_path',type=str,
        help="Path to output directory",required=True)

    args = parser.parse_args()

    mkdir(args.output_path)

    keys_dwi = [
        "DWI","DWI_gland_segmentations","DWI_lesion_segmentations"]
    keys_t2wax = [
        "T2WAx","T2WAx_gland_segmentations","T2WAx_lesion_segmentations"]

    path_dictionary = get_prostatex_path_dictionary(args.base_path)
    size_dict,spacing_dict = get_size_spacing_dict(
        path_dictionary,["DWI","T2WAx"])

    for k in size_dict:
        size_dict[k] = np.int32(
            np.median(np.array(size_dict[k]),axis=0))
        spacing_dict[k] = np.median(np.array(spacing_dict[k]),axis=0)

    intp = ["bilinear","nearest","nearest"]
    
    path_dictionary_dwi = {}
    for pid in path_dictionary:
        if all([k in path_dictionary[pid] for k in keys_dwi]):
            path_dictionary_dwi[pid] = {
                k:path_dictionary[pid][k] for k in keys_dwi}

    path_dictionary_t2wax = {}
    for pid in path_dictionary:
        if all([k in path_dictionary[pid] for k in keys_t2wax]):
            path_dictionary_t2wax[pid] = {
                k:path_dictionary[pid][k] for k in keys_t2wax}

    for pid in tqdm(path_dictionary_dwi):
        p = path_dictionary_dwi[pid]
        output_info = {}
        O = {}
        original_spacing = None
        for k in keys_dwi:
            path = p[k]
            sp = path.split(os.sep)
            D,filename = sp[-2],sp[-1]
            out_dir = os.path.join(args.output_path,D)
            out_path = os.path.join(out_dir,filename)
            mkdir(out_dir)
            output_info[k] = out_path
            img = sitk.ReadImage(p[k])
            orig_shape = img.GetSize()
            if 'segmentation' not in k:
                original_spacing = img.GetSpacing()
                O[k] = resample_image(img,tuple(spacing_dict["DWI"]),None)
                crop_coords = get_crop_coords(O[k],size_dict["DWI"])
                O[k] = crop(O[k],crop_coords[0],crop_coords[1])
                pad_coords = get_pad_coords(O[k],size_dict["DWI"])
                O[k] = pad(O[k],pad_coords[0],pad_coords[1])
            else:
                img.SetSpacing(original_spacing)
                O[k] = resample_image(img,tuple(spacing_dict["DWI"]),True)
                O[k] = crop(O[k],crop_coords[0],crop_coords[1])
                O[k] = pad(O[k],pad_coords[0],pad_coords[1])

        for k in output_info:
            out_path = output_info[k]
            img = O[k]
            sitk.WriteImage(img, out_path)
    
    for pid in tqdm(path_dictionary_t2wax):
        p = path_dictionary_t2wax[pid]
        output_info = {}
        O = {}
        original_spacing = None
        for k in keys_t2wax:
            path = p[k]
            sp = path.split(os.sep)
            D,filename = sp[-2],sp[-1]
            out_dir = os.path.join(args.output_path,D)
            out_path = os.path.join(out_dir,filename)
            mkdir(out_dir)
            output_info[k] = out_path
            img = sitk.ReadImage(p[k])
            orig_shape = img.GetSize()
            if 'segmentation' not in k:
                original_spacing = img.GetSpacing()
                O[k] = resample_image(img,tuple(spacing_dict["T2WAx"]),False)
                crop_coords = get_crop_coords(O[k],size_dict["T2WAx"])
                O[k] = crop(O[k],crop_coords[0],crop_coords[1])
                pad_coords = get_pad_coords(O[k],size_dict["T2WAx"])
                O[k] = pad(O[k],pad_coords[0],pad_coords[1])
            else:
                img.SetSpacing(original_spacing)
                O[k] = resample_image(img,tuple(spacing_dict["T2WAx"]),True)
                O[k] = crop(O[k],crop_coords[0],crop_coords[1])
                O[k] = pad(O[k],pad_coords[0],pad_coords[1])

        for k in output_info:
            out_path = output_info[k]
            img = O[k]
            sitk.WriteImage(img, out_path)