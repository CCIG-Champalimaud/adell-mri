import argparse
import re
import os
import numpy as np
import SimpleITK as sitk
from tqdm import tqdm
from pathlib import Path
from glob import glob
from skimage.measure import label
from skimage.morphology import remove_small_holes, remove_small_objects
from scipy import spatial


def remove_objects(mask):
    """Removes all objects from a binary mask except
    the largest one. (Assuming the largest object has
    the highest probability of being the correct
    segmentation)

    Parameters
    ----------
    mask : sitk object
        sitk binary mask.

    Returns
    -------
    sitk image
        sitk mask with only the largest object.
    """
    arr = sitk.GetArrayFromImage(mask)
    arr = arr.astype(dtype=bool)
    arr, nb_objects = label(arr, connectivity=3, return_num=True)
    min_size = 0
    while nb_objects > 1:
        min_size += 100
        new_arr = remove_small_holes(
            remove_small_objects(arr, min_size=min_size, connectivity=3),
            area_threshold=min_size,
            connectivity=3,
        )
        arr, nb_objects = label(new_arr, connectivity=3, return_num=True)
    arr = arr.astype(int)
    new_mask = sitk.GetImageFromArray(arr.astype(np.uint8))
    new_mask.CopyInformation(mask)
    return new_mask


def flood_fill_hull(sitk_image):
    """
    Flood fill an SITK image. Adapted from [1].
    [1] https://stackoverflow.com/a/46314485
    """
    # convert SITK image to np array
    image = sitk.GetArrayFromImage(sitk_image)
    # extract coordinates for all positive indices
    points = np.transpose(np.where(image))
    # calculate the convex hull (a geometrical object containing all points
    # whose vertices are defined from points)
    hull = spatial.ConvexHull(points)
    # calculate the Delaunay triangulation - in short, this divides the convex
    # hull into a set of triangles whose circumcircles do not contain those of
    # other triangles
    deln = spatial.Delaunay(points[hull.vertices])
    # construct a new array whose elements are the 3d coordinates of the image
    idx = np.stack(np.indices(image.shape), axis=-1)
    # find the simplices containing these indices (idx)
    out_idx = np.nonzero(deln.find_simplex(idx) + 1)
    # create new image
    out_img = np.zeros(image.shape)
    out_img[out_idx] = 1
    # convert to SITK image and copy information
    new_mask = sitk.GetImageFromArray(out_img.astype(np.uint8))
    new_mask.CopyInformation(sitk_image)
    # out_sitk_image = sitk.Cast(out_sitk_image, sitk.sitkUInt8)
    return new_mask


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument("--patterns", dest="patterns", nargs="+")
    parser.add_argument("--id_pattern", dest="id_pattern")
    parser.add_argument(
        "--mask_idx", dest="mask_idx", type=int, nargs="+", default=[]
    )
    parser.add_argument("--output_path", dest="output_path")
    parser.add_argument("--subsample_size", dest="subsample_size", type=int)
    parser.add_argument("--seed", dest="seed", default=42, type=int)

    args = parser.parse_args()

    id_pattern = re.compile(args.id_pattern)

    file_dict = {}
    for pattern_idx, pattern in enumerate(args.patterns):
        all_files = glob(pattern)
        for file in all_files:
            identifier = id_pattern.search(file)
            if identifier is not None:
                identifier = identifier.group()
                if identifier not in file_dict:
                    file_dict[identifier] = {}
                file_dict[identifier][pattern_idx] = file

    file_dict = {
        k: file_dict[k]
        for k in file_dict
        if len(file_dict[k]) == len(args.patterns)
    }

    if args.subsample_size is not None:
        file_dict_keys = list(file_dict.keys())
        rng = np.random.default_rng(args.seed)
        file_dict = {
            k: file_dict[k]
            for k in rng.choice(file_dict_keys, size=args.subsample_size)
        }

    for k in tqdm(file_dict):
        for file_key in file_dict[k]:
            file_path = file_dict[k][file_key]
            file_name = file_path.split(os.path.sep)[-1].split(".")[0]
            output_path = os.path.join(
                args.output_path, k, file_name + ".nii.gz"
            )
            Path(output_path).parent.mkdir(parents=True, exist_ok=True)
            image = sitk.ReadImage(file_path)
            if file_key in args.mask_idx:
                image = remove_objects(image)
                image = flood_fill_hull(image)
            sitk.WriteImage(image, output_path)
