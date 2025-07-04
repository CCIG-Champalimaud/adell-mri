import argparse
import os
import re
from glob import glob
from pathlib import Path
from typing import List

import numpy as np
import SimpleITK as sitk
from tqdm import tqdm

from adell_mri.utils.sitk_utils import resample_image_to_target

desc = "Merges two masks keeping pixels which are non-zero in either mask (like\
    an OR operator)."


def iou(a: np.ndarray, b: np.ndarray) -> float:
    a_unique = np.unique(a)
    b_unique = np.unique(b)
    a_unique = [x for x in a_unique if x != 0]
    b_unique = [x for x in b_unique if x != 0]
    uniq = list(set(a_unique + b_unique))
    result_list = []
    for u in uniq:
        intersection = np.where(np.logical_and(a == b, a == u), 1, 0)
        inter_value = np.sum(intersection)
        U = np.sum(a == u) + np.sum(b == u) - inter_value
        result_list.append(inter_value / U)
    return float(np.mean(result_list))


def merge_masks(path_list: List[str], argmax: bool) -> sitk.Image:
    images_ = [sitk.ReadImage(x) for x in path_list]
    images_[1:] = [
        resample_image_to_target(x, images_[0], True) for x in images_[1:]
    ]
    images = [sitk.GetArrayFromImage(x) for x in images_]
    image_out = np.stack([np.zeros_like(images[0]), *images])
    if argmax is True:
        image_out = np.argmax(image_out, 0)
    else:
        image_out = image_out.mean(0) > 0
    image_out = sitk.GetImageFromArray(image_out)
    image_out.CopyInformation(images_[0])
    return image_out


def main(arguments):
    parser = argparse.ArgumentParser(description=desc)

    parser.add_argument(
        "--input_paths",
        dest="input_paths",
        type=str,
        nargs="+",
        help="Path to directory containing masks.",
        required=True,
    )
    parser.add_argument(
        "--patterns",
        dest="patterns",
        type=str,
        nargs="+",
        default="*",
        help="Patterns to match masks",
    )
    parser.add_argument(
        "--pattern_id",
        dest="pattern_id",
        type=str,
        default="[0-9]+_[0-9]+",
        help="Regex pattern for the patient id.",
    )
    parser.add_argument(
        "--output_path",
        dest="output_path",
        type=str,
        required=True,
        help="Output path for images.",
    )
    parser.add_argument(
        "--strict",
        dest="strict",
        action="store_true",
        help="Stores only images if both masks are present.",
    )
    parser.add_argument(
        "--argmax",
        dest="argmax",
        action="store_true",
        help="Stores the images in argmax format.",
    )

    args = parser.parse_args(arguments)

    os.makedirs(args.output_path, exist_ok=True)

    if len(args.input_paths) == len(args.patterns):
        input_paths = args.input_paths
        patterns = args.patterns
    elif len(args.input_paths) == 1:
        input_paths = [args.input_paths[0] for _ in args.patterns]
        patterns = args.patterns
    else:
        raise Exception(
            "input_paths should have length 1 or length identical to patterns"
        )

    path_dict = {}
    for input_path, pattern in zip(input_paths, patterns):
        for path in glob(os.path.join(input_path, pattern)):
            study_id = re.search(args.pattern_id, path).group()
            if study_id not in path_dict:
                path_dict[study_id] = []
            path_dict[study_id].append(path)

    path_dict_keys = list(path_dict.keys())
    n = len(args.patterns)
    for patient_id in tqdm(path_dict_keys):
        if len(path_dict[patient_id]) == n or args.strict is False:
            image_out = merge_masks(path_dict[patient_id], args.argmax)
            image_out = sitk.Cast(image_out, sitk.sitkInt16)
            output_path_image = os.path.join(
                args.output_path, patient_id + ".nii.gz"
            )
            Path(output_path_image).parent.mkdir(parents=True, exist_ok=True)
            sitk.WriteImage(image_out, output_path_image)
