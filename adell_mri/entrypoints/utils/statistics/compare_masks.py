import argparse
import os
import re
from glob import glob

import numpy as np
import SimpleITK as sitk
from tqdm import tqdm

from adell_mri.utils.python_logging import get_logger
from adell_mri.utils.sitk_utils import resample_image_to_target

desc = "Calculates IoU of masks in two separate folders corresponding to the \
    same strcture and with the same identifier."


def iou(a, b):
    a_unique = np.unique(a)
    b_unique = np.unique(b)
    a_unique = [x for x in a_unique if x != 0]
    b_unique = [x for x in b_unique if x != 0]
    uniq = list(set(a_unique + b_unique))
    result_list = []
    for u in uniq:
        intersection_arr = np.where(np.logical_and(a == b, a == u), 1, 0)
        intersection = np.sum(intersection_arr)
        union = np.sum(a == u) + np.sum(b == u) - intersection
        result_list.append(intersection / union)
    return np.mean(result_list)


def main(arguments):
    logger = get_logger(__name__)
    parser = argparse.ArgumentParser(description=desc)

    parser.add_argument(
        "--input_paths",
        dest="input_paths",
        type=str,
        nargs=2,
        help="Path to directory containing masks.",
        required=True,
    )
    parser.add_argument(
        "--patterns",
        dest="patterns",
        type=str,
        default="*",
        nargs=2,
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
        "--binarize", dest="binarize", action="store_true", default=False
    )

    args = parser.parse_args(arguments)

    path_dict = {}
    for input_path, pattern in zip(args.input_paths, args.patterns):
        for path in glob(os.path.join(input_path, pattern)):
            study_id = re.search(args.pattern_id, path).group()
            if study_id not in path_dict:
                path_dict[study_id] = []
            path_dict[study_id].append(path)

    iou_list = []
    path_dict_keys = list(path_dict.keys())
    for patient_id in tqdm(path_dict_keys):
        if len(path_dict[patient_id]) == 2:
            image_1 = sitk.ReadImage(path_dict[patient_id][0])
            image_2 = sitk.ReadImage(path_dict[patient_id][1])
            image_2 = resample_image_to_target(image_2, image_1, True)
            image_1 = sitk.GetArrayFromImage(image_1)
            image_2 = sitk.GetArrayFromImage(image_2)
            if args.binarize is True:
                image_1 = np.where(image_1 > 0, 1, 0)
                image_2 = np.where(image_2 > 0, 1, 0)
            if np.count_nonzero(image_1) > 0 or np.count_nonzero(image_2) > 0:
                v = iou(image_1, image_2)
                iou_list.append(v)

    logger.info(
        "Quantiles (0%%,25%%,50%%,75%%,100%%): %s",
        np.quantile(iou_list, [0, 0.25, 0.5, 0.75, 1]),
    )
    logger.info("Mean: %s", np.mean(iou_list))
