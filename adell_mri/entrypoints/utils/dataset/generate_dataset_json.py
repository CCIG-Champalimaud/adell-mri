import json
import re
import argparse
import numpy as np
import monai
import os
from pathlib import Path
from skimage import measure
from tqdm import tqdm

from typing import List

desc = "Creates JSON file with paths and bounding boxes."


def value_range(x: np.ndarray) -> tuple[float, float]:
    """
    Retrieves range for array.

    Args:
        x (np.ndarray): array.

    Returns:
        tuple[float, float]: the minimum and maximum of x.
    """
    return x.min(), x.max()


def mask_to_bb(img: np.ndarray) -> tuple[list[np.array], list[int]]:
    """
    Converts a mask to a set of bounding boxes and their classes (one bounding
    box for each connected component).

    Args:
        img (np.ndarray): a binary image (it will be rounded in any case).

    Returns:
        tuple[list[np.array], list[int]]: list with bounding boxes with shape
            [[x1, x2], [y1, y2], ...] and list with classes (median value of
            each object in the mask).
    """
    img = np.round(img)
    labelled_image = measure.label(img, connectivity=3)
    uniq = np.unique(labelled_image)
    uniq = uniq[uniq != 0]

    bb_vertices = []
    c = []
    for u in uniq:
        C = np.where(labelled_image == u)
        bb = np.array([value_range(c) for c in C])
        if np.all(bb[:, 1] == bb[:, 0]) == False:  # noqa
            bb_vertices.append(bb)
            c.append(np.median(img[C]))

    return bb_vertices, c


def search_with_re(all_paths: List[str], re_pattern: str) -> list[str]:
    """
    Filters a list if there is a match with ``re_pattern``.

    Args:
        all_paths (List[str]): list of strings.
        re_pattern (str): pattern that will be searched.

    Returns:
        list[str]: filtered all_paths containing only elements with
            ``re_pattern``.
    """

    compiled_re_pattern = re.compile(re_pattern)
    output = all_paths
    output = [x for x in output if compiled_re_pattern.search(x) is not None]
    return output


def main(arguments):
    parser = argparse.ArgumentParser(description=desc)
    parser.add_argument(
        "--input_path",
        dest="input_path",
        required=True,
        help="Path to folder containing nibabel compatible files",
    )
    parser.add_argument(
        "--mask_path",
        dest="mask_path",
        default=None,
        help="Path to folder containing nibabel compatible masks",
    )
    parser.add_argument(
        "--mask_key",
        dest="mask_key",
        default="mask",
        help="Custom key for the mask. Helpful if later merging json files",
    )
    parser.add_argument(
        "--class_csv_path",
        dest="class_csv_path",
        default=None,
        help="Path to CSV with classes. Assumes first column is study ID and \
            last column is class.",
    )
    parser.add_argument(
        "--patterns",
        dest="patterns",
        default=[".*nii.gz"],
        nargs="+",
        help="Pattern to match for inputs (assumes each pattern corresponds to\
            a modality).",
    )
    parser.add_argument(
        "--mask_pattern",
        dest="mask_pattern",
        default="*nii.gz",
        help="Pattern to match for mask",
    )
    parser.add_argument(
        "--id_pattern",
        dest="id_pattern",
        default=".*",
        type=str,
        help="Pattern to extract IDs from image files",
    )
    parser.add_argument(
        "--output_json",
        dest="output_json",
        required=True,
        help="Path for output JSON file",
    )
    parser.add_argument(
        "--strict",
        dest="strict",
        action="store_true",
        help="Only includes images with a corresponding mask",
    )
    parser.add_argument(
        "--skip_bb",
        dest="skip_detection",
        action="store_true",
        help="Skips bounding box calculation from masks",
    )

    args = parser.parse_args(arguments)

    t = monai.transforms.Compose(
        [
            monai.transforms.LoadImaged(["image"], ensure_channel_first=True),
            monai.transforms.Orientationd(["image"], "RAS"),
        ]
    )

    bb_dict = {}
    all_paths = [str(x) for x in Path(args.input_path).rglob("*")]
    all_paths = {p: search_with_re(all_paths, p) for p in args.patterns}
    class_dict_csv = {}
    if args.class_csv_path:
        with open(args.class_csv_path, "r") as o:
            for line in o:
                line = line.strip().split(",")
                identifier, cl = line[0], line[-1]
                class_dict_csv[identifier] = cl
    if args.mask_path is not None:
        all_mask_paths = [str(x) for x in Path(args.mask_path).rglob("*")]
        mask_paths = search_with_re(all_mask_paths, args.mask_pattern)
    else:
        mask_paths = []
    for file_path in tqdm(all_paths[args.patterns[0]]):
        image_id = re.search(args.id_pattern, file_path).group()
        alt_file_paths = []
        for k in args.patterns[1:]:
            paths = all_paths[k]
            for alt_path in paths:
                if image_id in alt_path:
                    alt_file_paths.append(alt_path)
        mask_path = [
            p for p in mask_paths if image_id in p.replace(args.mask_path, "")
        ]
        if len(mask_path) == 0 and args.strict is True:
            continue

        bb_dict[image_id] = {"image": os.path.abspath(file_path)}
        if len(alt_file_paths) > 0:
            for i, p in enumerate(alt_file_paths):
                bb_dict[image_id]["image_" + str(i + 1)] = os.path.abspath(p)

        if len(mask_path) > 0:
            mask_path = mask_path[0]
            bb_dict[image_id][args.mask_key] = os.path.abspath(mask_path)
            if args.skip_detection is not True:
                bb_dict[image_id]["boxes"] = []
                bb_dict[image_id]["shape"] = ""
                bb_dict[image_id]["labels"] = []
                fdata = t({"image": mask_path})["image"][0]
                sh = np.array(fdata.shape)
                unique_labels = []
                for bb, c in zip(*mask_to_bb(fdata)):
                    c = int(c)
                    bb = [int(x) for x in bb.flatten("F")]
                    bb_dict[image_id]["boxes"].append(bb)
                    bb_dict[image_id]["labels"].append(c)
                    if c not in unique_labels:
                        unique_labels.append(c)
                bb_dict[image_id]["shape"] = [int(x) for x in sh]
                if len(unique_labels) == 0:
                    unique_labels = [0]
                bb_dict[image_id]["image_labels"] = unique_labels
        elif image_id in class_dict_csv:
            bb_dict[image_id] = {
                "image": os.path.abspath(file_path),
                "image_labels": class_dict_csv[image_id],
            }
            if len(alt_file_paths) > 0:
                for i, p in enumerate(alt_file_paths):
                    bb_dict[image_id]["image_" + str(i + 1)] = os.path.abspath(
                        p
                    )

    pretty_dict = json.dumps(bb_dict, indent=2)
    with open(args.output_json, "w") as o:
        o.write(pretty_dict + "\n")
