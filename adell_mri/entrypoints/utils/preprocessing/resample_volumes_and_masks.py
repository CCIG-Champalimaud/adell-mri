import argparse
import os
import re
from pathlib import Path

import SimpleITK as sitk
from tqdm import tqdm

from ....utils.sitk_utils import resample_image, resample_image_to_target

desc = "Resamples an image to a target spacing."


def main(arguments):
    parser = argparse.ArgumentParser(description=desc)

    parser.add_argument(
        "--image_path",
        dest="image_path",
        required=True,
        type=str,
        help="Path to folder containing fixed images.",
    )
    parser.add_argument(
        "--mask_path",
        dest="mask_path",
        required=True,
        type=str,
        help="Path to folder containing masks.",
    )
    parser.add_argument(
        "--image_patterns",
        dest="image_patterns",
        required=True,
        nargs="+",
        type=str,
        help="Pattern used to search for images \
            (must be regex compatible)",
    )
    parser.add_argument(
        "--mask_patterns",
        dest="mask_patterns",
        required=True,
        nargs="+",
        type=str,
        help="Pattern used to search for masks \
            (must be regex compatible)",
    )
    parser.add_argument(
        "--id_pattern",
        dest="id_pattern",
        required=True,
        type=str,
        help="Pattern used to extract ID (must be regex compatible)",
    )
    parser.add_argument(
        "--spacing",
        dest="spacing",
        required=True,
        nargs="+",
        type=float,
        help="Target spacing",
    )
    parser.add_argument(
        "--output_path",
        dest="output_path",
        required=True,
        type=str,
        help="Path to output file (moved images)",
    )

    args = parser.parse_args(arguments)

    # getting image paths
    path_dictionary = {}
    id_pattern = re.compile(args.id_pattern)
    for pattern in args.image_patterns:
        compiled_pattern = re.compile(pattern)
        image_paths = Path(args.image_path).rglob("*")
        image_paths = [
            x for x in image_paths if compiled_pattern.search(x) is not None
        ]
        for image_path in image_paths:
            identifier = id_pattern.match(image_path).group()
            if identifier not in path_dictionary:
                path_dictionary[identifier] = {}
            path_dictionary[identifier][pattern] = image_path

    # getting image paths
    for pattern in args.mask_patterns:
        compiled_pattern = re.compile(pattern)
        mask_paths = Path(args.mask_path).rglob("*")
        mask_paths = [
            x for x in mask_paths if compiled_pattern.search(x) is not None
        ]
        for mask_path in mask_paths:
            identifier = id_pattern.match(mask_path).group()
            if identifier not in path_dictionary:
                path_dictionary[identifier] = {}
            if "masks" not in path_dictionary[identifier]:
                path_dictionary[identifier]["masks"] = []
            path_dictionary[identifier]["masks"].append(mask_path)

    # removing cases where the first image pattern is not present
    new_path_dictionary = {}
    for identifier in path_dictionary:
        if args.image_patterns[0] in path_dictionary[identifier]:
            new_path_dictionary[identifier] = path_dictionary[identifier]
    new_path_dictionary = path_dictionary

    # resampling to common space
    os.makedirs(args.output_path, exist_ok=True)
    for identifier in tqdm(path_dictionary):
        output_base = os.path.join(args.output_path, identifier)
        os.makedirs(output_base, exist_ok=True)
        path_curr = path_dictionary[identifier][args.image_patterns[0]]
        fixed_image = sitk.ReadImage(
            path_dictionary[identifier][args.image_patterns[0]]
        )
        fixed_image = resample_image(fixed_image, args.spacing, args.is_label)
        sitk.WriteImage(
            fixed_image, os.path.join(output_base, path_curr.split(os.sep)[-1])
        )
        for pattern in args.image_patterns[1:]:
            path_curr = path_dictionary[identifier][pattern]
            image = sitk.ReadImage(path_curr)
            image = resample_image_to_target(image, fixed_image)
            sitk.WriteImage(
                image, os.path.join(output_base, path_curr.split(os.sep)[-1])
            )

        if "masks" in path_dictionary:
            for mask_path in path_dictionary[identifier]["masks"]:
                image = sitk.ReadImage(mask_path)
                image = resample_image_to_target(image, fixed_image, True)
                sitk.WriteImage(
                    image,
                    os.path.join(output_base, mask_path.split(os.sep)[-1]),
                    useCompression=True,
                    compressionLevel=9,
                )
