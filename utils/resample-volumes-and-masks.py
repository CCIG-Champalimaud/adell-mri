import os
import argparse
import re
import numpy as np
import SimpleITK as sitk
from pathlib import Path
from tqdm import tqdm


def resample_image(sitk_image, out_spacing=[1.0, 1.0, 1.0], is_label=False):
    original_spacing = sitk_image.GetSpacing()
    original_size = sitk_image.GetSize()

    out_size = [
        int(
            np.round(original_size[0] * (original_spacing[0] / out_spacing[0]))
        ),
        int(
            np.round(original_size[1] * (original_spacing[1] / out_spacing[1]))
        ),
        int(
            np.round(original_size[2] * (original_spacing[2] / out_spacing[2]))
        ),
    ]

    resample = sitk.ResampleImageFilter()
    resample.SetOutputSpacing(out_spacing)
    resample.SetSize(out_size)
    resample.SetOutputDirection(sitk_image.GetDirection())
    resample.SetOutputOrigin(sitk_image.GetOrigin())
    resample.SetTransform(sitk.Transform())
    resample.SetDefaultPixelValue(0.0)

    if is_label:
        resample.SetInterpolator(sitk.sitkNearestNeighbor)
    else:
        resample.SetInterpolator(sitk.sitkBSpline)

    output = resample.Execute(sitk_image)
    for k in sitk_image.GetMetaDataKeys():
        v = sitk_image.GetMetaData(k)
        output.SetMetaData(k, v)

    return output


def resample_to_target(image, fixed_image, is_label=False):
    if is_label:
        interpolator = sitk.sitkNearestNeighbor
    else:
        interpolator = sitk.sitkBSpline
    return sitk.Resample(
        image, fixed_image, sitk.Transform(), interpolator, 0.0
    )


def crop_image(sitk_image, size):
    size = np.array(size)
    curr_size = np.array(sitk_image.GetSize())
    # pad in case image is too small
    if any(curr_size < size):
        total_padding = np.maximum((0, 0, 0), size - curr_size)
        lower = np.int16(total_padding // 2)
        upper = np.int16(total_padding - lower)
        sitk_image = sitk.ConstantPad(
            sitk_image, lower.tolist(), upper.tolist(), 0.0
        )
    curr_size = np.array(sitk_image.GetSize())
    total_crop = np.maximum((0, 0, 0), curr_size - size)
    lower = np.int16(total_crop // 2)
    upper = np.int16((total_crop - lower))

    sitk_image = sitk.Crop(sitk_image, lower.tolist(), upper.tolist())
    return sitk_image


desc = """
Resamples an image to a target spacing.
"""

if __name__ == "__main__":
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

    args = parser.parse_args()

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
            image = resample_to_target(image, fixed_image)
            sitk.WriteImage(
                image, os.path.join(output_base, path_curr.split(os.sep)[-1])
            )

        if "masks" in path_dictionary:
            for mask_path in path_dictionary[identifier]["masks"]:
                image = sitk.ReadImage(mask_path)
                image = resample_to_target(image, fixed_image, True)
                sitk.WriteImage(
                    image,
                    os.path.join(output_base, mask_path.split(os.sep)[-1]),
                    useCompression=True,
                    compressionLevel=9,
                )
