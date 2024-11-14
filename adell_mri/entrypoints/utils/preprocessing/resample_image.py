import argparse

import SimpleITK as sitk

from ....utils.sitk_utils import crop_image, resample_image

desc = "Resamples an image to a target spacing."


def main(arguments):
    parser = argparse.ArgumentParser(description=desc)

    parser.add_argument(
        "--image_path",
        dest="image_path",
        required=True,
        type=str,
        help="Path to fixed image",
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
        "--crop_size",
        dest="crop_size",
        default=None,
        nargs="+",
        type=float,
        help="Center crops to specified size",
    )
    parser.add_argument(
        "--is_label",
        dest="is_label",
        action="store_true",
        default=False,
        help="Image is label map (uses NN interpolation)",
    )
    parser.add_argument(
        "--output_path",
        dest="output_path",
        required=True,
        type=str,
        help="Path to output file (moved images)",
    )

    args = parser.parse_args(arguments)

    # loading image

    image_path = args.image_path
    sf = sitk.sitkFloat32
    fixed_image = sitk.ReadImage(image_path, sf)

    # resampling to common space

    output_image = resample_image(fixed_image, args.spacing, args.is_label)
    if args.crop_size is not None:
        output_image = crop_image(output_image, args.crop_size)
    if args.is_label is True:
        output_image = sitk.Cast(output_image, sitk.sitkInt16)
    sitk.WriteImage(output_image, args.output_path)
