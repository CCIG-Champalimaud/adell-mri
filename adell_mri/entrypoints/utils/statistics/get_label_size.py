import argparse
import os
from glob import glob

import numpy as np
import SimpleITK as sitk
from tqdm import tqdm

from adell_mri.utils.sitk_utils import resample_image
from adell_mri.utils.python_logging import get_logger

desc = "Prints size of labels in a folder containing segmentation masks."


def main(arguments):
    logger = get_logger(__name__)
    parser = argparse.ArgumentParser(description=desc)

    parser.add_argument(
        "--input_path",
        dest="input_path",
        type=str,
        help="Path to directory containing masks.",
        required=True,
    )
    parser.add_argument(
        "--spacing",
        dest="spacing",
        type=float,
        nargs="+",
        help="Target spacing of the masks",
        default=None,
    )
    parser.add_argument(
        "--pattern",
        dest="pattern",
        type=str,
        default="*",
        help="Pattern to match masks",
    )

    args = parser.parse_args(arguments)

    for path in tqdm(glob(os.path.join(args.input_path, args.pattern))):
        image = sitk.ReadImage(path)
        if sitk.GetArrayFromImage(image).sum() > 0:
            if args.spacing is not None:
                image = resample_image(image, args.spacing, True)
            image = sitk.GetArrayFromImage(image)
            un, co = np.unique(image, return_counts=True)
            for u, c in zip(un, co):
                if u > 0:
                    logger.info("%s,%s,%s,%s", path, u, c, c / image.size)
