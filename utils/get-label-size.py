import argparse
import os
import SimpleITK as sitk
import numpy as np
from glob import glob
from tqdm import tqdm

desc = """
Counts which classes are present in a folder containing segmentation maps.
"""


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


if __name__ == "__main__":
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

    args = parser.parse_args()

    for path in tqdm(glob(os.path.join(args.input_path, args.pattern))):
        path_sub = path.split(os.sep)[-1]
        image = sitk.ReadImage(path)
        if sitk.GetArrayFromImage(image).sum() > 0:
            if args.spacing is not None:
                image = resample_image(image, args.spacing, True)
            image = sitk.GetArrayFromImage(image)
            un, co = np.unique(image, return_counts=True)
            for u, c in zip(un, co):
                if u > 0:
                    print("{},{},{},{}".format(path, u, c, c / image.size))
