import argparse
import os
from functools import partial
from glob import glob
from math import prod
from multiprocessing import Pool

import SimpleITK as sitk
from tqdm import tqdm

from adell_mri.utils.sitk_utils import resample_image

desc = "Prints size of labels in a folder containing segmentation masks."


def get_label_statistics(path: str, spacing: tuple | None = None) -> dict:
    output = []
    image = sitk.ReadImage(path)
    if sitk.GetArrayFromImage(image).sum() > 0:
        if spacing is not None:
            image = resample_image(image, spacing, True)
        cc = sitk.ConnectedComponent(image)
        lsf = sitk.LabelShapeStatisticsImageFilter()
        lsf.Execute(cc)
        if lsf.GetNumberOfLabels() == 0:
            return
        voxel_volume = prod(image.GetSpacing())
        for i in lsf.GetLabels():
            phy_size = lsf.GetPhysicalSize(i)
            vox_size = phy_size / voxel_volume
            bb = lsf.GetBoundingBox(i)
            span_x = bb[3]
            span_y = bb[4]
            span_z = bb[5]
            output.append(
                f"{path},{phy_size},{vox_size},{span_x},{span_y},{span_z}"
            )
    return output


def main(arguments):
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
    parser.add_argument(
        "--n_workers",
        dest="n_workers",
        type=int,
        default=0,
        help="Number of workers.",
    )

    args = parser.parse_args(arguments)

    rows = [f"path,physical_size,voxel_size,span_x,span_y,span_z"]

    mask_paths = glob(os.path.join(args.input_path, args.pattern))

    fn_with_args = partial(get_label_statistics, spacing=args.spacing)
    total = len(mask_paths)
    if args.n_workers < 2:
        map_fn = map(fn_with_args, mask_paths)
    else:
        pool = Pool(args.n_workers)
        map_fn = pool.imap_unordered(fn_with_args, mask_paths)

    for out in tqdm(map_fn, total=total):
        if out:
            rows.extend(out)

    for row in rows:
        print(row)
