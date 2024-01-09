import argparse
import numpy as np
from pydicom import dcmread
from pathlib import Path
from tqdm import tqdm
from skimage.transform import resize
from skimage.io import imsave

desc = "Generates a panel of random DICOM images in a folder"


def normalize(x: np.ndarray):
    return (x - x.min()) / (x.max() - x.min())


def crop_to_square(x: np.ndarray):
    sh = x.shape
    ms = np.min(sh)
    A = [(s - ms) // 2 for s in sh]
    B = [s - a for s, a in zip(sh, A)]
    x = x[A[0] : B[0], A[1] : B[1]]
    return x


def main(arguments):
    parser = argparse.ArgumentParser(description=desc)

    parser.add_argument(
        "--path",
        required=True,
        help="Path to folder containing a DICOM dataset",
    )
    parser.add_argument(
        "--structure",
        nargs=2,
        type=int,
        required=True,
        help="Number of images per row and column",
    )
    parser.add_argument(
        "--size",
        nargs=2,
        type=int,
        required=True,
        help="Size of each DICOM image in panel",
    )
    parser.add_argument(
        "--output_path",
        type=str,
        nargs="+",
        required=True,
        help="Output path for the generated panel",
    )

    args = parser.parse_args(arguments)
    h, w = args.structure
    n = h * w

    path_gen = Path(f"{args.path}").rglob("*dcm")

    all_dcm_paths = []
    for path in tqdm(path_gen):
        all_dcm_paths.append(path)

    for output_path in args.output_path:
        print(f"Making and saving {output_path}")
        path_subset = np.random.choice(all_dcm_paths, size=n, replace=False)

        all_dcm = [[] for _ in range(w)]
        for i, path in enumerate(path_subset):
            dcm = normalize(dcmread(path).pixel_array)
            if len(dcm.shape) == 3:
                print(dcm.shape)
                dcm = dcm[0]
            dcm = crop_to_square(dcm)
            all_dcm[i // w].append(
                np.uint8(resize(dcm, args.size, anti_aliasing=True) * 255)
            )

        all_dcm = [np.concatenate(dcm_list, axis=0) for dcm_list in all_dcm]
        all_dcm = np.concatenate(all_dcm, axis=1).astype(np.uint8)

        Path(output_path).parent.mkdir(exist_ok=True, parents=True)
        imsave(output_path, all_dcm)
