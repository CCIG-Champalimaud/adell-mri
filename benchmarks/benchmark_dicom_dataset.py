import os
import sys

sys.path.append(os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
import argparse
import json
import time

import monai
import numpy as np
from tqdm import trange

from adell_mri.transform_factory import SSLTransforms, get_augmentations_ssl
from adell_mri.utils.dicom_loader import DICOMDataset

if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument("--json_path", required=True)
    parser.add_argument("--crop_size", default=[256, 256], nargs=2, type=int)

    args = parser.parse_args()

    transform_factory = SSLTransforms(
        ["image"],
        ["image_copy"],
        adc_keys=[],
        non_adc_keys=["image"],
        target_spacing=None,
        crop_size=args.crop_size,
        pad_size=args.crop_size,
        n_dim=2,
    )
    augmentations = get_augmentations_ssl(
        ["image"], ["image_copy"], [256, 256], [224, 224], False, 2
    )

    with open(args.json_path) as o:
        dicom_dict = json.load(o)

    dicom_dataset = [dicom_dict[k] for k in dicom_dict]
    dicom_dataset = DICOMDataset(
        dicom_dataset,
        transform=transform_factory.transforms(augmentations),
    )

    times = []
    for i in trange(len(dicom_dataset)):
        a = time.time()
        dicom_dataset[i]
        b = time.time()
        times.append(b - a)

    print(np.mean(times) / len(times))
