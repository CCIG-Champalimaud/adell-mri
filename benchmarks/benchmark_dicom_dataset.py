import sys
import os
sys.path.append(os.path.join(os.path.dirname(os.path.abspath(__file__)),'..'))
import time
import json
import argparse
import numpy as np
import monai
from tqdm import trange

from lib.utils.dicom_loader import DICOMDataset,filter_orientations
from lib.monai_transforms import get_pre_transforms_ssl
from lib.monai_transforms import get_post_transforms_ssl
from lib.monai_transforms import get_augmentations_ssl

if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument("--json_path",required=True)
    parser.add_argument("--crop_size",default=[256,256],nargs=2,
                        type=int)
    
    args = parser.parse_args()

    pre_transforms = get_pre_transforms_ssl(["image"],["image_copy"],
                                            adc_keys=[],non_adc_keys=["image"],
                                            target_spacing=None,
                                            crop_size=args.crop_size,
                                            pad_size=args.crop_size,
                                            n_dim=2)
    post_transforms = get_post_transforms_ssl(["image"],["image_copy"])
    augmentations = get_augmentations_ssl(["image"],["image_copy"],
                                          [256,256],[224,224],False,2)

    with open(args.json_path) as o:
        dicom_dict = json.load(o)
        
    dicom_dataset = [dicom_dict[k] for k in dicom_dict]
    dicom_dataset = DICOMDataset(dicom_dataset,
                                 transform=monai.transforms.Compose(
                                     [*pre_transforms,
                                      *augmentations,
                                      *post_transforms]))
    
    times = []
    for i in trange(len(dicom_dataset)):
        a = time.time()
        dicom_dataset[i]
        b = time.time()
        times.append(b-a)
    
    print(np.mean(times) / len(times))