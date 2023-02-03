import sys
import os
sys.path.append(os.path.join(os.path.dirname(os.path.abspath(__file__)),'..'))
import time
import json
import argparse
import monai
from tqdm import tqdm

from lib.utils.dicom_loader import (
    DICOMDataset,SliceSampler,filter_bad_orientations)
from lib.monai_transforms import get_pre_transforms_ssl
from lib.monai_transforms import get_post_transforms_ssl

if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument("--json_path",required=True)
    parser.add_argument("--batch_size",default=32,type=int)
    parser.add_argument("--n_workers",default=0,type=int)
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

    with open(args.json_path) as o:
        dicom_dict = json.load(o)
    
    dicom_dict = filter_bad_orientations(dicom_dict)
    
    dicom_list = [dicom_dict[k] for k in dicom_dict]
    dicom_dataset = DICOMDataset(dicom_list,
                                 transform=monai.transforms.Compose(
                                     [*pre_transforms,
                                      *post_transforms]))
        
    dicom_sampler = SliceSampler(dicom_dataset=dicom_list)
    data_loader = monai.data.ThreadDataLoader(
        dicom_dataset,sampler=dicom_sampler,
        batch_size=args.batch_size,
        num_workers=args.n_workers,
        persistent_workers=args.n_workers>1)
    
    a = time.time()
    i = 0
    for element in tqdm(data_loader):
        element["image"].shape
        i += 1
    b = time.time()
    
    print((b-a) / i)