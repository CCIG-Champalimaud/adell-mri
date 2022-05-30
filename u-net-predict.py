import os
import argparse
import yaml
import numpy as np
import torch
import monai
from tqdm import tqdm

from lib.utils import (
    activation_factory,
    collate_last_slice,
    get_prostatex_path_dictionary,
    get_size_spacing_dict,
    SlicesToFirst,
    Index,PrintShaped)
from lib.modules.segmentation_pl import UNetPL

if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    # data
    parser.add_argument(
        '--input_path',dest='input_path',type=str,nargs='+',
        help="Path(s) to MRI scan(s)",required=True)
    parser.add_argument(
        '--index',dest='index',type=int,default=None,
        help="Index for image (in case >3D)")
    parser.add_argument(
        '--mod',dest='mod',type=str,choices=["T2WAx","DWI"],
        help="Key to be used",required=True)
    parser.add_argument(
        '--prostate_x_path',dest='prostate_x_path',type=str,
        help="Path to folder with training data",required=True)
    parser.add_argument(
        '--output_path',dest='output_path',type=str,
        help="Path to output folder",required=True)

    # network + training
    parser.add_argument(
        '--config_file',dest="config_file",
        help="Path to network configuration file (yaml)",required=True)
    parser.add_argument(
        '--checkpoint_path',dest="checkpoint_path",
        help="Path to U-Net checkpoint",required=True)

    # inference specific
    parser.add_argument(
        '--dev',dest='dev',choices=["cuda","cpu"],type=str,
        help="Device for PyTorch training")
    parser.add_argument(
        '--n_workers',dest='n_workers',
        help="No. of workers",default=1,type=int)
    parser.add_argument(
        '--downsample_rate',dest='downsample_rate',type=float,default=1.0,
        help="Resizes images with downsample rate")

    args = parser.parse_args()

    n_classes = 2
    
    path_dictionary = get_prostatex_path_dictionary(args.prostate_x_path)
    fpd = {}
    for pid in path_dictionary:
        if all([k in path_dictionary[pid] for k in [args.mod]]):
            fpd[pid] = {k:path_dictionary[pid][k] for k in [args.mod]}
    path_dictionary = fpd
    size_dict,spacing_dict = get_size_spacing_dict(path_dictionary,[args.mod])

    for k in size_dict:
        mm = np.median(np.array(size_dict[k]),axis=0)
        # downsample only non-depth dimensions
        mm[0] = mm[0]*args.downsample_rate
        mm[1] = mm[1]*args.downsample_rate
        size_dict[k] = np.int32(np.round(mm))
        spacing_dict[k] = np.median(np.array(spacing_dict[k]),axis=0)

    with open(args.config_file,'r') as o:
        network_config = yaml.safe_load(o)

    if "activation_fn" in network_config:
        network_config["activation_fn"] = activation_factory[
            network_config["activation_fn"]]

    if "spatial_dimensions" not in network_config:
        network_config["spatial_dimensions"] = 3

    if "batch_size" not in network_config:
        network_config["batch_size"] = 1

    intp = ["area"]

    transforms = [
        monai.transforms.LoadImaged(["image"]),
        Index(["image"],args.index,-1),
        monai.transforms.AddChanneld(["image"]),
        monai.transforms.Orientationd(["image"],"RAS"),
        monai.transforms.EnsureTyped(["image"]),
        monai.transforms.Spacingd(
            ["image"],tuple(spacing_dict[args.mod]),mode="bilinear"),
        monai.transforms.Resized(
            ["image"],tuple(size_dict[args.mod]),mode=intp),
        monai.transforms.ScaleIntensityd(["image"],0,1),
        monai.transforms.EnsureTyped(["image"])]

    if network_config["spatial_dimensions"] == 2:
        transforms.append(
            SlicesToFirst(["image"]))
        collate_fn = collate_last_slice
    else:
        collate_fn = None

    unet = UNetPL(image_key="image",n_classes=n_classes,**network_config)
    state_dict = torch.load(
        args.checkpoint_path,map_location=args.dev)['state_dict']
    inc = unet.load_state_dict(state_dict)
    print("Incompatible keys:",inc)

    unet = unet.to(args.dev)
    unet.eval()

    print("Predicting...")
    try: os.makedirs(args.output_path)
    except: pass
    transforms = monai.transforms.Compose(transforms)
    for path in tqdm(args.input_path):
        image = transforms({"image":path})
        # introduce batch dim
        if collate_fn is not None:
            image = collate_fn((image,))['image']
        else:
            image = torch.unsqueeze(image['image'],0)
        root_name = os.path.split(path)[-1].split('.')[0]
        output_path = os.path.join(args.output_path,root_name+'.npy')
        pred = unet(image.to(args.dev))
        # uniformize everything to [c,h,w,d], where c = 1 for binary classification
        if n_classes == 2:
            pred = pred.unsqueeze(-1).swapaxes(0,-1).squeeze(0)
        elif n_classes == 3:
            pred = pred.squeeze(0)
        image = pred.detach().numpy()
        pred = pred.detach().numpy()
        o = np.array([path,image,pred],dtype=object)
        np.save(output_path,o,allow_pickle=True)