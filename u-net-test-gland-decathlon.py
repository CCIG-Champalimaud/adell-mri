import argparse
import random
import yaml
import numpy as np
import nibabel as nib
import torch
import monai
from torchmetrics import JaccardIndex,Precision,FBetaScore
from tqdm import tqdm

from lib.utils import (
    activation_factory,
    collate_last_slice,
    get_prostatex_path_dictionary,
    get_size_spacing_dict,
    SlicesToFirst,
    Index,
    PrintShaped)
from lib.modules.segmentation_pl import UNetPL
from lib.dataoperations.lib.data_functions import MONAIDataset

if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    # data
    parser.add_argument(
        '--root_dir',dest='root_dir',type=str,
        help="Root directory for Decathlon",required=True)
    parser.add_argument(
        '--mod',dest='mod',type=str,choices=["T2WAx","DWI"],
        help="Key to be used",required=True)
    parser.add_argument(
        '--prostate_x_path',dest='prostate_x_path',type=str,
        help="Path to folder with training data",required=True)
    parser.add_argument(
        '--metrics_path',dest='metrics_path',type=str,
        help="Path to output folder",required=True)

    # network + training
    parser.add_argument(
        '--config_file',dest="config_file",
        help="Path to network configuration file (yaml)",required=True)
    parser.add_argument(
        '--checkpoint_path',dest="checkpoint_path",
        help="Path to U-Net checkpoint",required=True)

    # inference specific
    parser.add_argument('--dev',dest='dev',
        help="Device for PyTorch training",choices=["cuda","cpu"],type=str)
    parser.add_argument(
        '--n_workers',dest='n_workers',
        help="No. of workers",default=1,type=int)
    parser.add_argument(
        '--downsample_rate',dest='downsample_rate',type=float,default=1.0,
        help="Resizes images with downsample rate")

    args = parser.parse_args()

    n_classes = 2
    output_file = open(args.metrics_path,'w')
    
    mod = "image" 
    keys = [mod]
    label_keys = ["label"]
    all_keys = [mod,*label_keys]

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

    intp = ["area","nearest"]
    intp_spacing = ["bilinear","nearest"]

    transforms = [
        monai.transforms.LoadImaged(all_keys),
        Index(["image"],{"T2WAx":0,"DWI":1}[args.mod],-1),
        monai.transforms.AddChanneld(all_keys),
        monai.transforms.Orientationd(all_keys,"RAS"),
        monai.transforms.EnsureTyped(all_keys),
        monai.transforms.Spacingd(
            all_keys,tuple(spacing_dict[args.mod]),mode=intp_spacing),
        monai.transforms.Resized(
            all_keys,tuple(size_dict[args.mod]),mode=intp),
        monai.transforms.ScaleIntensityd(all_keys,0,1),
        monai.transforms.EnsureTyped(all_keys)]

    if network_config["spatial_dimensions"] == 2:
        transforms.append(
            SlicesToFirst(all_keys))
        collate_fn = collate_last_slice
    else:
        collate_fn = None

    dataset = monai.apps.DecathlonDataset(
        root_dir=args.root_dir,
        task="Task05_Prostate",
        section="training",
        download=True,
        transform=monai.transforms.Compose(transforms),
        val_frac=0.)

    loader = torch.utils.data.DataLoader(
        dataset,batch_size=2,
        num_workers=args.n_workers,collate_fn=collate_fn)

    unet = UNetPL(image_key=keys[0],label_key=label_keys[0],
                  n_classes=n_classes,**network_config)
    state_dict = torch.load(
        args.checkpoint_path,map_location=args.dev)['state_dict']
    inc = unet.load_state_dict(state_dict)
    print("Incompatible keys:",inc)

    unet = unet.to(args.dev)
    unet.eval()

    if n_classes > 2:
        metrics = {
            "iou":JaccardIndex(n_classes,reduction="none"),
            "prec":Precision(
                n_classes,mdmc_average="samplewise",average="none"),
            "f1-score":FBetaScore(
                n_classes,mdmc_average="samplewise",average="none")}
    else:
        metrics = {
            "iou":JaccardIndex(n_classes),
            "prec":Precision(None),
            "f1-score":FBetaScore(None,average="micro")}
        jaccard_validation = JaccardIndex(n_classes)
        precision_validation = Precision(None)
    
    metrics = {k:metrics[k].to(args.dev) for k in metrics}

    print("Validating...")
    for s in tqdm(loader):
        with torch.no_grad():
            pred = unet(s[mod].to(args.dev))
            y = torch.squeeze(s[label_keys[0]],1).to(args.dev)
            try: y = torch.round(y).int()
            except: pass

            for k in metrics:
                metrics[k].update(pred,y)

    for k in metrics:
        metrics[k] = metrics[k].compute().to("cpu")
    
    if n_classes > 2:
        # multi class scenario
        for k in metrics:
            for i,v in enumerate(metrics[k]):
                S = "{},{},{},{}".format(
                    k,0,i,float(v.numpy()))
                print(S)
                output_file.write(S+'\n')
            
    else:
        for k in metrics:
            S = "{},{},{},{}".format(
                k,0,0,float(metrics[k].numpy()))
            print(S)
            output_file.write(S+'\n')