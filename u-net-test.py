import argparse
import random
import json
import os

import numpy as np
import torch
import monai
import wandb
import gc
from copy import deepcopy
from sklearn.model_selection import KFold,train_test_split
from tqdm import tqdm

from pytorch_lightning import Trainer
from pytorch_lightning.callbacks import EarlyStopping,StochasticWeightAveraging
from pytorch_lightning.loggers import WandbLogger
from pytorch_lightning.callbacks.model_checkpoint import ModelCheckpoint
from pytorch_lightning.callbacks import RichProgressBar

from lib.utils import (
    PartiallyRandomSampler,
    get_loss_param_dict,
    collate_last_slice,
    RandomSlices,
    SlicesToFirst,
    safe_collate)
from lib.monai_transforms import get_transforms_unet as get_transforms
from lib.monai_transforms import get_augmentations_unet as get_augmentations
from lib.modules.layers import ResNet
from lib.modules.segmentation_pl import UNetPL,UNetPlusPlusPL,BrUNetPL
from lib.modules.config_parsing import parse_config_unet,parse_config_ssl

torch.backends.cudnn.benchmark = True

def if_none_else(x,obj):
    if x is None:
        return obj
    return x

def inter_size(a,b):
    return len(set.intersection(set(a),set(b)))

if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    # data
    parser.add_argument(
        '--dataset_json',dest='dataset_json',type=str,
        help="JSON containing dataset information",required=True)
    parser.add_argument(
        '--input_size',dest='input_size',type=float,nargs='+',default=None,
        help="Input size for network")
    parser.add_argument(
        '--resize_keys',dest='resize_keys',type=str,nargs='+',default=None,
        help="Keys that will be resized to input size")
    parser.add_argument(
        '--crop_size',dest='crop_size',action="store",
        default=None,type=float,nargs='+',
        help="Size of central crop after resizing (if none is specified then\
            no cropping is performed).")
    parser.add_argument(
        '--target_spacing',dest='target_spacing',action="store",default=None,
        help="Resamples all images to target spacing",nargs='+',type=float)
    parser.add_argument(
        '--image_keys',dest='image_keys',type=str,nargs='+',
        help="Image keys in the dataset JSON. First key is used as template",
        required=True)
    parser.add_argument(
        '--skip_key',dest='skip_key',type=str,default=None,
        nargs='+',
        help="Key for image in the dataset JSON that is concatenated to the \
            skip connections.")
    parser.add_argument(
        '--skip_mask_key',dest='skip_mask_key',type=str,
        nargs='+',default=None,
        help="Key for mask in the dataset JSON that is appended to the skip \
            connections (can be useful for including prior mask as feature).")
    parser.add_argument(
        '--mask_image_keys',dest='mask_image_keys',type=str,nargs='+',
        help="Keys corresponding to input images which are segmentation masks",
        default=None)
    parser.add_argument(
        '--adc_image_keys',dest='adc_image_keys',type=str,nargs='+',
        help="Keys corresponding to input images which are ADC maps \
            (normalized differently)",
        default=None)
    parser.add_argument(
        '--feature_keys',dest='feature_keys',type=str,nargs='+',
        help="Keys corresponding to tabular features in the JSON dataset",
        default=None)
    parser.add_argument(
        '--adc_factor',dest='adc_factor',type=float,default=1/3,
        help="Multiplies ADC images by this factor.")
    parser.add_argument(
        '--mask_keys',dest='mask_keys',type=str,nargs='+',
        help="Mask key in the dataset JSON.",
        required=True)
    parser.add_argument(
        '--bottleneck_classification',dest='bottleneck_classification',
        action="store_true",
        help="Predicts the maximum class in the output using the bottleneck \
            features.")
    parser.add_argument(
        '--deep_supervision',dest='deep_supervision',
        action="store_true",
        help="Triggers deep supervision.")
    parser.add_argument(
        '--possible_labels',dest='possible_labels',type=int,nargs='+',
        help="All the possible labels in the data.",
        required=True)
    parser.add_argument(
        '--positive_labels',dest='positive_labels',type=int,nargs='+',
        help="Labels that should be considered positive (binarizes labels)",
        default=None)
    parser.add_argument(
        '--missing_to_empty',dest='missing_to_empty',
        type=str,nargs="+",choices=["image","mask"],
        help="If some images or masks are missing, assume they are empty \
            tensors.")

    # network + training
    parser.add_argument(
        '--config_file',dest="config_file",
        help="Path to network configuration file (yaml)",
        required=True)
    parser.add_argument(
        '--unet_pp',dest='unet_pp',action="store_true",
        help="Uses U-Net++ rather than U-Net")
    parser.add_argument(
        '--brunet',dest='brunet',action="store_true",
        help="Uses BrU-Net rather than U-Net")
    parser.add_argument(
        '--res_config_file',dest='res_config_file',action="store",default=None,
        help="Uses a ResNet as a backbone (depths are inferred from this). \
            This config file is then used to parameterise the ResNet.")
    parser.add_argument(
        '--checkpoint',dest='checkpoint',action="store",required=True,
        help="Path to checkpoint.")
    
    # network, pipeline
    parser.add_argument(
        '--dev',dest='dev',type=str,
        help="Device for PyTorch training")
    parser.add_argument(
        '--n_workers',dest='n_workers',
        help="Number of workers",default=1,type=int)
    parser.add_argument(
        '--n_devices',dest='n_devices',
        help="Number of devices",default=1,type=int)
    parser.add_argument(
        '--picai_eval',dest='picai_eval',action="store_true",
        help="Validates model using PI-CAI metrics.")

    args = parser.parse_args()

    if args.possible_labels == 2 or args.positive_labels is not None:
        n_classes = 2
    else:
        n_classes = args.possible_labels

    keys = args.image_keys
    label_keys = args.mask_keys
    
    mask_image_keys = if_none_else(args.mask_image_keys,[])
    adc_image_keys = if_none_else(args.adc_image_keys,[])
    aux_keys = if_none_else(args.skip_key,[])
    aux_mask_keys = if_none_else(args.skip_mask_key,[])
    resize_keys = if_none_else(args.resize_keys,[])
    feature_keys = if_none_else(args.feature_keys,[])
    
    all_aux_keys = aux_keys + aux_mask_keys
    if len(all_aux_keys) > 0:
        aux_key_net = "aux_key"
    else:
        aux_key_net = None
    if len(feature_keys) > 0:
        feature_key_net = "tabular_features"
    else:
        feature_key_net = None

    adc_image_keys = [k for k in adc_image_keys if k in keys]
    intp = []
    intp_resampling_augmentations = []
    for k in keys:
        if k in mask_image_keys:
            intp.append("nearest")
            intp_resampling_augmentations.append("nearest")
        else:
            intp.append("area")
            intp_resampling_augmentations.append("bilinear")
    non_adc_keys = [k for k in keys if k not in adc_image_keys]
    intp.extend(["nearest"]*len(label_keys))
    intp.extend(["area"]*len(aux_keys))
    intp.extend(["nearest"]*len(aux_mask_keys))
    intp_resampling_augmentations.extend(["nearest"]*len(label_keys))
    intp_resampling_augmentations.extend(["bilinear"]*len(aux_keys))
    intp_resampling_augmentations.extend(["nearest"]*len(aux_mask_keys))
    all_keys = [*keys,*label_keys,*aux_keys,*aux_mask_keys]
    all_keys_t = [*all_keys,*feature_keys]
    if args.input_size is not None:
        args.input_size = [round(x) for x in args.input_size]

    data_dict = json.load(open(args.dataset_json,'r'))
    if args.missing_to_empty is None:
        data_dict = {
            k:data_dict[k] for k in data_dict
            if inter_size(data_dict[k],set(all_keys_t)) == len(all_keys_t)}
    else:
        if "image" in args.missing_to_empty:
            obl_keys = [*aux_keys,*aux_mask_keys,*feature_keys]
            opt_keys = keys
            data_dict = {
                k:data_dict[k] for k in data_dict
                if inter_size(data_dict[k],obl_keys) == len(obl_keys)}
            data_dict = {
                k:data_dict[k] for k in data_dict
                if inter_size(data_dict[k],opt_keys) > 0}
        if "mask" in args.missing_to_empty:
            data_dict = {
                k:data_dict[k] for k in data_dict
                if inter_size(data_dict[k],set(mask_image_keys)) >= 0}

    for kk in feature_keys:
        data_dict = {
            k:data_dict[k] for k in data_dict
            if np.isnan(data_dict[k][kk]) == False}

    all_pids = [k for k in data_dict]
    data_list = [data_dict[k] for k in all_pids]
    
    network_config,loss_key = parse_config_unet(
        args.config_file,len(keys),n_classes)
    
    label_mode = "binary" if n_classes == 2 else "cat"
    transform_arguments = {
        "all_keys": all_keys,
        "image_keys": keys,
        "label_keys": label_keys,
        "non_adc_keys": non_adc_keys,
        "adc_image_keys": adc_image_keys,
        "target_spacing": args.target_spacing,
        "intp": intp,
        "intp_resampling_augmentations": intp_resampling_augmentations,
        "possible_labels": args.possible_labels,
        "positive_labels": args.positive_labels,
        "adc_factor": args.adc_factor,
        "all_aux_keys": all_aux_keys,
        "resize_keys": resize_keys,
        "feature_keys": feature_keys,
        "aux_key_net": aux_key_net,
        "feature_key_net": feature_key_net,
        "input_size": args.input_size,
        "crop_size": args.crop_size,
        "label_mode": label_mode,
        "fill_missing": args.missing_to_empty is not None,
        "brunet": args.brunet}

    transforms = [
        *get_transforms("pre",**transform_arguments),
        *get_transforms("post",**transform_arguments)]

    if network_config["spatial_dimensions"] == 2:
        transforms.append(
            SlicesToFirst(["image","mask"]))
        collate_fn = collate_last_slice
    else:
        collate_fn = safe_collate

    
    torch.cuda.empty_cache()

    dataset = monai.data.Dataset(
        data_list,
        monai.transforms.Compose(transforms))

    # correctly assign devices
    if ":" in args.dev:
        devices = args.dev.split(":")[-1].split(",")
        devices = [int(i) for i in devices]
        dev = args.dev.split(":")[0]
    else:
        devices = args.n_devices
        dev = args.dev

    # calculate the mean/std of tabular features
    if feature_keys is not None:
        all_params = {"mean":[],"std":[]}
        for kk in feature_keys:
            f = np.array([x[kk] for x in train_list])
            all_params["mean"].append(np.mean(f))
            all_params["std"].append(np.std(f))
        all_params["mean"] = torch.as_tensor(
            all_params["mean"],dtype=torch.float32,device=dev)
        all_params["std"] = torch.as_tensor(
            all_params["std"],dtype=torch.float32,device=dev)
    else:
        all_params = None

    loader = monai.data.ThreadDataLoader(
        dataset,batch_size=1,
        num_workers=args.n_workers,
        collate_fn=collate_fn,pin_memory=True,
        persistent_workers=True,drop_last=True)

    if args.res_config_file is not None:
        _,network_config_ssl = parse_config_ssl(
            args.res_config_file,0.,len(keys),network_config["batch_size"])
        for k in ['weight_decay','learning_rate','batch_size']:
            if k in network_config_ssl:
                del network_config_ssl[k]
        if args.brunet == True:
            n = len(keys)
            nc = network_config_ssl["backbone_args"]["in_channels"]
            network_config_ssl["backbone_args"]["in_channels"] = nc // n
            res_net = [ResNet(**network_config_ssl) for _ in keys]
        else:
            res_net = [ResNet(**network_config_ssl)]
        backbone = [x.backbone for x in res_net]
        network_config['depth'] = [
            backbone[0].structure[0][0],
            *[x[0] for x in backbone[0].structure]]
        network_config['kernel_sizes'] = [3 for _ in network_config['depth']]
        # the last sum is for the bottleneck layer
        network_config['strides'] = [2]
        network_config['strides'].extend(network_config_ssl[
            "backbone_args"]["maxpool_structure"])
        res_ops = [[x.input_layer,*x.operations] for x in backbone]
        res_pool_ops = [[x.first_pooling,*x.pooling_operations]
                        for x in backbone]
        
        encoding_operations = [torch.nn.ModuleList([]) for _ in res_ops]
        for i in range(len(res_ops)):
            A = res_ops[i]
            B = res_pool_ops[i]
            for a,b in zip(A,B):
                encoding_operations[i].append(
                    torch.nn.ModuleList([a,b]))
        encoding_operations = torch.nn.ModuleList(encoding_operations)
    else:
        encoding_operations = [None]

    if args.brunet == True:
        nc = network_config["n_channels"]
        network_config["n_channels"] = nc // len(keys)
        unet = BrUNetPL(
            encoders=encoding_operations,
            image_keys=keys,label_key="mask",n_classes=n_classes,
            bottleneck_classification=args.bottleneck_classification,
            skip_conditioning=len(all_aux_keys),
            skip_conditioning_key=aux_key_net,
            feature_conditioning=len(feature_keys),
            feature_conditioning_params=all_params,
            feature_conditioning_key=feature_key_net,
            n_input_branches=len(keys),
            picai_eval=args.picai_eval,
            **network_config)
        if args.encoder_checkpoint is not None and args.res_config_file is None:
            for encoder,ckpt in zip(unet.encoders,args.encoder_checkpoint):
                encoder.load_state_dict(torch.load(ckpt)["state_dict"])
    elif args.unet_pp == True:
        encoding_operations = encoding_operations[0]
        unet = UNetPlusPlusPL(
            encoding_operations=encoding_operations,
            image_key="image",label_key="mask",n_classes=n_classes,
            bottleneck_classification=args.bottleneck_classification,
            skip_conditioning=len(all_aux_keys),
            skip_conditioning_key=aux_key_net,
            feature_conditioning=len(feature_keys),
            feature_conditioning_params=all_params,
            feature_conditioning_key=feature_key_net,
            picai_eval=args.picai_eval,
            **network_config)
    else:
        encoding_operations = encoding_operations[0]
        unet = UNetPL(
            encoding_operations=encoding_operations,
            image_key="image",label_key="mask",n_classes=n_classes,
            bottleneck_classification=args.bottleneck_classification,
            skip_conditioning=len(all_aux_keys),
            skip_conditioning_key=aux_key_net,
            feature_conditioning=len(feature_keys),
            feature_conditioning_params=all_params,
            feature_conditioning_key=feature_key_net,
            deep_supervision=args.deep_supervision,
            picai_eval=args.picai_eval,
            **network_config)
          
    unet.load_state_dict(torch.load(args.checkpoint)["state_dict"])
    trainer = Trainer(
        accelerator=dev,devices=devices)
    
    print("Validating...")
    test_metrics = trainer.test(
        unet,loader)[0]
    for k in test_metrics:
        out = test_metrics[k]
        if n_classes == 2:
            try:
                value = float(out.detach().numpy())
            except:
                value = float(out)
            x = "{},{},{},{}".format(args.checkpoint,k,0,0,value)
            print(x)
        else:
            for i,v in enumerate(out):
                x = "{},{},{},{}".format(args.checkpoint,k,0,i,v)
                print(x)
    gc.collect()
