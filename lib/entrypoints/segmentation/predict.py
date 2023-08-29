import argparse
import json
import sys
import gc
import os
import numpy as np
import torch
import SimpleITK as sitk
import monai
from pathlib import Path
from tqdm import trange

from ...utils import (
    collate_last_slice,
    SlicesToFirst,
    safe_collate)
from ...monai_transforms import get_transforms_unet as get_transforms
from ...modules.layers import ResNet
from ...utils.network_factories import get_segmentation_network
from ...modules.config_parsing import parse_config_unet,parse_config_ssl
from ...utils.parser import parse_ids
torch.backends.cudnn.benchmark = True

def if_none_else(x,obj):
    if x is None:
        return obj
    return x

def inter_size(a,b):
    return len(set.intersection(set(a),set(b)))

def main(arguments):
    parser = argparse.ArgumentParser()

    # data
    parser.add_argument(
        '--dataset_json',dest='dataset_json',type=str,
        help="JSON containing dataset information",required=True)
    parser.add_argument(
        '--pred_ids',dest='pred_ids',type=str,nargs="+",default=None,
        help="IDs used for prediction.")
    parser.add_argument(
        '--excluded_ids',dest="excluded_ids",nargs="+",default=None,
        help="Excludes these IDs from training and testing")
    parser.add_argument(
        '--resize_size',dest='resize_size',type=float,nargs='+',default=None,
        help="Input size for network")
    parser.add_argument(
        '--resize_keys',dest='resize_keys',type=str,nargs='+',default=None,
        help="Keys that will be resized to input size")
    parser.add_argument(
        '--pad_size',dest='pad_size',action="store",
        default=None,type=float,nargs='+',
        help="Padding size after resizing.")
    parser.add_argument(
        '--crop_size',dest='crop_size',action="store",
        default=None,type=float,nargs='+',
        help="Crop size after padding.")
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
        '--adc_keys',dest='adc_keys',type=str,nargs='+',
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
        '--bottleneck_classification',dest='bottleneck_classification',
        action="store_true",
        help="Predicts the maximum class in the output using the bottleneck \
            features.")
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
    
    # network + inference
    parser.add_argument(
        '--config_file',dest="config_file",
        help="Path to network configuration file (yaml)",
        required=True)
    parser.add_argument(
        '--unet_model',dest='unet_model',action="store",
        choices=["unet","unetpp","brunet","unetr","swin"],
        default="unet",help="Specifies which UNet model is used")
    parser.add_argument(
        '--mode',dest='mode',action='store',
        choices=["image","deep_features","bounding_box"],default="image")
    parser.add_argument(
        '--res_config_file',dest='res_config_file',action="store",default=None,
        help="Uses a ResNet as a backbone (depths are inferred from this). \
            This config file is then used to parameterise the ResNet.")
    parser.add_argument(
        '--checkpoints',dest='checkpoints',action="store",required=True,
        help="Paths to checkpoints.",nargs="+")
    parser.add_argument(
        '--paired',dest='paired',
        action="store_true",
        help="Test checkpoints only on the corresponding pred_ids.")
    parser.add_argument(
        '--per_sample',dest='per_sample',
        action="store_true",
        help="Also calculates metrics on a per sample basis.")

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
    parser.add_argument(
        '--output_path',dest='output_path',type=str,required=True,
        help='Path to output masks.')

    args = parser.parse_args(arguments)

    if args.possible_labels == 2 or args.positive_labels is not None:
        n_classes = 2
    else:
        n_classes = args.possible_labels

    keys = args.image_keys
    
    mask_image_keys = if_none_else(args.mask_image_keys,[])
    adc_keys = if_none_else(args.adc_keys,[])
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

    adc_keys = [k for k in adc_keys if k in keys]
    intp = []
    intp_resampling_augmentations = []
    for k in keys:
        if k in mask_image_keys:
            intp.append("nearest")
            intp_resampling_augmentations.append("nearest")
        else:
            intp.append("area")
            intp_resampling_augmentations.append("bilinear")
    non_adc_keys = [k for k in keys if k not in adc_keys]
    intp.extend(["area"]*len(aux_keys))
    intp.extend(["nearest"]*len(aux_mask_keys))
    intp_resampling_augmentations.extend(["bilinear"]*len(aux_keys))
    intp_resampling_augmentations.extend(["nearest"]*len(aux_mask_keys))
    all_keys = [*keys,*aux_keys,*aux_mask_keys]
    all_keys_t = [*all_keys,*feature_keys]
    if args.resize_size is not None:
        args.resize_size = [round(x) for x in args.resize_size]

    data_dict = json.load(open(args.dataset_json,'r'))
    if args.excluded_ids is not None:
        data_dict = {k:data_dict[k] for k in data_dict
                     if k not in args.excluded_ids}

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
    
    network_config,loss_key = parse_config_unet(
        args.config_file,len(keys),n_classes)
    
    label_mode = "binary" if n_classes == 2 else "cat"
    transform_arguments = {
        "all_keys": all_keys,
        "image_keys": keys,
        "label_keys": None,
        "non_adc_keys": non_adc_keys,
        "adc_keys": adc_keys,
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
        "resize_size": args.resize_size,
        "crop_size": args.crop_size,
        "label_mode": label_mode,
        "fill_missing": args.missing_to_empty is not None,
        "brunet": args.unet_model == "brunet",
        "track_meta":True,
        "convert_to_tensor":False}
    
    transforms = [
        *get_transforms("pre",**transform_arguments),
        *get_transforms("post",**transform_arguments)]

    if network_config["spatial_dimensions"] == 2:
        transforms.append(
            SlicesToFirst(["image"]))
        collate_fn = collate_last_slice
    else:
        collate_fn = safe_collate

    torch.cuda.empty_cache()

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
        all_feature_params = {}
        all_feature_params["mean"] = torch.zeros(
            len(feature_keys),dtype=torch.float32,device=dev)
        all_feature_params["std"] = torch.ones(
            len(feature_keys),dtype=torch.float32,device=dev)
    else:
        all_feature_params = None

    if args.res_config_file is not None:
        _,network_config_ssl = parse_config_ssl(
            args.res_config_file,0.,len(keys),network_config["batch_size"])
        for k in ['weight_decay','learning_rate','batch_size']:
            if k in network_config_ssl:
                del network_config_ssl[k]
        if args.unet_model == "brunet":
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

    unet = get_segmentation_network(
        net_type=args.unet_model,
        encoding_operations=encoding_operations,
        network_config=network_config,
        loss_params={},
        bottleneck_classification=args.bottleneck_classification,
        clinical_feature_keys=feature_keys,
        all_aux_keys=aux_keys,
        clinical_feature_params=all_feature_params,
        clinical_feature_key_net=feature_key_net,
        aux_key_net=aux_key_net,
        max_epochs=100,
        picai_eval=args.picai_eval,
        lr_encoder=None,
        cosine_decay=False,
        encoder_checkpoint=args.bottleneck_classification,
        res_config_file=args.res_config_file,
        deep_supervision=False,
        n_classes=n_classes,
        keys=keys,
        train_loader_call=None,
        random_crop_size=None,
        crop_size=args.crop_size,
        pad_size=args.pad_size,
        resize_size=args.resize_size)
        
    if args.pred_ids is not None:
        args.pred_ids = parse_ids(args.pred_ids)
    else:
        args.pred_ids = [[k for k in data_dict]]
    n_ckpt = len(args.checkpoints)
    n_data = len(args.pred_ids)
    output = {}
    for pred_idx in range(n_data):
        pred_ids = [k for k in args.pred_ids[pred_idx]
                    if k in data_dict]
        curr_dict = {k:data_dict[k] for k in pred_ids}
        data_list = [curr_dict[k] for k in curr_dict]
        
        transform_input = transforms[0]
        transforms_preprocess = monai.transforms.Compose(transforms[1:])
        transforms_postprocess = monai.transforms.Invertd(
            "image",transforms_preprocess)

        if args.paired == True:
            checkpoint_list = [args.checkpoints[pred_idx]]
        else:
            checkpoint_list = args.checkpoints
        
        networks = []
        for checkpoint in checkpoint_list:
            state_dict = torch.load(checkpoint)["state_dict"]
            state_dict = {k:state_dict[k] for k in state_dict
                          if "deep_supervision_ops" not in k}
            unet.load_state_dict(state_dict)
            unet = unet.to(args.dev)
            unet.eval()
            networks.append(unet)

        for i in trange(len(data_list)):
            data_element = data_list[i]
            data_element = transform_input(data_element)
            data_element = transforms_preprocess(data_element)
            for k in all_keys:
                data_element[k] = data_element[k].to(args.dev).unsqueeze(0)
            pred_id = pred_ids[i]

            if args.mode == "image" or args.mode == "bounding_box":
                pred = [
                    network.predict_step(data_element,0)[0][0]
                    for network in networks]
                pred = [transforms_postprocess({"image":p})["image"][0]
                        for p in pred]
            elif args.mode == "deep_features":
                pred = [
                    network.predict_step(
                        data_element,0,return_bottleneck=True)[2][0]
                    for network in networks]
                pred = [x.flatten(start_dim=1).max(1).values
                        for x in pred]

            pred = torch.stack(pred).mean(0).detach().cpu()
            if args.mode == "image":
                pred = np.int32(np.round(pred))
                pred = np.transpose(pred,[2,1,0])
                pred_image = sitk.GetImageFromArray(pred)
                pred_image.CopyInformation(
                    sitk.ReadImage(curr_dict[pred_id]["image"]))
                pred_image = sitk.Cast(pred_image,sitk.sitkInt16)
                output_path = Path(
                    os.path.join(args.output_path,pred_id + ".nii.gz"))
                output_path.parent.mkdir(parents=True,exist_ok=True)
                sitk.WriteImage(pred_image,str(output_path))
            elif args.mode == "deep_features":
                pred = pred
                output[pred_id] = pred.tolist()
            elif args.mode == "bounding_box":
                pred = monai.transforms.KeepLargestConnectedComponent()(pred)
                coords = np.where(pred > 0)
                bb_min = [x.min() for x in coords]
                bb_max = [x.max() for x in coords]
                output[pred_id] = [*bb_min,*bb_max]
            gc.collect()

    if args.mode in ["deep_features","bounding_box"]:
        with open(args.output_path,"w") as o:
            json.dump(output,o)