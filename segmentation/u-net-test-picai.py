import argparse
import random
import json
import time
import numpy as np
import torch
import monai
import SimpleITK as sitk
from tqdm import tqdm
from multiprocessing import Queue, Process

import sys
sys.path.append(r"..")
from lib.utils import (
    collate_last_slice,
    SlicesToFirst,
    safe_collate)
from lib.modules.layers import ResNet
from lib.modules.segmentation import UNet
from lib.modules.segmentation.unetpp import UNetPlusPlus
from lib.modules.config_parsing import parse_config_unet,parse_config_ssl
from lib.modules.extract_lesion_candidates import (
    extract_lesion_candidates,extract_lesion_candidates_dynamic)
from lib.monai_transforms import get_transforms_unet as get_transforms

from picai_eval import evaluate

def post_process_from_queue(queue,out_queue):
    while True:
        o = queue.get()
        if o == "DONE":
            break
        else:
            lesions,conf,dyn = extract_lesion_candidates(o[0])
            out_queue.put((lesions,o[1]))

def start_procs(in_q,num_of_reader_procs,out_q):
    all_reader_procs = list()
    for _ in range(0, num_of_reader_procs):
        reader_p = Process(target=post_process_from_queue,
                           args=(in_q,out_q))
        reader_p.daemon = True
        reader_p.start()
        all_reader_procs.append(reader_p)
    return all_reader_procs

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
        '--subsample_size',dest='subsample_size',type=int,
        help="Subsamples data to a given size",
        default=None)
    parser.add_argument(
        '--missing_to_empty',dest='missing_to_empty',
        type=str,nargs="+",choices=["image","mask"],
        help="If some images or masks are missing, assume they are empty \
            tensors.")

    # network
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
        '--checkpoint',dest='checkpoint',action="store",
        help="Checkpoint path")
    
    # prediction
    parser.add_argument(
        '--tta',dest='tta',
        help="TTA dimensions",default=None,nargs="+",type=str)

    # 
    parser.add_argument(
        '--dev',dest='dev',type=str,
        help="Device for PyTorch training")
    parser.add_argument(
        '--n_workers',dest='n_workers',
        help="Number of workers",default=1,type=int)

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
    resize_keys = []
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
    if args.subsample_size is not None:
        ss = np.random.choice(
            sorted(list(data_dict.keys())),args.subsample_size,replace=False)
        data_dict = {k:data_dict[k] for k in ss}

    all_pids = [k for k in data_dict]

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
        "input_size": None,
        "crop_size": args.crop_size,
        "label_mode": label_mode,
        "fill_missing": args.missing_to_empty is not None,
        "brunet": args.brunet}

    transforms_train = [
        *get_transforms("pre",**transform_arguments),
        *get_transforms("post",**transform_arguments)]

    if network_config["spatial_dimensions"] == 2:
        transforms_train.append(
            SlicesToFirst(["image","mask"],"mask"))
        collate_fn = collate_last_slice
    else:
        collate_fn = safe_collate

    data_list = [data_dict[i] for i in all_pids]
    dataset = monai.data.Dataset(
        data_list,
        monai.transforms.Compose(transforms_train))
        # calculate the mean/std of tabular features
    if feature_keys is not None:
        all_params = {"mean":[],"std":[]}
        for kk in feature_keys:
            f = np.array([x[kk] for x in data_list])
            all_params["mean"].append(np.mean(f))
            all_params["std"].append(np.std(f))
        all_params["mean"] = torch.as_tensor(
            all_params["mean"],dtype=torch.float32,device=args.dev)
        all_params["std"] = torch.as_tensor(
            all_params["std"],dtype=torch.float32,device=args.dev)
    else:
        all_params = None
        
    if args.res_config_file is not None:
        _,network_config_ssl = parse_config_ssl(
            args.res_config_file,0.,len(keys),network_config["batch_size"])
        for k in ['weight_decay','learning_rate','batch_size']:
            if k in network_config_ssl:
                del network_config_ssl[k]
        res_net = ResNet(**network_config_ssl)
        backbone = res_net.backbone
        network_config['depth'] = [
            backbone.structure[0][0],
            *[x[0] for x in backbone.structure]]
        network_config['kernel_sizes'] = [3 for _ in network_config['depth']]
        network_config['strides'] = [2 for _ in network_config['depth']]
        res_ops = [backbone.input_layer,*backbone.operations]
        res_pool_ops = [backbone.first_pooling,*backbone.pooling_operations]
        encoding_operations = torch.nn.ModuleList(
            [torch.nn.ModuleList([a,b]) 
                for a,b in zip(res_ops,res_pool_ops)])
    else:
        encoding_operations = None

    for k in ['weight_decay','learning_rate','batch_size','loss_fn']:
        if k in network_config:
            del network_config[k]

    if args.unet_pp == True:
        unet = UNetPlusPlus(
            encoding_operations=encoding_operations,
            n_classes=n_classes,
            bottleneck_classification=args.bottleneck_classification,
            skip_conditioning=len(all_aux_keys),
            feature_conditioning=len(feature_keys),
            feature_conditioning_params=all_params,
            **network_config)
    else:
        unet = UNet(
            encoding_operations=encoding_operations,
            n_classes=n_classes,
            bottleneck_classification=args.bottleneck_classification,
            skip_conditioning=len(all_aux_keys),
            feature_conditioning=len(feature_keys),
            feature_conditioning_params=all_params,
            deep_supervision=args.deep_supervision,
            **network_config)

    state_dict = torch.load(
        args.checkpoint,map_location=args.dev)['state_dict']
    inc = unet.load_state_dict(state_dict)
    unet.to(args.dev)

    y_pred = []
    y_true = []
    if args.tta is None:
        tta = []
    else:
        tta = [[int(y) for y in x.split(",")] for x in args.tta]
        
    n_processes = np.maximum(args.n_workers - 2,1)
    q = Queue()
    out_q = Queue()
    procs = start_procs(q,n_processes,out_q)
    for entry in tqdm(dataset):
        b = time.time()
        image = entry["image"].unsqueeze(0).to(args.dev)
        mask = entry["mask"].squeeze().numpy()

        if len(aux_keys) == 0: X_skip = None
        else: X_skip = entry[aux_key_net].unsqueeze(0).to(args.dev)
        if len(feature_keys) == 0: X_features = None
        else: X_features = entry["tabular_features"].unsqueeze(0).to(args.dev)

        output,bn_class,_ = unet.forward(image,X_skip,X_features)
        
        outputs = [output[0,0].cpu().detach().numpy()]
        bn_classes = [outputs[-1].max()]
        for d in tta:
            flipped_X = torch.flip(image,d)
            if X_skip is not None:
                flipped_X_skip = torch.flip(X_skip,d) 
            else:
                flipped_X_skip = None
            output,bn_class,_ = unet.forward(
                flipped_X,X_skip_layer=flipped_X_skip,
                X_feature_conditioning=X_features)
            output = torch.flip(output,d)[0].cpu().detach().numpy()
            outputs.append(output[0,0])
            bn_classes.append(outputs[-1].max())
        prob_map = sum(outputs) / len(outputs)
        q.put((prob_map,mask))
    for _ in range(n_processes):
        q.put("DONE")
    cont = True
    while cont == True:
        out = out_q.get()
        y_pred.append(out[0])
        y_true.append(out[1])
        if len(y_pred) == len(dataset):
            cont = False
    metrics = evaluate(
        y_pred,y_true,
        num_parallel_calls=args.n_workers)
    print_dict = {"ap":metrics.AP,"R":metrics.score,"auroc":metrics.auroc}
    for k in print_dict:
        print("{},{},{}".format(args.checkpoint,k,print_dict[k]))