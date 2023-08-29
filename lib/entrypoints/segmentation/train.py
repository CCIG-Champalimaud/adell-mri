import argparse
import random
import json
import os
import numpy as np
import torch
import monai
import gc
from sklearn.model_selection import KFold,train_test_split
from tqdm import tqdm

from lightning.pytorch import Trainer
from lightning.pytorch.callbacks import EarlyStopping
from lightning.pytorch.callbacks import RichProgressBar

import sys
from ...utils import (
    GetAllCropsd,
    PartiallyRandomSampler,
    get_loss_param_dict,
    collate_last_slice,
    RandomSlices,
    SlicesToFirst,
    safe_collate,
    safe_collate_crops)
from ...utils.pl_utils import get_ckpt_callback,get_logger,get_devices
from ...monai_transforms import get_transforms_unet as get_transforms
from ...monai_transforms import get_augmentations_unet as get_augmentations
from ...modules.layers import ResNet
from ...utils.network_factories import get_segmentation_network
from ...modules.config_parsing import parse_config_unet,parse_config_ssl
from ...utils.parser import parse_ids
from ...utils.sitk_utils import (
    spacing_values_from_dataset_json,get_spacing_quantile)

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
        '--dataset_json',dest='dataset_json',type=str,nargs="+",
        help="JSON containing dataset information",required=True)
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
        '--random_crop_size',dest='random_crop_size',action="store",
        default=None,type=float,nargs='+',
        help="Size of random crop (last step of the preprocessing pipeline).")
    parser.add_argument(
        '--n_crops',dest='n_crops',action="store",
        default=1,type=int,help="Number of random crops.")
    parser.add_argument(
        '--target_spacing',dest='target_spacing',action="store",default=None,
        help="Resamples all images to target spacing",nargs='+',type=str)
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
        '--t2_keys',dest='t2_keys',type=str,nargs='+',
        help="Keys corresponding to T2W images",default=None)
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
        '--constant_ratio',dest='constant_ratio',type=float,default=None,
        help="If there are masks with only one value, defines how many are\
            included relatively to masks with more than one value.")
    parser.add_argument(
        '--subsample_size',dest='subsample_size',type=int,
        help="Subsamples data to a given size",
        default=None)
    parser.add_argument(
        '--cache_rate',dest='cache_rate',type=float,
        help="Rate of samples to be cached",
        default=1.0)
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
        '--unet_model',dest='unet_model',action="store",
        choices=["unet","unetpp","brunet","unetr","swin"],
        default="unet",help="Specifies which UNet model is used")
    parser.add_argument(
        '--res_config_file',dest='res_config_file',action="store",default=None,
        help="Uses a ResNet as a backbone (depths are inferred from this). \
            This config file is then used to parameterise the ResNet.")
    parser.add_argument(
        '--encoder_checkpoint',dest='encoder_checkpoint',action="store",default=None,
        nargs="+",
        help="Checkpoint for encoder backbone checkpoint")
    parser.add_argument(
        '--lr_encoder',dest='lr_encoder',action="store",default=None,type=float,
        help="Sets learning rate for encoder.")
    parser.add_argument(
        '--cosine_decay',dest='cosine_decay',action="store_true",
        default=False,help="Decreases the LR using cosine decay.")
    parser.add_argument(
        '--from_checkpoint',dest='from_checkpoint',action="store",nargs="+",
        default=None,
        help="Uses this space-separated list of checkpoints as a starting\
            points for the network at each fold. If no checkpoint is provided\
            for a given fold, the network is initialized randomly.")
    parser.add_argument(
        '--resume_from_last',dest='resume_from_last',action="store_true",
        default=None,
        help="Resumes from the last checkpoint stored for a given fold.")
    
    # training
    parser.add_argument(
        '--dev',dest='dev',type=str,
        help="Device for PyTorch training")
    parser.add_argument(
        '--seed',dest='seed',help="Random seed",default=42,type=int)
    parser.add_argument(
        '--n_workers',dest='n_workers',
        help="Number of workers",default=1,type=int)
    parser.add_argument(
        '--n_devices',dest='n_devices',
        help="Number of devices",default=1,type=int)
    parser.add_argument(
        '--augment',dest='augment',action="store",nargs="+",
        help="Sets data augmentation.",default=[])
    parser.add_argument(
        '--loss_gamma',dest="loss_gamma",
        help="Gamma for focal loss",default=2.0,type=float)
    parser.add_argument(
        '--loss_comb',dest="loss_comb",
        help="Relative weight for combined losses",default=0.5,type=float)
    parser.add_argument(
        '--loss_scale',dest="loss_scale",
        help="Loss scale (helpful for 16bit trainign",default=1.0,type=float)
    parser.add_argument(
        '--learning_rate',dest="learning_rate",
        help="Learning rate (overrides the lr specified in network_config)",
        default=None,type=float)
    parser.add_argument(
        '--batch_size',dest="batch_size",
        help="Batch size (overrides the batch size specified in network_config)",
        default=None,type=int)
    parser.add_argument(
        '--gradient_clip_val',dest="gradient_clip_val",
        help="Value for gradient clipping",
        default=0.0,type=float)
    parser.add_argument(
        '--max_epochs',dest="max_epochs",
        help="Maximum number of training epochs",default=100,type=int)
    parser.add_argument(
        '--dataset_iterations_per_epoch',dest="dataset_iterations_per_epoch",
        help="Number of dataset iterations per epoch",default=1.0,type=float)
    parser.add_argument(
        '--precision',dest='precision',type=str,default="32",
        help="Floating point precision",choices=["16","32","bf16"])
    parser.add_argument(
        '--n_folds',dest="n_folds",
        help="Number of validation folds",default=5,type=int)
    parser.add_argument(
        '--folds',dest="folds",
        help="Specifies the comma separated IDs for each fold (overrides\
            n_folds)",default=None,type=str,nargs='+')
    parser.add_argument(
        '--excluded_ids',dest="excluded_ids",nargs="+",default=None,
        help="Excludes these IDs from training and testing")
    parser.add_argument(
        '--use_val_as_train_val',dest='use_val_as_train_val',action="store_true",
        help="Use validation set as training validation set.",default=False)
    parser.add_argument(
        '--check_val_every_n_epoch',dest='check_val_every_n_epoch',type=int,
        default=1,help="Epoch frequency of validation.")
    parser.add_argument(
        '--picai_eval',dest='picai_eval',action="store_true",
        help="Validates model using PI-CAI metrics.")
    parser.add_argument(
        '--accumulate_grad_batches',dest='accumulate_grad_batches',type=int,
        default=1,help="Accumulate batches to calculate gradients.")
    parser.add_argument(
        '--checkpoint_dir',dest='checkpoint_dir',type=str,default="models",
        help='Path to directory where checkpoints will be saved.')
    parser.add_argument(
        '--checkpoint_name',dest='checkpoint_name',type=str,default=None,
        help='Checkpoint ID.')
    parser.add_argument(
        '--monitor',dest='monitor',type=str,default="val_loss",
        help='Metric to monitor when saving the best ckpt.')
    parser.add_argument(
        '--summary_dir',dest='summary_dir',type=str,default="summaries",
        help='Path to summary directory (for wandb).')
    parser.add_argument(
        '--summary_name',dest='summary_name',type=str,default=None,
        help='Summary name.')
    parser.add_argument(
        '--project_name',dest='project_name',type=str,default=None,
        help='Project name for wandb.')
    parser.add_argument(
        '--resume',dest='resume',type=str,default="allow",
        choices=["allow","must","never","auto","none"],
        help='Whether wandb project should be resumed (check \
            https://docs.wandb.ai/ref/python/init for more details).')
    parser.add_argument(
        '--metric_path',dest='metric_path',type=str,default="metrics.csv",
        help='Path to file with CV metrics + information.')
    parser.add_argument(
        '--early_stopping',dest='early_stopping',type=int,default=None,
        help="No. of checks before early stop (defaults to no early stop).")
    parser.add_argument(
        '--class_weights',dest='class_weights',type=str,nargs='+',
        help="Class weights (by alphanumeric order).",default=[1.])

    args = parser.parse_args(arguments)

    torch.manual_seed(args.seed)
    random.seed(args.seed)
    np.random.seed(args.seed)
    g = torch.Generator()
    g.manual_seed(args.seed)
    
    accelerator,devices,strategy = get_devices(args.dev)
    dev = args.dev.split(":")[0]

    if args.possible_labels == 2 or args.positive_labels is not None:
        n_classes = 2
    else:
        n_classes = args.possible_labels

    keys = args.image_keys
    label_keys = args.mask_keys
    
    mask_image_keys = if_none_else(args.mask_image_keys,[])
    adc_keys = if_none_else(args.adc_keys,[])
    t2_keys = if_none_else(args.t2_keys,[])
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
    intp.extend(["nearest"]*len(label_keys))
    intp.extend(["area"]*len(aux_keys))
    intp.extend(["nearest"]*len(aux_mask_keys))
    intp_resampling_augmentations.extend(["nearest"]*len(label_keys))
    intp_resampling_augmentations.extend(["bilinear"]*len(aux_keys))
    intp_resampling_augmentations.extend(["nearest"]*len(aux_mask_keys))
    all_keys = [*keys,*label_keys,*aux_keys,*aux_mask_keys]
    all_keys_t = [*all_keys,*feature_keys]
    if args.resize_size is not None:
        args.resize_size = [round(x) for x in args.resize_size]
    if args.crop_size is not None:
        args.crop_size = [round(x) for x in args.crop_size]
    if args.pad_size is not None:
        args.pad_size = [round(x) for x in args.pad_size]
    if args.random_crop_size is not None:
        args.random_crop_size = [round(x) for x in args.random_crop_size]
    label_mode = "binary" if n_classes == 2 else "cat"

    data_dict = {}
    for dataset_json in args.dataset_json:
        cur_dataset_dict = json.load(open(dataset_json,'r'))
        for k in cur_dataset_dict:
            data_dict[k] = cur_dataset_dict[k]
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

    if args.excluded_ids is not None:
        args.excluded_ids = parse_ids(args.excluded_ids,
                                      output_format="list")
        print("Removing IDs specified in --excluded_ids")
        prev_len = len(data_dict)
        data_dict = {k:data_dict[k] for k in data_dict
                     if k not in args.excluded_ids}
        print("\tRemoved {} IDs".format(prev_len - len(data_dict)))

    if args.target_spacing[0] == "infer":
        target_spacing_dict = spacing_values_from_dataset_json(
            data_dict,key=keys[0],n_workers=args.n_workers)

    all_pids = [k for k in data_dict]

    network_config,loss_key = parse_config_unet(
        args.config_file,len(keys),n_classes)
    if args.learning_rate is not None:
        network_config["learning_rate"] = args.learning_rate
    if args.batch_size is not None:
        network_config["batch_size"] = args.batch_size
    
    if args.folds is None:
        if args.n_folds > 1:
            fold_generator = KFold(
                args.n_folds,shuffle=True,random_state=args.seed).split(all_pids)
        else:
            fold_generator = iter(
                [train_test_split(range(len(all_pids)),test_size=0.2)])
    else:
        args.folds = parse_ids(args.folds)
        folds = []
        for fold_idx,val_ids in enumerate(args.folds):
            train_idxs = [i for i,x in enumerate(all_pids) if x not in val_ids]
            val_idxs = [i for i,x in enumerate(all_pids) if x in val_ids]
            if len(train_idxs) == 0:
                print("No train samples in fold {}".format(fold_idx))
                continue
            if len(val_idxs) == 0:
                print("No val samples in fold {}".format(fold_idx))
                continue
            folds.append([train_idxs,val_idxs])
        args.n_folds = len(folds)
        fold_generator = iter(folds)
    
    output_file = open(args.metric_path,'w')
    for val_fold in range(args.n_folds):
        print("="*80)
        print("Starting fold={}".format(val_fold))
        
        train_idxs,val_idxs = next(fold_generator)
        if args.use_val_as_train_val == False:
            train_idxs,train_val_idxs = train_test_split(train_idxs,test_size=0.15)
        else:
            train_val_idxs = val_idxs
        train_pids = [all_pids[i] for i in train_idxs]
        train_val_pids = [all_pids[i] for i in train_val_idxs]
        val_pids = [all_pids[i] for i in val_idxs]
        train_list = [data_dict[pid] for pid in train_pids]
        train_val_list = [data_dict[pid] for pid in train_val_pids]
        val_list = [data_dict[pid] for pid in val_pids]

        if args.target_spacing[0] == "infer":
            target_spacing = get_spacing_quantile(
                {k:target_spacing_dict[k] for k in train_pids})
        else:
            target_spacing = [float(x) for x in args.target_spacing]

        transform_arguments = {
            "all_keys": all_keys,
            "image_keys": keys,
            "label_keys": label_keys,
            "non_adc_keys": non_adc_keys,
            "adc_keys": adc_keys,
            "target_spacing": target_spacing,
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
            "pad_size": args.pad_size,
            "crop_size": args.crop_size,
            "random_crop_size": args.random_crop_size,
            "label_mode": label_mode,
            "fill_missing": args.missing_to_empty is not None,
            "brunet": args.unet_model == "brunet"}
        transform_arguments_val = {k:transform_arguments[k]
                                for k in transform_arguments}
        transform_arguments_val["random_crop_size"] = None
        transform_arguments_val["crop_size"] = None
        augment_arguments = {
            "augment":args.augment,
            "all_keys":all_keys,
            "image_keys":keys,
            "t2_keys":t2_keys,
            "random_crop_size":args.random_crop_size,
            "n_crops":args.n_crops}
        if args.random_crop_size:
            get_all_crops_transform = [GetAllCropsd(
                args.image_keys + ["mask"],args.random_crop_size)]
        else:
            get_all_crops_transform = []
        transforms_train = [
            *get_transforms("pre",**transform_arguments),
            get_augmentations(**augment_arguments),
            *get_transforms("post",**transform_arguments)]

        transforms_train_val = [
            *get_transforms("pre",**transform_arguments_val),
            *get_all_crops_transform,
            *get_transforms("post",**transform_arguments_val)]

        transforms_val = [
            *get_transforms("pre",**transform_arguments_val),
            *get_transforms("post",**transform_arguments_val)]

        if network_config["spatial_dimensions"] == 2:
            transforms_train.append(
                RandomSlices(["image","mask"],"mask",n=8,base=0.05))
            transforms_train_val.append(
                SlicesToFirst(["image","mask"]))
            transforms_val.append(
                SlicesToFirst(["image","mask"]))
            collate_fn_train = collate_last_slice
            collate_fn_val = collate_last_slice
        elif args.random_crop_size is not None:
            collate_fn_train = safe_collate_crops
            collate_fn_val = safe_collate
        else:
            collate_fn_train = safe_collate
            collate_fn_val = safe_collate
            
        callbacks = [RichProgressBar()]
        ckpt_path = None

        ckpt_callback,ckpt_path,status = get_ckpt_callback(
            checkpoint_dir=args.checkpoint_dir,
            checkpoint_name=args.checkpoint_name,
            max_epochs=args.max_epochs,
            resume_from_last=args.resume_from_last,
            val_fold=val_fold,
            monitor=args.monitor,
            metadata={"transform_arguments":transform_arguments})
        ckpt = ckpt_callback is not None
        if status == "finished":
            continue
        if ckpt_callback is not None:
            callbacks.append(ckpt_callback)

        if args.from_checkpoint is not None:
            if len(args.from_checkpoint) >= (val_fold+1):
                ckpt_path = args.from_checkpoint[val_fold]
                print("Resuming training from checkpoint in {}".format(ckpt_path))
        
        train_dataset = monai.data.CacheDataset(
            train_list,
            monai.transforms.Compose(transforms_train),
            num_workers=args.n_workers,cache_rate=args.cache_rate)
        train_dataset_val = monai.data.CacheDataset(
            train_val_list,
            monai.transforms.Compose(transforms_train_val),
            num_workers=args.n_workers)
        validation_dataset = monai.data.Dataset(
            val_list,
            monai.transforms.Compose(transforms_val))

        # calculate the mean/std of tabular features
        if feature_keys is not None:
            all_feature_params = {"mean":[],"std":[]}
            for kk in feature_keys:
                f = np.array([x[kk] for x in train_list])
                all_feature_params["mean"].append(np.mean(f))
                all_feature_params["std"].append(np.std(f))
            all_feature_params["mean"] = torch.as_tensor(
                all_feature_params["mean"],dtype=torch.float32,device=dev)
            all_feature_params["std"] = torch.as_tensor(
                all_feature_params["std"],dtype=torch.float32,device=dev)
        else:
            all_feature_params = None
            
        n_samples = int(len(train_dataset) * args.dataset_iterations_per_epoch)
        sampler = torch.utils.data.RandomSampler(
            ["element" for _ in train_idxs],
            num_samples=n_samples,
            replacement=len(train_dataset) < n_samples,
            generator=g)
        if isinstance(args.class_weights[0],str):
            ad = "adaptive" in args.class_weights[0]
        else:
            ad = False
        # include some constant label images
        if args.constant_ratio is not None or ad:
            cl = []
            pos_pixel_sum = 0
            total_pixel_sum = 0
            with tqdm(train_list) as t:
                t.set_description("Setting up partially random sampler/adaptive weights")
                for x in t:
                    I = set.intersection(set(label_keys),set(x.keys()))
                    if len(I) > 0:
                        masks = monai.transforms.LoadImaged(
                            keys=label_keys,allow_missing_keys=True)(x)
                        total = []
                        for k in I:
                            for u,c in zip(*np.unique(masks[k],return_counts=True)):
                                if u not in total:
                                    total.append(u)
                                if u != 0:
                                    pos_pixel_sum += c
                                total_pixel_sum += c
                        if len(total) > 1:
                            cl.append(1)
                        else:
                            cl.append(0)
                    else:
                        cl.append(0)
            adaptive_weights = len(cl)/np.sum(cl)
            adaptive_pixel_weights = total_pixel_sum / pos_pixel_sum
            if args.constant_ratio is not None:
                sampler = PartiallyRandomSampler(
                    cl,non_keep_ratio=args.constant_ratio,seed=args.seed)
                if args.class_weights[0] == "adaptive":
                    adaptive_weights = 1 + args.constant_ratio
        # weights to tensor
        if args.class_weights[0] == "adaptive":
            weights = adaptive_weights
        elif args.class_weights[0] == "adaptive_pixel":
            weights = adaptive_pixel_weights
        else:
            weights = torch.as_tensor(
                np.float32(np.array(args.class_weights)),dtype=torch.float32,
                device=dev)
        print("Weights set to:",weights)
        
        # get loss function parameters
        loss_params = get_loss_param_dict(
            weights=weights,gamma=args.loss_gamma,
            comb=args.loss_comb,scale=args.loss_scale)[loss_key]
        if "eps" in loss_params and args.precision != "32":
            if loss_params["eps"] < 1e-4:
                loss_params["eps"] = 1e-4

        if isinstance(devices,list):
            nw = args.n_workers // len(devices)
        else:
            nw = args.n_workers

        def train_loader_call(batch_size): 
            return monai.data.ThreadDataLoader(
                dataset=train_dataset,batch_size=batch_size, # noqa: F821
                num_workers=nw,generator=g,sampler=sampler,
                collate_fn=collate_fn_train,pin_memory=True,
                persistent_workers=True,drop_last=True)

        train_loader = train_loader_call(network_config["batch_size"])
        train_val_loader = monai.data.ThreadDataLoader(
            train_dataset_val,batch_size=1,
            shuffle=False,num_workers=nw,
            collate_fn=collate_fn_train,persistent_workers=True)
        validation_loader = monai.data.ThreadDataLoader(
            validation_dataset,batch_size=1,
            shuffle=False,num_workers=args.n_workers,
            collate_fn=collate_fn_val,persistent_workers=True)

        if args.res_config_file is not None:
            if args.unet_model in ["unetr","swin"]:
                raise NotImplementedError("You can't use a ResNet backbone with\
                    a UNETR or SWINUNet model - what's the point?!")
            _,network_config_ssl = parse_config_ssl(
                args.res_config_file,0.,len(keys))
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
            if args.encoder_checkpoint is not None:
                for i in range(len(args.encoder_checkpoint)):
                    res_state_dict = torch.load(
                        args.encoder_checkpoint[i])['state_dict']
                    mismatched = res_net[i].load_state_dict(
                        res_state_dict,strict=False)
            backbone = [x.backbone for x in res_net]
            network_config['depth'] = [
                backbone[0].structure[0][0],
                *[x[0] for x in backbone[0].structure]]
            network_config['kernel_sizes'] = [3 for _ in network_config['depth']]
            # the last sum is for the bottleneck layer
            network_config['strides'] = [2]
            if "backbone_args" in network_config_ssl:
                mpl = network_config_ssl[
                    "backbone_args"]["maxpool_structure"]
            else:
                mpl = network_config_ssl["maxpool_structure"]
            network_config['strides'].extend(mpl)
            res_ops = [[x.input_layer,*x.operations] for x in backbone]
            res_pool_ops = [[x.first_pooling,*x.pooling_operations]
                            for x in backbone]
            if args.encoder_checkpoint is not None:
                # freezes training for resnet encoder
                for enc in res_ops:
                    for res_op in enc:
                        for param in res_op.parameters():
                            if args.lr_encoder == 0.:
                                param.requires_grad = False
            
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
            network_config=network_config,
            loss_params=loss_params,
            bottleneck_classification=args.bottleneck_classification,
            clinical_feature_keys=feature_keys,
            all_aux_keys=aux_keys,
            clinical_feature_params=all_feature_params,
            clinical_feature_key_net=feature_key_net,
            aux_key_net=aux_key_net,
            max_epochs=args.max_epochs,
            encoding_operations=encoding_operations,
            picai_eval=args.picai_eval,
            lr_encoder=args.lr_encoder,
            cosine_decay=args.cosine_decay,
            encoder_checkpoint=args.bottleneck_classification,
            res_config_file=args.res_config_file,
            deep_supervision=args.deep_supervision,
            n_classes=n_classes,
            keys=keys,
            train_loader_call=train_loader_call,
            random_crop_size=args.random_crop_size,
            crop_size=args.crop_size,
            pad_size=args.pad_size,
            resize_size=args.resize_size)

        if args.early_stopping is not None:
            early_stopping = EarlyStopping(
                'val_loss',patience=args.early_stopping,
                strict=True,mode="min")
            callbacks.append(early_stopping)
                
        logger = get_logger(args.summary_name,args.summary_dir,
                            args.project_name,args.resume,
                            fold=val_fold)

        precision = {"32":32,"16":16,"bf16":"bf16"}[args.precision]
        trainer = Trainer(
            accelerator=dev,
            devices=devices,logger=logger,callbacks=callbacks,
            strategy=strategy,max_epochs=args.max_epochs,
            enable_checkpointing=ckpt,
            accumulate_grad_batches=args.accumulate_grad_batches,
            check_val_every_n_epoch=args.check_val_every_n_epoch,
            log_every_n_steps=10,precision=precision,
            gradient_clip_val=args.gradient_clip_val,
            detect_anomaly=False)

        trainer.fit(unet,train_loader,train_val_loader,ckpt_path=ckpt_path)
        
        print("Validating...")
        if ckpt is True:
            ckpt_list = ["last","best"]
        else:
            ckpt_list = ["last"]
        for ckpt_key in ckpt_list:
            test_metrics = trainer.test(
                unet,validation_loader,ckpt_path=ckpt_key)[0]
            for k in test_metrics:
                out = test_metrics[k]
                if n_classes == 2:
                    try:
                        value = float(out.detach().numpy())
                    except Exception:
                        value = float(out)
                    x = "{},{},{},{},{}".format(k,ckpt_key,val_fold,0,value)
                    output_file.write(x+'\n')
                    print(x)
                else:
                    for i,v in enumerate(out):
                        x = "{},{},{},{},{}".format(k,ckpt_key,val_fold,i,v)
                        output_file.write(x+'\n')
                        print(x)

        print("="*80)
        gc.collect()
        
        # just for safety
        del trainer
        del train_dataset
        del train_loader
        del validation_dataset
        del validation_loader
        del train_dataset_val
        del train_val_loader