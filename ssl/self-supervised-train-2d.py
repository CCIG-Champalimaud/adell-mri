import sys, os
sys.path.append(os.path.abspath(r".."))
from lib.utils import safe_collate, ExponentialMovingAverage
from lib.utils.pl_utils import get_devices,get_ckpt_callback,get_logger
from lib.modules.config_parsing import parse_config_ssl,parse_config_unet
from lib.utils.dicom_loader import (
    DICOMDataset, SliceSampler)
from lib.utils.network_factories import get_ssl_network
from lib.monai_transforms import (
    get_pre_transforms_ssl,get_post_transforms_ssl,get_augmentations_ssl)

import argparse
import random
import json
import numpy as np
import torch
import monai
from copy import deepcopy
from lightning.pytorch import Trainer
from lightning.pytorch.callbacks import RichProgressBar

torch.backends.cudnn.benchmark = True

def keep_first_not_none(*args):
    for arg in args:
        if arg is not None:
            return arg

def force_cudnn_initialization():
    """Convenience function to initialise CuDNN (and avoid the lazy loading
    from PyTorch).
    """
    s = 16
    dev = torch.device('cuda')
    torch.nn.functional.conv2d(torch.zeros(s,s,s,s,device=dev), 
                               torch.zeros(s,s,s,s,device=dev))

def filter_dicom_dict_on_presence(data_dict,all_keys):
    def check_intersection(a,b):
        return len(set.intersection(set(a),set(b))) == len(set(b))
    for k in data_dict:
        for kk in data_dict[k]:
            data_dict[k][kk] = [
                element for element in data_dict[k][kk]
                if check_intersection(element.keys(),all_keys)]
    return data_dict

def filter_dicom_dict_by_size(data_dict,max_size):
    print("Filtering by size (max={})".format(
        max_size,len(data_dict)))
    output_dict = {}
    removed_series = 0
    for k in data_dict:
        for kk in data_dict[k]:
            if len(data_dict[k][kk]) < max_size:
                if k not in output_dict:
                    output_dict[k] = {}
                output_dict[k][kk] = data_dict[k][kk]
            else:
                removed_series += 1
    print("Removed={}".format(removed_series))
    return output_dict

if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    # data
    parser.add_argument(
        '--dataset_json',dest='dataset_json',type=str,
        help="JSON containing dataset information",required=True)
    parser.add_argument(
        '--train_pids',dest='train_pids',action="store",
        default=None,type=str,nargs='+',
        help="IDs in dataset_json used for training")
    parser.add_argument(
        '--pad_size',dest='pad_size',action="store",
        default=None,type=float,nargs='+',
        help="Size of central padded image after resizing (if none is specified\
            then no padding is performed).")
    parser.add_argument(
        '--crop_size',dest='crop_size',action="store",
        default=None,type=float,nargs='+',
        help="Size of central crop after resizing (if none is specified then\
            no cropping is performed).")
    parser.add_argument(
        '--random_crop_size',dest='random_crop_size',action="store",
        default=None,type=float,nargs='+',
        help="Size of crop with random centre.")
    parser.add_argument(
        '--scaled_crop_size',dest='scaled_crop_size',action="store",
        default=None,type=int,nargs='+',
        help="Crops a region with at least a quarter of the specified size \
            and then resizes them image to this size.")
    parser.add_argument(
        '--target_spacing',dest='target_spacing',action="store",default=None,
        help="Resamples all images to target spacing",nargs='+',type=float)
    parser.add_argument(
        '--image_keys',dest='image_keys',type=str,nargs='+',
        help="Image keys in the dataset JSON. First key is used as template",
        required=True)
    parser.add_argument(
        '--adc_image_keys',dest='adc_image_keys',type=str,nargs='+',
        help="Keys corresponding to input images which are ADC maps \
            (normalized differently)",
        default=None)
    parser.add_argument(
        '--adc_factor',dest='adc_factor',type=float,default=1/3,
        help="Multiplies ADC images by this factor.")
    parser.add_argument(
        '--subsample_size',dest='subsample_size',type=int,
        help="Subsamples data to a given size",
        default=None)
    parser.add_argument(
        '--cache_rate',dest='cache_rate',type=float,
        help="Cache rate for CacheDataset",
        default=1.0)
    parser.add_argument(
        '--precision',dest='precision',type=str,
        help="Floating point precision",default="32")
    parser.add_argument(
        '--net_type',dest='net_type',
        choices=["resnet","unet_encoder","convnext","vit"],
        help="Which network should be trained.")
    parser.add_argument(
        '--batch_size',dest='batch_size',type=int,default=None,
        help="Overrides batch size in config file")
    parser.add_argument(
        '--max_slices',dest='max_slices',type=int,default=None,
        help="Excludes studies with over max_slices")

    # network + training
    parser.add_argument(
        '--config_file',dest="config_file",
        help="Path to network configuration file (yaml)",
        required=True)
    parser.add_argument(
        '--ema',dest="ema",action="store_true",
        help="Includes exponential moving average teacher (like BYOL)")
    parser.add_argument(
        '--stop_gradient',dest="stop_gradient",action="store_true",
        help="Stops gradient")
    parser.add_argument(
        '--lars',dest="lars",action="store_true",
        help="Wraps optimizer with LARS")
    parser.add_argument(
        '--from_checkpoint',dest='from_checkpoint',action="store",
        help="Uses this checkpoint as a starting point for the network")
    parser.add_argument(
        '--n_series_iterations',dest="n_series_iterations",default=2,type=int,
        help="Number of iterations over each series per epoch")
    parser.add_argument(
        '--n_transforms',dest="n_transforms",default=3,type=int,
        help="Number of augmentations for each image")
    
    # training
    parser.add_argument(
        '--dev',dest='dev',type=str,default="cuda",
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
        '--max_epochs',dest="max_epochs",
        help="Maximum number of training epochs",default=100,type=int)
    parser.add_argument(
        '--check_val_every_n_epoch',dest="check_val_every_n_epoch",
        help="Validation check frequency",default=5,type=int)
    parser.add_argument(
        '--steps_per_epoch',dest="steps_per_epoch",
        help="Number of steps per epoch",default=None,type=int)
    parser.add_argument(
        '--warmup_epochs',dest='warmup_epochs',type=float,default=0.0,
        help="Number of warmup epochs (if SWA is triggered it starts after\
            this number of epochs).")
    parser.add_argument(
        '--accumulate_grad_batches',dest="accumulate_grad_batches",
        help="Number batches to accumulate before backpropgating gradient",
        default=1,type=int)
    parser.add_argument(
        '--gradient_clip_val',dest="gradient_clip_val",
        help="Value for gradient clipping",
        default=0.0,type=float)
    parser.add_argument(
        '--checkpoint_dir',dest='checkpoint_dir',type=str,default="models",
        help='Path to directory where checkpoints will be saved.')
    parser.add_argument(
        '--checkpoint_name',dest='checkpoint_name',type=str,default=None,
        help='Checkpoint ID.')
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
        '--dropout_param',dest='dropout_param',type=float,
        help="Parameter for dropout.",default=0.0)
    parser.add_argument(
        "--ssl_method",dest="ssl_method",type=str,
        help="SSL method",choices=["simsiam","byol","simclr",
                                   "vicreg","vicregl","ijepa"])
    parser.add_argument(
        '--resume_from_last',dest='resume_from_last',action="store_true",
        help="Resumes training from last checkpoint.")

    args = parser.parse_args()

    torch.manual_seed(args.seed)
    random.seed(args.seed)
    np.random.seed(args.seed)
    g = torch.Generator()
    g.manual_seed(args.seed)
    
    n_iterations = args.n_series_iterations
    accelerator,devices,strategy = get_devices(args.dev)

    output_file = open(args.metric_path,'w')

    keys = args.image_keys
    copied_keys = [k+"_copy" for k in keys]
    if args.adc_image_keys is None:
        adc_image_keys = []
    adc_image_keys = [k for k in adc_image_keys if k in keys]
    non_adc_keys = [k for k in keys if k not in adc_image_keys]
    all_keys = [*keys]

    data_dict = json.load(open(args.dataset_json,'r'))
    if len(data_dict) == 0:
        print("No data in dataset JSON")
        exit()
    #data_dict = filter_orientations(data_dict)
    data_dict = filter_dicom_dict_on_presence(data_dict,all_keys)
    if args.max_slices is not None:
        data_dict = filter_dicom_dict_by_size(data_dict,args.max_slices)

    if args.subsample_size is not None:
        ss = np.random.choice(
            list(data_dict.keys()),args.subsample_size,replace=False)
        data_dict = {k:data_dict[k] for k in ss}
    for k in data_dict:
        for kk in data_dict[k]:
            for i in range(len(data_dict[k][kk])):  
                data_dict[k][kk][i]["pid"] = k

    if args.net_type == "unet_encoder":
        network_config,_ = parse_config_unet(
            args.config_file,len(keys),2)
        network_config_correct = deepcopy(network_config)
        for k in network_config:
            if k in ["loss_fn"]:
                del network_config_correct[k]
    else:
        network_config,network_config_correct = parse_config_ssl(
            args.config_file,args.dropout_param,len(keys),
            args.ssl_method=="ijepa")

    if args.batch_size is not None:
        network_config["batch_size"] = args.batch_size
        network_config_correct["batch_size"] = args.batch_size

    if args.random_crop_size is None:
        roi_size = [128,128]
    else:
        roi_size = [int(x) for x in args.random_crop_size]

    all_pids = [k for k in data_dict]
    
    is_ijepa = args.ssl_method == "ijepa"
    pre_transform_args = {
        "all_keys":all_keys,
        "copied_keys":copied_keys,
        "adc_keys":adc_image_keys,
        "non_adc_keys":non_adc_keys,
        "target_spacing":args.target_spacing,
        "crop_size":args.crop_size,
        "pad_size":args.pad_size,
        "n_channels":1,
        "n_dim":2,
        "skip_augmentations":is_ijepa}
    
    post_transform_args = {
        "all_keys":all_keys,
        "copied_keys":copied_keys,
        "skip_augmentations":is_ijepa}
    
    augmentation_args = {
        "all_keys":all_keys,
        "copied_keys":copied_keys if is_ijepa is False else [],
        "scaled_crop_size":args.scaled_crop_size,
        "roi_size":roi_size,
        "vicregl":args.ssl_method == "vicregl",
        "n_transforms":args.n_transforms,
        "n_dim":2,
        "skip_augmentations":is_ijepa}
    
    if is_ijepa is True:
        image_size = keep_first_not_none(
            args.scaled_crop_size,args.crop_size)
        patch_size = network_config_correct["backbone_args"]["patch_size"]
        feature_map_size = [i//pi for i,pi in zip(image_size,patch_size)]
        network_config_correct["backbone_args"]["image_size"] = image_size
        network_config_correct["feature_map_dimensions"] = feature_map_size

    transforms = [
        *get_pre_transforms_ssl(**pre_transform_args),
        *get_augmentations_ssl(**augmentation_args),
        *get_post_transforms_ssl(**post_transform_args)]

    if args.train_pids is not None:
        train_pids = args.train_pids
    else:
        train_pids = [pid for pid in data_dict]
    train_list = [data_dict[pid] for pid in data_dict
                  if pid in train_pids]
    
    # split workers across cache construction and data loading
    train_dataset = DICOMDataset(
        train_list,
        monai.transforms.Compose(transforms))
    if args.steps_per_epoch is not None:
        n_samples = args.steps_per_epoch * network_config["batch_size"]
    else:
        n_samples = None
    sampler = SliceSampler(
        train_list,n_iterations=n_iterations,n_samples=n_samples)
    val_sampler = SliceSampler(
        train_list,n_iterations=n_iterations,n_samples=n_samples)

    n_devices = len(devices) if isinstance(devices,list) else 1
    agb = args.accumulate_grad_batches
    if args.steps_per_epoch is not None:
        steps_per_epoch = args.steps_per_epoch
        steps_per_epoch_optim = int(np.ceil(args.steps_per_epoch / agb))
        max_steps = args.max_epochs * steps_per_epoch
        max_epochs = -1
        max_steps_optim = max_epochs * steps_per_epoch_optim
        warmup_steps = args.warmup_epochs * steps_per_epoch_optim
        check_val_every_n_epoch = None
        val_check_interval = args.check_val_every_n_epoch * steps_per_epoch
    else:
        bs = network_config_correct["batch_size"]
        steps_per_epoch = len(sampler) // (bs * n_devices)
        steps_per_epoch = int(np.ceil(steps_per_epoch / agb))
        max_epochs = args.max_epochs
        max_steps = -1
        max_steps_optim = args.max_epochs * steps_per_epoch
        warmup_steps = args.warmup_epochs * steps_per_epoch
        check_val_every_n_epoch = args.check_val_every_n_epoch
        val_check_interval = None
    warmup_steps = int(warmup_steps)
    max_steps_optim = int(max_steps_optim)
    
    if args.ema is True:
        if is_ijepa is True:
            ema_params = {
                "decay":0.99,
                "final_decay":1.0,
                "n_steps":max_steps_optim}
        else:
            ema_params = {
                "decay":0.996,
                "final_decay":1.0,
                "n_steps":max_steps_optim}
        ema = ExponentialMovingAverage(**ema_params)
    else:
        ema = None

    if isinstance(devices,list):
        n_workers = args.n_workers // len(devices)
    else:
        n_workers = args.n_workers // devices
    def train_loader_call(batch_size,shuffle=True):
        return monai.data.ThreadDataLoader(
            train_dataset,batch_size=batch_size,
            num_workers=n_workers,sampler=sampler,
            collate_fn=safe_collate,pin_memory=True,
            persistent_workers=n_workers>1,drop_last=True)

    train_loader = train_loader_call(
        network_config_correct["batch_size"],False)
    val_loader = monai.data.ThreadDataLoader(
        train_dataset,batch_size=network_config_correct["batch_size"],
        num_workers=n_workers,collate_fn=safe_collate,
        sampler=val_sampler,drop_last=True)

    ssl = get_ssl_network(train_loader_call=train_loader_call,
                          max_epochs=max_epochs,
                          max_steps_optim=max_steps_optim,
                          warmup_steps=warmup_steps,
                          ssl_method=args.ssl_method,
                          ema=ema,
                          net_type=args.net_type,
                          network_config_correct=network_config_correct,
                          stop_gradient=args.stop_gradient)

    if args.from_checkpoint is not None:
        state_dict = torch.load(
            args.from_checkpoint,map_location=args.dev)['state_dict']
        inc = ssl.load_state_dict(state_dict)
    
    callbacks = [RichProgressBar()]

    epochs_ckpt = max_epochs
    ckpt_callback,ckpt_path,status = get_ckpt_callback(
        checkpoint_dir=args.checkpoint_dir,
        checkpoint_name=args.checkpoint_name,
        max_epochs=epochs_ckpt,
        resume_from_last=args.resume_from_last,
        val_fold=None,
        monitor="val_loss")
    if ckpt_callback is not None:   
        callbacks.append(ckpt_callback)
    ckpt = ckpt_callback is not None
    if status == "finished":
        print("Training has finished")
        exit()

    logger = get_logger(summary_name=args.summary_name,
                        summary_dir=args.summary_dir,
                        project_name=args.project_name,
                        resume=args.resume,
                        fold=None)

    precision = args.precision
    trainer = Trainer(
        accelerator=accelerator,
        devices=devices,logger=logger,callbacks=callbacks,
        strategy=strategy,max_epochs=max_epochs,max_steps=max_steps,
        sync_batchnorm=True if strategy is not None else False,
        enable_checkpointing=ckpt,
        val_check_interval=val_check_interval,
        check_val_every_n_epoch=check_val_every_n_epoch,
        precision=precision,
        accumulate_grad_batches=args.accumulate_grad_batches,
        gradient_clip_val=args.gradient_clip_val)
    
    torch.cuda.empty_cache()
    force_cudnn_initialization()
    trainer.fit(ssl,val_dataloaders=val_loader,ckpt_path=ckpt_path)
    
    print("Validating...")
    test_metrics = trainer.test(ssl,val_loader)[0]
    for k in test_metrics:
        out = test_metrics[k]
        try:
            value = float(out.detach().numpy())
        except Exception:
            value = float(out)
        x = "{},{},{},{}".format(k,0,0,value)
        output_file.write(x+'\n')
    x = "{},{},{},{}".format(
        "train_ids",0,0,':'.join(train_pids))
    output_file.write(x+'\n')

    # just in case
    torch.cuda.empty_cache()

    output_file.close()
