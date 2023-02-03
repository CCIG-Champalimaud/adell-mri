from copy import deepcopy
import argparse
import random
import json
import numpy as np
import torch
import monai

from pytorch_lightning import Trainer
from pytorch_lightning.callbacks import RichProgressBar

import sys
sys.path.append(r"..")
from lib.utils import safe_collate,ExponentialMovingAverage
from lib.utils.pl_utils import get_devices,get_ckpt_callback,get_logger
from lib.modules.augmentations import *
from lib.modules.self_supervised.pl import (
    NonContrastiveSelfSLPL,NonContrastiveSelfSLUNetPL)
from lib.modules.config_parsing import parse_config_ssl,parse_config_unet
from lib.monai_transforms import (
    get_pre_transforms_ssl,get_post_transforms_ssl,get_augmentations_ssl)

torch.backends.cudnn.benchmark = True

def force_cudnn_initialization():
    """Convenience function to initialise CuDNN (and avoid the lazy loading
    from PyTorch).
    """
    s = 16
    dev = torch.device('cuda')
    torch.nn.functional.conv2d(torch.zeros(s,s,s,s,device=dev), 
                               torch.zeros(s,s,s,s,device=dev))

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
        help="Floating point precision",choices=["16","32","bf16"],
        default="32")
    parser.add_argument(
        '--unet_encoder',dest='unet_encoder',action="store_true",
        help="Trains a UNet encoder")
    parser.add_argument(
        '--batch_size',dest='batch_size',type=int,default=None,
        help="Overrides batch size in config file")

    # network + training
    parser.add_argument(
        '--config_file',dest="config_file",
        help="Path to network configuration file (yaml)",
        required=True)
    parser.add_argument(
        '--ema',dest="ema",action="store_true",
        help="Includes exponential moving average teacher (like BYOL)")
    parser.add_argument(
        '--from_checkpoint',dest='from_checkpoint',action="store",
        help="Uses this checkpoint as a starting point for the network")
    
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
        '--max_epochs',dest="max_epochs",
        help="Maximum number of training epochs",default=100,type=int)
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
        '--vicreg',dest='vicreg',action="store_true",
        help="Use VICReg loss")
    parser.add_argument(
        '--vicregl',dest='vicregl',action="store_true",
        help="Use VICRegL loss")
    parser.add_argument(
        '--resume_from_last',dest='resume_from_last',action="store_true",
        help="Resumes training from last checkpoint.")

    args = parser.parse_args()

    torch.manual_seed(args.seed)
    random.seed(args.seed)
    np.random.seed(args.seed)
    g = torch.Generator()
    g.manual_seed(args.seed)
    
    accelerator,devices,strategy = get_devices(args.dev,
                                               find_unused_parameters=False,
                                               static_graph=True)

    output_file = open(args.metric_path,'w')

    keys = args.image_keys
    copied_keys = [k+"_copy" for k in keys]
    if args.adc_image_keys is None:
        adc_image_keys = []
    adc_image_keys = [k for k in adc_image_keys if k in keys]
    non_adc_keys = [k for k in keys if k not in adc_image_keys]
    all_keys = [*keys]

    data_dict = json.load(open(args.dataset_json,'r'))
    data_dict = {
        k:data_dict[k] for k in data_dict
        if len(set.intersection(set(data_dict[k]),
                                set(all_keys))) == len(all_keys)}
    if args.subsample_size is not None:
        ss = np.random.choice(
            list(data_dict.keys()),args.subsample_size,replace=False)
        data_dict = {k:data_dict[k] for k in ss}
    for k in data_dict:
        data_dict[k]["pid"] = k

    if args.unet_encoder == True:
        network_config,_ = parse_config_unet(
            args.config_file,len(keys),2)
        network_config_correct = deepcopy(network_config)
        for k in network_config:
            if k in ["loss_fn"]:
                del network_config_correct[k]
    else:
        network_config,network_config_correct = parse_config_ssl(
            args.config_file,args.dropout_param,len(keys))

    if (args.batch_size is not None) and (args.batch_size != "tune"):
        network_config["batch_size"] = args.batch_size
        network_config_correct["batch_size"] = args.batch_size

    if args.ema == True:
        bs = network_config_correct["batch_size"]
        ema_params = {
            "decay":0.99,
            "final_decay":1.0,
            "n_steps":args.max_epochs*len(data_dict)/bs}
        ema = ExponentialMovingAverage(**ema_params)
    else:
        ema = None

    if args.random_crop_size is None:
        roi_size = [256,256]
    else:
        roi_size = [int(x) for x in args.random_crop_size]

    all_pids = [k for k in data_dict]
    
    pre_transforms_args = {
        "all_keys":all_keys,
        "copied_keys":copied_keys,
        "adc_keys":adc_image_keys,
        "non_adc_keys":non_adc_keys,
        "target_spacing":args.target_spacing,
        "crop_size":args.crop_size,
        "pad_size":args.pad_size}
    
    post_transform_args = {
        "all_keys":all_keys,
        "copied_keys":copied_keys}
    
    augmentation_args = {
        "all_keys":all_keys,
        "copied_keys":copied_keys,
        "crop_size":args.crop_size,
        "roi_size":roi_size,
        "vicregl":args.vicregl
    }
    
    transforms = [
        *get_pre_transforms_ssl(**pre_transforms_args),
        *get_augmentations_ssl(),
        *get_post_transforms_ssl(**post_transform_args)]

    collate_fn = safe_collate

    if args.train_pids is not None:
        train_pids = args.train_pids
    else:
        train_pids = [pid for pid in data_dict]
    train_list = [data_dict[pid] for pid in data_dict
                  if pid in train_pids]

    # split workers across cache construction and data loading
    a = args.n_workers // 2
    b = args.n_workers - a
    train_dataset = monai.data.CacheDataset(
        train_list,
        monai.transforms.Compose(transforms),
        cache_rate=args.cache_rate,num_workers=a)

    def train_loader_call(batch_size,shuffle=True): 
        return monai.data.ThreadDataLoader(
            train_dataset,batch_size=batch_size,
            shuffle=shuffle,num_workers=b,generator=g,
            collate_fn=collate_fn,pin_memory=True,
            persistent_workers=True,drop_last=True)

    train_loader = train_loader_call(
        network_config_correct["batch_size"],False)

    if args.unet_encoder == True:
        ssl = NonContrastiveSelfSLUNetPL(
            training_dataloader_call=train_loader_call,
            aug_image_key_1="augmented_image_1",
            aug_image_key_2="augmented_image_2",
            box_key_1="box_1",
            box_key_2="box_2",
            n_epochs=args.max_epochs,
            vic_reg=args.vicreg,
            vic_reg_local=args.vicregl,
            ema=ema,
            **network_config_correct)
    else:
        ssl = NonContrastiveSelfSLPL(
            training_dataloader_call=train_loader_call,
            aug_image_key_1="augmented_image_1",
            aug_image_key_2="augmented_image_2",
            box_key_1="box_1",
            box_key_2="box_2",
            n_epochs=args.max_epochs,
            vic_reg=args.vicreg,
            vic_reg_local=args.vicregl,
            ema=ema,
            **network_config_correct)

    if args.from_checkpoint is not None:
        state_dict = torch.load(
            args.from_checkpoint,map_location=args.dev)['state_dict']
        inc = ssl.load_state_dict(state_dict)
    
    callbacks = [RichProgressBar()]

    ckpt_callback,ckpt_path,status = get_ckpt_callback(
        checkpoint_dir=args.checkpoint_dir,
        checkpoint_name=args.checkpoint_name,
        max_epochs=args.max_epochs,
        resume_from_last=args.resume_from_last,
        val_fold=None,
        monitor=args.monitor)
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

    precision = {"16":16,"32":32,"bf16":"bf16"}[args.precision]
    trainer = Trainer(
        accelerator=accelerator,
        devices=devices,logger=logger,callbacks=callbacks,
        strategy=strategy,max_epochs=args.max_epochs,
        sync_batchnorm=True if strategy is not None else False,
        enable_checkpointing=ckpt,check_val_every_n_epoch=5,
        precision=precision,resume_from_checkpoint=ckpt_path,
        auto_scale_batch_size="power" if args.batch_size == "tune" else None,
        accumulate_grad_batches=args.accumulate_grad_batches,
        gradient_clip_val=args.gradient_clip_val)
    if strategy is None and args.batch_size == "tune":
        bs = trainer.tune(ssl,scale_batch_size_kwargs={"steps_per_trial":2,
                                                       "init_val":16})
    
    torch.cuda.empty_cache()
    force_cudnn_initialization()
    trainer.fit(ssl,val_dataloaders=train_loader)
    
    print("Validating...")
    test_metrics = trainer.test(ssl,train_loader)[0]
    for k in test_metrics:
        out = test_metrics[k]
        try:
            value = float(out.detach().numpy())
        except:
            value = float(out)
        x = "{},{},{},{}".format(k,0,0,value)
        output_file.write(x+'\n')
    x = "{},{},{},{}".format(
        "train_ids",0,0,':'.join(train_pids))
    output_file.write(x+'\n')

    # just in case
    torch.cuda.empty_cache()

    output_file.close()
