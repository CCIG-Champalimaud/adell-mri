from copy import deepcopy
import os
import argparse
import random
import yaml
import json
import numpy as np
import torch
import monai
import wandb

from pytorch_lightning import Trainer
from pytorch_lightning.loggers import WandbLogger
from pytorch_lightning.callbacks.model_checkpoint import ModelCheckpoint
from pytorch_lightning.callbacks import RichProgressBar

from lib.utils import (
    CopyEntryd,
    PrintShaped,
    collate_last_slice,
    RandomSlices,
    ConditionalRescalingd,
    ExposeTransformKeyd,
    safe_collate)
from lib.modules.augmentations import *
from lib.modules.self_supervised_pl import NonContrastiveSelfSLPL,NonContrastiveSelfSLUNetPL
from lib.utils import ExponentialMovingAverage
from lib.modules.config_parsing import parse_config_ssl,parse_config_unet

torch.backends.cudnn.benchmark = True

def force_cudnn_initialization():
    """Convenience function to initialise CuDNN (and avoid the lazy loading
    from PyTorch).
    """
    s = 16
    dev = torch.device('cuda')
    torch.nn.functional.conv2d(torch.zeros(s,s,s,s,device=dev), 
                               torch.zeros(s,s,s,s,device=dev))

def flatten_box(box):
    return np.array(box).T.reshape(-1).astype(np.float32)

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
        '--precision',dest='precision',type=str,
        help="Floating point precision",choices=["16","32","bf16"],
        default="32")
    parser.add_argument(
        '--unet_encoder',dest='unet_encoder',action="store_true",
        help="Trains a UNet encoder")


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
        help="Parameter for dropout.",default=0.1)
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

    output_file = open(args.metric_path,'w')

    keys = args.image_keys
    copied_keys = [k+"_copy" for k in keys]
    key_correspondence = {k:kk for k,kk in zip(keys,copied_keys)}
    if args.adc_image_keys is None:
        args.adc_image_keys = []
    args.adc_image_keys = [k for k in args.adc_image_keys if k in keys]
    intp = []
    intp_resampling_augmentations = []
    for k in keys:
        intp.append("area")
        intp_resampling_augmentations.append("bilinear")
    non_adc_keys = [k for k in keys if k not in args.adc_image_keys]
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
        n_dims = network_config["spatial_dimensions"]
    else:
        network_config,network_config_correct = parse_config_ssl(
            args.config_file,args.dropout_param,len(keys),args.n_devices)
        n_dims = network_config["backbone_args"]["spatial_dim"]

    if args.ema == True:
        bs = network_config["batch_size"]
        ema_params = {
            "decay":0.99,
            "final_decay":1.0,
            "n_steps":args.max_epochs*len(data_dict)/bs}
        ema = ExponentialMovingAverage(**ema_params)
    else:
        ema = None

    if args.random_crop_size is None:
        roi_size = [64,64,8]
    else:
        roi_size = [int(x) for x in args.random_crop_size]

    def get_transforms(x):
        if args.target_spacing is not None:
            rs = [
                monai.transforms.Spacingd(
                    keys=all_keys,pixdim=args.target_spacing,
                    mode=intp_resampling_augmentations)]
        else:
            rs = []
        scaling_ops = []
        if len(non_adc_keys) > 0:
            scaling_ops.append(
                monai.transforms.ScaleIntensityd(non_adc_keys,0,1))
        if len(args.adc_image_keys) > 0:
            scaling_ops.append(
                ConditionalRescalingd(args.adc_image_keys,1000,0.001))
            scaling_ops.append(
                monai.transforms.ScaleIntensityd(
                    args.adc_image_keys,None,None,-(1-args.adc_factor)))
        if args.crop_size is not None:
            crop_op = [
                monai.transforms.CenterSpatialCropd(
                    all_keys,[int(j) for j in args.crop_size]),
                monai.transforms.SpatialPadd(
                    all_keys,[int(j) for j in args.crop_size])]
        else:
            crop_op = []
        if x == "pre":
            return [
                monai.transforms.LoadImaged(all_keys),
                monai.transforms.AddChanneld(all_keys),
                monai.transforms.Orientationd(all_keys,"RAS"),
                *rs,
                *crop_op,
                *scaling_ops,
                monai.transforms.EnsureTyped(all_keys),
                CopyEntryd(all_keys,key_correspondence)]
        elif x == "post":
            return [
                monai.transforms.ConcatItemsd(keys,"augmented_image_1"),
                monai.transforms.ConcatItemsd(copied_keys,"augmented_image_2"),
                monai.transforms.ToTensord(
                    ["augmented_image_1","augmented_image_2"])]

    def get_augmentations():
        aug_list = generic_augments+mri_specific_augments+spatial_augments
        if len(roi_size) == 0:
            cropping_strategy = []
        if args.vicregl == True:
            cropping_strategy = [
                monai.transforms.RandSpatialCropd(
                    all_keys,roi_size=roi_size,random_size=False),
                monai.transforms.RandSpatialCropd(
                    copied_keys,roi_size=roi_size,random_size=False),
                ExposeTransformKeyd(all_keys[0]+"_transforms",
                                    "RandSpatialCropd",["extra_info","slices"],
                                    "box_1"),
                ExposeTransformKeyd(copied_keys[0]+"_transforms",
                                    "RandSpatialCropd",["extra_info","slices"],
                                    "box_2"),
                monai.transforms.Lambdad(["box_1","box_2"],flatten_box),
            ]
        else:
            cropping_strategy = [
                monai.transforms.RandSpatialCropd(
                    all_keys+copied_keys,roi_size=roi_size,random_size=False)
                ]
        return [
            *cropping_strategy,
            AugmentationWorkhorsed(
                augmentations=aug_list,
                keys=all_keys,mask_keys=[],max_mult=0.5,N=3),
            AugmentationWorkhorsed(
                augmentations=aug_list,
                keys=copied_keys,mask_keys=[],max_mult=0.5,N=3),
            ]

    all_pids = [k for k in data_dict]
    
    transforms = [
        *get_transforms("pre"),
        *get_augmentations(),
        *get_transforms("post")]

    if n_dims == 2:
        transforms.append(
            RandomSlices(["image"],None,8,base=0.001))
        collate_fn = collate_last_slice
    else:
        collate_fn = safe_collate

    if args.train_pids is not None:
        train_pids = args.train_pids
    else:
        train_pids = [pid for pid in data_dict]
    train_list = [data_dict[pid] for pid in data_dict
                  if pid in train_pids]

    train_dataset = monai.data.CacheDataset(
        train_list,
        monai.transforms.Compose(transforms),
        num_workers=args.n_workers)

    def train_loader_call(batch_size,shuffle=True): 
        return monai.data.ThreadDataLoader(
            train_dataset,batch_size=batch_size,
            shuffle=shuffle,num_workers=args.n_workers,generator=g,
            pin_memory=True,collate_fn=collate_fn,
            persistent_workers=True,drop_last=True)

    train_loader = train_loader_call(
        network_config["batch_size"])
    validation_loader = train_loader_call(
        network_config["batch_size"],False)

    force_cudnn_initialization()
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

    ckpt_path = None
    if args.checkpoint_name is not None:
        ckpt_name = args.checkpoint_name
        ckpt_name = ckpt_name + "_{epoch:03d}"
        ckpt_name = ckpt_name + "_{val_loss:.3f}"
        ckpt_callback = ModelCheckpoint(
            dirpath=args.checkpoint_dir,
            filename=ckpt_name,monitor="val_loss",
            save_last=True,save_top_k=1,mode="min")
        ckpt_last = args.checkpoint_name + "_last"
        ckpt_callback.CHECKPOINT_NAME_LAST = ckpt_last
        callbacks.append(ckpt_callback)
        ckpt_last_full = os.path.join(
            args.checkpoint_dir,ckpt_last+'.ckpt')
        if os.path.exists(ckpt_last_full) and args.resume_from_last == True:
            ckpt_path = ckpt_last_full
            epoch = torch.load(ckpt_path)["epoch"]
            if epoch >= (args.max_epochs-1):
                print("Training has finished, exiting")
                exit()
            else:
                print("Resuming training from checkpoint in {} (epoch={})".format(
                    ckpt_path,epoch))
        ckpt = True
    else:
        ckpt = False
    
    if args.summary_name is not None and args.project_name is not None:
        wandb.finish()
        wandb_resume = args.resume
        if wandb_resume == "none":
            wandb_resume = None
        run_name = args.summary_name.replace(':','_')
        logger = WandbLogger(
            save_dir=args.summary_dir,project=args.project_name,
            name=run_name,version=run_name,reinit=True,resume=wandb_resume)
    else:
        logger = None

    if ":" in args.dev:
        devices = args.dev.split(":")[-1].split(",")
        devices = [int(i) for i in devices]
        if len(devices) > 1:
            strategy = "ddp"
        else:
            strategy = None
    else:
        devices = args.n_devices
        if devices > 1:
            strategy = "ddp"
        else:
            strategy = None

    precision = {"16":16,"32":32,"bf16":"bf16"}[args.precision]
    torch.autograd.set_detect_anomaly(True)
    trainer = Trainer(
        accelerator="gpu" if "cuda" in args.dev else "cpu",
        devices=devices,logger=logger,callbacks=callbacks,
        strategy=strategy,max_epochs=args.max_epochs,
        sync_batchnorm=True if strategy is not None else False,
        enable_checkpointing=ckpt,check_val_every_n_epoch=5,
        precision=precision,resume_from_checkpoint=ckpt_path,
        auto_scale_batch_size=True,
        accumulate_grad_batches=args.accumulate_grad_batches)
    if strategy is None:
        bs = trainer.tune(ssl)
    torch.cuda.empty_cache()
    
    trainer.fit(ssl,val_dataloaders=validation_loader)
    
    print("Validating...")
    test_metrics = trainer.test(ssl,validation_loader)[0]
    for k in test_metrics:
        out = test_metrics[k]
        try:
            value = float(out.detach().numpy())
        except:
            value = float(out)
        x = "{},{},{},{}".format(k,0,0,value)
        output_file.write(x+'\n')
        print(x)
    x = "{},{},{},{}".format(
        "train_ids",0,0,':'.join(train_pids))
    output_file.write(x+'\n')
    print(x)
    output_file.write(x+'\n')

    # just in case
    torch.cuda.empty_cache()

    output_file.close()