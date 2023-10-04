import os
import argparse
import random
import json
import numpy as np
import torch
import monai
import wandb
from copy import deepcopy

from lightning.pytorch import Trainer
from lightning.pytorch.loggers import WandbLogger
from lightning.pytorch.strategies import DDPStrategy
from lightning.pytorch.callbacks.model_checkpoint import ModelCheckpoint
from lightning.pytorch.callbacks import RichProgressBar

from ...entrypoints.assemble_args import Parser
from ...utils import (
    CopyEntryd,
    collate_last_slice,
    RandomSlices,
    ConditionalRescalingd,
    ExposeTransformKeyMetad,
    safe_collate)
from ...utils.pl_utils import get_ckpt_callback,get_logger,get_devices
from ...modules.augmentations import (
    AugmentationWorkhorsed,generic_augments,mri_specific_augments,
    spatial_augments)
from ...modules.self_supervised.pl import (
    SelfSLResNetPL,SelfSLUNetPL,SelfSLConvNeXtPL)
from ...utils import ExponentialMovingAverage
from ...modules.config_parsing import parse_config_ssl,parse_config_unet

torch.backends.cudnn.benchmark = True

def force_cudnn_initialization():
    """Convenience function to initialise CuDNN (and avoid the lazy loading
    from PyTorch).
    """
    s = 16
    dev = torch.device('cuda')
    torch.nn.functional.conv2d(torch.zeros(s,s,s,s,device=dev), 
                               torch.zeros(s,s,s,s,device=dev))

def main(arguments):
    parser = Parser()

    parser.add_argument_by_key([
        "dataset_json",
        "image_keys",
        ("adc_keys","adc_image_keys"),
        "train_pids",
        "target_spacing",
        "pad_size","crop_size","random_crop_size",
        "subsample_size",
        "cache_rate",
        "precision",
        "unet_encoder",
        "batch_size",
        "config_file","ssl_method","ema",
        "checkpoint_dir","checkpoint_name","checkpoint","resume_from_last",
        "project_name","resume","summary_name","summary_dir","metric_path",
        "dev","n_workers",
        "seed",
        "max_epochs",
        "accumulate_grad_batches","gradient_clip_val",
        "dropout_param"
    ])
    
    args = parser.parse_args(arguments)

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

    if args.unet_encoder is True:
        network_config,_ = parse_config_unet(
            args.config_file,len(keys),2)
        network_config_correct = deepcopy(network_config)
        for k in network_config:
            if k in ["loss_fn"]:
                del network_config_correct[k]
        n_dims = network_config["spatial_dimensions"]
    else:
        network_config,network_config_correct = parse_config_ssl(
            args.config_file,args.dropout_param,len(keys))
        n_dims = network_config["backbone_args"]["spatial_dim"]

    if (args.batch_size is not None) and (args.batch_size != "tune"):
        network_config["batch_size"] = args.batch_size
        network_config_correct["batch_size"] = args.batch_size

    if args.ema is True:
        bs = network_config_correct["batch_size"]
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
                    args.adc_image_keys,None,None,-(1-1/3)))
        crop_op = []
        if args.crop_size is not None:
            crop_op.append(
                monai.transforms.CenterSpatialCropd(
                    all_keys,[int(j) for j in args.crop_size]))
        if args.pad_size is not None:
            crop_op.append(
                monai.transforms.SpatialPadd(
                    all_keys,[int(j) for j in args.pad_size]))
        if x == "pre":
            return [
                monai.transforms.LoadImaged(
                    all_keys,ensure_channel_first=True,image_only=True),
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
        def flatten_box(box):
            box1 = np.array(box[::2])
            box2 = np.array(args.crop_size) - np.array(box[1::2])
            out = np.concatenate([box1,box2]).astype(np.float32)
            return out
        
        transforms_to_remove = [
            # not super slow but not really a common artefact
            "gaussian_smooth_x","gaussian_smooth_y","gaussian_smooth_z",
            # the sharpens are remarkably slow, not worth it imo
            "gaussian_sharpen_x","gaussian_sharpen_y","gaussian_sharpen_z"]
        aug_list = generic_augments+mri_specific_augments+spatial_augments
        aug_list = [x for x in aug_list if x not in transforms_to_remove]
        
        if len(roi_size) == 0:
            cropping_strategy = []
        if args.ssl_method == "vicregl":
            cropping_strategy = [
                monai.transforms.RandSpatialCropd(
                    all_keys,roi_size=roi_size,random_size=False),
                monai.transforms.RandSpatialCropd(
                    copied_keys,roi_size=roi_size,random_size=False),
                # exposes the value associated with the random crop as a key
                # in the data element dict
                ExposeTransformKeyMetad(
                    all_keys[0],"RandSpatialCrop",
                    ["extra_info","cropped"],"box_1"),
                ExposeTransformKeyMetad(
                    copied_keys[0],"RandSpatialCrop",
                    ["extra_info","cropped"],"box_2"),
                # transforms the bounding box into (x1,y1,z1,x2,y2,z2) format
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
                keys=all_keys,mask_keys=[],max_mult=0.5,N=2),
            AugmentationWorkhorsed(
                augmentations=aug_list,
                keys=copied_keys,mask_keys=[],max_mult=0.5,N=2),
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

    accelerator,devices,strategy = get_devices(args.dev)

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

    if args.unet_encoder is True:
        ssl = SelfSLUNetPL(
            training_dataloader_call=train_loader_call,
            aug_image_key_1="augmented_image_1",
            aug_image_key_2="augmented_image_2",
            box_key_1="box_1",
            box_key_2="box_2",
            n_epochs=args.max_epochs,
            vic_reg=args.ssl_method == "vicreg",
            vic_reg_local=args.ssl_method == "vicregl",
            ema=ema,
            **network_config_correct)
    else:
        ssl = SelfSLResNetPL(
            training_dataloader_call=train_loader_call,
            aug_image_key_1="augmented_image_1",
            aug_image_key_2="augmented_image_2",
            box_key_1="box_1",
            box_key_2="box_2",
            n_epochs=args.max_epochs,
            vic_reg=args.ssl_method == "vicreg",
            vic_reg_local=args.ssl_method == "vicregl",
            ema=ema,
            **network_config_correct)
    if "cuda" in args.dev:
        ssl = ssl.to("cuda")

    if args.checkpoint is not None:
        state_dict = torch.load(
            args.checkpoint,map_location=args.dev)['state_dict']
        inc = ssl.load_state_dict(state_dict)
    
    callbacks = [RichProgressBar()]

    ckpt_callback,ckpt_path,status = get_ckpt_callback(
        checkpoint_dir=args.checkpoint_dir,
        checkpoint_name=args.checkpoint_name,
        max_epochs=args.max_epochs,
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

    precision = {"16":16,"32":32,"bf16":"bf16"}[args.precision]
    trainer = Trainer(
        accelerator="gpu" if "cuda" in args.dev else "cpu",
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
