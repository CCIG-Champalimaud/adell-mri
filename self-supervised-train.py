import os
import argparse
import random
import yaml
import json
import numpy as np
import torch
import monai
import wandb
from sklearn.model_selection import KFold,train_test_split

from pytorch_lightning import Trainer
from pytorch_lightning.loggers import WandbLogger
from pytorch_lightning.callbacks.model_checkpoint import ModelCheckpoint

from lib.utils import (
    CopyEntryd,
    PrintShaped,
    collate_last_slice,
    RandomSlices,
    SlicesToFirst,
    ConditionalRescalingd,
    FastResample,
    safe_collate)
from lib.modules.augmentations import *
from lib.modules.self_supervised_pl import BootstrapYourOwnLatentPL
from lib.modules.self_supervised import ExponentialMovingAverage
from lib.modules.config_parsing import parse_config_ssl

torch.backends.cudnn.benchmark = True

def force_cudnn_initialization():
    """Convenience function to initialise CuDNN (and avoid the lazy loading
    from PyTorch).
    """
    s = 16
    dev = torch.device('cuda')
    torch.nn.functional.conv2d(torch.zeros(s, s, s, s, device=dev), 
                               torch.zeros(s,s,s,s,device=dev))

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
        '--resize',dest='resize',type=float,nargs='+',default=None,
        help="Scales the input size by a float")
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
        help="Device for PyTorch training",choices=["cuda","cpu"])
    parser.add_argument(
        '--seed',dest='seed',help="Random seed",default=42,type=int)
    parser.add_argument(
        '--n_workers',dest='n_workers',
        help="Number of workers",default=1,type=int)
    parser.add_argument(
        '--n_devices',dest='n_devices',
        help="Number of devices",default=1,type=int)
    parser.add_argument(
        '--pre_load',dest='pre_load',action="store_true",
        help="Load and process data to memory at the beginning of training",
        default=False)
    parser.add_argument(
        '--max_epochs',dest="max_epochs",
        help="Maximum number of training epochs",default=100,type=int)
    parser.add_argument(
        '--n_folds',dest="n_folds",
        help="Number of validation folds",default=5,type=int)
    parser.add_argument(
        '--checkpoint_dir',dest='checkpoint_dir',type=str,default="models",
        help='Path to directory where checkpoints will be saved.')
    parser.add_argument(
        '--checkpoint_name',dest='checkpoint_name',type=str,default=None,
        help='Checkpoint ID.')
    parser.add_argument(
        '--summary_dir',dest='summary_dir',type=str,default="summaries",
        help='Path to summary directory (for tensorboard).')
    parser.add_argument(
        '--summary_name',dest='summary_name',type=str,default=None,
        help='Summary name.')
    parser.add_argument(
        '--project_name',dest='project_name',type=str,default="summaries",
        help='Project name for wandb.')
    parser.add_argument(
        '--metric_path',dest='metric_path',type=str,default="metrics.csv",
        help='Path to file with CV metrics + information.')
    parser.add_argument(
        '--dropout_param',dest='dropout_param',type=float,
        help="Parameter for dropout.",default=0.1)

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
    if args.input_size is not None:
        if args.resize is None:
            R = [1 for _ in args.input_size]
        else:
            R = args.resize
        args.input_size = [
            int(x*y) for x,y in zip(args.input_size,R)]

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
    
    network_config,network_config_correct = parse_config_ssl(
        args.config_file,args.dropout_param,len(keys),args.n_devices)

    if args.ema == True:
        bs = network_config["batch_size"]
        ema_params = {
            "decay":0.99,
            "final_decay":1.0,
            "n_steps":args.max_epochs*len(data_dict)/bs}
        ema = ExponentialMovingAverage(**ema_params)
    else:
        ema = None

    print("Setting up transforms...")
    def get_transforms(x):
        if args.target_spacing is not None:
            rs = [
                FastResample(keys=all_keys,target=args.target_spacing,
                             mode=intp)]
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
                    args.adc_image_keys,None,None,1-args.adc_factor))
        if args.input_size is not None:
            resize = [monai.transforms.Resized(
                all_keys,tuple(args.input_size),mode=intp)]
        else:
            resize = []
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
                *rs,
                monai.transforms.Orientationd(all_keys,"RAS"),
                *resize,
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
        return [
            AugmentationWorkhorsed(
                augmentations=aug_list,
                keys=all_keys,mask_keys=[],max_mult=0.5,N=3),
            AugmentationWorkhorsed(
                augmentations=aug_list,
                keys=copied_keys,mask_keys=[],max_mult=0.5,N=3)]

    all_pids = [k for k in data_dict]
    if args.pre_load == False:
        transforms_train = [
            *get_transforms("pre"),
            *get_augmentations(),
            *get_transforms("post")]

        transforms_train_val = [
            *get_transforms("pre"),
            *get_augmentations(),
            *get_transforms("post")]

        transforms_val = [
            *get_transforms("pre"),
            *get_augmentations(),
            *get_transforms("post")]
    else:
        transforms_train = [
            *get_augmentations(),
            *get_transforms("post")]

        transforms_train_val = [
            *get_augmentations(),
            *get_transforms("post")]

        transforms_val = [
            *get_augmentations(),
            *get_transforms("post")]
        
        transform_all_data = get_transforms("pre")

        path_list = [data_dict[k] for k in all_pids]
        # load using cache dataset because it handles parallel processing
        # and then convert to list
        dataset_full = monai.data.CacheDataset(
            path_list,
            monai.transforms.Compose(transform_all_data),
            num_workers=args.n_workers)
        dataset_full_ = []
        all_pids = []
        for x in dataset_full:
            dataset_full_.append({k:x[k] for k in x if "transforms" not in k})
            all_pids.append(x["pid"])
        dataset_full = dataset_full_

    if network_config["backbone_args"]["spatial_dim"] == 2:
        transforms_train.append(
            RandomSlices(["image"],None,8,base=0.001))
        transforms_train_val.append(
            SlicesToFirst(["image"]))
        transforms_val.append(
            SlicesToFirst(["image"]))
        collate_fn = collate_last_slice
    else:
        collate_fn = safe_collate

    if args.n_folds > 1:
        fold_generator = KFold(
            args.n_folds,shuffle=True,random_state=args.seed).split(all_pids)
    else:
        fold_generator = iter(
            [train_test_split(range(len(all_pids)),test_size=0.2)])

    for val_fold in range(args.n_folds):
        train_idxs,val_idxs = next(fold_generator)
        train_idxs,train_val_idxs = train_test_split(train_idxs,test_size=0.15)
        if args.pre_load == False:
            train_pids = [all_pids[i] for i in train_idxs]
            train_val_pids = [all_pids[i] for i in train_val_idxs]
            val_pids = [all_pids[i] for i in val_idxs]
            train_list = [data_dict[pid] for pid in train_pids]
            train_val_list = [data_dict[pid] for pid in train_val_pids]
            val_list = [data_dict[pid] for pid in val_pids]
            train_dataset = monai.data.CacheDataset(
                train_list,
                monai.transforms.Compose(transforms_train),
                num_workers=args.n_workers)
            train_dataset_val = monai.data.CacheDataset(
                train_val_list,
                monai.transforms.Compose(transforms_train_val),
                num_workers=args.n_workers)
            validation_dataset = monai.data.CacheDataset(
                val_list,
                monai.transforms.Compose(transforms_val),
                num_workers=args.n_workers)

        else:
            train_pids = [all_pids[i] for i in train_idxs]
            train_val_pids = [all_pids[i] for i in train_val_idxs]
            val_pids = [all_pids[i] for i in val_idxs]
            train_list = [dataset_full[i] for i in train_idxs]
            train_val_list = [dataset_full[i] for i in train_val_idxs]
            val_list = [dataset_full[i] for i in val_idxs]
            train_dataset = monai.data.Dataset(
                train_list,
                monai.transforms.Compose(transforms_train))
            train_dataset_val = monai.data.Dataset(
                train_val_list,
                monai.transforms.Compose(transforms_train_val))
            validation_dataset = monai.data.Dataset(
                val_list,
                monai.transforms.Compose(transforms_val))

        def train_loader_call(): 
            return monai.data.ThreadDataLoader(
                train_dataset,batch_size=network_config["batch_size"],
                shuffle=True,num_workers=args.n_workers,generator=g,
                collate_fn=collate_fn,pin_memory=True,
                persistent_workers=True,drop_last=True)

        train_loader = train_loader_call()
        train_val_loader = monai.data.ThreadDataLoader(
            train_dataset_val,batch_size=network_config["batch_size"],
            shuffle=False,num_workers=args.n_workers,generator=g,
            collate_fn=collate_fn,persistent_workers=True,
            drop_last=True)
        validation_loader = monai.data.ThreadDataLoader(
            validation_dataset,batch_size=1,
            shuffle=False,num_workers=args.n_workers,generator=g,
            collate_fn=collate_fn,persistent_workers=True,
            drop_last=True)

        print("Setting up training...")
        force_cudnn_initialization()
        ssl = BootstrapYourOwnLatentPL(
            training_dataloader_call=train_loader_call,
            aug_image_key_1="augmented_image_1",
            aug_image_key_2="augmented_image_2",
            n_epochs=args.max_epochs,
            ema=ema,**network_config_correct)
        if args.from_checkpoint is not None:
            state_dict = torch.load(
                args.from_checkpoint,map_location=args.dev)['state_dict']
            inc = ssl.load_state_dict(state_dict)
        
        callbacks = []

        if args.checkpoint_name is not None:
            ckpt_name = args.checkpoint_name + "_fold" + str(val_fold)
            ckpt_name = ckpt_name + "_{epoch:03d}"
            ckpt_name = ckpt_name + "_{val_loss:.3f}"
            ckpt_callback = ModelCheckpoint(
                dirpath=args.checkpoint_dir,
                filename=ckpt_name,monitor="val_loss",
                save_last=True,save_top_k=1,mode="min")
            ckpt_callback.CHECKPOINT_NAME_LAST = \
                args.checkpoint_name + "_fold" + str(val_fold) + "_last"
            callbacks.append(ckpt_callback)
        
        if args.summary_name is not None:
            wandb.finish()
            run_name = args.summary_name.replace(':','_') + "_fold_{}".format(val_fold)
            logger = WandbLogger(
                save_dir=args.summary_dir,project=args.project_name,
                name=run_name,version=run_name,reinit=True)
        else:
            logger = None

        trainer = Trainer(
            accelerator="gpu" if args.dev=="cuda" else "cpu",
            devices=args.n_devices,logger=logger,callbacks=callbacks,
            strategy='ddp' if args.n_devices > 1 else None,
            max_epochs=args.max_epochs,
            check_val_every_n_epoch=1,log_every_n_steps=10)

        trainer.fit(ssl, train_loader, train_val_loader)
        
        print("Validating...")
        test_metrics = trainer.test(ssl,validation_loader)[0]
        for k in test_metrics:
            out = test_metrics[k]
            try:
                value = float(out.detach().numpy())
            except:
                value = float(out)
            x = "{},{},{},{}".format(k,val_fold,0,value)
            output_file.write(x+'\n')
            print(x)
        x = "{},{},{},{}".format(
            "train_ids",val_fold,0,':'.join(train_pids))
        output_file.write(x+'\n')
        print(x)
        x = "{},{},{},{}".format(
            "train_val_ids",val_fold,0,':'.join(train_val_pids))
        output_file.write(x+'\n')
        x = "{},{},{},{}".format(
            "val_ids",val_fold,0,':'.join(val_pids))
        output_file.write(x+'\n')

        # just in case
        torch.cuda.empty_cache()

    output_file.close()