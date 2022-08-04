import os
import argparse
import random
import yaml
import json
import numpy as np
import torch
import monai
import wandb
import SimpleITK as sitk
from sklearn.model_selection import KFold,train_test_split
from tqdm import tqdm

from pytorch_lightning import Trainer
from pytorch_lightning.callbacks import EarlyStopping,StochasticWeightAveraging
from pytorch_lightning.loggers import WandbLogger
from pytorch_lightning.callbacks.model_checkpoint import ModelCheckpoint

from lib.utils import (
    loss_factory,
    activation_factory,
    PartiallyRandomSampler,
    LabelOperatorSegmentationd,
    get_loss_param_dict,
    collate_last_slice,
    RandomSlices,
    SlicesToFirst,
    CombineBinaryLabelsd,
    ConditionalRescalingd,
    safe_collate)
from lib.modules.layers import ResNet
from lib.modules.segmentation_pl import UNetPL,UNetPlusPlusPL
from lib.modules.config_parsing import parse_config_unet,parse_config_ssl

if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    # data
    parser.add_argument(
        '--dataset_json',dest='dataset_json',type=str,
        help="JSON containing dataset information",required=True)
    parser.add_argument(
        '--input_size',dest='input_size',type=float,nargs='+',
        help="Input size for network",required=True)
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
        '--mask_image_keys',dest='mask_image_keys',type=str,nargs='+',
        help="Keys corresponding to input images which are segmentation masks",
        default=None)
    parser.add_argument(
        '--adc_image_keys',dest='adc_image_keys',type=str,nargs='+',
        help="Keys corresponding to input images which are ADC maps \
            (normalized differently)",
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

    # network + training
    parser.add_argument(
        '--config_file',dest="config_file",
        help="Path to network configuration file (yaml)",
        required=True)
    parser.add_argument(
        '--unet_pp',dest='unet_pp',action="store_true",
        help="Uses U-Net++ rather than U-Net")
    parser.add_argument(
        '--res_config_file',dest='res_config_file',action="store",default=None,
        help="Uses a ResNet as a backbone (depths are inferred from this). \
            This config file is then used to parameterise the ResNet.")
    parser.add_argument(
        '--res_checkpoint',dest='res_checkpoint',action="store",default=None,
        help="Checkpoint for SSL backbone (if --res_config_file is specified.")
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
        '--augment',dest='augment',action="store_true",
        help="Use data augmentations",default=False)
    parser.add_argument(
        '--pre_load',dest='pre_load',action="store_true",
        help="Load and process data to memory at the beginning of training",
        default=False)
    parser.add_argument(
        '--loss_gamma',dest="loss_gamma",
        help="Gamma for focal loss",default=2.0,type=float)
    parser.add_argument(
        '--loss_comb',dest="loss_comb",
        help="Relative weight for combined losses",default=0.5,type=float)
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
        '--metric_path',dest='metric_path',type=str,default="metrics.csv",
        help='Path to file with CV metrics + information.')
    parser.add_argument(
        '--early_stopping',dest='early_stopping',type=int,default=None,
        help="No. of checks before early stop (defaults to no early stop).")
    parser.add_argument(
        '--swa',dest='swa',action="store_true",
        help="Use stochastic gradient averaging.",default=False)
    parser.add_argument(
        '--class_weights',dest='class_weights',type=float,nargs='+',
        help="Class weights (by alphanumeric order).",default=1.)
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

    if args.possible_labels == 2 or args.positive_labels is not None:
        n_classes = 2
    else:
        n_classes = args.possible_labels

    keys = args.image_keys
    label_keys = args.mask_keys
    if args.mask_image_keys is None:
        args.mask_image_keys = []
    if args.adc_image_keys is None:
        args.adc_image_keys = []
    args.adc_image_keys = [k for k in args.adc_image_keys if k in keys]
    intp = []
    intp_resampling_augmentations = []
    for k in keys:
        if k in args.mask_image_keys:
            intp.append("nearest")
            intp_resampling_augmentations.append("nearest")
        else:
            intp.append("area")
            intp_resampling_augmentations.append("bilinear")
    non_adc_keys = [k for k in keys if k not in args.adc_image_keys]
    intp.extend(["nearest"]*len(label_keys))
    intp_resampling_augmentations.extend(["nearest"]*len(label_keys))
    all_keys = [*keys,*label_keys]
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
    
    network_config,loss_key = parse_config_unet(
        args.config_file,len(keys),n_classes)

    print("Setting up transforms...")
    def get_transforms(x,label_mode=None):
        if args.target_spacing is not None:
            rs = [
                monai.transforms.Spacingd(
                    all_keys,args.target_spacing,
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
                    args.adc_image_keys,None,None,1-args.adc_factor))
        if args.crop_size is not None:
            crop_op = [
                monai.transforms.CenterSpatialCropd(
                    all_keys,[int(j) for j in args.crop_size])]
        else:
            crop_op = []

        if x == "pre":
            return [
                monai.transforms.LoadImaged(all_keys),
                monai.transforms.AddChanneld(all_keys),
                *rs,
                monai.transforms.Orientationd(all_keys,"RAS"),
                monai.transforms.Resized(
                    all_keys,tuple(args.input_size),mode=intp),
                *crop_op,
                *scaling_ops,
                monai.transforms.EnsureTyped(all_keys),
                CombineBinaryLabelsd(label_keys,"any","mask"),
                LabelOperatorSegmentationd(
                    ["mask"],args.possible_labels,
                    mode=label_mode,positive_labels=args.positive_labels)]
        elif x == "post":
            return [
                monai.transforms.ConcatItemsd(keys,"image"),
                monai.transforms.ToTensord(["image","mask"])]

    def get_augmentations(augment):
        if augment == True:
            return [
                monai.transforms.RandAffined(
                    all_keys,
                    scale_range=[0.1,0.1,0.1],
                    rotate_range=[np.pi/8,np.pi/8,np.pi/16],
                    translate_range=[10,10,1],
                    prob=0.2,mode=intp_resampling_augmentations),
                monai.transforms.RandFlipd(
                    all_keys,spatial_axis=[0,1,2])]
        else:
            return []

    label_mode = "binary" if n_classes == 2 else "cat"
    all_pids = [k for k in data_dict]
    if args.pre_load == False:
        transforms_train = [
            *get_transforms("pre",label_mode),
            *get_augmentations(args.augment),
            *get_transforms("post",label_mode)]

        transforms_train_val = [
            *get_transforms("pre",label_mode),
            *get_transforms("post",label_mode)]

        transforms_val = [
            *get_transforms("pre",label_mode),
            *get_transforms("post",label_mode)]
    else:
        transforms_train = [
            *get_augmentations(args.augment),
            *get_transforms("post",label_mode)]

        transforms_train_val = [
            *get_transforms("post",label_mode)]

        transforms_val = [
            *get_transforms("post",label_mode)]
        
        transform_all_data = get_transforms("pre",label_mode)

        path_list = [data_dict[k] for k in all_pids]
        # load using cache dataset because it handles parallel processing
        # and then convert to list
        dataset_full = monai.data.CacheDataset(
            path_list,
            monai.transforms.Compose(transform_all_data),
            num_workers=args.n_workers)
        dataset_full = [{k:x[k] for k in x if "transforms" not in k} 
                         for x in dataset_full]

    if network_config["spatial_dimensions"] == 2:
        transforms_train.append(
            RandomSlices(["image","mask"],"mask",8,base=0.001))
        transforms_train_val.append(
            SlicesToFirst(["image","mask"]))
        transforms_val.append(
            SlicesToFirst(["image","mask"]))
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
        train_idxs,train_val_idxs = train_test_split(train_idxs,test_size=0.2)
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

        print("Calculating class weights...")        
        weights = torch.as_tensor(
            np.array(args.class_weights),dtype=torch.float32,
            device=args.dev)
        
        loss_params = get_loss_param_dict(
            weights=weights,gamma=args.loss_gamma,
            comb=args.loss_comb)[loss_key]

        if args.constant_ratio is not None:
            cl = []
            for x in train_dataset:
                if len(np.unique(x["mask"])) > 1:
                    cl.append(1)
                else:
                    cl.append(0)
            sampler = PartiallyRandomSampler(
                cl,non_keep_ratio=args.constant_ratio,seed=args.seed)
        else:
            sampler = None

        def train_loader_call(): 
            return monai.data.ThreadDataLoader(
                train_dataset,batch_size=network_config["batch_size"],
                num_workers=args.n_workers,generator=g,sampler=sampler,
                collate_fn=collate_fn,pin_memory=True,
                persistent_workers=True)

        train_loader = train_loader_call()
        train_val_loader = monai.data.ThreadDataLoader(
            train_dataset_val,batch_size=network_config["batch_size"],
            shuffle=False,num_workers=args.n_workers,generator=g,
            collate_fn=collate_fn,persistent_workers=True)
        validation_loader = monai.data.ThreadDataLoader(
            validation_dataset,batch_size=1,
            shuffle=False,num_workers=args.n_workers,generator=g,
            collate_fn=collate_fn,persistent_workers=True)

        print("Setting up training...")
        if args.res_config_file is not None:
            _,network_config_ssl = parse_config_ssl(
                args.res_config_file,0.,len(keys),network_config["batch_size"])
            for k in ['weight_decay','learning_rate','batch_size']:
                if k in network_config_ssl:
                    del network_config_ssl[k]
            res_net = ResNet(**network_config_ssl)
            if args.res_checkpoint is not None:
                res_state_dict = torch.load(
                    args.res_checkpoint)['state_dict']
                mismatched = res_net.load_state_dict(
                    res_state_dict,strict=False)
            backbone = res_net.backbone
            network_config['depth'] = [
                backbone.structure[0][0],
                *[x[0] for x in backbone.structure]]
            network_config['kernel_sizes'] = [3 for _ in network_config['depth']]
            network_config['strides'] = [2 for _ in network_config['depth']]
            res_ops = [backbone.input_layer,*backbone.operations]
            res_pool_ops = [backbone.first_pooling,*backbone.pooling_operations]
            if args.res_checkpoint is not None:
                # freezes training for resnet encoder
                for res_op in res_ops:
                    for param in res_op.parameters():
                        pass #param.requires_grad = False
            encoding_operations = torch.nn.ModuleList(
                [torch.nn.ModuleList([a,b]) 
                 for a,b in zip(res_ops,res_pool_ops)])
        else:
            encoding_operations = None
        if args.unet_pp == True:
            unet = UNetPlusPlusPL(
                training_dataloader_call=train_loader_call,
                encoding_operations=encoding_operations,
                image_key="image",label_key="mask",
                loss_params=loss_params,n_classes=n_classes,
                bottleneck_classification=args.bottleneck_classification,
                **network_config)
        else:
            unet = UNetPL(
                training_dataloader_call=train_loader_call,
                encoding_operations=encoding_operations,
                image_key="image",label_key="mask",
                loss_params=loss_params,n_classes=n_classes,
                bottleneck_classification=args.bottleneck_classification,
                **network_config)

        if args.from_checkpoint is not None:
            state_dict = torch.load(
                args.from_checkpoint,map_location=args.dev)['state_dict']
            inc = unet.load_state_dict(state_dict)
        
        callbacks = []

        if args.early_stopping is not None:
            early_stopping = EarlyStopping(
                'val_loss',patience=args.early_stopping,
                strict=True,mode="min")
            callbacks.append(early_stopping)
        if args.swa == True:
            swa_callback = StochasticWeightAveraging()
            callbacks.append(swa_callback)

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
            project_name = "unet_" + args.summary_name.replace(":","_")
            logger = WandbLogger(
                save_dir=args.summary_dir,project=project_name,
                name="fold_{}".format(val_fold),
                version="fold_{}".format(val_fold),reinit=True)
        else:
            logger = None

        trainer = Trainer(
            accelerator="gpu" if args.dev=="cuda" else "cpu",
            devices=args.n_devices,logger=logger,callbacks=callbacks,
            strategy='ddp' if args.n_devices > 1 else None,
            max_epochs=args.max_epochs,
            check_val_every_n_epoch=1,log_every_n_steps=10)

        trainer.fit(unet, train_loader, train_val_loader)
        
        print("Validating...")
        test_metrics = trainer.test(unet,validation_loader)[0]
        for k in test_metrics:
            out = test_metrics[k]
            if n_classes == 2:
                try:
                    value = float(out.detach().numpy())
                except:
                    value = float(out)
                x = "{},{},{},{}".format(k,val_fold,0,value)
                output_file.write(x+'\n')
                print(x)
            else:
                for i,v in enumerate(out):
                    x = "{},{},{},{}".format(k,val_fold,i,v)
                    output_file.write(x+'\n')
                    print(x)