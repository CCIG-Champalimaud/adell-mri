import argparse
import random
import json
import os
import numpy as np
import torch
import monai
import wandb
from copy import deepcopy
from sklearn.model_selection import KFold,train_test_split
from tqdm import tqdm

from pytorch_lightning import Trainer
from pytorch_lightning.callbacks import EarlyStopping,StochasticWeightAveraging
from pytorch_lightning.loggers import WandbLogger
from pytorch_lightning.callbacks.model_checkpoint import ModelCheckpoint

from lib.utils import (
    PartiallyRandomSampler,
    LabelOperatorSegmentationd,
    get_loss_param_dict,
    collate_last_slice,
    RandomSlices,
    SlicesToFirst,
    CombineBinaryLabelsd,
    ConditionalRescalingd,
    PrintShaped,
    PrintRanged,
    safe_collate)
from lib.modules.layers import ResNet
from lib.modules.segmentation_pl import UNetPL,UNetPlusPlusPL
from lib.modules.config_parsing import parse_config_unet,parse_config_ssl

torch.backends.cudnn.benchmark = True

def if_none_else(x,obj):
    if x is None:
        return obj
    return x

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
        '--learning_rate',dest="learning_rate",
        help="Learning rate (overrides the lr specified in network_config)",
        default=None,type=float)
    parser.add_argument(
        '--max_epochs',dest="max_epochs",
        help="Maximum number of training epochs",default=100,type=int)
    parser.add_argument(
        '--precision',dest='precision',type=str,default="32",
        help="Floating point precision",choices=["16","32","bf16"])
    parser.add_argument(
        '--n_folds',dest="n_folds",
        help="Number of validation folds",default=5,type=int)
    parser.add_argument(
        '--folds',dest="folds",
        help="Specifies the comma separated IDs for each fold (overrides\
            n_folds)",
        default=None,type=str,nargs='+')
    parser.add_argument(
        '--use_val_as_train_val',dest='use_val_as_train_val',action="store_true",
        help="Use validation set as training validation set.",default=False)
    parser.add_argument(
        '--check_val_every_n_epoch',dest='check_val_every_n_epoch',type=int,
        default=1,help="Epoch frequency of validation.")
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
        '--summary_dir',dest='summary_dir',type=str,default="summaries",
        help='Path to summary directory (for wandb).')
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
        '--early_stopping',dest='early_stopping',type=int,default=None,
        help="No. of checks before early stop (defaults to no early stop).")
    parser.add_argument(
        '--swa',dest='swa',action="store_true",
        help="Use stochastic gradient averaging.",default=False)
    parser.add_argument(
        '--class_weights',dest='class_weights',type=str,nargs='+',
        help="Class weights (by alphanumeric order).",default=1.)

    args = parser.parse_args()

    torch.manual_seed(args.seed)
    random.seed(args.seed)
    np.random.seed(args.seed)
    g = torch.Generator()
    g.manual_seed(args.seed)

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
    data_dict = {
        k:data_dict[k] for k in data_dict
        if len(set.intersection(set(data_dict[k]),
                                set(all_keys_t))) == len(all_keys_t)}
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
    if args.learning_rate is not None:
        network_config["learning_rate"] = args.learning_rate
    
    def get_transforms(x,label_mode=None):
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
        if len(adc_image_keys) > 0:
            scaling_ops.append(
                ConditionalRescalingd(adc_image_keys,1000,0.001))
            scaling_ops.append(
                monai.transforms.ScaleIntensityd(
                    adc_image_keys,None,None,-(1-args.adc_factor)))
        if args.input_size is not None and len(resize_keys) > 0:
            intp_ = [k for k,kk in zip(intp,all_keys) 
                     if kk in resize_keys]
            resize = [monai.transforms.Resized(
                resize_keys,tuple(args.input_size),mode=intp_)]
        else:
            resize = []
        if args.crop_size is not None:
            crop_size = [int(args.crop_size[0]),
                         int(args.crop_size[1]),
                         int(args.crop_size[2])]
            crop_op = [
                monai.transforms.CenterSpatialCropd(all_keys,crop_size),
                monai.transforms.SpatialPadd(all_keys,crop_size)]
        else:
            crop_op = []

        if x == "pre":
            return [
                monai.transforms.LoadImaged(
                    all_keys,ensure_channel_first=True),
                monai.transforms.Orientationd(all_keys,"RAS"),
                *rs,
                *scaling_ops,
                *resize,
                *crop_op,
                monai.transforms.EnsureTyped(all_keys),
                CombineBinaryLabelsd(label_keys,"any","mask"),
                LabelOperatorSegmentationd(
                    ["mask"],args.possible_labels,
                    mode=label_mode,positive_labels=args.positive_labels)]
        elif x == "post":
            if len(all_aux_keys) > 0:
                aux_concat = [monai.transforms.ConcatItemsd(
                    all_aux_keys,aux_key_net)]
            else:
                aux_concat = []
            if len(feature_keys) > 0:
                feature_concat = [
                    monai.transforms.EnsureTyped(
                        feature_keys,dtype=np.float32),
                    monai.transforms.Lambdad(
                        feature_keys,
                        func=lambda x:np.reshape(x,[1])),
                    monai.transforms.ConcatItemsd(
                        feature_keys,feature_key_net)]
            else:
                feature_concat = []
            return [
                monai.transforms.ConcatItemsd(keys,"image"),
                *aux_concat,
                *feature_concat,
                monai.transforms.ToTensord(["image","mask"])]

    def get_augmentations(augment):
        if augment == True:
            return [
                monai.transforms.RandBiasFieldd(keys,degree=2,prob=0.1),
                monai.transforms.RandAdjustContrastd(keys,prob=0.25),
                monai.transforms.RandRicianNoised(
                    keys,std=0.05,prob=0.25),
                monai.transforms.RandGibbsNoised(
                    keys,alpha=(0.0,0.6),prob=0.25),
                monai.transforms.RandGaussianSmoothd(keys,prob=0.25),
                monai.transforms.RandAffined(
                    all_keys,
                    scale_range=[0.1,0.1,0.1],
                    rotate_range=[np.pi/8,np.pi/8,np.pi/16],
                    translate_range=[10,10,1],
                    shear_range=((0.9,1.1),(0.9,1.1),(0.9,1.1)),
                    prob=0.5,mode=intp_resampling_augmentations,
                    padding_mode="zeros"),
                monai.transforms.RandFlipd(
                    all_keys,prob=0.5,spatial_axis=[0,1,2])]
        else:
            return []

    label_mode = "binary" if n_classes == 2 else "cat"
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
            RandomSlices(["image","mask"],"mask",8,base=0.05))
        transforms_train_val.append(
            SlicesToFirst(["image","mask"]))
        transforms_val.append(
            SlicesToFirst(["image","mask"]))
        collate_fn = collate_last_slice
    else:
        collate_fn = safe_collate

    if args.folds is None:
        if args.n_folds > 1:
            fold_generator = KFold(
                args.n_folds,shuffle=True,random_state=args.seed).split(all_pids)
        else:
            fold_generator = iter(
                [train_test_split(range(len(all_pids)),test_size=0.2)])
    else:
        args.n_folds = len(args.folds)
        folds = []
        for val_ids in args.folds:
            val_ids = val_ids.split(',')
            train_idxs = [i for i,x in enumerate(all_pids) if x not in val_ids]
            val_idxs = [i for i,x in enumerate(all_pids) if x in val_ids]
            folds.append([train_idxs,val_idxs])
        fold_generator = iter(folds)

    output_file = open(args.metric_path,'w')
    for val_fold in range(args.n_folds):
        print("="*80)
        print("Starting fold={}".format(val_fold))
        callbacks = []

        ckpt_path = None

        if args.checkpoint_name is not None:
            ckpt_name = args.checkpoint_name + "_fold" + str(val_fold)
            ckpt_name = ckpt_name + "_best"
            ckpt_callback = ModelCheckpoint(
                dirpath=args.checkpoint_dir,
                filename=ckpt_name,monitor="val_loss",
                save_last=True,save_top_k=1,mode="min")
            ckpt_last = \
                args.checkpoint_name + "_fold" + str(val_fold) + "_last"
            ckpt_callback.CHECKPOINT_NAME_LAST = ckpt_last
            ckpt_last_full = os.path.join(
                args.checkpoint_dir,ckpt_last+'.ckpt')
            if os.path.exists(ckpt_last_full) and args.resume_from_last == True:
                ckpt_path = ckpt_last_full
                epoch = torch.load(ckpt_path)["epoch"]
                if epoch >= (args.max_epochs-1):
                    print("Training has finished for this fold, skipping")
                    continue
                else:
                    print("Resuming training from checkpoint in {} (epoch={})".format(
                        ckpt_path,epoch))
            callbacks.append(ckpt_callback)

            ckpt = True
        else:
            ckpt = False

        if args.from_checkpoint is not None:
            if len(args.from_checkpoint) >= (val_fold+1):
                ckpt_path = args.from_checkpoint[val_fold]
                print("Resuming training from checkpoint in {}".format(ckpt_path))

        train_idxs,val_idxs = next(fold_generator)
        if args.use_val_as_train_val == False:
            train_idxs,train_val_idxs = train_test_split(train_idxs,test_size=0.15)
        else:
            train_val_idxs = val_idxs
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

        # calculate the mean/std of tabular features
        if feature_keys is not None:
            all_params = {"mean":[],"std":[]}
            for kk in feature_keys:
                f = np.array([x[kk] for x in train_list])
                all_params["mean"].append(np.mean(f))
                all_params["std"].append(np.std(f))
            all_params["mean"] = torch.as_tensor(
                all_params["mean"],dtype=torch.float32,device=args.dev)
            all_params["std"] = torch.as_tensor(
                all_params["std"],dtype=torch.float32,device=args.dev)
        else:
            all_params = None
            
        # include some constant label images
        sampler = None
        if args.constant_ratio is not None or args.class_weights[0] == "adaptive":
            cl = []
            with tqdm(train_list) as t:
                t.set_description("Setting up partially random sampler")
                for x in t:
                    masks = monai.transforms.LoadImaged(keys=label_keys)(x)
                    total = []
                    for k in label_keys:
                        for u in np.unique(masks[k]):
                            if u not in total:
                                total.append(u)
                    if len(total) > 1:
                        cl.append(1)
                    else:
                        cl.append(0)
            adaptive_weights = len(cl)/np.sum(cl)
            if args.constant_ratio is not None:
                sampler = PartiallyRandomSampler(
                    cl,non_keep_ratio=args.constant_ratio,seed=args.seed)

        # weights to tensor
        if args.class_weights[0] != "adaptive":
            weights = torch.as_tensor(
                np.float32(np.array(args.class_weights)),dtype=torch.float32,
                device=args.dev)
        else:
            weights = adaptive_weights
        
        # get loss function parameters
        loss_params = get_loss_param_dict(
            weights=weights,gamma=args.loss_gamma,
            comb=args.loss_comb)[loss_key]

        def train_loader_call(): 
            return monai.data.ThreadDataLoader(
                train_dataset,batch_size=network_config["batch_size"],
                num_workers=args.n_workers,generator=g,sampler=sampler,
                collate_fn=collate_fn,pin_memory=True,
                persistent_workers=True,drop_last=True)

        train_loader = train_loader_call()
        train_val_loader = monai.data.ThreadDataLoader(
            train_dataset_val,batch_size=network_config["batch_size"],
            shuffle=False,num_workers=args.n_workers,generator=g,
            collate_fn=collate_fn,persistent_workers=True)
        validation_loader = monai.data.ThreadDataLoader(
            validation_dataset,batch_size=1,
            shuffle=False,num_workers=args.n_workers,generator=g,
            collate_fn=collate_fn,persistent_workers=True)

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
                        pass
                        #param.requires_grad = False
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
                skip_conditioning=len(all_aux_keys),
                skip_conditioning_key=aux_key_net,
                feature_conditioning=len(feature_keys),
                feature_conditioning_params=all_params,
                feature_conditioning_key=feature_key_net,
                n_epochs=args.max_epochs,
                **network_config)
        else:
            unet = UNetPL(
                training_dataloader_call=train_loader_call,
                encoding_operations=encoding_operations,
                image_key="image",label_key="mask",
                loss_params=loss_params,n_classes=n_classes,
                bottleneck_classification=args.bottleneck_classification,
                skip_conditioning=len(all_aux_keys),
                skip_conditioning_key=aux_key_net,
                feature_conditioning=len(feature_keys),
                feature_conditioning_params=all_params,
                feature_conditioning_key=feature_key_net,
                deep_supervision=args.deep_supervision,
                n_epochs=args.max_epochs,
                **network_config)
        
        if args.early_stopping is not None:
            early_stopping = EarlyStopping(
                'val_loss',patience=args.early_stopping,
                strict=True,mode="min")
            callbacks.append(early_stopping)
        if args.swa == True:
            swa_callback = StochasticWeightAveraging()
            callbacks.append(swa_callback)
                
        if args.summary_name is not None and args.project_name is not None:
            wandb.finish()
            run_name = args.summary_name.replace(':','_') + "_fold_{}".format(val_fold)
            logger = WandbLogger(
                save_dir=args.summary_dir,project=args.project_name,
                name=run_name,version=run_name,reinit=True,resume="allow")
        else:
            logger = None

        # correctly assign devices
        if ":" in args.dev:
            devices = args.dev.split(":")[-1].split(",")
            devices = [int(i) for i in devices]
        else:
            devices = args.n_devices

        trainer = Trainer(
            accelerator="gpu" if "cuda" in args.dev else "cpu",
            devices=devices,logger=logger,callbacks=callbacks,
            strategy='ddp' if args.n_devices > 1 else None,
            max_epochs=args.max_epochs,enable_checkpointing=ckpt,
            accumulate_grad_batches=3,resume_from_checkpoint=ckpt_path,
            check_val_every_n_epoch=args.check_val_every_n_epoch,
            log_every_n_steps=10,precision=args.precision)

        trainer.fit(unet,train_loader,train_val_loader)
        
        print("Validating...")
        test_metrics = trainer.test(
            unet,validation_loader)[0]
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
        print("="*80)