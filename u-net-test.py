import argparse
import random
import json
import numpy as np
import torch
import monai
import wandb
from sklearn.model_selection import KFold,train_test_split

from pytorch_lightning import Trainer
from pytorch_lightning.callbacks import EarlyStopping,StochasticWeightAveraging
from pytorch_lightning.loggers import WandbLogger
from pytorch_lightning.callbacks.model_checkpoint import ModelCheckpoint

from lib.utils import (
    PartiallyRandomSampler,
    LabelOperatorSegmentationd,
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
        '--possible_labels',dest='possible_labels',type=int,nargs='+',
        help="All the possible labels in the data.",
        required=True)
    parser.add_argument(
        '--positive_labels',dest='positive_labels',type=int,nargs='+',
        help="Labels that should be considered positive (binarizes labels)",
        default=None)

    # network + interence
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
        '--tta',dest='tta',action="store_true",
        help="Use test-time augmentations",default=False)

    # inference
    parser.add_argument(
        '--dev',dest='dev',type=str,
        help="Device for PyTorch training",choices=["cuda","cpu"])
    parser.add_argument(
        '--seed',dest='seed',help="Random seed",default=42,type=int)
    parser.add_argument(
        '--n_workers',dest='n_workers',
        help="Number of workers",default=1,type=int)
    parser.add_argument(
        '--folds',dest="folds",
        help="Specifies the comma separated validation IDs for each fold",
        default=None,
        type=str,nargs='+')
    parser.add_argument(
        '--checkpoint_path',dest='checkpoint_path',type=str,default="models",
        help='Path to directory where checkpoints will be saved.')

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
    if args.mask_image_keys is None:
        args.mask_image_keys = []
    if args.adc_image_keys is None:
        args.adc_image_keys = []
    if args.skip_key is None:
        args.skip_key = []
    if args.skip_mask_key is None:
        args.skip_mask_key = []
    if args.resize_keys is None:
        args.resize_keys = []
    if args.feature_keys is None:
        args.feature_keys = []
    aux_keys = args.skip_key
    aux_mask_keys = args.skip_mask_key
    all_aux_keys = aux_keys + aux_mask_keys
    if len(all_aux_keys) > 0:
        aux_key_net = "aux_key"
    else:
        aux_key_net = None
    if len(args.feature_keys) > 0:
        feature_key_net = "tabular_features"
    else:
        feature_key_net = None

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
    intp.extend(["area"]*len(aux_keys))
    intp.extend(["nearest"]*len(aux_mask_keys))
    intp_resampling_augmentations.extend(["nearest"]*len(label_keys))
    intp_resampling_augmentations.extend(["bilinear"]*len(aux_keys))
    intp_resampling_augmentations.extend(["nearest"]*len(aux_mask_keys))
    all_keys = [*keys,*label_keys,*aux_keys,*aux_mask_keys]
    all_keys_t = [*all_keys,*args.feature_keys]
    if args.input_size is not None:
        args.input_size = [round(x) for x in args.input_size]

    data_dict = json.load(open(args.dataset_json,'r'))
    data_dict = {
        k:data_dict[k] for k in data_dict
        if len(set.intersection(set(data_dict[k]),
                                set(all_keys_t))) == len(all_keys_t)}
    for kk in args.feature_keys:
        data_dict = {
            k:data_dict[k] for k in data_dict
            if np.isnan(data_dict[k][kk]) == False}
    
    network_config,loss_key = parse_config_unet(
        args.config_file,len(keys),n_classes)

    print("Setting up transforms...")
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
        if len(args.adc_image_keys) > 0:
            scaling_ops.append(
                ConditionalRescalingd(args.adc_image_keys,1000,0.001))
            scaling_ops.append(
                monai.transforms.ScaleIntensityd(
                    args.adc_image_keys,None,None,1-args.adc_factor))
        if args.input_size is not None and len(args.resize_keys) > 0:
            intp_ = [k for k,kk in zip(intp,all_keys) 
                     if kk in args.resize_keys]
            resize = [monai.transforms.Resized(
                args.resize_keys,tuple(args.input_size),mode=intp_)]
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
                monai.transforms.Orientationd(all_keys,"RAS"),
                *rs,
                *resize,
                *crop_op,
                *scaling_ops,
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
            if len(args.feature_keys) > 0:
                feature_concat = [
                    monai.transforms.EnsureTyped(
                        args.feature_keys,dtype=np.float32),
                    monai.transforms.Lambdad(
                        args.feature_keys,
                        func=lambda x:np.reshape(x,[1])),
                    monai.transforms.ConcatItemsd(
                        args.feature_keys,feature_key_net)]
            else:
                feature_concat = []
            return [
                monai.transforms.ConcatItemsd(keys,"image"),
                *aux_concat,
                *feature_concat,
                monai.transforms.ToTensord(["image","mask"])]

    label_mode = "binary" if n_classes == 2 else "cat"
    all_pids = [k for k in data_dict]

    transforms_val = [
        *get_transforms("pre",label_mode),
        *get_transforms("post",label_mode)]
    
    if network_config["spatial_dimensions"] == 2:
        transforms_val.append(
            SlicesToFirst(["image","mask"]))
        collate_fn = collate_last_slice
    else:
        collate_fn = safe_collate

    n_folds = len(args.folds)
    folds = []
    for val_ids in args.folds:
        val_ids = val_ids.split(',')
        train_idxs = [i for i,x in enumerate(all_pids) if x not in val_ids]
        val_idxs = [i for i,x in enumerate(all_pids) if x in val_ids]
        folds.append([train_idxs,val_idxs])
    fold_generator = iter(folds)

    for val_fold in range(n_folds):
        val_list = [data_dict[i] for i in val_idxs]
        validation_dataset = monai.data.Dataset(
            val_list,
            monai.transforms.Compose(transforms_val))

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

        if len(args.feature_keys) > 0:
            all_params = {
                "mean":torch.zeros([len(args.feature_keys)]),
                "std":torch.ones([len(args.feature_keys)])}
        else:
            all_params = None
            
        if args.unet_pp == True:
            unet = UNetPlusPlusPL(
                encoding_operations=encoding_operations,
                image_key="image",label_key="mask",
                loss_params={},n_classes=n_classes,
                bottleneck_classification=args.bottleneck_classification,
                skip_conditioning=len(all_aux_keys),
                skip_conditioning_key=aux_key_net,
                feature_conditioning=len(args.feature_keys),
                feature_conditioning_params=all_params,
                feature_conditioning_key=feature_key_net,
                tta=args.tta,
                **network_config)
        else:
            unet = UNetPL(
                encoding_operations=encoding_operations,
                image_key="image",label_key="mask",
                loss_params={},n_classes=n_classes,
                bottleneck_classification=args.bottleneck_classification,
                skip_conditioning=len(all_aux_keys),
                skip_conditioning_key=aux_key_net,
                feature_conditioning=len(args.feature_keys),
                feature_conditioning_params=all_params,
                feature_conditioning_key=feature_key_net,
                tta=args.tta,
                **network_config)

        state_dict = torch.load(
            args.checkpoint,map_location=args.dev)['state_dict']
        inc = unet.load_state_dict(state_dict)

        trainer = Trainer(
            accelerator="gpu" if args.dev=="cuda" else "cpu",
            logger=None,enable_checkpointing=False)
        
        test_metrics = trainer.test(unet,validation_loader)[0]
        for k in test_metrics:
            out = test_metrics[k]
            if n_classes == 2:
                try:
                    value = float(out.detach().numpy())
                except:
                    value = float(out)
                x = "{},{},{},{}".format(k,val_fold,0,value)
                print(x)
            else:
                for i,v in enumerate(out):
                    x = "{},{},{},{}".format(k,val_fold,i,v)
                    print(x)
