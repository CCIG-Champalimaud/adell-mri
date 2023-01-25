import re
import os
import argparse
import random
import json
import numpy as np
import torch
import monai
import SimpleITK as sitk
from skimage import filters
from tqdm import tqdm

import sys
sys.path.append(r"..")
from lib.utils import (
    ConditionalRescalingd,split,resample_image)
from lib.modules.layers import ResNet
from lib.modules.segmentation import UNet
from lib.modules.segmentation.unetpp import UNetPlusPlus
from lib.modules.config_parsing import parse_config_unet,parse_config_ssl

if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    # data
    parser.add_argument(
        '--image_paths',dest='image_paths',type=str,nargs="+",
        help="Paths to input images (if more than one input sequence, these \
            should be comma-separated)",required=True)
    parser.add_argument(
        '--resize_keys',dest='resize_keys',type=int,nargs='+',default=None,
        help="Keys that will be resized to input size")
    parser.add_argument(
        '--target_spacing',dest='target_spacing',action="store",default=None,
        help="Resamples all images to target spacing",nargs='+',type=float)
    parser.add_argument(
        '--input_idx',dest='input_idx',type=int,nargs='+',
        help="Indices for images which will be input",
        required=True)
    parser.add_argument(
        '--skip_idx',dest='skip_idx',type=int,default=None,
        nargs='+',
        help="Indices for images used for skip connection conditioning.")
    parser.add_argument(
        '--mask_idx',dest='mask_idx',type=int,nargs='+',
        help="Indices for images which are masks (changes interpolation)",
        default=None)
    parser.add_argument(
        '--adc_idx',dest='adc_idx',type=int,nargs='+',
        help="Indices for ADC images (normalized differently)",
        default=None)
    parser.add_argument(
        '--features',dest='features',type=str,nargs='+',
        help="Comma-separated features used in conditioning",
        default=None)
    parser.add_argument(
        '--input_size',dest='input_size',type=float,nargs='+',default=None,
        help="Input size for network")
    parser.add_argument(
        '--crop_size',dest='crop_size',action="store",
        default=None,type=float,nargs='+',
        help="Size of central crop after resizing (if none is specified then\
            no cropping is performed).")
    parser.add_argument(
        '--adc_factor',dest='adc_factor',type=float,default=1/3,
        help="Multiplies ADC images by this factor.")

    # network
    parser.add_argument(
        '--config_file',dest="config_file",
        help="Path to network configuration file (yaml)",
        required=True)
    parser.add_argument(
        '--unet_pp',dest='unet_pp',action="store_true",
        help="Uses U-Net++ rather than U-Net")
    parser.add_argument(
        '--n_classes',dest='n_classes',type=int,
        help='Number of classes')
    parser.add_argument(
        '--res_config_file',dest='res_config_file',action="store",default=None,
        help="Uses a ResNet as a backbone (depths are inferred from this). \
            This config file is then used to parameterise the ResNet.")
    parser.add_argument(
        '--checkpoints',dest='checkpoints',action="store",nargs="+",
        help="Checkpoint for network")
    
    # inference
    parser.add_argument(
        '--dev',dest='dev',type=str,
        help="Device for PyTorch training",
        choices=["cuda","cuda:0","cuda:1","cpu"])
    parser.add_argument(
        '--n_workers',dest='n_workers',
        help="Number of workers",default=1,type=int)
    parser.add_argument(
        '--tta',dest='tta',action="store_true",
        help="Use test-time augmentations",default=False)
    parser.add_argument(
        '--output_dir',dest='output_dir',
        help='Output directory for predictions')
    parser.add_argument(
        '--output_regex',dest='output_regex',
        help='Regex used on images to get an ID which will be used as the \
            file name',default='[0-9]+_[0-9]+')

    args = parser.parse_args()

    n_classes = args.n_classes

    if args.features is None:
        args.features = ["" for _ in args.image_paths]
    if args.mask_idx is None:
        args.mask_idx = []
    if args.skip_idx is None:
        args.skip_idx = []
    if args.adc_idx is None:
        args.adc_idx = []

    data_dict = {}
    keys = []
    feature_keys = []
    interpolation = []
    interpolation_resample = []
    for image_set,feature_set in zip(args.image_paths,args.features):
        image_set = image_set.split(',')
        identifier = re.search(args.output_regex,image_set[0]).group()
        data_dict[identifier] = {}
        for i,im in enumerate(image_set):
            k = "image_{}".format(i)
            data_dict[identifier][k] = im
            if k not in keys:
                keys.append(k)
                if i in args.mask_idx:
                    interpolation_resample.append("nearest")
                else:
                    interpolation_resample.append("bilinear")
        for i,im in enumerate(feature_set.split(",")):
            k = "feature_{}".format(i)
            if k not in feature_keys:
                feature_keys.append(k)
            data_dict[identifier][k] = float(im)

    input_keys = [x for i,x in enumerate(keys) if i in args.input_idx]
    aux_keys = [x for i,x in enumerate(keys) if i in args.skip_idx]

    if len(aux_keys) > 0:
        aux_key_net = "aux_key"
    else:
        aux_key_net = None
    adc_input_keys = [x for i,x in enumerate(keys) if i in args.adc_idx]
    non_adc_keys = [x for i,x in enumerate(keys) if i not in args.adc_idx]

    if args.input_size is not None:
        args.input_size = [round(x) for x in args.input_size]

    network_config,loss_key = parse_config_unet(
        args.config_file,len(input_keys),n_classes)

    def get_transforms():
        if args.target_spacing is not None:
            rs = [
                monai.transforms.Spacingd(
                    keys=keys,pixdim=args.target_spacing,
                    mode=interpolation_resample)]
        else:
            rs = []
        scaling_ops = []
        if len(non_adc_keys) > 0:
            scaling_ops.append(
                monai.transforms.ScaleIntensityd(non_adc_keys,0,1))
        if len(adc_input_keys) > 0:
            scaling_ops.append(
                ConditionalRescalingd(adc_input_keys,1000,0.001))
            scaling_ops.append(
                monai.transforms.ScaleIntensityd(
                    adc_input_keys,None,None,1-args.adc_factor))
        if args.crop_size is not None:
            crop_op = [
                monai.transforms.CenterSpatialCropd(
                    keys,[int(j) for j in args.crop_size]),
                monai.transforms.SpatialPadd(
                    keys,[int(j) for j in args.crop_size])]
        else:
            crop_op = []

        if len(aux_keys) > 0:
            aux_concat = [monai.transforms.ConcatItemsd(aux_keys,aux_key_net)]
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
                    feature_keys,"tabular_features")]
        else:
            feature_concat = []
        return [
                monai.transforms.LoadImaged(keys),
                monai.transforms.AddChanneld(keys),
                monai.transforms.Orientationd(keys,"RAS"),
                *rs,
                *crop_op,
                *scaling_ops,
                monai.transforms.ConcatItemsd(input_keys,"image"),
                *aux_concat,
                *feature_concat,
                monai.transforms.ToTensord(
                    ["image",aux_key_net,"tabular_features"],
                    allow_missing_keys=True),
                monai.transforms.EnsureTyped(
                    ["image",aux_key_net,"tabular_features"],
                    device=args.dev,allow_missing_keys=True),]

    all_pids = [k for k in data_dict]

    print("Setting up networks...")
    for k in ['weight_decay','learning_rate','batch_size','loss_fn']:
        if k in network_config:
            del network_config[k]

    if args.res_config_file is not None:
        _,network_config_ssl = parse_config_ssl(
            args.res_config_file,0.,len(input_keys),1)
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
    
    if len(feature_keys) > 0:
        all_params = {
            "mean":torch.zeros([len(feature_keys)]),
            "std":torch.ones([len(feature_keys)])}
    else:
        all_params = None

    if args.unet_pp == True:
        unet = UNetPlusPlus(
            encoding_operations=encoding_operations,
            n_classes=n_classes,
            bottleneck_classification=False,
            skip_conditioning=len(aux_keys),
            feature_conditioning=len(feature_keys),
            feature_conditioning_params=all_params,
            **network_config).to(args.dev)
    else:
        unet = UNet(
            encoding_operations=encoding_operations,
            n_classes=n_classes,
            bottleneck_classification=False,
            skip_conditioning=len(aux_keys),
            feature_conditioning=len(feature_keys),
            feature_conditioning_params=all_params,
            **network_config).to(args.dev)
        unet.eval()
    
    print("Setting up data...")
    transforms = monai.transforms.Compose(get_transforms())

    checkpoint_outputs = {}
    for checkpoint in args.checkpoints:     
        state_dict = torch.load(
            checkpoint,map_location=args.dev)['state_dict']
        inc = unet.load_state_dict(state_dict)
        print(inc)
        for k in tqdm(data_dict):
            d = transforms(data_dict[k])

            X = torch.unsqueeze(d["image"],0)
            x_cond = None
            x_fc = None
            if len(aux_keys) > 0:
                x_cond = torch.unsqueeze(d[aux_key_net],0)
            if len(feature_keys) > 0:
                x_fc = torch.unsqueeze(d["tabular_features"],0)
                
            if args.tta == True:
                output,_ = unet.forward(
                    X,X_skip_layer=x_cond,
                    X_feature_conditioning=x_fc)
                outputs = [output[0].cpu().detach().numpy()]
                for d in [(2,),(3,),(4,)]:
                    flipped_X = torch.flip(X,d)
                    if x_cond is not None:
                        flipped_X_cond = torch.flip(x_cond,d) 
                    else:
                        flipped_X_cond = None
                    output,_ = unet.forward(
                        flipped_X,X_skip_layer=flipped_X_cond,
                        X_feature_conditioning=x_fc)
                    output = torch.flip(output,d)[0].cpu().detach().numpy()
                    outputs.append(output)

                output = sum(outputs)/len(outputs)
            else:
                output,_ = unet.forward(
                        X,X_skip_layer=x_cond,X_feature_conditioning=x_fc)
                output = output[0].cpu().detach().numpy()

            if k not in checkpoint_outputs:
                checkpoint_outputs[k] = []
            checkpoint_outputs[k].append(output)

    for k in checkpoint_outputs:
        outputs = checkpoint_outputs[k]
        output = sum(outputs)/len(outputs)
        if n_classes == 2:
            output = filters.apply_hysteresis_threshold(
                output, 0.5, 0.5)[0].astype(np.int16)
        else:
            output = np.argmax(output,axis=0)
                    
        target_image = sitk.ReadImage(data_dict[k]["image_0"])
        target_spacing = target_image.GetSpacing()
        target_size = target_image.GetSize()
        padding_size = np.multiply(
            target_size,
            np.array(target_spacing)/np.array(args.target_spacing)).astype(np.uint32)
        output = monai.transforms.SpatialPad(padding_size)(output[np.newaxis])[0]
        output = sitk.GetImageFromArray(output.swapaxes(0,2)[:,::-1,::-1])
        output.SetSpacing(args.target_spacing)
        output.SetDirection(target_image.GetDirection())
        output.SetOrigin(target_image.GetOrigin())
        output = resample_image(
            output,target_image.GetSpacing(),target_image.GetSize(),True)
        output.CopyInformation(target_image)
        output = sitk.Cast(output,sitk.sitkInt16)
        os.makedirs(args.output_dir,exist_ok=True)
        sitk.WriteImage(
            output,
            os.path.join(args.output_dir,"{}.nii.gz".format(k)))
