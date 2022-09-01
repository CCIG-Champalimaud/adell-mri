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

from lib.utils import (
    ConditionalRescalingd,split)
from lib.modules.layers import ResNet
from lib.modules.segmentation import UNet,UNetPlusPlus
from lib.modules.config_parsing import parse_config_unet,parse_config_ssl

if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    # data
    parser.add_argument(
        '--image_paths',dest='image_paths',type=str,
        help="Paths to input images (if more than one input sequence, these \
            should be comma-separated)",required=True)
    parser.add_argument(
        '--resize_keys',dest='resize_keys',type=str,nargs='+',default=None,
        help="Keys that will be resized to input size")
    parser.add_argument(
        '--target_spacing',dest='target_spacing',action="store",default=None,
        help="Resamples all images to target spacing",nargs='+',type=float)
    parser.add_argument(
        '--input_idx',dest='input_idx',type=str,nargs='+',
        help="Indices for images which will be input",
        required=True)
    parser.add_argument(
        '--skip_idx',dest='skip_idx',type=str,default=None,
        nargs='+',
        help="Indices for images used for skip connection conditioning.")
    parser.add_argument(
        '--mask_idx',dest='mask_idx',type=str,nargs='+',
        help="Indices for images which are masks (changes interpolation)",
        default=None)
    parser.add_argument(
        '--adc_image_idx',dest='adc_image_idx',type=str,nargs='+',
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
    parser.add_argument('n_classes',dest='n_classes',type=int,
        help='Number of classes')
    parser.add_argument(
        '--res_config_file',dest='res_config_file',action="store",default=None,
        help="Uses a ResNet as a backbone (depths are inferred from this). \
            This config file is then used to parameterise the ResNet.")
    parser.add_argument(
        '--checkpoint',dest='checkpoint',action="store",
        help="Checkpoint for network")
    
    # inference
    parser.add_argument(
        '--dev',dest='dev',type=str,
        help="Device for PyTorch training",choices=["cuda","cpu"])
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

    torch.manual_seed(args.seed)
    random.seed(args.seed)
    np.random.seed(args.seed)
    g = torch.Generator()
    g.manual_seed(args.seed)

    n_classes = args.n_classes

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
                    interpolation.append("nearest")
                    interpolation_resample.append("nearest")
                else:
                    interpolation.append("area")
                    interpolation_resample.append("bilinear")
        for i,im in enumerate(feature_set):
            if k not in feature_keys:
                feature_keys.append(k)
            data_dict[identifier]["feature_{}".format(i)] = im

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
        args.config_file,len(keys),n_classes)

    def get_transforms(x,label_mode=None):
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
        if args.input_size is not None and len(args.resize_keys) > 0:
            intp_ = [k for k,kk in zip(interpolation,keys) 
                     if kk in args.resize_keys]
            resize = [monai.transforms.Resized(
                args.resize_keys,tuple(args.input_size),mode=intp_)]
        else:
            resize = []
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
        return [
                monai.transforms.LoadImaged(keys),
                monai.transforms.AddChanneld(keys),
                monai.transforms.Orientationd(keys,"RAS"),
                *rs,
                *resize,
                *crop_op,
                *scaling_ops,
                monai.transforms.EnsureTyped(keys),
                monai.transforms.ConcatItemsd(keys,"image"),
                *aux_concat,
                *feature_concat,
                monai.transforms.ToTensord(["image"])]

    label_mode = "binary" if n_classes == 2 else "cat"
    all_pids = [k for k in data_dict]

    print("Setting up networks...")
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
    
    if len(feature_keys) > 0:
        all_params = {
            "mean":torch.zeros([len(feature_keys)]),
            "std":torch.ones([len(feature_keys)])}

    if args.unet_pp == True:
        unet = UNetPlusPlus(
            encoding_operations=encoding_operations,
            n_classes=n_classes,
            bottleneck_classification=False,
            skip_conditioning=len(aux_keys),
            feature_conditioning=len(args.feature_keys),
            feature_conditioning_params=all_params,
            **network_config)
    else:
        unet = UNet(
            encoding_operations=encoding_operations,
            n_classes=n_classes,
            bottleneck_classification=False,
            skip_conditioning=len(aux_keys),
            feature_conditioning=len(args.feature_keys),
            feature_conditioning_params=all_params,
            **network_config)

    state_dict = torch.load(
        args.checkpoint,map_location=args.dev)['state_dict']
    inc = unet.load_state_dict(state_dict)
    
    print(inc)

    print("Setting up data...")
    transforms = get_transforms()
                
    for k in tqdm(data_dict):
        d = transforms(data_dict[k])

        X = torch.unsqueeze(d["image"],0)
        x_cond = None
        x_fc = None
        if len(aux_keys) is not None:
            x_cond = torch.unsqueeze(d[aux_key_net],0)
        if len(feature_keys) is not None:
            x_fc = torch.unsqueeze(d["tabular_features"],0)            

        if args.tta == True:
            X = torch.cat([X,X[:,:,::-1]])
            x_cond = torch.cat([x_cond,x_cond[:,:,::-1]])

        output = unet.forward(
            X,X_skip_layer=x_cond,X_feature_conditioning=x_fc)
        if args.tta == True:
            output = split(output,2,0)
            output = (output[0] + output[1])/2
        output = output.squeeze(0)
        output = output.detach().cpu().numpy()
        if n_classes == 2:
            output = filters.apply_hysteresis_threshold(
                output,0.45, 0.5)[0]
        else:
            output = np.argmax(output,axis=0)
        
        target_image = sitk.ReadImage(data_dict[k]["image"])
        output = sitk.GetImageFromArray(output)
        output.CopyInformation(target_image)
        output = sitk.Cast(output,sitk.sitkInt16)
        sitk.WriteImage(
            output,
            os.path.join(args.output_path,"{}.mha".format(k)))
