import numpy as np
import monai

from typing import List
from .utils import (
    ConditionalRescalingd,
    CombineBinaryLabelsd,
    LabelOperatorSegmentationd,
    CreateImageAndWeightsd)

def get_transforms_unet(x,
                        all_keys: List[str],
                        image_keys: List[str],
                        label_keys: List[str],
                        non_adc_keys: List[str],
                        adc_image_keys: List[str],
                        target_spacing: List[float],
                        intp: List[str],
                        intp_resampling_augmentations: List[str],
                        possible_labels: List[str],
                        positive_labels: List[str],
                        adc_factor: float,
                        all_aux_keys: List[str]=[],
                        resize_keys: List[str]=[],
                        feature_keys: List[str]=[],
                        aux_key_net: str=None,
                        feature_key_net: str=None,
                        input_size: List[int]=None,
                        crop_size: List[int]=None,
                        label_mode: str=None,
                        fill_missing: bool=False,
                        brunet: bool=False):
    if target_spacing is not None:
        rs = [
            monai.transforms.Spacingd(
                keys=all_keys,pixdim=target_spacing,
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
                adc_image_keys,None,None,-(1-adc_factor)))
    if input_size is not None and len(resize_keys) > 0:
        intp_ = [k for k,kk in zip(intp,all_keys) 
                    if kk in resize_keys]
        resize = [monai.transforms.Resized(
            resize_keys,tuple(input_size),mode=intp_)]
    else:
        resize = []
    if crop_size is not None:
        crop_size = [int(crop_size[0]),
                        int(crop_size[1]),
                        int(crop_size[2])]
        crop_op = [
            monai.transforms.CenterSpatialCropd(all_keys,crop_size),
            monai.transforms.SpatialPadd(all_keys,crop_size)]
    else:
        crop_op = []

    if fill_missing == True:
        create_images_op = [CreateImageAndWeightsd(
            all_keys,[1] + crop_size)]
    else:
        create_images_op = []
    
    if x == "pre":
        return [
            monai.transforms.LoadImaged(
                all_keys,ensure_channel_first=True,
                allow_missing_keys=fill_missing),
            *create_images_op,
            monai.transforms.Orientationd(all_keys,"RAS"),
            *rs,
            *scaling_ops,
            *resize,
            *crop_op,
            monai.transforms.EnsureTyped(all_keys),
            CombineBinaryLabelsd(label_keys,"any","mask"),
            LabelOperatorSegmentationd(
                ["mask"],possible_labels,
                mode=label_mode,positive_labels=positive_labels)]
    
    elif x == "post":
        if brunet == False:
            concat_op = [monai.transforms.ConcatItemsd(image_keys,"image")]
            totensor_op = [monai.transforms.ToTensord(["image","mask"])]
        else:
            concat_op = []
            totensor_op = [monai.transforms.ToTensord(image_keys + ["mask"])]

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
            *concat_op,
            *aux_concat,
            *feature_concat,
            *totensor_op]

def get_augmentations_unet(augment,
                           all_keys,
                           image_keys,
                           intp_resampling_augmentations):
    if augment == True:
        return [
            monai.transforms.RandBiasFieldd(image_keys,degree=2,prob=0.1),
            monai.transforms.RandAdjustContrastd(image_keys,prob=0.25),
            monai.transforms.RandRicianNoised(
                image_keys,std=0.05,prob=0.25),
            monai.transforms.RandGibbsNoised(
                image_keys,alpha=(0.0,0.6),prob=0.25),
            monai.transforms.RandGaussianSmoothd(image_keys,prob=0.25),
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
