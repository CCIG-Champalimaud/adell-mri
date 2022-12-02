import numpy as np
import monai

from typing import List
from .utils import (
    ConditionalRescalingd,
    CombineBinaryLabelsd,
    LabelOperatorSegmentationd,
    CreateImageAndWeightsd,
    PrintShaped)

def unbox(x):
    if isinstance(x,list):
        return x[0]
    else:
        return x

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
                        resize_size: List[int]=None,
                        crop_size: List[int]=None,
                        random_crop_size: List[int]=None,
                        label_mode: str=None,
                        fill_missing: bool=False,
                        brunet: bool=False):
    if x == "pre":
        transforms = [
            monai.transforms.LoadImaged(
                all_keys,ensure_channel_first=True,
                allow_missing_keys=fill_missing,image_only=True)]
        # "creates" empty images/masks if necessary
        if fill_missing == True:
            transforms.append(CreateImageAndWeightsd(
                all_keys,[1] + crop_size))
        # sets orientation
        transforms.append(monai.transforms.Orientationd(all_keys,"RAS"))
            
        # sets target spacing
        if target_spacing is not None:
            transforms.append(monai.transforms.Spacingd(
                keys=all_keys,pixdim=target_spacing,
                mode=intp_resampling_augmentations))
        # sets intensity transforms for ADC and other sequence types
        if len(non_adc_keys) > 0:
            transforms.append(
                monai.transforms.ScaleIntensityd(non_adc_keys,0,1))
        if len(adc_image_keys) > 0:
            transforms.append(
                ConditionalRescalingd(adc_image_keys,1000,0.001))
            transforms.append(
                monai.transforms.ScaleIntensityd(
                    adc_image_keys,None,None,-(1-adc_factor)))
        # sets resize
        if resize_size is not None and len(resize_keys) > 0:
            intp_ = [k for k,kk in zip(intp,all_keys) 
                     if kk in resize_keys]
            transforms.append(monai.transforms.Resized(
                resize_keys,tuple(resize_size),mode=intp_))
        # sets crop op
        if crop_size is not None:
            crop_size = [int(crop_size[0]),
                         int(crop_size[1]),
                         int(crop_size[2])]

            transforms.extend([
                monai.transforms.CenterSpatialCropd(all_keys,crop_size),
                monai.transforms.SpatialPadd(all_keys,crop_size)])
        # sets indices for random crop op
        if random_crop_size is not None:
            transforms.append(monai.transforms.FgBgToIndicesd("mask"))
        transforms.extend([monai.transforms.EnsureTyped(all_keys),
                           CombineBinaryLabelsd(label_keys,"any","mask"),
                           LabelOperatorSegmentationd(
                               ["mask"],possible_labels,
                               mode=label_mode,positive_labels=positive_labels)])
        return transforms
    
    elif x == "post":
        transforms = []
        if brunet == False:
            transforms.append(
                monai.transforms.ConcatItemsd(image_keys,"image"))
            
        if len(all_aux_keys) > 0:
            transforms.append(monai.transforms.ConcatItemsd(
                all_aux_keys,aux_key_net))
        if len(feature_keys) > 0:
            transforms.extend([
                monai.transforms.EnsureTyped(
                    feature_keys,dtype=np.float32),
                monai.transforms.Lambdad(
                    feature_keys,
                    func=lambda x:np.reshape(x,[1])),
                monai.transforms.ConcatItemsd(
                    feature_keys,feature_key_net)])
        if brunet == False:
            transforms.append(monai.transforms.ToTensord(["image","mask"],
                                                         track_meta=False))
        else:
            transforms.append(monai.transforms.ToTensord(image_keys + ["mask"],
                                                         track_meta=False))
        return transforms

def get_augmentations_unet(augment,
                           all_keys,
                           image_keys,
                           intp_resampling_augmentations,
                           random_crop_size: List[int]=None):
    valid_arg_list = ["full","light","lightest","none"]
    if augment not in valid_arg_list:
        raise NotImplementedError(
            "augment must be one of {}".format(valid_arg_list))
    if augment == "full":
        augments = [
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
                padding_mode="zeros")]
    elif augment == "light":
        augments = [
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
                prob=0.5,mode=intp_resampling_augmentations,
                padding_mode="zeros")]

    elif augment == "lightest":
        augments = [
            monai.transforms.RandAdjustContrastd(image_keys,prob=0.25),
            monai.transforms.RandRicianNoised(
                image_keys,std=0.05,prob=0.25),
            monai.transforms.RandGibbsNoised(
                image_keys,alpha=(0.0,0.6),prob=0.25),
            monai.transforms.RandGaussianSmoothd(image_keys,prob=0.25)]

    #augments.append(
    #    monai.transforms.RandFlipd(
    #        all_keys,prob=0.25,spatial_axis=[0,1,2]))

    if random_crop_size is not None:
        augments.append(
            monai.transforms.RandCropByPosNegLabeld(
                ["image","mask"],"mask",random_crop_size,
                fg_indices_key="mask_fg_indices",
                bg_indices_key="mask_fg_indices"))
    
    return augments