import numpy as np
import monai

from typing import List
from .utils import (
    ConditionalRescalingd,
    CombineBinaryLabelsd,
    LabelOperatorSegmentationd,
    CreateImageAndWeightsd,
    LabelOperatord,
    CopyEntryd,
    ExposeTransformKeyMetad)
from lib.modules.augmentations import (
    generic_augments,mri_specific_augments,spatial_augments,
    AugmentationWorkhorsed)

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
        if fill_missing is True:
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
                ConditionalRescalingd(adc_image_keys,500,0.001))
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
        if brunet is False:
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
        if brunet is False:
            transforms.append(monai.transforms.ToTensord(["image","mask"],
                                                         track_meta=False))
        else:
            transforms.append(monai.transforms.ToTensord(image_keys + ["mask"],
                                                         track_meta=False))
        return transforms

def get_transforms_classification(x,
                                  keys,
                                  adc_keys,
                                  clinical_feature_keys,
                                  target_spacing,
                                  crop_size,
                                  pad_size,
                                  possible_labels,
                                  positive_labels,
                                  label_key,
                                  label_mode=None):
    non_adc_keys = [k for k in keys if k not in adc_keys]
    if x == "pre":
        transforms = [
            monai.transforms.LoadImaged(keys,ensure_channel_first=True),
            monai.transforms.Orientationd(keys,"RAS")]
        if len(non_adc_keys) > 0:
            transforms.append(
                monai.transforms.ScaleIntensityd(non_adc_keys,0,1))
        if len(adc_keys) > 0:
            transforms.append(
                ConditionalRescalingd(adc_keys,500,0.001))
            transforms.append(
                monai.transforms.ScaleIntensityd(
                    adc_keys,None,None,-2/3))
        if target_spacing is not None:
            transforms.append(
                monai.transforms.Spacingd(keys,pixdim=target_spacing))
        if pad_size is not None:
            transforms.append(
                monai.transforms.SpatialPadd(
                    keys,[int(j) for j in pad_size]))
        # initial crop with margin allows for rotation transforms to not create
        # black pixels around the image (these transforms do not need to applied
        # to the whole image)
        if crop_size is not None:
            transforms.append(
                monai.transforms.CenterSpatialCropd(
                    keys,[int(j)+16 for j in crop_size]))
        transforms.append(monai.transforms.EnsureTyped(keys))
    elif x == "post":
        transforms = []
        if crop_size is not None:
            transforms.append(
                monai.transforms.CenterSpatialCropd(
                    keys,[int(j) for j in crop_size]))
        transforms.append(
            monai.transforms.ConcatItemsd(keys,"image"))
        if len(clinical_feature_keys) > 0:
            transforms.extend(
                [monai.transforms.EnsureTyped(
                    clinical_feature_keys,dtype=np.float32),
                 monai.transforms.Lambdad(
                     clinical_feature_keys,
                     func=lambda x:np.reshape(x,[1])),
                 monai.transforms.ConcatItemsd(
                     clinical_feature_keys,"tabular")])
        if isinstance(positive_labels,int):
            positive_labels = [positive_labels]
        transforms.append(
            LabelOperatord(
                [label_key],possible_labels,
                mode=label_mode,positive_labels=positive_labels,
                output_keys={label_key:"label"}))
    return transforms

def get_pre_transforms_ssl(all_keys,
                           copied_keys,
                           adc_keys,
                           non_adc_keys,
                           target_spacing,
                           crop_size,
                           pad_size,
                           n_dim=3):
    intp = []
    intp_resampling_augmentations = []
    key_correspondence = {k:kk for k,kk in zip(all_keys,copied_keys)}
    for k in all_keys:
        intp.append("area")
        intp_resampling_augmentations.append("bilinear")

    transforms = [
        monai.transforms.LoadImaged(
            all_keys,ensure_channel_first=True,image_only=True),
        monai.transforms.SqueezeDimd(all_keys,-1,update_meta=False)]
    if n_dim == 3:
        transforms.append(monai.transforms.Orientationd(all_keys,"RAS"))
    if target_spacing is not None:
        intp_resampling_augmentations = ["bilinear" for _ in all_keys]
        transforms.append(
            monai.transforms.Spacingd(
                keys=all_keys,pixdim=target_spacing,
                mode=intp_resampling_augmentations))
    if len(non_adc_keys) > 0:
        transforms.append(
            monai.transforms.ScaleIntensityd(non_adc_keys,0,1))
    if len(adc_keys) > 0:
        transforms.extend([
            ConditionalRescalingd(adc_keys,1000,0.001),
            monai.transforms.ScaleIntensityd(
                adc_keys,None,None,-2/3)])
    if crop_size is not None:
        transforms.append(
            monai.transforms.CenterSpatialCropd(
                all_keys,[int(j) for j in crop_size]))
    if pad_size is not None:
        transforms.append(
            monai.transforms.SpatialPadd(
                all_keys,[int(j) for j in pad_size]))
    transforms.append(monai.transforms.EnsureTyped(all_keys))
    transforms.append(CopyEntryd(all_keys,key_correspondence))
    return transforms

def get_post_transforms_ssl(all_keys,
                            copied_keys):
    return [
        monai.transforms.ConcatItemsd(all_keys,"augmented_image_1"),
        monai.transforms.ConcatItemsd(copied_keys,"augmented_image_2"),
        monai.transforms.ToTensord(
            ["augmented_image_1","augmented_image_2"],
            track_meta=False)]

def get_augmentations_unet(augment,
                           all_keys,
                           image_keys,
                           intp_resampling_augmentations,
                           random_crop_size: List[int]=None):
    valid_arg_list = ["full","light","lightest","none",True]
    if augment not in valid_arg_list:
        raise NotImplementedError(
            "augment must be one of {}".format(valid_arg_list))
    if augment == "full" or augment is True:
        augments = [
            monai.transforms.RandBiasFieldd(image_keys,degree=3,prob=0.1),
            monai.transforms.RandAdjustContrastd(image_keys,prob=0.25),
            monai.transforms.RandRicianNoised(
                image_keys,std=0.05,prob=0.25),
            monai.transforms.RandGibbsNoised(
                image_keys,alpha=(0.0,0.6),prob=0.25),
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
                image_keys,alpha=(0.0,0.6),prob=0.25)]

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

def get_augmentations_class(augment,
                            all_keys,
                            image_keys,
                            t2_keys,
                            intp_resampling_augmentations):
    valid_arg_list = ["intensity","noise","rbf","affine","shear","flip",
                      "trivial"]
    for a in augment:
        if a not in valid_arg_list:
            raise NotImplementedError(
                "augment can only contain {}".format(valid_arg_list))
    augments = []
    
    prob = 0.1
    if "trivial" in augment:
        augments.append(monai.transforms.Identityd(all_keys))
        prob = 1.0

    if "intensity" in augment:
        augments.extend([
            monai.transforms.RandAdjustContrastd(
                image_keys,gamma=(0.5,1.5),prob=prob),
            monai.transforms.RandStdShiftIntensityd(
                image_keys,factors=0.1,prob=prob),
            monai.transforms.RandShiftIntensityd(
                image_keys,offsets=0.1,prob=prob)])
    
    if "noise" in augment:
        augments.extend([
            monai.transforms.RandRicianNoised(
                image_keys,std=0.02,prob=prob)])
        
    if "flip" in augment:
        augments.append(
            monai.transforms.RandFlipd(image_keys,prob=prob,spatial_axis=0))

    if "rbf" in augment and len(t2_keys) > 0:
        augments.append(
            monai.transforms.RandBiasFieldd(t2_keys,degree=3,prob=prob))
        
    if "affine" in augment:
        augments.append(
            monai.transforms.RandAffined(
                all_keys,
                scale_range=[0.05 for _ in range(3)],
                translate_range=[4,4,1],
                rotate_range=[np.pi/16],
                prob=prob,mode=intp_resampling_augmentations,
                padding_mode="zeros"))
        
    if "shear" in augment:
        augments.append(
            monai.transforms.RandAffined(
                all_keys,
                shear_range=((0.9,1.1),(0.9,1.1),(0.9,1.1)),
                prob=prob,mode=intp_resampling_augmentations,
                padding_mode="zeros"))
    
    if "trivial" in augment:
        augments = monai.transforms.OneOf(augments)
    else:
        augments = monai.transforms.Compose(augments)
    return augments

def get_augmentations_ssl(all_keys:List[str],
                          copied_keys:List[str],
                          scaled_crop_size:List[int],
                          roi_size:List[int],
                          vicregl:bool,
                          n_dim:int=3):
    def flatten_box(box):
        box1 = np.array(box[::2])
        box2 = np.array(roi_size) - np.array(box[1::2])
        out = np.concatenate([box1,box2]).astype(np.float32)
        return out
    
    scaled_crop_size = tuple([int(x) for x in scaled_crop_size])
    roi_size = tuple([int(x) for x in roi_size])
    
    transforms_to_remove = [
        # not super slow but not really a common artefact
        "gaussian_smooth_x","gaussian_smooth_y","gaussian_smooth_z",
        # the sharpens are remarkably slow, not worth it imo
        "gaussian_sharpen_x","gaussian_sharpen_y","gaussian_sharpen_z"]
    if vicregl == True:
        # cannot get the transform information out of MONAI at the moment
        # so the best step forward is to avoid comparing regions which *may*
        # not correspond to one another
        transforms_to_remove.extend(spatial_augments)
    if n_dim == 2:
        transforms_to_remove.extend(
            ["rotate_z","translate_z","shear_z","scale_z"])
    aug_list = generic_augments+mri_specific_augments+spatial_augments
    aug_list = [x for x in aug_list if x not in transforms_to_remove]
    
    cropping_strategy = []

    if scaled_crop_size is not None:
        cropping_strategy.extend([
            monai.transforms.RandSpatialCropd(
                all_keys+copied_keys,
                roi_size=[x//4 for x in scaled_crop_size],
                random_size=True),
            monai.transforms.Resized(all_keys+copied_keys,
                                     scaled_crop_size)])

    if vicregl == True:
        cropping_strategy.extend([
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
            monai.transforms.Lambdad(["box_1","box_2"],flatten_box)])
    else:
        cropping_strategy.append(
            monai.transforms.RandSpatialCropd(
                all_keys+copied_keys,roi_size=roi_size,random_size=False))
    return [
        *cropping_strategy,
        AugmentationWorkhorsed(
            augmentations=aug_list,
            keys=all_keys,mask_keys=[],max_mult=0.5,N=2,
            dropout_size=(8,8)),
        AugmentationWorkhorsed(
            augmentations=aug_list,
            keys=copied_keys,mask_keys=[],max_mult=0.5,N=2,
            dropout_size=(8,8)),
        ]
