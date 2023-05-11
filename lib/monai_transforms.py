import numpy as np
import torch
import monai

from typing import List
from .utils import (
    ConditionalRescalingd,
    CombineBinaryLabelsd,
    LabelOperatorSegmentationd,
    CreateImageAndWeightsd,
    LabelOperatord,
    CopyEntryd,
    ExposeTransformKeyMetad,
    Offsetd,
    BBToAdjustedAnchorsd,
    MasksToBBd,
    RandRotateWithBoxesd,
    SampleChannelDimd)
from .modules.augmentations import (
    generic_augments,mri_specific_augments,spatial_augments,
    AugmentationWorkhorsed)

class TransformWrapper:
    def __init__(self,data_dictionary,transform):
        self.data_dictionary = data_dictionary
        self.transform = transform
    
    def __call__(self,key):
        return key,self.transform(self.data_dictionary[key])

def unbox(x):
    if isinstance(x,list):
        return x[0]
    else:
        return x

def get_transforms_unet_seg(seg_keys,
                            target_spacing,
                            resize_size,
                            resize_keys,
                            pad_size,
                            crop_size,
                            label_mode,
                            positive_labels):
    intp = ["nearest" for _ in seg_keys]
    transforms = [
        monai.transforms.LoadImaged(
            seg_keys,ensure_channel_first=True,
            allow_missing_keys=seg_keys,image_only=True)]
    # sets orientation
    transforms.append(monai.transforms.Orientationd(seg_keys,"RAS"))
    if target_spacing is not None:
        transforms.append(monai.transforms.Spacingd(
            keys=seg_keys,pixdim=target_spacing,
            mode=intp))
    # sets resize
    if resize_size is not None and len(resize_keys) > 0:
        intp_ = [k for k,kk in zip(intp,seg_keys) 
                 if kk in resize_keys]
        transforms.append(monai.transforms.Resized(
            resize_keys,tuple(resize_size),mode=intp_))
    # sets pad op
    if pad_size is not None:
        transforms.append(
            monai.transforms.SpatialPadd(seg_keys,pad_size))
    # sets crop op
    if crop_size is not None:
        transforms.extend(
            monai.transforms.CenterSpatialCropd(seg_keys,crop_size))
    transforms.extend([CombineBinaryLabelsd(seg_keys,"any","mask"),
                       LabelOperatorSegmentationd(
                           ["mask"],seg_keys,
                           mode=label_mode,positive_labels=positive_labels)])
    return transforms

def get_transforms_unet(x,
                        all_keys: List[str],
                        image_keys: List[str],
                        label_keys: List[str],
                        non_adc_keys: List[str],
                        adc_keys: List[str],
                        target_spacing: List[float],
                        intp: List[str],
                        intp_resampling_augmentations: List[str],
                        possible_labels: List[str]=[0,1],
                        positive_labels: List[str]=[1],
                        adc_factor: float=1.0,
                        all_aux_keys: List[str]=[],
                        resize_keys: List[str]=[],
                        feature_keys: List[str]=[],
                        aux_key_net: str=None,
                        feature_key_net: str=None,
                        resize_size: List[int]=None,
                        crop_size: List[int]=None,
                        pad_size: List[int]=None,
                        random_crop_size: List[int]=None,
                        label_mode: str=None,
                        fill_missing: bool=False,
                        brunet: bool=False):
    if x == "pre":
        transforms = [
            monai.transforms.LoadImaged(
                all_keys,ensure_channel_first=True,
                allow_missing_keys=fill_missing)]
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
                monai.transforms.NormalizeIntensityd(non_adc_keys,0,1))
        if len(adc_keys) > 0:
            transforms.append(
                ConditionalRescalingd(adc_keys,500,0.001))
            transforms.append(
                monai.transforms.ScaleIntensityd(
                    adc_keys,None,None,-(1-adc_factor)))
        # sets resize
        if resize_size is not None and len(resize_keys) > 0:
            intp_ = [k for k,kk in zip(intp,all_keys) 
                     if kk in resize_keys]
            transforms.append(monai.transforms.Resized(
                resize_keys,tuple(resize_size),mode=intp_))
        # sets pad op
        if pad_size is not None:
            transforms.append(
                monai.transforms.SpatialPadd(all_keys,pad_size))
        # sets crop op
        if crop_size is not None:
            transforms.append(
                monai.transforms.CenterSpatialCropd(all_keys,crop_size))
        transforms.append(
            monai.transforms.EnsureTyped(all_keys,dtype=torch.float32))
        if label_keys is not None:
            transforms.extend([
                CombineBinaryLabelsd(label_keys,"any","mask"),
                LabelOperatorSegmentationd(
                    ["mask"],possible_labels,
                    mode=label_mode,positive_labels=positive_labels)
            ])
        # sets indices for random crop op
        if random_crop_size is not None:
            transforms.append(monai.transforms.FgBgToIndicesd("mask"))
        return transforms
    
    elif x == "post":
        keys = []
        transforms = []
        if brunet is False:
            transforms.append(
                monai.transforms.ConcatItemsd(image_keys,"image"))
            
        if len(all_aux_keys) > 0:
            keys.append(all_aux_keys)
            transforms.append(monai.transforms.ConcatItemsd(
                all_aux_keys,aux_key_net))
        if len(feature_keys) > 0:
            keys.append(feature_keys)
            transforms.extend([
                monai.transforms.EnsureTyped(
                    feature_keys,dtype=np.float32),
                monai.transforms.Lambdad(
                    feature_keys,
                    func=lambda x:np.reshape(x,[1])),
                monai.transforms.ConcatItemsd(
                    feature_keys,feature_key_net)])
        if label_keys is not None:
            mask_key = ["mask"]
        else:
            mask_key = []
        if brunet is False:
            keys.append("image")
            transforms.append(monai.transforms.ToTensord(["image"] + mask_key,
                                                         track_meta=False))
        else:
            keys.extend(image_keys)
            transforms.append(monai.transforms.ToTensord(image_keys + mask_key,
                                                         track_meta=False))
        transforms.append(monai.transforms.SelectItemsd(keys + mask_key))
        return transforms

def get_transforms_detection_pre(keys:List[str],
                                 adc_keys:List[str],
                                 input_size:List[int],
                                 box_class_key:str,
                                 shape_key:str,
                                 box_key:str,
                                 mask_key:str,
                                 mask_mode:str="mask_is_labels",
                                 target_spacing:List[float]=None):
    intp_resampling = ["area" for _ in keys]
    non_adc_keys = [k for k in keys if k not in adc_keys]
    if mask_key is not None:
        image_keys = keys + [mask_key]
        spacing_mode = ["bilinear" if k != mask_key else "nearest"
                        for k in image_keys]
    else:
        image_keys = keys
        spacing_mode = ["bilinear" for k in image_keys]
    transforms = [
        monai.transforms.LoadImaged(image_keys,ensure_channel_first=True),
        monai.transforms.Orientationd(image_keys,"RAS")]
    if target_spacing is not None:
        transforms.append(
            monai.transforms.Spacingd(image_keys,target_spacing,
                                      mode=spacing_mode))
    if len(non_adc_keys) > 0:
        transforms.append(
            monai.transforms.ScaleIntensityd(non_adc_keys,0,1))
    if len(adc_keys) > 0:
        transforms.append(
            ConditionalRescalingd(adc_keys,500,0.001))
        transforms.append(
            Offsetd(adc_keys,None))
        transforms.append(
            monai.transforms.ScaleIntensityd(adc_keys,None,None,-2/3))
    transforms.extend([
        monai.transforms.SpatialPadd(keys,input_size),
        monai.transforms.CenterSpatialCropd(keys,input_size)])
    if mask_key is not None:
        transforms.append(
            MasksToBBd(keys=[mask_key],
                       bounding_box_key=box_key,
                       classes_key=box_class_key,
                       shape_key=shape_key,
                       mask_mode=mask_mode))
    return transforms

def get_transforms_detection_post(keys:List[str],
                                  t2_keys:List[str],
                                  anchor_array,
                                  input_size:List[int],
                                  output_size:List[int],
                                  iou_threshold:float,
                                  box_class_key:str,
                                  shape_key:str,
                                  box_key:str,
                                  augments:bool,
                                  predict=False):
    intp_resampling = ["area" for _ in keys]
    transforms = []
    transforms.append(
        get_augmentations_detection(augments,keys,[box_key],
                                    t2_keys,intp_resampling))
    if predict == False:
        transforms.append(BBToAdjustedAnchorsd(
            anchor_sizes=anchor_array,input_sh=input_size,
            output_sh=output_size,iou_thresh=iou_threshold,
            bb_key=box_key,class_key=box_class_key,shape_key=shape_key,
            output_key="bb_map"))
    transforms.append(
        monai.transforms.ConcatItemsd(keys,"image"))
    transforms.append(monai.transforms.EnsureTyped(keys))
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
                                  target_size=None,
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
                Offsetd(adc_keys,None))
            transforms.append(
                monai.transforms.ScaleIntensityd(adc_keys,None,None,-2/3))
        if target_spacing is not None:
            transforms.extend(
                [   
                    monai.transforms.Spacingd(
                        keys,pixdim=target_spacing,dtype=torch.float32),
                ])
        if target_size is not None:
            transforms.append(
                monai.transforms.Resized(keys=keys,spatial_size=target_size))
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
                           n_channels=1,
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
        SampleChannelDimd(all_keys,n_channels),
        SampleChannelDimd(all_keys,1,3),
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
            ConditionalRescalingd(adc_keys,500,0.001),
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
                           t2_keys,
                           random_crop_size: List[int]=None,
                           n_crops: int=1):
    valid_arg_list = ["intensity","noise","rbf","affine","shear","flip",
                      "blur","trivial"]
    interpolation = ["bilinear" if k in image_keys else "nearest" 
                     for k in all_keys]
    for a in augment:
        if a not in valid_arg_list:
            raise NotImplementedError(
                "augment can only contain {}".format(valid_arg_list))
    augments = []
    
    prob = 0.2
    if "trivial" in augment:
        augments.append(monai.transforms.Identityd(image_keys))
        prob = 1.0

    if "intensity" in augment:
        augments.extend([
            monai.transforms.RandAdjustContrastd(
                image_keys,gamma=(0.5,1.5),prob=prob),
            monai.transforms.RandStdShiftIntensityd(
                image_keys,factors=0.1,prob=prob),
            monai.transforms.RandShiftIntensityd(
                image_keys,offsets=0.1,prob=prob)])
    
    if "blur" in augment:
        augments.extend([
            monai.transforms.RandGaussianSmoothd(image_keys)
        ])
    
    if "noise" in augment:
        augments.extend([
            monai.transforms.RandRicianNoised(
                image_keys,std=0.02,prob=prob),
            monai.transforms.RandGibbsNoised(
                image_keys,alpha=(0.0,0.6),prob=0.25)])
        
    if "flip" in augment:
        augments.append(
            monai.transforms.RandFlipd(
                all_keys,prob=prob,spatial_axis=0))

    if "rbf" in augment and len(t2_keys) > 0:
        augments.append(
            monai.transforms.RandBiasFieldd(
                t2_keys,degree=3,prob=prob))

    if "affine" in augment:
        augments.append(
            monai.transforms.RandAffined(
                all_keys,
                translate_range=[4,4,1],
                rotate_range=[np.pi/16],
                prob=prob,mode=interpolation,
                padding_mode="zeros"))
        
    if "shear" in augment:
        augments.append(
            monai.transforms.RandAffined(
                all_keys,
                shear_range=((0.9,1.1),(0.9,1.1),(0.9,1.1)),
                prob=prob,mode=interpolation,
                padding_mode="zeros"))
    
    if "trivial" in augment:
        augments = monai.transforms.OneOf(augments)
    else:
        augments = monai.transforms.Compose(augments)

    if random_crop_size is not None:
        # do a first larger crop that prevents artefacts introduced by 
        # affine transforms and then crop the rest
        pre_final_size = [int(i * 1.10) for i in random_crop_size]
        augments = [
            monai.transforms.RandCropByPosNegLabeld(
                [*image_keys,"mask"],"mask",pre_final_size,
                allow_smaller=True,num_samples=n_crops,
                fg_indices_key="mask_fg_indices",
                bg_indices_key="mask_bg_indices"),
            augments,
            monai.transforms.CenterSpatialCropd([*image_keys,"mask"],
                                                random_crop_size)
            ]
        augments = monai.transforms.Compose(augments)
        
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

def get_augmentations_detection(augment,
                                image_keys,
                                box_keys,
                                t2_keys,
                                intp_resampling_augmentations):
    valid_arg_list = ["intensity","noise","rbf","rotate","trivial",
                      "distortion"]
    for a in augment:
        if a not in valid_arg_list:
            raise NotImplementedError(
                "augment can only contain {}".format(valid_arg_list))
    
    augments = []
    prob = 0.1
    if "trivial" in augment:
        augments.append(monai.transforms.Identityd(image_keys))
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
        
    if "rbf" in augment and len(t2_keys) > 0:
        augments.append(
            monai.transforms.RandBiasFieldd(t2_keys,degree=3,prob=prob))

    if "rotate" in augment:
        augments.append(
            RandRotateWithBoxesd(
                image_keys=image_keys,
                box_keys=box_keys,
                rotate_range=[np.pi/16],
                prob=prob,mode=["bilinear" for _ in image_keys],
                padding_mode="zeros"))
    
    if "distortion" in augment:
        augments.append(
            monai.transforms.RandGridDistortion(image_keys))
    
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
                          n_transforms=3,
                          n_dim:int=3):
    def flatten_box(box):
        box1 = np.array(box[::2])
        box2 = np.array(roi_size) - np.array(box[1::2])
        out = np.concatenate([box1,box2]).astype(np.float32)
        return out
    
    scaled_crop_size = tuple([int(x) for x in scaled_crop_size])
    roi_size = tuple([int(x) for x in roi_size])
    
    transforms_to_remove = []
    if vicregl == True:
        transforms_to_remove.extend(spatial_augments)
    if n_dim == 2:
        transforms_to_remove.extend(
            ["rotate_z","translate_z","shear_z","scale_z"])
    else:
        # the sharpens are remarkably slow, not worth it imo
        transforms_to_remove.extend(
            ["gaussian_sharpen_x","gaussian_sharpen_y","gaussian_sharpen_z"])
    aug_list = generic_augments+mri_specific_augments+spatial_augments
    aug_list = [x for x in aug_list if x not in transforms_to_remove]
    
    cropping_strategy = []

    if scaled_crop_size is not None:
        small_crop_size = [x//2 for x in scaled_crop_size]
        cropping_strategy.extend([
            monai.transforms.SpatialPadd(all_keys + copied_keys,
                                         small_crop_size),
            monai.transforms.RandSpatialCropd(
                all_keys+copied_keys,
                roi_size=small_crop_size,
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
    dropout_size = tuple([x // 10 for x in roi_size])
    return [
        *cropping_strategy,
        AugmentationWorkhorsed(
            augmentations=aug_list,keys=all_keys,
            mask_keys=[],max_mult=0.5,N=n_transforms,
            dropout_size=dropout_size),
        AugmentationWorkhorsed(
            augmentations=aug_list,keys=copied_keys,
            mask_keys=[],max_mult=0.5,N=n_transforms,
            dropout_size=dropout_size),
        ]
