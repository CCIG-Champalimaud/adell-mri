import numpy as np
import torch
import monai

from typing import List, Dict, Any
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
    SampleChannelDimd,
    AdjustSizesd,
    CropFromMaskd,
    ConvexHulld,
)
from .modules.augmentations import (
    generic_augments,
    mri_specific_augments,
    spatial_augments,
    AugmentationWorkhorsed,
)
from .modules.semi_supervised_segmentation.utils import (
    convert_arguments_pre,
    convert_arguments_post,
    convert_arguments_augment_all,
    convert_arguments_augment_individual,
)

ADC_FACTOR = -2 / 3


class TransformWrapper:
    def __init__(self, data_dictionary, transform):
        self.data_dictionary = data_dictionary
        self.transform = transform

    def __call__(self, key):
        return key, self.transform(self.data_dictionary[key])


def unbox(x):
    if isinstance(x, list):
        return x[0]
    else:
        return x


def box(x):
    if isinstance(x, float):
        return np.array([x], dtype=np.float32)
    else:
        return np.array([x])


def get_transforms_unet_seg(
    seg_keys,
    target_spacing,
    resize_size,
    resize_keys,
    pad_size,
    crop_size,
    label_mode,
    positive_labels,
):
    intp = ["nearest" for _ in seg_keys]
    transforms = [
        monai.transforms.LoadImaged(
            seg_keys,
            ensure_channel_first=True,
            allow_missing_keys=seg_keys,
            image_only=True,
        )
    ]
    # sets orientation
    transforms.append(monai.transforms.Orientationd(seg_keys, "RAS"))
    if target_spacing is not None:
        transforms.append(
            monai.transforms.Spacingd(
                keys=seg_keys, pixdim=target_spacing, mode=intp
            )
        )
    # sets resize
    if resize_size is not None and len(resize_keys) > 0:
        intp_ = [k for k, kk in zip(intp, seg_keys) if kk in resize_keys]
        transforms.append(
            monai.transforms.Resized(
                resize_keys, tuple(resize_size), mode=intp_
            )
        )
    # sets pad op
    if pad_size is not None:
        transforms.append(monai.transforms.SpatialPadd(seg_keys, pad_size))
    # sets crop op
    if crop_size is not None:
        transforms.extend(
            monai.transforms.CenterSpatialCropd(seg_keys, crop_size)
        )
    transforms.extend(
        [
            CombineBinaryLabelsd(seg_keys, "any", "mask"),
            LabelOperatorSegmentationd(
                ["mask"],
                seg_keys,
                mode=label_mode,
                positive_labels=positive_labels,
            ),
        ]
    )
    return transforms


def get_transforms_unet(
    x,
    all_keys: List[str],
    image_keys: List[str],
    label_keys: List[str],
    non_adc_keys: List[str],
    adc_keys: List[str],
    target_spacing: List[float],
    intp: List[str],
    intp_resampling_augmentations: List[str],
    output_image_key: str = "image",
    possible_labels: List[str] = [0, 1],
    positive_labels: List[str] = [1],
    all_aux_keys: List[str] = [],
    resize_keys: List[str] = [],
    feature_keys: List[str] = [],
    aux_key_net: str = None,
    feature_key_net: str = None,
    resize_size: List[int] = None,
    crop_size: List[int] = None,
    pad_size: List[int] = None,
    random_crop_size: List[int] = None,
    label_mode: str = None,
    fill_missing: bool = False,
    brunet: bool = False,
    track_meta: bool = False,
    convert_to_tensor: bool = True,
):
    if x == "pre":
        transforms = [
            monai.transforms.LoadImaged(
                all_keys,
                ensure_channel_first=True,
                allow_missing_keys=fill_missing,
            )
        ]
        # "creates" empty images/masks if necessary
        if fill_missing is True:
            transforms.append(
                CreateImageAndWeightsd(all_keys, [1] + crop_size)
            )
        # sets orientation
        transforms.append(monai.transforms.Orientationd(all_keys, "RAS"))

        # sets target spacing
        if target_spacing is not None:
            transforms.append(
                monai.transforms.Spacingd(
                    keys=all_keys,
                    pixdim=target_spacing,
                    mode=intp_resampling_augmentations,
                )
            )

        # sets intensity transforms for ADC and other sequence types
        if len(non_adc_keys) > 0:
            transforms.append(
                monai.transforms.ScaleIntensityd(
                    non_adc_keys, minv=0.0, maxv=1.0
                )
            )
        if len(adc_keys) > 0:
            transforms.append(ConditionalRescalingd(adc_keys, 500, 0.001))
            transforms.append(
                monai.transforms.ScaleIntensityd(
                    adc_keys, None, None, ADC_FACTOR
                ),
            )
        # sets resize
        if resize_size is not None and len(resize_keys) > 0:
            intp_ = [k for k, kk in zip(intp, all_keys) if kk in resize_keys]
            transforms.append(
                monai.transforms.Resized(
                    resize_keys, tuple(resize_size), mode=intp_
                )
            )
        # sets pad op
        if pad_size is not None:
            transforms.append(monai.transforms.SpatialPadd(all_keys, pad_size))
        # sets crop op
        if crop_size is not None:
            transforms.append(
                monai.transforms.CenterSpatialCropd(all_keys, crop_size)
            )
        transforms.append(
            monai.transforms.EnsureTyped(all_keys, dtype=torch.float32)
        )
        if label_keys is not None:
            transforms.extend(
                [
                    CombineBinaryLabelsd(label_keys, "any", "mask"),
                    LabelOperatorSegmentationd(
                        ["mask"],
                        possible_labels,
                        mode=label_mode,
                        positive_labels=positive_labels,
                    ),
                ]
            )
        # sets indices for random crop op
        if random_crop_size is not None:
            if label_keys is not None:
                transforms.append(
                    AdjustSizesd([*image_keys, "mask"], mode="crop")
                )
                transforms.append(monai.transforms.FgBgToIndicesd("mask"))
            else:
                transforms.append(AdjustSizesd(image_keys, mode="crop"))
        return transforms

    elif x == "post":
        keys = []
        transforms = []
        if brunet is False:
            transforms.append(
                monai.transforms.ConcatItemsd(image_keys, output_image_key)
            )

        if len(all_aux_keys) > 0:
            keys.append(all_aux_keys)
            transforms.append(
                monai.transforms.ConcatItemsd(all_aux_keys, aux_key_net)
            )
        if len(feature_keys) > 0:
            keys.append(feature_keys)
            transforms.extend(
                [
                    monai.transforms.EnsureTyped(
                        feature_keys, dtype=np.float32
                    ),
                    monai.transforms.Lambdad(
                        feature_keys, func=lambda x: np.reshape(x, [1])
                    ),
                    monai.transforms.ConcatItemsd(
                        feature_keys, feature_key_net
                    ),
                ]
            )
        if label_keys is not None:
            mask_key = ["mask"]
        else:
            mask_key = []
        if convert_to_tensor is True:
            if brunet is False:
                keys.append(output_image_key)
                transforms.append(
                    monai.transforms.ToTensord(
                        [output_image_key] + mask_key,
                        track_meta=track_meta,
                        dtype=torch.float32,
                    )
                )
            else:
                keys.extend(image_keys)
                transforms.append(
                    monai.transforms.ToTensord(
                        image_keys + mask_key,
                        track_meta=track_meta,
                        dtype=torch.float32,
                    )
                )
        else:
            keys.extend(image_keys)
        if track_meta is False:
            transforms.append(monai.transforms.SelectItemsd(keys + mask_key))

        return transforms


def get_transforms_detection_pre(
    keys: List[str],
    adc_keys: List[str],
    pad_size: List[int],
    crop_size: List[int],
    box_class_key: str,
    shape_key: str,
    box_key: str,
    mask_key: str,
    mask_mode: str = "mask_is_labels",
    target_spacing: List[float] = None,
):
    intp_resampling = ["area" for _ in keys]
    non_adc_keys = [k for k in keys if k not in adc_keys]
    if mask_key is not None:
        image_keys = keys + [mask_key]
        spacing_mode = [
            "bilinear" if k != mask_key else "nearest" for k in image_keys
        ]
    else:
        image_keys = keys
        spacing_mode = ["bilinear" for k in image_keys]
    transforms = [
        monai.transforms.LoadImaged(image_keys, ensure_channel_first=True),
        monai.transforms.Orientationd(image_keys, "RAS"),
    ]
    if target_spacing is not None:
        transforms.append(
            monai.transforms.Spacingd(
                image_keys, target_spacing, mode=spacing_mode
            )
        )
    if len(non_adc_keys) > 0:
        transforms.append(monai.transforms.ScaleIntensityd(non_adc_keys, 0, 1))
    if len(adc_keys) > 0:
        transforms.append(ConditionalRescalingd(adc_keys, 500, 0.001))
        transforms.append(Offsetd(adc_keys, None))
        transforms.append(
            monai.transforms.ScaleIntensityd(adc_keys, None, None, ADC_FACTOR)
        )
    transforms.extend(
        [
            monai.transforms.SpatialPadd(keys, pad_size),
            monai.transforms.CenterSpatialCropd(keys, crop_size),
        ]
    )
    if mask_key is not None:
        transforms.append(
            MasksToBBd(
                keys=[mask_key],
                bounding_box_key=box_key,
                classes_key=box_class_key,
                shape_key=shape_key,
                mask_mode=mask_mode,
            )
        )
    return transforms


def get_transforms_detection_post(
    keys: List[str],
    t2_keys: List[str],
    anchor_array,
    input_size: List[int],
    output_size: List[int],
    iou_threshold: float,
    box_class_key: str,
    shape_key: str,
    box_key: str,
    augments: bool,
    predict=False,
):
    intp_resampling = ["area" for _ in keys]
    transforms = []
    transforms.append(
        get_augmentations_detection(
            augments, keys, [box_key], t2_keys, intp_resampling
        )
    )
    if predict is False:
        transforms.append(
            BBToAdjustedAnchorsd(
                anchor_sizes=anchor_array,
                input_sh=input_size,
                output_sh=output_size,
                iou_thresh=iou_threshold,
                bb_key=box_key,
                class_key=box_class_key,
                shape_key=shape_key,
                output_key="bb_map",
            )
        )
    transforms.append(monai.transforms.ConcatItemsd(keys, "image"))
    transforms.append(monai.transforms.EnsureTyped(keys))
    return transforms


def get_transforms_classification(
    x,
    keys: List[str],
    adc_keys: List[str],
    clinical_feature_keys: List[str],
    target_spacing: List[float],
    crop_size: List[int],
    pad_size: List[int],
    image_masking: bool = False,
    image_crop_from_mask: bool = False,
    mask_key: str = None,
    branched: bool = False,
    possible_labels: List[int] = None,
    positive_labels: List[int] = None,
    label_groups: List[int | List[int]] = None,
    label_key: str = None,
    target_size: List[int] = None,
    label_mode: str = None,
    cat_confounder_keys: List[str] = None,
    cont_confounder_keys: List[str] = None,
):
    non_adc_keys = [k for k in keys if k not in adc_keys]
    all_keys = [k for k in keys]
    if mask_key is not None:
        all_keys.append(mask_key)
    if x == "pre":
        transforms = [
            monai.transforms.LoadImaged(all_keys, ensure_channel_first=True)
        ]
        if mask_key is not None:
            transforms.append(
                monai.transforms.ResampleToMatchd(
                    mask_key, key_dst=keys[0], mode="nearest"
                )
            )
        transforms.append(monai.transforms.Orientationd(all_keys, "RAS"))

        if len(non_adc_keys) > 0:
            transforms.append(
                monai.transforms.ScaleIntensityd(non_adc_keys, 0, 1)
            )
        if len(adc_keys) > 0:
            transforms.append(ConditionalRescalingd(adc_keys, 500, 0.001))
            transforms.append(Offsetd(adc_keys, None))
            transforms.append(
                monai.transforms.ScaleIntensityd(
                    adc_keys, None, None, ADC_FACTOR
                )
            )
        if target_spacing is not None:
            transforms.extend(
                [
                    monai.transforms.Spacingd(
                        all_keys,
                        pixdim=target_spacing,
                        dtype=torch.float32,
                        mode=[
                            "bilinear" if k != mask_key else "nearest"
                            for k in all_keys
                        ],
                    ),
                ]
            )
        if target_size is not None:
            transforms.append(
                monai.transforms.Resized(
                    keys=all_keys,
                    spatial_size=target_size,
                    mode=[
                        "bilinear" if k != mask_key else "nearest"
                        for k in all_keys
                    ],
                )
            )
        if pad_size is not None:
            transforms.append(
                monai.transforms.SpatialPadd(
                    all_keys, [int(j) + 16 for j in crop_size]
                )
            )
        # initial crop with margin allows for rotation transforms to not create
        # black pixels around the image (these transforms do not need to applied
        # to the whole image)
        if image_masking is True:
            transforms.append(ConvexHulld([mask_key]))
        if image_crop_from_mask is True:
            transforms.append(
                CropFromMaskd(
                    all_keys,
                    mask_key=mask_key,
                    output_size=[int(j) + 16 for j in crop_size],
                )
            )
        elif crop_size is not None:
            transforms.append(
                monai.transforms.CenterSpatialCropd(
                    all_keys, [int(j) + 16 for j in crop_size]
                )
            )
        transforms.append(monai.transforms.EnsureTyped(all_keys))
    elif x == "post":
        transforms = []
        if crop_size is not None:
            transforms.append(
                monai.transforms.CenterSpatialCropd(
                    all_keys, [int(j) for j in crop_size]
                )
            )
        if image_masking is True:
            transforms.append(
                monai.transforms.MaskIntensityd(keys, mask_key=mask_key)
            )
        if branched is not True:
            transforms.append(monai.transforms.ConcatItemsd(all_keys, "image"))
        if len(clinical_feature_keys) > 0:
            transforms.extend(
                [
                    monai.transforms.EnsureTyped(
                        clinical_feature_keys, dtype=np.float32
                    ),
                    monai.transforms.Lambdad(
                        clinical_feature_keys,
                        func=lambda x: np.reshape(x, [1]),
                    ),
                    monai.transforms.ConcatItemsd(
                        clinical_feature_keys, "tabular"
                    ),
                ]
            )
        if isinstance(positive_labels, int):
            positive_labels = [positive_labels]
        if label_key is not None:
            transforms.append(
                LabelOperatord(
                    [label_key],
                    possible_labels,
                    mode=label_mode,
                    positive_labels=positive_labels,
                    label_groups=label_groups,
                    output_keys={label_key: "label"},
                )
            )
        if cat_confounder_keys is not None:
            transforms.append(
                monai.transforms.Lambdad(
                    cat_confounder_keys, box, track_meta=False
                )
            )
            transforms.append(
                monai.transforms.ConcatItemsd(
                    cat_confounder_keys, "cat_confounder"
                )
            )
        if cont_confounder_keys is not None:
            transforms.append(
                monai.transforms.Lambdad(cont_confounder_keys, box)
            )
            transforms.append(
                monai.transforms.ConcatItemsd(
                    cont_confounder_keys, "cont_confounder"
                )
            )
    return transforms


def get_pre_transforms_generation(keys, target_spacing, crop_size, pad_size):
    transforms = [
        monai.transforms.LoadImaged(
            keys, ensure_channel_first=True, image_only=True
        ),
        monai.transforms.Orientationd(keys, "RAS"),
    ]
    if target_spacing is not None:
        transforms.extend(
            [
                monai.transforms.Spacingd(
                    keys, pixdim=target_spacing, dtype=torch.float32
                ),
            ]
        )
    transforms.append(monai.transforms.ScaleIntensityd(keys))
    if pad_size is not None:
        transforms.append(
            monai.transforms.SpatialPadd(keys, [int(j) for j in pad_size])
        )
    if crop_size is not None:
        transforms.append(
            monai.transforms.CenterSpatialCropd(
                keys, [int(j) + 16 for j in crop_size]
            )
        )
    transforms.append(monai.transforms.EnsureTyped(keys))
    return transforms


def get_post_transforms_generation(
    image_keys: List[str],
    crop_size: List[int] = None,
    cat_keys: List[str] = None,
    num_keys: List[str] = None,
):
    transforms = []
    transforms.append(monai.transforms.ConcatItemsd(image_keys, "image"))
    if crop_size is not None:
        transforms.append(
            monai.transforms.CenterSpatialCropd(
                image_keys, [int(j) for j in crop_size]
            )
        )
    if cat_keys is not None:
        transforms.append(
            monai.transforms.Lambdad(cat_keys, box, track_meta=False)
        )
        transforms.append(monai.transforms.ConcatItemsd(cat_keys, "cat"))
    if num_keys is not None:
        transforms.append(monai.transforms.Lambdad(num_keys, box))
        transforms.append(monai.transforms.ConcatItemsd(num_keys, "num"))
    return transforms


def get_pre_transforms_ssl(
    all_keys: List[str],
    copied_keys: List[str],
    adc_keys: List[str],
    non_adc_keys: List[str],
    target_spacing: List[float],
    crop_size: List[int],
    pad_size: List[int],
    n_channels: int = 1,
    n_dim: int = 3,
    skip_augmentations: bool = False,
    jpeg_dataset: bool = False,
):
    intp = []
    intp_resampling_augmentations = []
    key_correspondence = {k: kk for k, kk in zip(all_keys, copied_keys)}
    for k in all_keys:
        intp.append("area")
        intp_resampling_augmentations.append("bilinear")

    transforms = [
        monai.transforms.LoadImaged(
            all_keys, ensure_channel_first=True, image_only=True
        ),
        SampleChannelDimd(all_keys, n_channels),
    ]
    if jpeg_dataset is False and n_dim == 2:
        transforms.extend(
            [
                SampleChannelDimd(all_keys, 1, 3),
                monai.transforms.SqueezeDimd(all_keys, -1, update_meta=False),
            ]
        )
    if n_dim == 3:
        transforms.append(monai.transforms.Orientationd(all_keys, "RAS"))
    if target_spacing is not None:
        intp_resampling_augmentations = ["bilinear" for _ in all_keys]
        transforms.append(
            monai.transforms.Spacingd(
                keys=all_keys,
                pixdim=target_spacing,
                mode=intp_resampling_augmentations,
            )
        )
    if len(non_adc_keys) > 0:
        transforms.append(monai.transforms.ScaleIntensityd(non_adc_keys, 0, 1))
    if len(adc_keys) > 0:
        transforms.extend(
            [
                ConditionalRescalingd(adc_keys, 500, 0.001),
                monai.transforms.ScaleIntensityd(
                    adc_keys, None, None, ADC_FACTOR
                ),
            ]
        )
    if crop_size is not None:
        transforms.append(
            monai.transforms.CenterSpatialCropd(
                all_keys, [int(j) for j in crop_size]
            )
        )
    if pad_size is not None:
        transforms.append(
            monai.transforms.SpatialPadd(all_keys, [int(j) for j in pad_size])
        )
    transforms.append(monai.transforms.EnsureTyped(all_keys))
    if skip_augmentations is False:
        transforms.append(CopyEntryd(all_keys, key_correspondence))
    return transforms


def get_post_transforms_ssl(
    all_keys: List[str],
    copied_keys: List[str],
    skip_augmentations: bool = False,
):
    if skip_augmentations is False:
        return [
            monai.transforms.ConcatItemsd(all_keys, "augmented_image_1"),
            monai.transforms.ConcatItemsd(copied_keys, "augmented_image_2"),
            monai.transforms.ToTensord(
                ["augmented_image_1", "augmented_image_2"], track_meta=False
            ),
        ]
    else:
        return [
            monai.transforms.ConcatItemsd(all_keys, "image"),
            monai.transforms.ToTensord(["image"], track_meta=False),
        ]


def get_augmentations_unet(
    augment,
    all_keys,
    image_keys,
    t2_keys,
    random_crop_size: List[int] = None,
    has_label: bool = True,
    n_crops: int = 1,
    flip_axis: list[int] = [0, 1],
):
    valid_arg_list = [
        "intensity",
        "noise",
        "rbf",
        "affine",
        "shear",
        "flip",
        "blur",
        "distort",
        "lowres",
        "trivial",
    ]
    interpolation = [
        "bilinear" if k in image_keys else "nearest" for k in all_keys
    ]
    for a in augment:
        if a not in valid_arg_list:
            raise NotImplementedError(
                "augment can only contain {}".format(valid_arg_list)
            )
    augments = []

    prob = 0.2
    if "trivial" in augment:
        augments.append(monai.transforms.Identityd(image_keys))
        prob = 1.0

    if "distort" in augment:
        augments.append(
            monai.transforms.RandGridDistortiond(
                all_keys,
                distort_limit=0.05,
                prob=prob,
                mode=interpolation,
            )
        )

    if "intensity" in augment:
        augments.extend(
            [
                monai.transforms.RandAdjustContrastd(
                    image_keys, gamma=(0.5, 1.5), prob=prob
                ),
                monai.transforms.RandStdShiftIntensityd(
                    image_keys, factors=0.1, prob=prob
                ),
            ]
        )

    if "blur" in augment:
        augments.extend([monai.transforms.RandGaussianSmoothd(image_keys)])

    if "noise" in augment:
        augments.extend(
            [
                monai.transforms.RandRicianNoised(
                    image_keys, std=0.02, prob=prob
                ),
                monai.transforms.RandGibbsNoised(
                    image_keys, alpha=(0.3, 0.6), prob=prob
                ),
            ]
        )

    if "rbf" in augment and len(t2_keys) > 0:
        augments.append(
            monai.transforms.RandBiasFieldd(t2_keys, degree=3, prob=prob)
        )

    if "affine" in augment:
        augments.append(
            monai.transforms.RandAffined(
                all_keys,
                rotate_range=[np.pi / 8, np.pi / 8, np.pi / 16],
                prob=prob,
                mode=interpolation,
            )
        )

    if "shear" in augment:
        augments.append(
            monai.transforms.RandAffined(
                all_keys,
                shear_range=((0.9, 1.1), (0.9, 1.1), (0.9, 1.1)),
                prob=prob,
                mode=interpolation,
            )
        )
    if "lowres" in augment:
        augments.append(
            monai.transforms.RandSimulateLowResolutiond(
                image_keys,
                zoom_range=[0.8, 1.2],
                prob=prob,
            )
        )

    # ensures that flips are applied regardless of TrivialAugment trigger
    if "flip" in augment:
        flip_transform = [
            monai.transforms.RandFlipd(all_keys, spatial_axis=[axis], prob=0.5)
            for axis in flip_axis
        ]
    else:
        flip_transform = []

    if "trivial" in augment:
        augments = monai.transforms.Compose(
            [monai.transforms.OneOf(augments), *flip_transform]
        )
    else:
        augments = monai.transforms.Compose([*augments, *flip_transform])

    if random_crop_size is not None:
        # do a first larger crop that prevents artefacts introduced by
        # affine transforms and then crop the rest
        pre_final_size = [int(i * 1.10) for i in random_crop_size]
        new_augments = []
        if has_label is True:
            new_augments.append(
                monai.transforms.RandCropByPosNegLabeld(
                    [*image_keys, "mask"],
                    "mask",
                    pre_final_size,
                    allow_smaller=True,
                    num_samples=n_crops,
                    fg_indices_key="mask_fg_indices",
                    bg_indices_key="mask_bg_indices",
                )
            )
        else:
            new_augments.append(
                monai.transforms.RandSpatialCropd(
                    image_keys,
                    pre_final_size,
                )
            )

        augments = [
            *new_augments,
            augments,
            monai.transforms.CenterSpatialCropd(
                [*image_keys, "mask"] if has_label else image_keys,
                random_crop_size,
            ),
        ]

        augments = monai.transforms.Compose(augments)

    return augments


def get_augmentations_class(
    augment, all_keys, mask_key, image_keys, t2_keys, flip_axis=[0, 1]
):
    valid_arg_list = [
        "intensity",
        "noise",
        "rbf",
        "affine",
        "shear",
        "flip",
        "blur",
        "trivial",
    ]
    all_keys_with_mask = [k for k in all_keys]
    if mask_key is not None:
        all_keys_with_mask.append(mask_key)
    intp_resampling_augmentations = [
        "bilinear" if k != mask_key else "nearest" for k in all_keys_with_mask
    ]
    for a in augment:
        if a not in valid_arg_list:
            raise NotImplementedError(
                "augment can only contain {}".format(valid_arg_list)
            )
    augments = []

    prob = 0.1
    if "trivial" in augment:
        augments.append(monai.transforms.Identityd(all_keys))
        prob = 1.0

    if "intensity" in augment:
        augments.extend(
            [
                monai.transforms.RandAdjustContrastd(
                    image_keys, gamma=(0.5, 1.5), prob=prob
                ),
                monai.transforms.RandStdShiftIntensityd(
                    image_keys, factors=0.1, prob=prob
                ),
                monai.transforms.RandShiftIntensityd(
                    image_keys, offsets=0.1, prob=prob
                ),
            ]
        )

    if "noise" in augment:
        augments.extend(
            [
                monai.transforms.RandRicianNoised(
                    image_keys, std=0.02, prob=prob
                )
            ]
        )

    if "flip" in augment:
        augments.append(
            monai.transforms.RandFlipd(
                all_keys_with_mask, prob=prob, spatial_axis=flip_axis
            )
        )

    if "blur" in augment:
        augments.extend([monai.transforms.RandGaussianSmoothd(image_keys)])

    if "rbf" in augment and len(t2_keys) > 0:
        augments.append(
            monai.transforms.RandBiasFieldd(t2_keys, degree=3, prob=prob)
        )

    if "affine" in augment:
        augments.append(
            monai.transforms.RandAffined(
                all_keys_with_mask,
                translate_range=[4, 4, 1],
                rotate_range=[np.pi / 16],
                prob=prob,
                mode=intp_resampling_augmentations,
                padding_mode="zeros",
            )
        )

    if "shear" in augment:
        augments.append(
            monai.transforms.RandAffined(
                all_keys_with_mask,
                shear_range=((0.9, 1.1), (0.9, 1.1), (0.9, 1.1)),
                prob=prob,
                mode=intp_resampling_augmentations,
                padding_mode="zeros",
            )
        )

    if "trivial" in augment:
        augments = monai.transforms.OneOf(augments)
    else:
        augments = monai.transforms.Compose(augments)
    return augments


def get_augmentations_detection(
    augment, image_keys, box_keys, t2_keys, intp_resampling_augmentations
):
    valid_arg_list = [
        "intensity",
        "noise",
        "rbf",
        "rotate",
        "trivial",
        "distortion",
    ]
    for a in augment:
        if a not in valid_arg_list:
            raise NotImplementedError(
                "augment can only contain {}".format(valid_arg_list)
            )

    augments = []
    prob = 0.1
    if "trivial" in augment:
        augments.append(monai.transforms.Identityd(image_keys))
        prob = 1.0

    if "intensity" in augment:
        augments.extend(
            [
                monai.transforms.RandAdjustContrastd(
                    image_keys, gamma=(0.5, 1.5), prob=prob
                ),
                monai.transforms.RandStdShiftIntensityd(
                    image_keys, factors=0.1, prob=prob
                ),
                monai.transforms.RandShiftIntensityd(
                    image_keys, offsets=0.1, prob=prob
                ),
            ]
        )

    if "noise" in augment:
        augments.extend(
            [
                monai.transforms.RandRicianNoised(
                    image_keys, std=0.02, prob=prob
                )
            ]
        )

    if "rbf" in augment and len(t2_keys) > 0:
        augments.append(
            monai.transforms.RandBiasFieldd(t2_keys, degree=3, prob=prob)
        )

    if "rotate" in augment:
        augments.append(
            RandRotateWithBoxesd(
                image_keys=image_keys,
                box_keys=box_keys,
                rotate_range=[np.pi / 16],
                prob=prob,
                mode=["bilinear" for _ in image_keys],
                padding_mode="zeros",
            )
        )

    if "distortion" in augment:
        augments.append(monai.transforms.RandGridDistortion(image_keys))

    if "trivial" in augment:
        augments = monai.transforms.OneOf(augments)
    else:
        augments = monai.transforms.Compose(augments)
    return augments


def get_augmentations_ssl(
    all_keys: List[str],
    copied_keys: List[str],
    scaled_crop_size: List[int],
    roi_size: List[int],
    vicregl: bool,
    different_crop: bool,
    n_transforms=3,
    n_dim: int = 3,
    skip_augmentations: bool = False,
):
    def flatten_box(box):
        box1 = np.array(box[::2])
        box2 = np.array(roi_size) - np.array(box[1::2])
        out = np.concatenate([box1, box2]).astype(np.float32)
        return out

    scaled_crop_size = tuple([int(x) for x in scaled_crop_size])
    roi_size = tuple([int(x) for x in roi_size])

    transforms_to_remove = []
    if vicregl is True:
        transforms_to_remove.extend(spatial_augments)
    if n_dim == 2:
        transforms_to_remove.extend(
            ["rotate_z", "translate_z", "shear_z", "scale_z"]
        )
    else:
        # the sharpens are remarkably slow, not worth it imo
        transforms_to_remove.extend(
            ["gaussian_sharpen_x", "gaussian_sharpen_y", "gaussian_sharpen_z"]
        )
    aug_list = generic_augments + mri_specific_augments + spatial_augments
    aug_list = [x for x in aug_list if x not in transforms_to_remove]

    cropping_strategy = []

    if scaled_crop_size is not None:
        small_crop_size = [x // 2 for x in scaled_crop_size]
        cropping_strategy.extend(
            [
                monai.transforms.SpatialPadd(
                    all_keys + copied_keys, small_crop_size
                ),
                monai.transforms.RandSpatialCropd(
                    all_keys + copied_keys,
                    roi_size=small_crop_size,
                    random_size=True,
                ),
                monai.transforms.Resized(
                    all_keys + copied_keys, scaled_crop_size
                ),
            ]
        )

    if skip_augmentations is True:
        return cropping_strategy

    if vicregl is True:
        cropping_strategy.extend(
            [
                monai.transforms.RandSpatialCropd(
                    all_keys, roi_size=roi_size, random_size=False
                ),
                monai.transforms.RandSpatialCropd(
                    copied_keys, roi_size=roi_size, random_size=False
                ),
                # exposes the value associated with the random crop as a key
                # in the data element dict
                ExposeTransformKeyMetad(
                    all_keys[0],
                    "RandSpatialCrop",
                    ["extra_info", "cropped"],
                    "box_1",
                ),
                ExposeTransformKeyMetad(
                    copied_keys[0],
                    "RandSpatialCrop",
                    ["extra_info", "cropped"],
                    "box_2",
                ),
                # transforms the bounding box into (x1,y1,z1,x2,y2,z2) format
                monai.transforms.Lambdad(["box_1", "box_2"], flatten_box),
            ]
        )
    elif different_crop is True:
        cropping_strategy.extend(
            [
                monai.transforms.RandSpatialCropd(
                    all_keys, roi_size=roi_size, random_size=False
                ),
                monai.transforms.RandSpatialCropd(
                    copied_keys, roi_size=roi_size, random_size=False
                ),
            ]
        )
    else:
        cropping_strategy.append(
            monai.transforms.RandSpatialCropd(
                all_keys + copied_keys, roi_size=roi_size, random_size=False
            )
        )
    dropout_size = tuple([x // 10 for x in roi_size])
    transforms = [
        *cropping_strategy,
        AugmentationWorkhorsed(
            augmentations=aug_list,
            keys=all_keys,
            mask_keys=[],
            max_mult=0.5,
            N=n_transforms,
            dropout_size=dropout_size,
        ),
        AugmentationWorkhorsed(
            augmentations=aug_list,
            keys=copied_keys,
            mask_keys=[],
            max_mult=0.5,
            N=n_transforms,
            dropout_size=dropout_size,
        ),
    ]
    return transforms


def get_semi_sl_transforms(
    transform_arguments: Dict[str, Any],
    augment_arguments: Dict[str, Any],
    keys: List[str],
):
    transform_arguments_semi_sl_pre = convert_arguments_pre(
        transform_arguments, keys
    )
    transform_arguments_semi_sl_post_1 = convert_arguments_post(
        transform_arguments, 1, keys
    )
    transform_arguments_semi_sl_post_2 = convert_arguments_post(
        transform_arguments, 2, keys
    )
    augment_arguments_semi_sl_all = convert_arguments_augment_all(
        augment_arguments, keys
    )
    augment_arguments_semi_sl_1 = convert_arguments_augment_individual(
        augment_arguments, image_keys=keys, idx=1
    )
    augment_arguments_semi_sl_2 = convert_arguments_augment_individual(
        augment_arguments, image_keys=keys, idx=2
    )

    transforms_semi_sl = [
        *get_transforms_unet("pre", **transform_arguments_semi_sl_pre),
        CopyEntryd(keys, {k: f"{k}_aug_1" for k in keys}),
        CopyEntryd(keys, {k: f"{k}_aug_2" for k in keys}),
        get_augmentations_unet(**augment_arguments_semi_sl_all),
        get_augmentations_unet(**augment_arguments_semi_sl_1),
        get_augmentations_unet(**augment_arguments_semi_sl_2),
        *get_transforms_unet("post", **transform_arguments_semi_sl_post_1),
        *get_transforms_unet("post", **transform_arguments_semi_sl_post_2),
        monai.transforms.SelectItemsd(["semi_sl_image_1", "semi_sl_image_2"]),
    ]
    return transforms_semi_sl
