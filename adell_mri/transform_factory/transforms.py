from dataclasses import dataclass

import monai
import monai.transforms
import numpy as np
import torch

from adell_mri.utils.monai_transforms import (
    AdjustSizesd,
    BBToAdjustedAnchorsd,
    CombineBinaryLabelsd,
    ConditionalRescalingd,
    ConvexHulld,
    CopyEntryd,
    CreateImageAndWeightsd,
    CropFromMaskd,
    LabelOperatord,
    LabelOperatorSegmentationd,
    MasksToBBd,
    Offsetd,
    SampleChannelDimd,
)

ADC_FACTOR = -2 / 3


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


@dataclass
class TransformMixin:
    def pre_transforms(self):
        raise NotImplementedError("pre_transform must be implemented")

    def post_transforms(self):
        raise NotImplementedError("post_transform must be implemented")

    def transforms(
        self,
        augmentations: monai.transforms.Transform | list | None = None,
        final_transforms: tuple[monai.transforms.Transform] | None = None,
    ):
        transforms = [*self.pre_transforms()]
        if augmentations:
            if isinstance(augmentations, monai.transforms.Transform):
                transforms.append(augmentations)
            else:
                transforms.extend(augmentations)
        transforms.extend(self.post_transforms())
        if final_transforms is not None:
            transforms.extend(final_transforms)
        return monai.transforms.Compose(transforms)


@dataclass
class SegmentationTransforms(TransformMixin):
    all_keys: tuple[str]
    image_keys: tuple[str]
    label_keys: tuple[str]
    non_adc_keys: tuple[str]
    adc_keys: tuple[str]
    target_spacing: tuple[float]
    intp: tuple[str]
    intp_resampling_augmentations: tuple[str]
    output_image_key: str = "image"
    possible_labels: tuple[str] = (0, 1)
    positive_labels: tuple[str] = (1,)
    all_aux_keys: tuple[str] = tuple()
    resize_keys: tuple[str] = tuple()
    feature_keys: tuple[str] = tuple()
    aux_key_net: str = None
    feature_key_net: str = None
    resize_size: tuple[int] = None
    crop_size: tuple[int] = None
    pad_size: tuple[int] = None
    random_crop_size: tuple[int] = None
    label_mode: str = None
    fill_missing: bool = False
    brunet: bool = False
    track_meta: bool = False
    convert_to_tensor: bool = True

    def __post_init__(self):
        self.transform_keys = []
        if not self.brunet:
            self.transform_keys.append(self.output_image_key)
        if self.all_aux_keys:
            self.transform_keys.append(self.all_aux_keys)
        if self.feature_keys:
            self.transform_keys.append(self.feature_keys)
        if self.brunet:
            self.transform_keys.extend(self.image_keys)
        self.mask_key = ["mask"] if self.label_keys is not None else []

    def pre_transforms(self):
        transforms = [
            monai.transforms.LoadImaged(
                self.all_keys,
                ensure_channel_first=True,
                allow_missing_keys=self.fill_missing,
            )
        ]
        # "creates" empty images/masks if necessary
        if self.fill_missing is True:
            transforms.append(
                CreateImageAndWeightsd(self.all_keys, [1] + self.crop_size)
            )
        # sets orientation
        transforms.append(
            monai.transforms.Orientationd(
                self.all_keys,
                "RAS",
                labels=(("L", "R"), ("P", "A"), ("I", "S")),
            )
        )

        # sets target spacing
        if self.target_spacing is not None:
            transforms.append(
                monai.transforms.Spacingd(
                    keys=self.all_keys,
                    pixdim=self.target_spacing,
                    mode=self.intp_resampling_augmentations,
                )
            )

        # sets intensity transforms for ADC and other sequence types
        if self.non_adc_keys:
            transforms.append(
                monai.transforms.ScaleIntensityd(
                    self.non_adc_keys, minv=0.0, maxv=1.0
                )
            )
        if self.adc_keys:
            transforms.append(ConditionalRescalingd(self.adc_keys, 500, 0.001))
            transforms.append(
                monai.transforms.ScaleIntensityd(
                    self.adc_keys, None, None, ADC_FACTOR
                ),
            )
        # sets resize
        if self.resize_size is not None and self.resize_keys:
            intp_ = [
                k
                for k, kk in zip(self.intp, self.all_keys)
                if kk in self.resize_keys
            ]
            transforms.append(
                monai.transforms.Resized(
                    self.resize_keys, tuple(self.resize_size), mode=intp_
                )
            )
        # sets pad op
        if self.pad_size is not None:
            transforms.append(
                monai.transforms.SpatialPadd(self.all_keys, self.pad_size)
            )
        # sets crop op
        if self.crop_size is not None:
            transforms.append(
                monai.transforms.CenterSpatialCropd(
                    self.all_keys, self.crop_size
                )
            )
        transforms.append(
            monai.transforms.EnsureTyped(self.all_keys, dtype=torch.float32)
        )
        if self.label_keys is not None:
            transforms.extend(
                [
                    CombineBinaryLabelsd(self.label_keys, "any", "mask"),
                    LabelOperatorSegmentationd(
                        ["mask"],
                        self.possible_labels,
                        mode=self.label_mode,
                        positive_labels=self.positive_labels,
                    ),
                ]
            )
        # sets indices for random crop op
        if self.random_crop_size is not None:
            if self.label_keys is not None:
                transforms.append(
                    AdjustSizesd([*self.image_keys, "mask"], mode="crop")
                )
                transforms.append(monai.transforms.FgBgToIndicesd("mask"))
            else:
                transforms.append(AdjustSizesd(self.image_keys, mode="crop"))
        return transforms

    def post_transforms(self):
        transforms = []

        if not self.brunet:
            transforms.append(
                monai.transforms.ConcatItemsd(
                    self.image_keys, self.output_image_key
                )
            )

        if self.all_aux_keys:
            transforms.append(
                monai.transforms.ConcatItemsd(
                    self.all_aux_keys, self.aux_key_net
                )
            )

        if self.feature_keys:
            transforms.extend(
                [
                    monai.transforms.EnsureTyped(
                        self.feature_keys, dtype=np.float32
                    ),
                    monai.transforms.Lambdad(
                        self.feature_keys, func=lambda x: np.reshape(x, [1])
                    ),
                    monai.transforms.ConcatItemsd(
                        self.feature_keys, self.feature_key_net
                    ),
                ]
            )

        if self.convert_to_tensor is True:
            if self.brunet is False:
                transforms.append(
                    monai.transforms.ToTensord(
                        [self.output_image_key] + self.mask_key,
                        track_meta=self.track_meta,
                        dtype=torch.float32,
                    )
                )
            else:
                transforms.append(
                    monai.transforms.ToTensord(
                        self.image_keys + self.mask_key,
                        track_meta=self.track_meta,
                        dtype=torch.float32,
                    )
                )

        if not self.track_meta:
            transforms.append(
                monai.transforms.SelectItemsd(
                    self.transform_keys + self.mask_key
                )
            )

        return transforms


@dataclass
class DetectionTransforms(TransformMixin):
    keys: tuple[str]
    adc_keys: tuple[str]
    pad_size: tuple[int]
    crop_size: tuple[int]
    box_class_key: str
    shape_key: str
    box_key: str
    mask_key: str
    anchor_array: np.ndarray | None = None
    input_size: tuple[int] | None = None
    output_size: tuple[int] | None = None
    iou_threshold: float | None = 0.5
    mask_mode: str = "mask_is_labels"
    target_spacing: tuple[float] = None
    predict: bool = False

    def __post_init__(self):
        self.non_adc_keys = [k for k in self.keys if k not in self.adc_keys]
        self.image_keys = (
            self.keys + [self.mask_key]
            if self.mask_key is not None
            else self.keys
        )
        self.spacing_mode = [
            "bilinear" if k != self.mask_key else "nearest"
            for k in self.image_keys
        ]

    def pre_transforms(self):
        transforms = [
            monai.transforms.LoadImaged(
                self.image_keys, ensure_channel_first=True
            ),
            monai.transforms.Orientationd(
                self.image_keys,
                "RAS",
                labels=(("L", "R"), ("P", "A"), ("I", "S")),
            ),
        ]
        if self.target_spacing is not None:
            transforms.append(
                monai.transforms.Spacingd(
                    self.image_keys, self.target_spacing, mode=self.spacing_mode
                )
            )
        if self.non_adc_keys:
            transforms.append(
                monai.transforms.ScaleIntensityd(
                    self.non_adc_keys, minv=0.0, maxv=1.0
                )
            )
        if self.adc_keys:
            transforms.append(ConditionalRescalingd(self.adc_keys, 500, 0.001))
            transforms.append(Offsetd(self.adc_keys, None))
            transforms.append(
                monai.transforms.ScaleIntensityd(
                    self.adc_keys, None, None, ADC_FACTOR
                )
            )
        transforms.extend(
            [
                monai.transforms.SpatialPadd(self.keys, self.pad_size),
                monai.transforms.CenterSpatialCropd(self.keys, self.crop_size),
            ]
        )
        if self.mask_key is not None:
            transforms.append(
                MasksToBBd(
                    keys=[self.mask_key],
                    bounding_box_key=self.box_key,
                    classes_key=self.box_class_key,
                    shape_key=self.shape_key,
                    mask_mode=self.mask_mode,
                )
            )
        return transforms

    def post_transforms(self):
        transforms = []
        if self.predict is False:
            transforms.append(
                BBToAdjustedAnchorsd(
                    anchor_sizes=self.anchor_array,
                    input_sh=self.input_size,
                    output_sh=self.output_size,
                    iou_thresh=self.iou_threshold,
                    bb_key=self.box_key,
                    class_key=self.box_class_key,
                    shape_key=self.shape_key,
                    output_key="bb_map",
                )
            )
        transforms.append(monai.transforms.ConcatItemsd(self.keys, "image"))
        transforms.append(monai.transforms.EnsureTyped(self.keys))
        return transforms


@dataclass
class ClassificationTransforms(TransformMixin):
    keys: tuple[str]
    adc_keys: tuple[str]
    clinical_feature_keys: tuple[str]
    target_spacing: tuple[float]
    crop_size: tuple[int]
    pad_size: tuple[int]
    image_masking: bool = False
    image_crop_from_mask: bool = False
    mask_key: str = None
    branched: bool = False
    possible_labels: tuple[int] = None
    positive_labels: tuple[int] = None
    label_groups: tuple[int | tuple[int]] = None
    label_key: str = None
    target_size: tuple[int] = None
    label_mode: str = None
    cat_confounder_keys: tuple[str] = None
    cont_confounder_keys: tuple[str] = None

    def __post_init__(self):
        self.non_adc_keys = [k for k in self.keys if k not in self.adc_keys]
        self.all_keys = [k for k in self.keys]
        if self.mask_key is not None:
            self.all_keys.append(self.mask_key)
        self.interpolation = [
            "bilinear" if k != self.mask_key else "nearest"
            for k in self.all_keys
        ]
        if isinstance(self.positive_labels, int):
            self.positive_labels = [self.positive_labels]
        self.crop_size_with_margin = (
            [int(j) + 16 for j in self.crop_size]
            if self.crop_size is not None
            else None
        )
        self.crop_size_final = (
            [int(j) for j in self.crop_size]
            if self.crop_size is not None
            else None
        )

    def pre_transforms(self):
        transforms = [
            monai.transforms.LoadImaged(
                self.all_keys, ensure_channel_first=True
            )
        ]
        if self.mask_key is not None:
            transforms.append(
                monai.transforms.ResampleToMatchd(
                    self.mask_key, key_dst=self.keys[0], mode="nearest"
                )
            )
        transforms.append(
            monai.transforms.Orientationd(
                self.all_keys,
                "RAS",
                labels=(("L", "R"), ("P", "A"), ("I", "S")),
            )
        )

        if len(self.non_adc_keys) > 0:
            transforms.append(
                monai.transforms.ScaleIntensityd(
                    self.non_adc_keys, minv=0.0, maxv=1.0
                )
            )
        if len(self.adc_keys) > 0:
            transforms.append(ConditionalRescalingd(self.adc_keys, 500, 0.001))
            transforms.append(Offsetd(self.adc_keys, None))
            transforms.append(
                monai.transforms.ScaleIntensityd(
                    self.adc_keys, None, None, ADC_FACTOR
                )
            )
        if self.target_spacing is not None:
            transforms.extend(
                [
                    monai.transforms.Spacingd(
                        self.all_keys,
                        pixdim=self.target_spacing,
                        dtype=torch.float32,
                        mode=self.interpolation,
                    ),
                ]
            )
        if self.target_size is not None:
            transforms.append(
                monai.transforms.Resized(
                    keys=self.all_keys,
                    spatial_size=self.target_size,
                    mode=self.interpolation,
                )
            )
        if self.pad_size is not None:
            transforms.append(
                monai.transforms.SpatialPadd(
                    self.all_keys, self.crop_size_with_margin
                )
            )
        # initial crop with margin allows for rotation transforms to not create
        # black pixels around the image (these transforms do not need to applied
        # to the whole image)
        if self.image_masking is True:
            transforms.append(ConvexHulld([self.mask_key]))
        if self.image_crop_from_mask is True:
            transforms.append(
                CropFromMaskd(
                    self.all_keys,
                    mask_key=self.mask_key,
                    output_size=self.crop_size_with_margin,
                )
            )
        elif self.crop_size is not None:
            transforms.append(
                monai.transforms.CenterSpatialCropd(
                    self.all_keys, self.crop_size_with_margin
                )
            )
        transforms.append(monai.transforms.EnsureTyped(self.all_keys))

        return transforms

    def post_transforms(self):
        transforms = []
        if self.crop_size is not None:
            transforms.append(
                monai.transforms.CenterSpatialCropd(
                    self.all_keys, self.crop_size_final
                )
            )
        if self.image_masking is True:
            transforms.append(
                monai.transforms.MaskIntensityd(
                    self.keys, mask_key=self.mask_key
                )
            )
        if self.branched is not True:
            transforms.append(
                monai.transforms.ConcatItemsd(self.all_keys, "image")
            )
        if len(self.clinical_feature_keys) > 0:
            transforms.extend(
                [
                    monai.transforms.EnsureTyped(
                        self.clinical_feature_keys, dtype=np.float32
                    ),
                    monai.transforms.Lambdad(
                        self.clinical_feature_keys,
                        func=lambda x: np.reshape(x, [1]),
                    ),
                    monai.transforms.ConcatItemsd(
                        self.clinical_feature_keys, "tabular"
                    ),
                ]
            )
        if self.label_key is not None:
            transforms.append(
                LabelOperatord(
                    [self.label_key],
                    possible_labels=self.possible_labels,
                    mode=self.label_mode,
                    positive_labels=self.positive_labels,
                    label_groups=self.label_groups,
                    output_keys={self.label_key: "label"},
                )
            )
        if self.cat_confounder_keys is not None:
            transforms.append(
                monai.transforms.Lambdad(
                    self.cat_confounder_keys, box, track_meta=False
                )
            )
            transforms.append(
                monai.transforms.ConcatItemsd(
                    self.cat_confounder_keys, "cat_confounder"
                )
            )
        if self.cont_confounder_keys is not None:
            transforms.append(
                monai.transforms.Lambdad(self.cont_confounder_keys, box)
            )
            transforms.append(
                monai.transforms.ConcatItemsd(
                    self.cont_confounder_keys, "cont_confounder"
                )
            )

        return transforms


@dataclass
class GenerationTransforms(TransformMixin):
    keys: tuple[str]
    target_spacing: tuple[int] | None = None
    crop_size: tuple[int] | None = None
    pad_size: tuple[int] | None = None
    n_dim: int = 3
    cat_keys: tuple[str] | None = None
    num_keys: tuple[str] | None = None

    def pre_transforms(self):
        transforms = [
            monai.transforms.LoadImaged(
                self.keys, ensure_channel_first=True, image_only=True
            )
        ]
        if self.n_dim == 2:
            transforms.extend(
                [
                    SampleChannelDimd(self.keys, 1, 3),
                    monai.transforms.SqueezeDimd(
                        self.keys, -1, update_meta=False
                    ),
                ]
            )
        if self.n_dim == 3:
            transforms.append(
                monai.transforms.Orientationd(
                    self.keys,
                    "RAS",
                    labels=(("L", "R"), ("P", "A"), ("I", "S")),
                )
            )

        if self.target_spacing is not None:
            transforms.extend(
                [
                    monai.transforms.Spacingd(
                        self.keys,
                        pixdim=self.target_spacing,
                        dtype=torch.float32,
                    ),
                ]
            )
        transforms.append(
            monai.transforms.ScaleIntensityd(self.keys, minv=0.0, maxv=1.0)
        )
        if self.pad_size is not None:
            transforms.append(
                monai.transforms.SpatialPadd(
                    self.keys, [int(j) for j in self.pad_size]
                )
            )
        if self.crop_size is not None:
            transforms.append(
                monai.transforms.CenterSpatialCropd(
                    self.keys, [int(j) + 16 for j in self.crop_size]
                )
            )
        transforms.append(monai.transforms.EnsureTyped(self.keys))
        return transforms

    def post_transforms(self):
        transforms = []
        transforms.append(monai.transforms.ConcatItemsd(self.keys, "image"))
        if self.crop_size is not None:
            transforms.append(
                monai.transforms.CenterSpatialCropd(
                    "image", [int(j) for j in self.crop_size]
                )
            )
        if self.cat_keys is not None:
            transforms.append(
                monai.transforms.Lambdad(self.cat_keys, box, track_meta=False)
            )
            transforms.append(
                monai.transforms.ConcatItemsd(self.cat_keys, "cat")
            )
        if self.num_keys is not None:
            transforms.append(monai.transforms.Lambdad(self.num_keys, box))
            transforms.append(
                monai.transforms.ConcatItemsd(self.num_keys, "num")
            )
        return transforms


@dataclass
class SSLTransforms(TransformMixin):
    all_keys: tuple[str]
    copied_keys: tuple[str]
    adc_keys: tuple[str]
    non_adc_keys: tuple[str]
    target_spacing: tuple[float]
    crop_size: tuple[int] = None
    pad_size: tuple[int] = None
    resize_size: tuple[int] = None
    in_channels: int = 1
    n_dim: int = 3
    skip_augmentations: bool = False
    jpeg_dataset: bool = False

    def __post_init__(self):
        self.output_keys = (
            ["image"]
            if self.skip_augmentations
            else ["augmented_image_1", "augmented_image_2"]
        )
        self.concat_keys = (
            [self.all_keys]
            if self.skip_augmentations
            else [self.all_keys, self.copied_keys]
        )

    def pre_transforms(self):
        intp = []
        intp_resampling_augmentations = []
        for k in self.all_keys:
            intp.append("area")
            intp_resampling_augmentations.append("bilinear")

        transforms = [
            monai.transforms.LoadImaged(
                self.all_keys, ensure_channel_first=True, image_only=True
            ),
            SampleChannelDimd(self.all_keys, self.in_channels),
        ]
        if self.jpeg_dataset is False and self.n_dim == 2:
            transforms.extend(
                [
                    SampleChannelDimd(self.all_keys, 1, 3),
                    monai.transforms.SqueezeDimd(
                        self.all_keys, -1, update_meta=False
                    ),
                ]
            )
        if self.n_dim == 3:
            transforms.append(
                monai.transforms.Orientationd(
                    self.all_keys,
                    "RAS",
                    labels=(("L", "R"), ("P", "A"), ("I", "S")),
                )
            )
        if self.target_spacing is not None:
            intp_resampling_augmentations = ["bilinear" for _ in self.all_keys]
            transforms.append(
                monai.transforms.Spacingd(
                    keys=self.all_keys,
                    pixdim=self.target_spacing,
                    mode=intp_resampling_augmentations,
                )
            )
        if len(self.non_adc_keys) > 0:
            transforms.append(
                monai.transforms.ScaleIntensityd(
                    self.non_adc_keys, minv=0.0, maxv=1.0
                )
            )
        if len(self.adc_keys) > 0:
            transforms.extend(
                [
                    ConditionalRescalingd(self.adc_keys, 500, 0.001),
                    monai.transforms.ScaleIntensityd(
                        self.adc_keys, None, None, ADC_FACTOR
                    ),
                ]
            )
        if self.crop_size is not None:
            transforms.append(
                monai.transforms.CenterSpatialCropd(
                    self.all_keys, [int(j) for j in self.crop_size]
                )
            )
        if self.pad_size is not None:
            transforms.append(
                monai.transforms.SpatialPadd(
                    self.all_keys, [int(j) for j in self.pad_size]
                )
            )
        if self.resize_size is not None:
            transforms.append(
                monai.transforms.Resized(
                    self.all_keys, [int(j) for j in self.resize_size]
                )
            )
        transforms.append(monai.transforms.EnsureTyped(self.all_keys))
        if self.skip_augmentations is False:
            key_correspondence = {
                k: kk for k, kk in zip(self.all_keys, self.copied_keys)
            }
            transforms.append(CopyEntryd(self.all_keys, key_correspondence))
        return transforms

    def post_transforms(self):
        transforms = []
        for keys, output_key in zip(self.concat_keys, self.output_keys):
            transforms.append(monai.transforms.ConcatItemsd(keys, output_key))
        transforms.append(
            monai.transforms.ToTensord(self.output_keys, track_meta=False)
        )
        return transforms
