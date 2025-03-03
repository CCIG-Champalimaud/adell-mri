import itertools

import monai
import monai.transforms
import numpy as np

from ..modules.augmentations import (
    AugmentationWorkhorsed,
    generic_augments,
    mri_specific_augments,
    spatial_augments,
)
from ..utils.monai_transforms import (
    ExposeTransformKeyMetad,
    RandRotateWithBoxesd,
)


def get_augmentations_unet(
    augment,
    all_keys,
    image_keys,
    t2_keys,
    random_crop_size: list[int] = None,
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
            monai.transforms.RandFlipd(all_keys, spatial_axis=[axis], prob=0.25)
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
    augment: list[str],
    image_keys: list[str],
    mask_key: str,
    t2_keys: list[str],
    flip_axis: list[int] = [0, 1],
    prob: float = 0.1,
    n_transforms_trivial: int = 1,
):
    valid_arg_list = [
        "intensity",
        "noise",
        "rbf",
        "affine",
        "shear",
        "flip",
        "blur",
        "lowres",
        "distort",
        "trivial",
    ]
    all_keys_with_mask = [k for k in image_keys]
    if mask_key is not None:
        all_keys_with_mask.append(mask_key)
    intp = [
        "bilinear" if k != mask_key else "nearest" for k in all_keys_with_mask
    ]
    for a in augment:
        if a not in valid_arg_list:
            raise NotImplementedError(
                "augment can only contain {}".format(valid_arg_list)
            )
    augments = []

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
            ]
        )

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

    if "lowres" in augment:
        augments.append(
            monai.transforms.RandSimulateLowResolutiond(
                image_keys,
                zoom_range=[0.8, 1.2],
                prob=prob,
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
                monai.transforms.RandShiftIntensityd(
                    image_keys, offsets=0.1, prob=prob
                ),
            ]
        )

    if "flip" in augment:
        if isinstance(flip_axis, int):
            flip_axis = [flip_axis]
        flips = []
        for i in range(len(flip_axis)):
            axes_to_flip = itertools.combinations(flip_axis, i + 1)
            for axis_to_flip in axes_to_flip:
                flips.append(
                    monai.transforms.RandFlipd(
                        all_keys_with_mask, prob=prob, spatial_axis=axis_to_flip
                    )
                )
        augments.append(monai.transforms.OneOf(flips))

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
                scale_range=[0.1, 0.1, 0.05],
                prob=prob,
                mode=intp,
                padding_mode="zeros",
            )
        )

    if "shear" in augment:
        augments.append(
            monai.transforms.RandAffined(
                all_keys_with_mask,
                shear_range=((0.9, 1.1), (0.9, 1.1), (0.9, 1.1)),
                prob=prob,
                mode=intp,
                padding_mode="zeros",
            )
        )

    if "distort" in augment:
        augments.append(
            monai.transforms.RandGridDistortiond(
                all_keys_with_mask,
                distort_limit=0.05,
                prob=prob,
                mode=intp,
                padding_mode="zeros",
            )
        )

    if "trivial" in augment:
        augments = monai.transforms.SomeOf(
            augments, num_transforms=n_transforms_trivial
        )
    else:
        augments = monai.transforms.Compose(augments)
    return augments


def get_augmentations_detection(augment, image_keys, box_keys, t2_keys):
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
            [monai.transforms.RandRicianNoised(image_keys, std=0.02, prob=prob)]
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
    all_keys: list[str],
    copied_keys: list[str],
    scaled_crop_size: list[int],
    roi_size: list[int],
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
        scaled_crop_size = tuple([int(x) for x in scaled_crop_size])
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
    ]
    if len(copied_keys) > 0:
        transforms.append(
            AugmentationWorkhorsed(
                augmentations=aug_list,
                keys=copied_keys,
                mask_keys=[],
                max_mult=0.5,
                N=n_transforms,
                dropout_size=dropout_size,
            )
        )
    return transforms
