import itertools

import monai
import monai.transforms
import numpy as np

from adell_mri.modules.augmentations import (
    AugmentationWorkhorsed,
    generic_augments,
    mri_specific_augments,
    spatial_augments,
)
from adell_mri.utils.monai_transforms import (
    ExposeTransformKeyMetad,
    RandRotateWithBoxesd,
)


def get_augmentations(
    augment: list[str],
    image_keys: list[str],
    label_keys: list[str] | None = None,
    t2_keys: list[str] | None = None,
    random_crop_size: list[int] | None = None,
    has_label: bool = True,
    n_crops: int = 1,
    flip_axis: list[int] | tuple[int, ...] = (0, 1),
    prob: float = 0.2,
    n_transforms_trivial: int = 1,
    data_range: tuple[float, float] | None = None,
):
    """Builds a task-agnostic set of on-the-fly data augmentations.

    The augmentations selected via ``augment`` are applied either
    independently (one random transform per call) or jointly (a random
    sub-set of transforms), depending on whether ``"trivial"`` is present.
    Intensity-based transforms only touch ``image_keys`` while spatial
    transforms are shared across ``image_keys`` and ``label_keys`` (labels
    are interpolated with nearest-neighbour sampling). An optional random
    crop around positive/negative label samples is appended when
    ``random_crop_size`` is provided and, finally, an unconditional
    intensity clamp to ``data_range`` can be added to guarantee the values
    stay within the expected interval.

    Args:
        augment (list[str]): Sub-set of augmentations to enable from
            ``["intensity", "noise", "rbf", "affine", "shear", "flip",
            "blur", "distort", "lowres", "trivial"]``.
        image_keys (list[str]): Keys of the images to augment. These are
            interpolated with bilinear sampling and receive the
            intensity-based augmentations.
        label_keys (list[str], optional): Keys of label/auxiliary volumes to
            spatially augment alongside the images. These are interpolated
            with nearest-neighbour sampling and are never intensity
            augmented. Defaults to None.
        t2_keys (list[str], optional): Keys on which to apply the random
            bias field. Only used when ``"rbf"`` is in ``augment``. Defaults
            to None.
        random_crop_size (list[int], optional): If provided, a random crop
            centred around positive/negative label samples is applied before
            the augmentations, followed by a central crop to this size.
            Defaults to None.
        has_label (bool): Whether ``label_keys`` contain a foreground mask to
            guide the random crop. Only used when ``random_crop_size`` is
            provided. Defaults to True.
        n_crops (int): Number of random crops to sample per volume. Only used
            when ``random_crop_size`` is provided. Defaults to 1.
        flip_axis (list[int], optional): Axes that may be flipped when
            ``"flip"`` is in ``augment``. Defaults to ``(0, 1)``.
        prob (float): Probability of applying each individual augmentation.
            Forced to 1.0 when ``"trivial"`` is in ``augment``. Defaults to
            0.2.
        n_transforms_trivial (int): Number of transforms to randomly compose
            when ``"trivial"`` is in ``augment``. Defaults to 1.
        data_range (tuple[float, float], optional): When provided, the
            resulting images are clamped into ``(min, max)`` so the data is
            guaranteed to be within this interval. Defaults to None.

    Returns:
        monai.transforms.Compose: The composition of the requested
        augmentations.

    Raises:
        NotImplementedError: If ``augment`` contains an unknown augmentation.
        ValueError: If ``random_crop_size`` is provided together with
            ``has_label`` but no ``label_keys`` were supplied.
    """
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
    for a in augment:
        if a not in valid_arg_list:
            raise NotImplementedError(
                f"augment can only contain {valid_arg_list}"
            )
    if random_crop_size is not None and has_label is True and not label_keys:
        raise ValueError(
            "random_crop_size requires label_keys when has_label is True"
        )
    label_keys = [] if label_keys is None else label_keys
    t2_keys = [] if t2_keys is None else t2_keys
    if isinstance(flip_axis, int):
        flip_axis = [flip_axis]
    flip_axis = list(flip_axis)
    spatial_keys = [*image_keys, *label_keys]
    interpolation = [
        "bilinear" if k in image_keys else "nearest" for k in spatial_keys
    ]
    augments = []

    if "trivial" in augment:
        augments.append(monai.transforms.Identityd(image_keys))
        prob = 1.0

    if "rbf" in augment and len(t2_keys) > 0:
        augments.append(
            monai.transforms.RandBiasFieldd(t2_keys, degree=3, prob=prob)
        )

    if "affine" in augment or "shear" in augment:
        kwargs = {}
        if "affine" in augment:
            kwargs["translate_range"] = [8, 8, 2]
            kwargs["rotate_range"] = [np.pi / 16, np.pi / 16, 0]
            kwargs["scale_range"] = [0.1]
        if "shear" in augment:
            kwargs["shear_range"] = (0.1 for _ in range(3))
        augments.append(
            monai.transforms.RandAffined(
                spatial_keys,
                **kwargs,
                prob=prob,
                mode=interpolation,
                padding_mode="zeros",
            )
        )

    if "distort" in augment:
        augments.append(
            monai.transforms.RandGridDistortiond(
                spatial_keys,
                distort_limit=0.2,
                prob=prob,
                mode=interpolation,
                padding_mode="zeros",
            )
        )

    if "flip" in augment:
        flips = []
        for i in range(len(flip_axis)):
            axes_to_flip = itertools.combinations(flip_axis, i + 1)
            for axis_to_flip in axes_to_flip:
                flips.append(
                    monai.transforms.RandFlipd(
                        spatial_keys,
                        prob=prob,
                        spatial_axis=axis_to_flip,
                    )
                )
        augments.append(monai.transforms.OneOf(flips))

    if "intensity" in augment:
        augments.extend(
            [
                monai.transforms.RandAdjustContrastd(
                    image_keys, gamma=(0.5, 1.5), prob=prob
                ),
                monai.transforms.RandScaleIntensityd(
                    image_keys, factors=0.1, prob=prob
                ),
            ]
        )

    if "blur" in augment:
        augments.append(
            monai.transforms.RandGaussianSmoothd(
                image_keys,
                prob=prob,
                sigma_x=(0.5, 1.5),
                sigma_y=(0.5, 1.5),
                sigma_z=(0.5, 1.5),
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

    if "noise" in augment:
        augments.extend(
            [
                monai.transforms.RandRicianNoised(
                    image_keys, std=0.05, prob=prob
                ),
                monai.transforms.RandGibbsNoised(
                    image_keys, alpha=(0.5, 0.7), prob=prob
                ),
            ]
        )

    if "trivial" in augment:
        transform = monai.transforms.SomeOf(
            augments, num_transforms=n_transforms_trivial
        )
    else:
        transform = monai.transforms.Compose(augments)

    if random_crop_size is not None:
        # do a first larger crop that prevents artefacts introduced by
        # affine transforms and then crop the rest
        pre_final_size = [int(i * 1.10) for i in random_crop_size]
        new_augments = []
        if has_label is True:
            new_augments.append(
                monai.transforms.RandCropByPosNegLabeld(
                    spatial_keys,
                    label_keys[0],
                    pre_final_size,
                    allow_smaller=True,
                    num_samples=n_crops,
                    fg_indices_key=f"{label_keys[0]}_fg_indices",
                    bg_indices_key=f"{label_keys[0]}_bg_indices",
                )
            )
        else:
            new_augments.append(
                monai.transforms.RandSpatialCropd(
                    spatial_keys,
                    pre_final_size,
                )
            )
        transform = monai.transforms.Compose(
            [
                *new_augments,
                transform,
                monai.transforms.CenterSpatialCropd(
                    spatial_keys, random_crop_size
                ),
            ]
        )

    if data_range is not None:
        transform = monai.transforms.Compose(
            [
                transform,
                monai.transforms.ScaleIntensityRanged(
                    keys=image_keys,
                    a_min=data_range[0],
                    a_max=data_range[1],
                    b_min=data_range[0],
                    b_max=data_range[1],
                    clip=True,
                ),
            ]
        )
    return transform


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
                f"augment can only contain {valid_arg_list}"
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
