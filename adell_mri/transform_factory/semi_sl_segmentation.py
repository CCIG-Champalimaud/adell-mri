from typing import Any

import monai
import monai.transforms

from ..modules.semi_supervised_segmentation.utils import (
    convert_arguments_augment_all,
    convert_arguments_augment_individual,
    convert_arguments_post,
    convert_arguments_pre,
)
from ..utils.monai_transforms import CopyEntryd
from .augmentations import get_augmentations_unet
from .transforms import SegmentationTransforms


def get_semi_sl_transforms(
    transform_arguments: dict[str, Any],
    augment_arguments: dict[str, Any],
    keys: list[str],
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
        *SegmentationTransforms(
            **transform_arguments_semi_sl_pre
        ).pre_transforms(),
        CopyEntryd(keys, {k: f"{k}_aug_1" for k in keys}),
        CopyEntryd(keys, {k: f"{k}_aug_2" for k in keys}),
        get_augmentations_unet(**augment_arguments_semi_sl_all),
        get_augmentations_unet(**augment_arguments_semi_sl_1),
        get_augmentations_unet(**augment_arguments_semi_sl_2),
        *SegmentationTransforms(
            **transform_arguments_semi_sl_post_1
        ).post_transforms(),
        *SegmentationTransforms(
            **transform_arguments_semi_sl_post_2
        ).post_transforms(),
        monai.transforms.SelectItemsd(["semi_sl_image_1", "semi_sl_image_2"]),
    ]
    return monai.transforms.Compose(transforms_semi_sl)
