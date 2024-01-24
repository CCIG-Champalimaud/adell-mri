from typing import Dict, Any, List
from itertools import product


def convert_arguments_pre(
    transform_arguments: Dict[str, Any], image_keys: List[str]
):
    """
    Converts the transform arguments dictionary for use in semi-supervised learning.
    Used internally by get_semi_sl_transforms.

    This takes the original transform arguments dict and modifies it for use in
    semi-supervised learning, where unlabeled images will be augmented. It removes
    the label keys, sets new image keys for the augmented images, and truncates some
    augmentation parameters to match the number of input image keys.

    Args:
        transform_arguments: The original transform arguments dict.
        image_keys: A list of the image keys that transformations will be applied to.

    Returns:
        A new dict with modified arguments for semi-supervised learning.
    """
    transform_arguments_semi_sl = {
        k: transform_arguments[k] for k in transform_arguments
    }
    transform_arguments_semi_sl["label_keys"] = None
    transform_arguments_semi_sl["all_keys"] = image_keys
    transform_arguments_semi_sl["image_keys"] = image_keys
    transform_arguments_semi_sl["intp"] = transform_arguments_semi_sl["intp"][
        : len(image_keys)
    ]
    transform_arguments_semi_sl[
        "intp_resampling_augmentations"
    ] = transform_arguments_semi_sl["intp_resampling_augmentations"][
        : len(image_keys)
    ]
    return transform_arguments_semi_sl


def convert_arguments_post(
    transform_arguments: Dict[str, Any], idx: int, image_keys: List[str]
):
    """
    Converts the transform arguments dictionary for use in semi-supervised learning.
    Used internally by get_semi_sl_transforms.

    This takes the original transform arguments dict and modifies it
    for use in semi-supervised learning, where unlabeled images will be
    augmented. It removes the label keys, sets a new output image key,
    tracks metadata, and sets new image keys for the augmented images.

    Args:
        transform_arguments: The original transform arguments dict.
        idx: The index of the augmentation pass.
        image_keys: A list of the image keys that transformations will be applied to.

    Returns:
        A new dict with modified arguments for semi-supervised learning.
    """

    transform_arguments_semi_sl = {
        k: transform_arguments[k] for k in transform_arguments
    }
    transform_arguments_semi_sl["label_keys"] = None
    transform_arguments_semi_sl["all_keys"] = image_keys
    transform_arguments_semi_sl["track_meta"] = True
    transform_arguments_semi_sl["output_image_key"] = f"semi_sl_image_{idx}"
    transform_arguments_semi_sl["image_keys"] = [
        f"{k}_aug_{idx}" for k in image_keys
    ]
    return transform_arguments_semi_sl


def convert_arguments_augment_all(
    augment_arguments: Dict[str, Any], image_keys: List[str]
):
    """
    Converts the augmentation transform arguments dictionary for use in semi-
    supervised learning. Used internally by get_semi_sl_transforms.

    This takes the original transform arguments dict and modifies it
    for use in semi-supervised learning, where unlabeled images will be
    augmented. It removes the label keys, sets a new output image key,
    tracks metadata, and sets new image keys for the augmented images.

    Args:
        transform_arguments: The original transform arguments dict.
        idx: The index of the augmentation pass.
        image_keys: A list of the image keys that transformations will be applied to.

    Returns:
        A new dict with modified arguments for semi-supervised learning.
    """
    augment_arguments_semi_sl = {
        k: augment_arguments[k] for k in augment_arguments
    }
    augment_arguments_semi_sl["augment"] = [
        "affine",
        "shear",
        "flip",
    ]
    augment_arguments_semi_sl["all_keys"] = [
        f"{k}_aug_{idx}" for k, idx in product(image_keys, [1, 2])
    ]
    augment_arguments_semi_sl["image_keys"] = [
        f"{k}_aug_{idx}" for k, idx in product(image_keys, [1, 2])
    ]
    augment_arguments_semi_sl["has_label"] = False
    return augment_arguments_semi_sl


def convert_arguments_augment_individual(
    augment_arguments: Dict[str, Any], idx: int, image_keys: List[str]
):
    """
    Converts the augmentation transform arguments dictionary for use in semi-
    supervised learning. Used internally by get_semi_sl_transforms.

    This takes the original transform arguments dict and modifies it
    for use in semi-supervised learning, where unlabeled images will be
    augmented. It removes the label keys, sets a new output image key,
    tracks metadata, and sets new image keys for the augmented images.

    Args:
        transform_arguments (dict[str, Any]): The original transform arguments
            dict.
        idx (int): The index of the augmentation pass.
        image_keys [list[str]]: A list of the image keys that transformations
            will be applied to.

    Returns:
        A new dict with modified arguments for semi-supervised learning.
    """
    augment_arguments_semi_sl = {
        k: augment_arguments[k] for k in augment_arguments
    }
    augment_arguments_semi_sl["augment"] = [
        "intensity",
        "noise",
        "rbf",
        "blur",
        "trivial",
    ]
    augment_arguments_semi_sl["all_keys"] = [
        f"{k}_aug_{idx}" for k in image_keys
    ]
    augment_arguments_semi_sl["image_keys"] = [
        f"{k}_aug_{idx}" for k in image_keys
    ]
    augment_arguments_semi_sl["has_label"] = False
    return augment_arguments_semi_sl
