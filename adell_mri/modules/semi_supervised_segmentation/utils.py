from typing import Dict, Any, List
from itertools import product


def convert_arguments_pre(
    transform_arguments: Dict[str, Any], image_keys: List[str]
):
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


def convert_arguments_augment(
    augment_arguments: Dict[str, Any], image_keys: List[str]
):
    augment_arguments_semi_sl = {
        k: augment_arguments[k] for k in augment_arguments
    }
    augment_arguments_semi_sl["augment"] = [
        "intensity",
        "noise",
        "rbf",
        "affine",
        "shear",
        "flip",
        "blur",
        "trivial",
    ]
    augment_arguments_semi_sl["all_keys"] = [
        f"{k}_aug_{idx}" for k, idx in product(image_keys, [1, 2])
    ]
    augment_arguments_semi_sl["image_keys"] = [
        f"{k}_aug_{idx}" for k, idx in product(image_keys, [1, 2])
    ]
    augment_arguments_semi_sl["has_label"] = False
    return augment_arguments_semi_sl
