from typing import Dict, Any


def convert_arguments_post(transform_arguments: Dict[str, Any], idx: int):
    transform_arguments_semi_sl = {
        k: transform_arguments[k]
        for k in transform_arguments
        if k != "label_keys"
    }
    transform_arguments_semi_sl["output_image_key"] = f"semi_sl_image_{idx}"
    transform_arguments_semi_sl["image_keys"] = [
        f"{k}_aug_{idx}" for k in transform_arguments["all_keys"]
    ]
    return transform_arguments_semi_sl


def convert_arguments_augment(augment_arguments: Dict[str, Any], idx: int):
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
    augment_arguments_semi_sl = {
        k: augment_arguments[k] for k in augment_arguments_semi_sl
    }
    augment_arguments_semi_sl["all_keys"] = [
        f"{k}_aug_{idx}" for k in augment_arguments["all_keys"]
    ]
    return augment_arguments_semi_sl
