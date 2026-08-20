import os
import sys

sys.path.append(os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))

from adell_mri.modules.semi_supervised_segmentation.utils import (
    convert_arguments_augment_all,
    convert_arguments_augment_individual,
    convert_arguments_post,
    convert_arguments_pre,
)

TRANSFORM_ARGUMENTS = {
    "label_keys": ["mask"],
    "all_keys": ["image", "mask"],
    "image_keys": ["image"],
    "non_adc_keys": ["image"],
    "adc_keys": [],
    "target_spacing": 1.0,
    "intp": ["area", "nearest"],
    "intp_resampling_augmentations": ["bilinear", "nearest"],
    "possible_labels": 2,
    "positive_labels": None,
    "all_aux_keys": [],
    "resize_keys": [],
    "feature_keys": [],
    "aux_key_net": None,
    "feature_key_net": None,
    "resize_size": None,
    "pad_size": None,
    "crop_size": None,
    "random_crop_size": None,
    "label_mode": "binary",
    "fill_missing": False,
    "brunet": False,
}

AUGMENT_ARGUMENTS = {
    "augment": ["affine"],
    "all_keys": ["image", "mask"],
    "image_keys": ["image"],
    "t2_keys": [],
    "random_crop_size": None,
    "n_crops": 1,
    "flip_axis": [0, 1, 2],
}


def test_convert_arguments_pre():
    out = convert_arguments_pre(TRANSFORM_ARGUMENTS, ["image"])
    assert out["label_keys"] is None
    assert out["image_keys"] == ["image"]
    assert out["all_keys"] == ["image"]
    assert out["intp"] == ["area"]
    assert out["intp_resampling_augmentations"] == ["bilinear"]
    assert TRANSFORM_ARGUMENTS["label_keys"] == ["mask"]
    assert TRANSFORM_ARGUMENTS["intp"] == ["area", "nearest"]


def test_convert_arguments_post():
    out = convert_arguments_post(TRANSFORM_ARGUMENTS, 2, ["image"])
    assert out["label_keys"] is None
    assert out["image_keys"] == ["image_aug_2"]
    assert out["all_keys"] == ["image"]
    assert out["output_image_key"] == "semi_sl_image_2"
    assert out["track_meta"] is True


def test_convert_arguments_augment_all():
    out = convert_arguments_augment_all(AUGMENT_ARGUMENTS, ["image"])
    assert out["augment"] == ["affine", "shear", "flip"]
    assert out["image_keys"] == ["image_aug_1", "image_aug_2"]
    assert out["all_keys"] == ["image_aug_1", "image_aug_2"]
    assert out["has_label"] is False


def test_convert_arguments_augment_individual():
    out = convert_arguments_augment_individual(AUGMENT_ARGUMENTS, 2, ["image"])
    assert out["augment"] == ["intensity", "noise", "rbf", "blur", "trivial"]
    assert out["image_keys"] == ["image_aug_2"]
    assert out["all_keys"] == ["image_aug_2"]
    assert out["has_label"] is False
