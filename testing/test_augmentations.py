"""
Tests for the unified get_augmentations transform factory.
"""

import monai.transforms
import pytest

from adell_mri.transform_factory.augmentations import get_augmentations

ALL_AUGMENT = [
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


def _class_names(transform) -> list[str]:
    """
    Recursively collects the class names of ``transform`` and any nested
    sub-transforms (i.e. those held inside a ``Compose``/``SomeOf``/``OneOf``).
    """
    names = [type(transform).__name__]
    for sub in getattr(transform, "transforms", []) or []:
        names.extend(_class_names(sub))
    return names


@pytest.mark.parametrize("augment", ALL_AUGMENT)
def test_each_augment_builds(augment):
    """Every supported augmentation can be selected in isolation."""
    transform = get_augmentations(
        [augment],
        image_keys=["image"],
        label_keys=["mask"],
        t2_keys=["image"],
    )
    assert transform is not None


@pytest.mark.parametrize("augment", ALL_AUGMENT)
def test_each_augment_builds_without_label_keys(augment):
    """Augmentations also build when no label keys are supplied."""
    transform = get_augmentations([augment], image_keys=["image"])
    assert transform is not None


def test_unknown_augment_raises():
    with pytest.raises(NotImplementedError):
        get_augmentations(["bogus"], image_keys=["image"])


def test_random_crop_requires_label_keys():
    with pytest.raises(ValueError):
        get_augmentations(
            ["affine"], image_keys=["image"], random_crop_size=[64, 64, 64]
        )


def test_trivial_augment_returns_someof():
    transform = get_augmentations(["trivial"], image_keys=["image"])
    assert isinstance(transform, monai.transforms.SomeOf)


def test_random_crop_compose():
    transform = get_augmentations(
        ["affine"],
        image_keys=["image"],
        label_keys=["mask"],
        random_crop_size=[64, 64, 64],
    )
    names = _class_names(transform)
    assert "RandCropByPosNegLabeld" in names
    assert "CenterSpatialCropd" in names


def test_random_crop_without_label_uses_spatial_crop():
    transform = get_augmentations(
        ["affine"],
        image_keys=["image"],
        random_crop_size=[64, 64, 64],
        has_label=False,
    )
    names = _class_names(transform)
    assert "RandSpatialCropd" in names
    assert "RandCropByPosNegLabeld" not in names


def test_data_range_appends_clamp():
    transform = get_augmentations(
        ["affine"],
        image_keys=["image"],
        label_keys=["mask"],
        data_range=(0.0, 1.0),
    )
    names = _class_names(transform)
    assert isinstance(transform, monai.transforms.Compose)
    assert "ScaleIntensityRanged" in names


def test_no_data_range_no_clamp():
    transform = get_augmentations(
        ["affine"],
        image_keys=["image"],
        label_keys=["mask"],
        data_range=None,
    )
    names = _class_names(transform)
    assert "ScaleIntensityRanged" not in names


def test_spatial_augment_covers_label_keys():
    """Affine must be applied to the label keys together with the images."""
    transform = get_augmentations(
        ["affine"],
        image_keys=["image"],
        label_keys=["mask"],
        data_range=None,
    )
    names = _class_names(transform)
    affine = [
        t
        for t in _flatten_transforms(transform)
        if isinstance(t, monai.transforms.RandAffined)
    ]
    assert affine
    keys = list(affine[0].keys) if hasattr(affine[0], "keys") else []
    assert "image" in keys
    assert "mask" in keys
    assert "RandAffined" in names


def _flatten_transforms(transform) -> list:
    """Flattens nested compositions into a flat list of transform instances."""
    out = [transform]
    for sub in getattr(transform, "transforms", []) or []:
        out.extend(_flatten_transforms(sub))
    return out
