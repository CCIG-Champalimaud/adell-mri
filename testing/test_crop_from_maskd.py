import os
import sys

sys.path.append(os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))

import monai.transforms
import numpy as np
import torch
from monai.data import MetaTensor

from adell_mri.transform_factory.transforms import ClassificationTransforms
from adell_mri.utils.monai_transforms import CropFromMaskd

IMAGE_KEY = "image"
MASK_KEY = "mask"


def _make_sample(spatial_shape, mask_slices, n_channels=1):
    """
    Helper: creates a uniquely-valued image + binary mask dict.
    """
    n_vox = n_channels * int(torch.tensor(spatial_shape).prod())
    img = MetaTensor(
        torch.arange(n_vox, dtype=torch.float32).reshape(
            n_channels, *spatial_shape
        )
    )
    mask = MetaTensor(torch.zeros(1, *spatial_shape))
    mask[tuple([slice(None)] + list(mask_slices))] = 1.0
    return {IMAGE_KEY: img, MASK_KEY: mask}


def _assert_content_preserved(restored, original, mask_slices):
    """Values inside the mask window and overall shape must match the original."""
    s = (slice(None), *mask_slices)
    assert (
        restored.shape == original.shape
    ), f"Shape mismatch: {restored.shape} != {original.shape}"
    assert torch.allclose(
        restored[s], original[s]
    ), "Values inside the crop window changed after crop + inverse"


def _assert_cropped_content_preserved(cropped, original, crop_slices):
    """Compare cropped region with corresponding region in original.

    Args:
        cropped: The cropped tensor (smaller shape)
        original: The original tensor (larger shape)
        crop_slices: Slices defining the crop region in original
    """
    s = (slice(None), *crop_slices)
    original_region = original[s]
    assert cropped.shape == original_region.shape, (
        f"Shape mismatch: cropped {cropped.shape} != "
        f"original region {original_region.shape}"
    )
    assert torch.allclose(
        cropped, original_region
    ), "Cropped content does not match original region"


def test_3d_mask_driven_content():
    """Mask-driven crop: values inside the bounding box survive round-trip."""
    spatial = (32, 32, 32)
    mask_slices = [slice(3, 13), slice(1, 15), slice(4, 7)]
    sample = _make_sample(spatial, mask_slices)
    orig = sample[IMAGE_KEY].clone()
    t = CropFromMaskd(keys=[IMAGE_KEY], mask_key=MASK_KEY)
    cropped = t(sample)
    inverted = t.inverse(cropped)[IMAGE_KEY]
    _assert_content_preserved(inverted, orig, mask_slices)


def test_3d_fixed_size_content():
    """Fixed output_size crop: values inside the crop window survive round-trip."""
    spatial = (64, 64, 64)
    mask_slices = [slice(24, 40), slice(24, 40), slice(24, 40)]
    sample = _make_sample(spatial, mask_slices)
    orig = sample[IMAGE_KEY].clone()
    t = CropFromMaskd(
        keys=[IMAGE_KEY], mask_key=MASK_KEY, output_size=[16, 16, 16]
    )
    _assert_content_preserved(
        t.inverse(t(sample))[IMAGE_KEY], orig, mask_slices
    )


def test_3d_fixed_size_clamped_content():
    """Crop centred near the boundary still restores content correctly."""
    spatial = (32, 32, 32)
    mask_slices = [slice(0, 4), slice(0, 4), slice(0, 4)]
    sample = _make_sample(spatial, mask_slices)
    orig = sample[IMAGE_KEY].clone()
    t = CropFromMaskd(
        keys=[IMAGE_KEY], mask_key=MASK_KEY, output_size=[16, 16, 16]
    )
    _assert_content_preserved(
        t.inverse(t(sample))[IMAGE_KEY], orig, mask_slices
    )


def test_3d_empty_mask_centre_crop_content():
    """All-zero mask falls back to a centre crop; content must survive round-trip."""
    spatial = (32, 32, 32)
    orig = MetaTensor(
        torch.arange(32**3, dtype=torch.float32).reshape(1, *spatial)
    )
    sample = {
        IMAGE_KEY: orig.clone(),
        MASK_KEY: MetaTensor(torch.zeros(1, *spatial)),
    }
    t = CropFromMaskd(
        keys=[IMAGE_KEY], mask_key=MASK_KEY, output_size=[16, 16, 16]
    )
    restored = t.inverse(t(sample))[IMAGE_KEY]
    centre = [slice(8, 24), slice(8, 24), slice(8, 24)]
    _assert_content_preserved(restored, orig, centre)


def test_2d_mask_driven_content():
    """2-D mask-driven crop: values inside the bounding box survive round-trip."""
    spatial = (64, 64)
    mask_slices = [slice(20, 40), slice(10, 30)]
    sample = _make_sample(spatial, mask_slices)
    orig = sample[IMAGE_KEY].clone()
    t = CropFromMaskd(keys=[IMAGE_KEY], mask_key=MASK_KEY)
    _assert_content_preserved(
        t.inverse(t(sample))[IMAGE_KEY], orig, mask_slices
    )


def test_2d_fixed_size_content():
    """2-D fixed output_size crop: content survives round-trip."""
    spatial = (64, 64)
    mask_slices = [slice(20, 44), slice(20, 44)]
    sample = _make_sample(spatial, mask_slices)
    orig = sample[IMAGE_KEY].clone()
    t = CropFromMaskd(keys=[IMAGE_KEY], mask_key=MASK_KEY, output_size=[24, 24])
    _assert_content_preserved(
        t.inverse(t(sample))[IMAGE_KEY], orig, mask_slices
    )


def test_multi_key_content():
    """All keys must have their content preserved through crop + inverse."""
    spatial = (32, 32, 32)
    mask_slices = [slice(10, 22), slice(10, 22), slice(10, 22)]
    orig1 = MetaTensor(
        torch.arange(32**3, dtype=torch.float32).reshape(1, *spatial)
    )
    orig2 = MetaTensor(
        torch.arange(2 * 32**3, dtype=torch.float32).reshape(2, *spatial)
    )
    mask = MetaTensor(torch.zeros(1, *spatial))
    mask[tuple([slice(None)] + mask_slices)] = 1.0
    sample = {"img1": orig1.clone(), "img2": orig2.clone(), MASK_KEY: mask}
    t = CropFromMaskd(
        keys=["img1", "img2"], mask_key=MASK_KEY, output_size=[12, 12, 12]
    )
    restored = t.inverse(t(sample))
    _assert_content_preserved(restored["img1"], orig1, mask_slices)
    _assert_content_preserved(restored["img2"], orig2, mask_slices)


def test_classification_transforms_center_crop():
    """ClassificationTransforms with center crop - verifies spatial correctness."""
    spatial = (64, 64, 64)
    orig = MetaTensor(
        torch.arange(64**3, dtype=torch.float32).reshape(1, *spatial)
    )
    sample = {IMAGE_KEY: orig.clone()}

    transform_factory = ClassificationTransforms(
        keys=(IMAGE_KEY,),
        adc_keys=(),
        clinical_feature_keys=(),
        target_spacing=None,
        crop_size=(16, 16, 16),
        pad_size=None,
    )

    transforms = monai.transforms.Compose(
        [
            *transform_factory.pre_transforms()[1:],
            *transform_factory.post_transforms(),
        ]
    )
    final = transforms(sample)
    restored = final["image"]

    assert restored.shape == (1, 16, 16, 16)
    assert restored.min() >= 0.0 and restored.max() <= 1.0
    assert restored.max() < 2.0, f"Values not scaled, max={restored.max()}"


def test_classification_transforms_resampling_center_crop():
    """ClassificationTransforms with resampling and center crop - full pipeline."""
    spatial = (64, 64, 64)
    orig = MetaTensor(
        torch.arange(64**3, dtype=torch.float32).reshape(1, *spatial)
    )
    orig.affine = np.eye(4)
    sample = {IMAGE_KEY: orig.clone()}

    transform_factory = ClassificationTransforms(
        keys=(IMAGE_KEY,),
        adc_keys=(),
        clinical_feature_keys=(),
        target_spacing=(0.5, 0.5, 0.5),
        crop_size=(16, 16, 16),
        pad_size=None,
    )

    transforms = monai.transforms.Compose(
        [
            *transform_factory.pre_transforms()[1:],
            *transform_factory.post_transforms(),
        ]
    )
    final = transforms(sample)
    restored = final["image"]

    assert restored.shape == (
        1,
        16,
        16,
        16,
    ), f"Expected (1, 16, 16, 16), got {restored.shape}"


def test_classification_transforms_mask_crop():
    """ClassificationTransforms with mask-driven crop - verifies spatial correctness."""
    spatial = (64, 64, 64)
    mask_slices = [slice(20, 36), slice(20, 36), slice(20, 36)]
    orig = MetaTensor(
        torch.arange(64**3, dtype=torch.float32).reshape(1, *spatial)
    )
    mask = MetaTensor(torch.zeros(1, *spatial))
    mask[tuple([slice(None)] + mask_slices)] = 1.0
    sample = {IMAGE_KEY: orig.clone(), MASK_KEY: mask}

    transform_factory = ClassificationTransforms(
        keys=(IMAGE_KEY,),
        adc_keys=(),
        clinical_feature_keys=(),
        target_spacing=None,
        crop_size=(16, 16, 16),
        pad_size=None,
        image_crop_from_mask=True,
        mask_key=MASK_KEY,
    )

    transforms = monai.transforms.Compose(
        [
            *transform_factory.pre_transforms()[1:],
            *transform_factory.post_transforms(),
        ]
    )
    final = transforms(sample)
    restored = final["image"]

    assert restored.shape == (2, 16, 16, 16)
    assert torch.all((restored[1] == 0) | (restored[1] == 1))


def test_classification_transforms_full_pipeline():
    """ClassificationTransforms full pipeline with resampling, mask crop, and center crop."""
    spatial = (64, 64, 64)
    mask_slices = [slice(20, 44), slice(20, 44), slice(20, 44)]
    orig = MetaTensor(
        torch.arange(64**3, dtype=torch.float32).reshape(1, *spatial)
    )
    orig.affine = np.eye(4)
    mask = MetaTensor(torch.zeros(1, *spatial))
    mask[tuple([slice(None)] + mask_slices)] = 1.0
    sample = {IMAGE_KEY: orig.clone(), MASK_KEY: mask}

    transform_factory = ClassificationTransforms(
        keys=(IMAGE_KEY,),
        adc_keys=(),
        clinical_feature_keys=(),
        target_spacing=(0.8, 0.8, 0.8),
        crop_size=(16, 16, 16),
        pad_size=None,
        image_crop_from_mask=True,
        mask_key=MASK_KEY,
    )

    transforms = monai.transforms.Compose(
        [
            *transform_factory.pre_transforms()[1:],
            *transform_factory.post_transforms(),
        ]
    )
    final = transforms(sample)
    restored = final["image"]

    assert restored.shape == (
        2,
        16,
        16,
        16,
    ), f"Expected (2, 16, 16, 16), got {restored.shape}"


def test_classification_transforms_mask_crop_with_inverse():
    """
    ClassificationTransforms full pipeline: forward + inverse restores content.

    Uses Compose([*pre, *post]) as the full pipeline, then applies inverse
    to verify content preservation through the entire transform cycle.
    """
    spatial = (64, 64, 64)
    mask_slices = [slice(23, 31), slice(4, 52), slice(10, 29)]
    orig = MetaTensor(
        torch.arange(64**3, dtype=torch.float32).reshape(1, *spatial)
    )
    mask = MetaTensor(torch.zeros(1, *spatial))
    mask[tuple([slice(None)] + mask_slices)] = 1.0
    sample = {IMAGE_KEY: orig.clone(), MASK_KEY: mask}

    transform_factory = ClassificationTransforms(
        keys=(IMAGE_KEY,),
        adc_keys=(),
        clinical_feature_keys=(),
        target_spacing=None,
        crop_size=(16, 16, 16),
        pad_size=None,
        image_crop_from_mask=True,
        mask_key=MASK_KEY,
    )

    pre_transforms = transform_factory.pre_transforms()[1:]
    post_transforms = transform_factory.post_transforms()
    full_pipeline = monai.transforms.Compose(
        [*pre_transforms, *post_transforms]
    )

    transformed = full_pipeline(sample)

    assert transformed["image"].shape == (
        2,
        16,
        16,
        16,
    ), f"Expected (2, 16, 16, 16), got {transformed['image'].shape}"

    restored_sample = full_pipeline.inverse(transformed)
    restored_img = restored_sample["image"]

    assert (
        restored_img.shape[1:] == orig.shape[1:]
    ), f"Spatial shape not restored: {restored_img.shape[1:]} != {orig.shape[1:]}"

    s = (slice(None), *mask_slices)
    orig_region = orig[s]
    restored_region = restored_img[0:1][s]

    valid_mask = orig_region > 1e-6
    if valid_mask.any():
        ratios = restored_region[valid_mask] / orig_region[valid_mask]
        unique_ratios = torch.unique(ratios)
        unique_ratios = unique_ratios[unique_ratios != 0]
        assert (
            len(unique_ratios) == 1
        ), f"Scaling ratio not constant: {unique_ratios}"
        assert ratios.std() < 0.01, (
            f"Scaling ratio not constant: std={ratios.std()}, "
            f"mean={ratios.mean()}. Values were not preserved through scaling."
        )
        assert (
            1e-6 < ratios.mean() < 1.0
        ), f"Unexpected scaling factor: {ratios.mean()}"
