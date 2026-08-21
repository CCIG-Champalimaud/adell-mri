import os
import sys

sys.path.append(os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))

import numpy as np
import pytest
import torch

from adell_mri.utils.masking import (
    GenericTransformerMasker,
    TransformerMasker,
    _check_patch_size,
)

IMAGE_DIMENSIONS_2D = [8, 8]
MIN_PATCH = [2, 2]
MAX_PATCH = [4, 4]
N_FEATURES = 6


def test_check_patch_size_raises_on_oversized_patch():
    with pytest.raises(ValueError):
        _check_patch_size([8, 8], [8, 9])


def test_masker_deterministic_with_seed():
    X = torch.rand(1, 64, N_FEATURES)
    outputs = []
    for _ in range(2):
        masker = TransformerMasker(
            image_dimensions=IMAGE_DIMENSIONS_2D,
            min_patch_size=MIN_PATCH,
            max_patch_size=MAX_PATCH,
            seed=42,
        )
        masked, patches, coords = masker(X.clone(), n_patches=3)
        outputs.append((masked, [np.asarray(p) for p in patches], coords))
    (m1, p1, c1), (m2, p2, c2) = outputs
    assert torch.equal(m1, m2)
    assert all(np.array_equal(a, b) for a, b in zip(p1, p2))
    assert np.array_equal(np.array(c1), np.array(c2))


def test_transformer_masker_returns_patches():
    masker = TransformerMasker(
        image_dimensions=IMAGE_DIMENSIONS_2D,
        min_patch_size=MIN_PATCH,
        max_patch_size=MAX_PATCH,
        n_features=N_FEATURES,
        seed=42,
    )
    X = torch.rand(1, 64, N_FEATURES)
    _, patches, coords = masker(X.clone(), n_patches=2)
    assert len(patches) == 2
    assert len(coords) == 2
    for patch in patches:
        assert patch.shape[0] == 1
        assert patch.shape[-1] == N_FEATURES
        assert 0 < patch.shape[1] < 64


def test_retrieve_patch_indices_are_spatially_consistent():
    masker = TransformerMasker(
        image_dimensions=[4, 4],
        min_patch_size=[1, 1],
        max_patch_size=[2, 2],
        seed=0,
    )
    X = torch.arange(16, dtype=torch.float32).reshape(1, 16, 1)
    patch, long_coords = masker.retrieve_patch(X, [0, 0, 2, 2])
    expected_idx = np.array([0, 1, 4, 5])
    assert np.array_equal(np.sort(np.asarray(long_coords)), expected_idx)
    assert torch.equal(patch[0, :, 0], torch.tensor([0.0, 1.0, 4.0, 5.0]))


def test_masking_with_mask_vector_replaces_tokens():
    masker = TransformerMasker(
        image_dimensions=[4, 4],
        min_patch_size=[1, 1],
        max_patch_size=[2, 2],
        n_features=N_FEATURES,
        seed=0,
    )
    X = torch.ones(1, 16, N_FEATURES)
    mask_vector = torch.zeros(1, 1, N_FEATURES)
    masked, _, _ = masker(X.clone(), mask_vector=mask_vector, n_patches=1)
    changed = not torch.equal(masked, X)
    assert changed
    n_zero_tokens = int((masked.abs().sum(-1) == 0).sum())
    assert n_zero_tokens > 0


def test_generic_masker_matches_base_sampling():
    base = TransformerMasker(
        image_dimensions=IMAGE_DIMENSIONS_2D,
        min_patch_size=MIN_PATCH,
        max_patch_size=MAX_PATCH,
        seed=42,
    )
    generic = GenericTransformerMasker(
        image_dimensions=IMAGE_DIMENSIONS_2D,
        min_patch_size=MIN_PATCH,
        max_patch_size=MAX_PATCH,
        seed=42,
    )
    base_coords = np.array(base.sample_patches(5))
    generic_coords = np.array(generic.sample_patches(5))
    assert np.array_equal(base_coords, generic_coords)


def test_generic_masker_call_masks_correctly():
    masker = GenericTransformerMasker(
        image_dimensions=[4, 4],
        min_patch_size=[1, 1],
        max_patch_size=[2, 2],
        seed=0,
    )
    X = torch.ones(1, 16, N_FEATURES)
    mask_vector = torch.zeros(1, 1, N_FEATURES)
    masked, _ = masker(X.clone(), mask_vector=mask_vector, n_patches=1)
    assert not torch.equal(masked, X)


def test_maskers_support_3d():
    for Masker in [TransformerMasker, GenericTransformerMasker]:
        masker = Masker(
            image_dimensions=[4, 4, 4],
            min_patch_size=[1, 1, 1],
            max_patch_size=[2, 2, 2],
            seed=0,
        )
        X = torch.rand(1, 64, N_FEATURES)
        out = masker(X.clone(), n_patches=2)
        assert out[0].shape == X.shape
