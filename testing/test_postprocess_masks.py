import sys
from pathlib import Path

import numpy as np
import pytest
import SimpleITK as sitk

sys.path.append(str(Path(__file__).resolve().parents[1]))

from adell_mri.entrypoints.utils.postprocessing.postprocess_masks import (
    exclude_touching_border,
    exclude_with_distance,
    exclude_with_size,
    fill_holes,
    get_label_image,
    keep_labels,
    main,
    remove_labels,
)


def _make_mask(
    objects: list[tuple[int, tuple[int, int, int, int, int, int]]],
    size: tuple[int, int, int] = (100, 100, 100),
) -> sitk.Image:
    array = np.zeros(size, dtype=np.uint8)
    for label, (x, y, z, sx, sy, sz) in objects:
        array[z : z + sz, y : y + sy, x : x + sx] = label
    return sitk.GetImageFromArray(array)


def _to_array(image: sitk.Image) -> np.ndarray:
    return sitk.GetArrayFromImage(image)


@pytest.fixture
def distance_mask() -> sitk.Image:
    return _make_mask(
        [
            (1, (48, 48, 48, 4, 4, 4)),
            (2, (0, 0, 0, 4, 4, 4)),
        ]
    )


def test_get_label_image_assigns_unique_labels():
    mask = _make_mask([(1, (0, 0, 0, 4, 4, 4)), (2, (20, 20, 20, 4, 4, 4))])
    label_image = get_label_image(mask)
    labels = sorted(_to_array(label_image).flatten())
    assert len(set(labels)) == 3
    assert 0 in set(labels)


def test_exclude_with_distance_removes_distant_object(distance_mask):
    result, _ = exclude_with_distance(distance_mask)
    assert set(np.unique(_to_array(result))) == {0, 1}


def test_exclude_with_distance_absolute_reference(distance_mask):
    result, _ = exclude_with_distance(
        distance_mask, reference=(50, 50, 50), max_distance=20
    )
    assert set(np.unique(_to_array(result))) == {0, 1}


def test_exclude_with_distance_keeps_all_with_large_distance(distance_mask):
    result, _ = exclude_with_distance(distance_mask, max_distance=90)
    assert set(np.unique(_to_array(result))) == {0, 1, 2}


def test_exclude_with_distance_raises_on_dimension_mismatch(distance_mask):
    with pytest.raises(ValueError):
        exclude_with_distance(distance_mask, reference=(0.5, 0.5))


def test_exclude_with_distance_raises_on_mixed_types(distance_mask):
    with pytest.raises(TypeError):
        exclude_with_distance(distance_mask, reference=(0.5, 100, 0.5))


def test_exclude_with_size_minimum_maximum():
    mask = _make_mask(
        [
            (1, (0, 0, 0, 4, 4, 4)),
            (2, (20, 20, 20, 8, 8, 8)),
            (3, (50, 50, 50, 12, 12, 12)),
        ]
    )
    result, _ = exclude_with_size(mask, minimum_size=100, maximum_size=1000)
    assert set(np.unique(_to_array(result))) == {0, 2}


def test_exclude_with_size_keep_n_largest():
    mask = _make_mask(
        [
            (1, (0, 0, 0, 4, 4, 4)),
            (2, (20, 20, 20, 8, 8, 8)),
            (3, (50, 50, 50, 12, 12, 12)),
        ]
    )
    result, _ = exclude_with_size(mask, keep_n_largest=1)
    assert set(np.unique(_to_array(result))) == {0, 3}


def test_remove_labels():
    mask = _make_mask(
        [
            (1, (0, 0, 0, 4, 4, 4)),
            (2, (20, 20, 20, 4, 4, 4)),
            (3, (50, 50, 50, 4, 4, 4)),
        ]
    )
    result, _ = remove_labels(mask, [2])
    assert set(np.unique(_to_array(result))) == {0, 1, 3}


def test_keep_labels():
    mask = _make_mask(
        [
            (1, (0, 0, 0, 4, 4, 4)),
            (2, (20, 20, 20, 4, 4, 4)),
        ]
    )
    result, _ = keep_labels(mask, [2])
    assert set(np.unique(_to_array(result))) == {0, 2}


def test_exclude_touching_border():
    mask = _make_mask(
        [
            (1, (0, 0, 0, 4, 4, 4)),
            (2, (48, 48, 48, 4, 4, 4)),
        ]
    )
    result, _ = exclude_touching_border(mask)
    assert set(np.unique(_to_array(result))) == {0, 2}


def test_fill_holes():
    array = np.zeros((20, 20, 20), dtype=np.uint8)
    array[2:18, 2:18, 2:18] = 1
    array[7:13, 7:13, 7:13] = 0
    mask = sitk.GetImageFromArray(array)

    result = fill_holes(mask)
    result_array = _to_array(result)
    assert result_array[10, 10, 10] == 1
    assert np.sum(result_array) == 16**3


def test_fill_holes_preserves_labels():
    array = np.zeros((20, 20, 20), dtype=np.uint8)
    array[2:10, 2:10, 2:10] = 1
    array[4:7, 4:7, 4:7] = 0
    array[2:10, 12:20, 2:10] = 2
    array[4:7, 14:17, 4:7] = 0
    mask = sitk.GetImageFromArray(array)

    result = fill_holes(mask)
    result_array = _to_array(result)
    assert result_array[5, 5, 5] == 1
    assert result_array[5, 15, 5] == 2


def test_main_writes_postprocessed_masks(tmp_path):
    input_dir = tmp_path / "masks"
    input_dir.mkdir()
    output_dir = tmp_path / "output"

    mask = _make_mask(
        [
            (1, (0, 0, 0, 4, 4, 4)),
            (2, (20, 20, 20, 4, 4, 4)),
        ]
    )
    sitk.WriteImage(mask, str(input_dir / "mask_1.nii.gz"))

    main(
        [
            "--input_paths",
            str(input_dir),
            "--patterns",
            "*.nii.gz",
            "--output_path",
            str(output_dir),
            "--remove_labels",
            "2",
        ]
    )

    output_path = output_dir / "masks" / "mask_1.nii.gz"
    assert output_path.exists()
    result = sitk.ReadImage(str(output_path))
    assert set(np.unique(_to_array(result))) == {0, 1}


def test_main_exclude_with_distance(tmp_path):
    input_dir = tmp_path / "masks"
    input_dir.mkdir()
    output_dir = tmp_path / "output"

    mask = _make_mask(
        [
            (1, (48, 48, 48, 4, 4, 4)),
            (2, (0, 0, 0, 4, 4, 4)),
        ]
    )
    sitk.WriteImage(mask, str(input_dir / "mask_1.nii.gz"))

    main(
        [
            "--input_paths",
            str(input_dir),
            "--patterns",
            "*.nii.gz",
            "--output_path",
            str(output_dir),
            "--exclude_with_distance",
            "--reference",
            "0.5",
            "--max_distance",
            "0.2",
        ]
    )

    output_path = output_dir / "masks" / "mask_1.nii.gz"
    assert output_path.exists()
    result = sitk.ReadImage(str(output_path))
    assert set(np.unique(_to_array(result))) == {0, 1}
