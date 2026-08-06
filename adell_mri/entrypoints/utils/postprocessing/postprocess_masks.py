"""
Postprocesses mask predictions using a set of user-defined heuristics.

This module provides both a set of standalone functions (``remove_labels``,
``exclude_with_distance``, ``exclude_with_size``, ``exclude_touching_border``
and ``fill_holes``) and a command-line interface (``main``) which applies a
user-defined sequence of these heuristics to a folder of masks.

The component-based operations (``exclude_with_distance``,
``exclude_with_size`` and ``exclude_touching_border``) operate on connected
components. Each component is assigned a unique label using
``get_label_image`` and objects are removed based on a user-defined criterion.
The label-value operations (``remove_labels`` and ``keep_labels``) operate
directly on the mask label values.
"""

import argparse
from collections.abc import Sequence
from functools import partial
from math import inf
from multiprocessing import Pool
from pathlib import Path

import numpy as np
import SimpleITK as sitk
from tqdm import tqdm

from adell_mri.utils.python_logging import get_logger

logger = get_logger(__name__)

desc = "Postprocesses mask predictions using a set of user-defined heuristics."


def get_label_image(mask: sitk.Image) -> sitk.Image:
    """
    Generates a label image where each connected component in ``mask`` is
    assigned a unique integer label.

    Args:
        mask (sitk.Image): binary or labelled mask.

    Returns:
        sitk.Image: label image where each connected component has a unique
            label. The background (value 0) is not labelled.
    """
    filter = sitk.ConnectedComponentImageFilter()
    return filter.Execute(mask)


def _get_unique_values(mask: sitk.Image) -> list[int]:
    """
    Returns the unique non-zero label values present in ``mask``.

    Args:
        mask (sitk.Image): binary or labelled mask.

    Returns:
        list[int]: unique non-zero label values present in ``mask``.
    """
    filter = sitk.LabelStatisticsImageFilter()
    filter.Execute(mask, mask)
    return filter.GetLabels()


def _get_component_centroid(
    shape_filter: sitk.LabelShapeStatisticsImageFilter, label_idx: int
) -> np.ndarray:
    """
    Computes the centroid of a connected component as the centre of its
    bounding box.

    Args:
        shape_filter (sitk.LabelShapeStatisticsImageFilter): statistics filter
            which has been executed on the label image.
        label_idx (int): label of the component.

    Returns:
        np.ndarray: centroid of the component in voxel coordinates.
    """
    bounding_box = shape_filter.GetBoundingBox(label_idx)
    n_dim = len(bounding_box) // 2
    return np.array(
        [bounding_box[i] + bounding_box[n_dim + i] / 2 for i in range(n_dim)]
    )


def _reference_to_absolute(
    reference: int | float | Sequence[int | float], size: Sequence[int]
) -> np.ndarray:
    """
    Converts a reference point to absolute voxel coordinates.

    Floats are interpreted as relative coordinates (fractions of the image
    size along each dimension) and ints as absolute voxel coordinates. A
    scalar is broadcast to all dimensions; a sequence must have as many values
    as the number of mask dimensions and be homogeneous in type.

    Args:
        reference (int | float | Sequence[int | float]): point of reference.
        size (Sequence[int]): image size in voxels along each dimension.

    Returns:
        np.ndarray: absolute voxel coordinates.

    Raises:
        ValueError: if the number of values in ``reference`` does not match
            the number of mask dimensions.
        TypeError: if ``reference`` mixes ints and floats.
    """
    if isinstance(reference, (int, np.integer)):
        return np.full(len(size), float(reference))
    if isinstance(reference, (float, np.floating)):
        return float(reference) * np.asarray(size, dtype=float)

    values = np.asarray(reference)
    if values.ndim != 1 or len(values) != len(size):
        raise ValueError(
            "reference must be a scalar or a sequence with as many values as "
            "the number of mask dimensions"
        )
    has_int = any(isinstance(v, (int, np.integer)) for v in reference)
    has_float = any(isinstance(v, (float, np.floating)) for v in reference)
    if has_int and has_float:
        raise TypeError(
            "reference values must be either all ints (absolute voxel "
            "coordinates) or all floats (relative coordinates)"
        )
    if has_float:
        return values.astype(float) * np.asarray(size, dtype=float)
    return values.astype(float)


def _distance_to_absolute(
    max_distance: int | float, size: Sequence[int]
) -> float:
    """
    Converts a maximum distance to an absolute distance in voxels.

    Floats are interpreted as a fraction of the size of the first mask
    dimension and ints as absolute voxel counts.

    Args:
        max_distance (int | float): maximum allowed distance.
        size (Sequence[int]): image size in voxels along each dimension.

    Returns:
        float: absolute maximum distance in voxels.

    Raises:
        TypeError: if ``max_distance`` is neither an int nor a float.
    """
    if isinstance(max_distance, (int, np.integer)):
        return float(max_distance)
    if isinstance(max_distance, (float, np.floating)):
        return float(max_distance) * size[0]
    raise TypeError(
        "max_distance must be an int (absolute) or a float (relative)"
    )


def _apply_keep_mask(
    mask: sitk.Image, label_image: sitk.Image, keep_mask: sitk.Image
) -> tuple[sitk.Image, sitk.Image]:
    """
    Zeroes out the voxels where ``keep_mask`` is zero in both ``mask`` and
    ``label_image``.

    Args:
        mask (sitk.Image): binary or labelled mask.
        label_image (sitk.Image): label image.
        keep_mask (sitk.Image): binary mask where 1 marks voxels to keep.

    Returns:
        tuple[sitk.Image, sitk.Image]: the masked ``mask`` and ``label_image``.
    """
    mask = sitk.Mask(mask, keep_mask)
    label_image = sitk.Mask(label_image, keep_mask)
    return mask, label_image


def _remove_component_labels(
    mask: sitk.Image,
    label_image: sitk.Image,
    component_labels: Sequence[int],
) -> tuple[sitk.Image, sitk.Image]:
    """
    Removes connected components identified by ``component_labels`` from both
    ``mask`` and ``label_image``.

    Args:
        mask (sitk.Image): binary or labelled mask.
        label_image (sitk.Image): label image where each connected component
            has a unique label.
        component_labels (Sequence[int]): component labels to remove.

    Returns:
        tuple[sitk.Image, sitk.Image]: the postprocessed ``mask`` and
            ``label_image``.
    """
    for label_idx in component_labels:
        keep_mask = label_image != label_idx
        mask, label_image = _apply_keep_mask(mask, label_image, keep_mask)
    return mask, label_image


def _remove_label_values(
    mask: sitk.Image,
    label_image: sitk.Image,
    values_to_remove: Sequence[int],
) -> tuple[sitk.Image, sitk.Image]:
    """
    Removes all voxels whose label value is in ``values_to_remove`` from both
    ``mask`` and ``label_image``.

    Args:
        mask (sitk.Image): binary or labelled mask.
        label_image (sitk.Image): label image.
        values_to_remove (Sequence[int]): label values to remove.

    Returns:
        tuple[sitk.Image, sitk.Image]: the postprocessed ``mask`` and
            ``label_image``.
    """
    for value in values_to_remove:
        keep_mask = mask != value
        mask, label_image = _apply_keep_mask(mask, label_image, keep_mask)
    return mask, label_image


def _parse_number(value: str) -> int | float:
    """
    Parses a string into an int when possible, otherwise into a float.

    Args:
        value (str): string to parse.

    Returns:
        int | float: parsed number.
    """
    if value == inf:
        return value
    try:
        return int(value)
    except ValueError:
        return float(value)


def exclude_with_distance(
    mask: sitk.Image,
    reference: int | float | Sequence[int | float] = 0.5,
    max_distance: int | float = 0.2,
    label_image: sitk.Image | None = None,
) -> tuple[sitk.Image, sitk.Image]:
    """
    Excludes objects whose centroid is too distant from a point of reference.

    Both ``reference`` and ``max_distance`` can be provided in absolute
    (voxel) or relative (fraction of the image size) units. The unit is
    inferred from the value type: floats are treated as relative and ints as
    absolute. A scalar ``reference`` is broadcast to all dimensions; a
    sequence must have as many values as the number of mask dimensions. A
    relative ``max_distance`` is expressed as a fraction of the size of the
    first mask dimension.

    Args:
        mask (sitk.Image): binary or labelled mask.
        reference (int | float | Sequence[int | float], optional): point of
            reference. Relative references (floats) are fractions of the image
            size along each dimension and absolute references (ints) are
            voxel coordinates. Defaults to 0.5 (image centre).
        max_distance (int | float, optional): maximum allowed distance between
            an object centroid and the reference point. Relative distances
            (floats) are fractions of the size of the first mask dimension
            and absolute distances (ints) are numbers of voxels. Defaults to
            0.2.
        label_image (sitk.Image, optional): label image where each connected
            component has a unique label. If not provided it is computed from
            ``mask``. Defaults to None.

    Returns:
        tuple[sitk.Image, sitk.Image]: the postprocessed mask and the
            corresponding label image.

    Raises:
        ValueError: if ``reference`` does not match the number of mask
            dimensions.
        TypeError: if ``reference`` or ``max_distance`` mixes ints and floats.
    """
    logger.debug("Running exclude_with_distance")

    if label_image is None:
        label_image = get_label_image(mask)

    shape_filter = sitk.LabelShapeStatisticsImageFilter()
    shape_filter.Execute(label_image)
    if shape_filter.GetNumberOfLabels() == 0:
        logger.debug("No objects found, skipping exclusion")
        return mask, label_image

    size = mask.GetSize()
    abs_reference = _reference_to_absolute(reference, size)
    abs_max_distance = _distance_to_absolute(max_distance, size)
    logger.debug("Calculated an absolute reference of %s", abs_reference)
    logger.debug("Calculated an absolute max distance of %s", abs_max_distance)

    labels_to_remove = []
    for label_idx in shape_filter.GetLabels():
        centroid = _get_component_centroid(shape_filter, label_idx)
        distance = np.sqrt(np.sum(np.square(centroid - abs_reference)))
        if distance > abs_max_distance:
            labels_to_remove.append(label_idx)

    mask, label_image = _remove_component_labels(
        mask, label_image, labels_to_remove
    )
    n_removed = len(labels_to_remove)
    n_kept = shape_filter.GetNumberOfLabels() - n_removed
    logger.debug("exclude_with_distance removed %s objects", n_removed)
    logger.debug("exclude_with_distance kept %s objects", n_kept)
    return mask, label_image


def exclude_with_size(
    mask: sitk.Image,
    minimum_size: int | float = 0,
    maximum_size: int | float = inf,
    keep_n_largest: int | None = None,
    label_image: sitk.Image | None = None,
) -> tuple[sitk.Image, sitk.Image]:
    """
    Excludes objects based on their physical size.

    Objects with a physical size (in mm^3) smaller than ``minimum_size`` or
    larger than ``maximum_size`` are removed. If ``keep_n_largest`` is
    provided, only the ``keep_n_largest`` largest remaining objects are kept.
    This happens after minimum/maximum size-based removal.

    Args:
        mask (sitk.Image): binary or labelled mask.
        minimum_size (int | float, optional): minimum object size in mm^3.
            Defaults to 0.
        maximum_size (int | float, optional): maximum object size in mm^3.
            Defaults to inf.
        keep_n_largest (int, optional): number of largest remaining objects to
            keep. If None, all objects within the size thresholds are kept.
            Defaults to None.
        label_image (sitk.Image, optional): label image where each connected
            component has a unique label. If not provided it is computed from
            ``mask``. Defaults to None.

    Returns:
        tuple[sitk.Image, sitk.Image]: the postprocessed mask and the
            corresponding label image.
    """
    logger.debug("Running exclude_with_size")

    if label_image is None:
        label_image = get_label_image(mask)

    shape_filter = sitk.LabelShapeStatisticsImageFilter()
    shape_filter.Execute(label_image)
    if shape_filter.GetNumberOfLabels() == 0:
        logger.debug("No objects found, skipping exclusion")
        return mask, label_image

    label_sizes = {
        label_idx: shape_filter.GetPhysicalSize(label_idx)
        for label_idx in shape_filter.GetLabels()
    }
    labels_to_remove = [
        label_idx
        for label_idx, size in label_sizes.items()
        if size < minimum_size or size > maximum_size
    ]

    if keep_n_largest is not None:
        largest = sorted(label_sizes, key=label_sizes.get, reverse=True)[
            :keep_n_largest
        ]
        labels_to_remove.extend(
            label_idx for label_idx in label_sizes if label_idx not in largest
        )

    labels_to_remove = set(labels_to_remove)
    mask, label_image = _remove_component_labels(
        mask, label_image, labels_to_remove
    )
    n_removed = len(labels_to_remove)
    n_kept = shape_filter.GetNumberOfLabels() - n_removed
    logger.debug("exclude_with_size removed %s objects", n_removed)
    logger.debug("exclude_with_size kept %s objects", n_kept)
    return mask, label_image


def remove_labels(
    mask: sitk.Image,
    labels_to_remove: Sequence[int],
    label_image: sitk.Image | None = None,
) -> tuple[sitk.Image, sitk.Image]:
    """
    Removes specific label values from ``mask``.

    This removes all voxels whose label value is in ``labels_to_remove``,
    regardless of the number of connected components they contain. It is
    useful to remove specific classes from a multi-label segmentation.

    Args:
        mask (sitk.Image): binary or labelled mask.
        labels_to_remove (Sequence[int]): label values to remove.
        label_image (sitk.Image, optional): label image. If not provided it is
            computed from ``mask``. Defaults to None.

    Returns:
        tuple[sitk.Image, sitk.Image]: the postprocessed mask and the
            corresponding label image.
    """
    logger.debug("Running remove_labels with labels %s", labels_to_remove)

    if label_image is None:
        label_image = get_label_image(mask)
    mask, label_image = _remove_label_values(
        mask, label_image, labels_to_remove
    )
    logger.debug("remove_labels removed %s labels", len(labels_to_remove))
    return mask, label_image


def keep_labels(
    mask: sitk.Image,
    labels_to_keep: Sequence[int],
    label_image: sitk.Image | None = None,
) -> tuple[sitk.Image, sitk.Image]:
    """
    Keeps only the specified label values in ``mask``, removing all others.

    Args:
        mask (sitk.Image): binary or labelled mask.
        labels_to_keep (Sequence[int]): label values to keep.
        label_image (sitk.Image, optional): label image. If not provided it is
            computed from ``mask``. Defaults to None.

    Returns:
        tuple[sitk.Image, sitk.Image]: the postprocessed mask and the
            corresponding label image.
    """
    logger.debug("Running keep_labels with labels %s", labels_to_keep)

    if label_image is None:
        label_image = get_label_image(mask)

    values_to_remove = [
        value
        for value in _get_unique_values(mask)
        if value not in set(labels_to_keep)
    ]
    mask, label_image = _remove_label_values(
        mask, label_image, values_to_remove
    )
    logger.debug("keep_labels removed %s labels", len(values_to_remove))
    return mask, label_image


def exclude_touching_border(
    mask: sitk.Image, label_image: sitk.Image | None = None
) -> tuple[sitk.Image, sitk.Image]:
    """
    Excludes objects which touch the image border.

    Uses the bounding box of each connected component and removes components
    whose bounding box intersects any image boundary. This is useful to remove
    objects which are truncated by the field of view.

    Args:
        mask (sitk.Image): binary or labelled mask.
        label_image (sitk.Image, optional): label image where each connected
            component has a unique label. If not provided it is computed from
            ``mask``. Defaults to None.

    Returns:
        tuple[sitk.Image, sitk.Image]: the postprocessed mask and the
            corresponding label image.
    """
    logger.debug("Running exclude_touching_border")

    if label_image is None:
        label_image = get_label_image(mask)

    shape_filter = sitk.LabelShapeStatisticsImageFilter()
    shape_filter.Execute(label_image)
    if shape_filter.GetNumberOfLabels() == 0:
        logger.debug("No objects found, skipping exclusion")
        return mask, label_image

    size = mask.GetSize()
    n_dim = mask.GetDimension()
    labels_to_remove = []
    for label_idx in shape_filter.GetLabels():
        bounding_box = shape_filter.GetBoundingBox(label_idx)
        touches_border = any(
            bounding_box[i] == 0
            or bounding_box[i] + bounding_box[n_dim + i] == size[i]
            for i in range(n_dim)
        )
        if touches_border:
            labels_to_remove.append(label_idx)

    mask, label_image = _remove_component_labels(
        mask, label_image, labels_to_remove
    )
    n_removed = len(labels_to_remove)
    n_kept = shape_filter.GetNumberOfLabels() - n_removed
    logger.debug("exclude_touching_border removed %s objects", n_removed)
    logger.debug("exclude_touching_border kept %s objects", n_kept)
    return mask, label_image


def fill_holes(mask: sitk.Image) -> sitk.Image:
    """
    Fills holes inside the objects of ``mask``.

    Holes are filled independently for each label value present in ``mask``,
    so the output preserves the original label values. The mask is returned as
    is if it does not contain any objects.

    Args:
        mask (sitk.Image): binary or labelled mask.

    Returns:
        sitk.Image: mask with holes filled.
    """
    logger.debug("Running fill_holes")

    labels = _get_unique_values(mask)
    if len(labels) == 0:
        return mask

    fill_filter = sitk.BinaryFillholeImageFilter()
    fill_filter.SetForegroundValue(1)

    result = sitk.Image(mask)
    for label in labels:
        binary = mask == label
        filled = fill_filter.Execute(binary)
        new_region = sitk.Cast(filled, sitk.sitkUInt8) - sitk.Cast(
            binary, sitk.sitkUInt8
        )
        result = result + sitk.Cast(new_region, mask.GetPixelID()) * label
    return sitk.Cast(result, mask.GetPixelID())


def wrapper(input_path_and_mask_path: str, **kwargs) -> sitk.Image:
    input_path, mask_path = input_path_and_mask_path
    mask = sitk.ReadImage(mask_path)
    label_image = None
    if kwargs.get("remove_labels", []):
        mask, label_image = remove_labels(
            mask,
            kwargs["remove_labels"],
            label_image=label_image,
        )
    if kwargs.get("keep_labels", []):
        mask, label_image = keep_labels(
            mask,
            kwargs["keep_labels"],
            label_image=label_image,
        )
    if kwargs.get("exclude_touching_border", False):
        mask, label_image = exclude_touching_border(
            mask, label_image=label_image
        )
    if kwargs.get("exclude_with_distance", False):
        mask, label_image = exclude_with_distance(
            mask,
            reference=kwargs["reference"],
            max_distance=kwargs["max_distance"],
            label_image=label_image,
        )
    if kwargs.get("exclude_with_size", False):
        mask, label_image = exclude_with_size(
            mask,
            minimum_size=kwargs["minimum_size"],
            maximum_size=kwargs["maximum_size"],
            keep_n_largest=kwargs["keep_n_largest"],
            label_image=label_image,
        )
    if kwargs.get("fill_holes", False):
        mask = fill_holes(mask)
    return mask, input_path, mask_path


def main(arguments: list[str] | None = None) -> None:
    """
    Applies a user-defined sequence of postprocessing heuristics to a folder
    of masks.

    The following steps are applied in a fixed order, each only if the
    corresponding flag is set:

    1. ``remove_labels`` or ``keep_labels`` (value-based removal).
    2. ``exclude_touching_border`` (border-touching component removal).
    3. ``exclude_with_distance`` (distance-based component removal).
    4. ``exclude_with_size`` (size-based component removal).
    5. ``fill_holes`` (hole filling).

    Postprocessed masks are written to ``output_path`` mirroring the directory
    structure of the input, prefixed with the input directory name.

    Args:
        arguments (list[str], optional): command-line arguments. If None,
            arguments are read from ``sys.argv``. Defaults to None.
    """
    parser = argparse.ArgumentParser(description=desc)

    parser.add_argument(
        "--input_paths",
        dest="input_paths",
        type=str,
        nargs="+",
        help="Path to directories containing masks.",
        required=True,
    )
    parser.add_argument(
        "--patterns",
        dest="patterns",
        type=str,
        nargs="+",
        default=["*"],
        help="Patterns to match masks within each input directory.",
    )
    parser.add_argument(
        "--output_path",
        dest="output_path",
        type=str,
        help="Path to directory where postprocessed masks will be written.",
        required=True,
    )
    parser.add_argument(
        "--n_workers",
        dest="n_workers",
        type=int,
        default=0,
        help="Number of workers.",
    )

    label_value_group = parser.add_mutually_exclusive_group()
    label_value_group.add_argument(
        "--remove_labels",
        dest="remove_labels",
        type=int,
        nargs="+",
        default=None,
        help="Label values to remove from the masks.",
    )
    label_value_group.add_argument(
        "--keep_labels",
        dest="keep_labels",
        type=int,
        nargs="+",
        default=None,
        help="Label values to keep in the masks (all others are removed).",
    )

    parser.add_argument(
        "--exclude_with_distance",
        dest="exclude_with_distance",
        action="store_true",
        default=False,
        help="Excludes objects whose centroid is too distant from a reference "
        "point.",
    )
    parser.add_argument(
        "--reference",
        dest="reference",
        type=str,
        nargs="+",
        default=["0.5"],
        help="Reference point for distance-based exclusion. Floats are "
        "relative (fraction of the image size along each dimension) and ints "
        "are absolute (voxel coordinates). A scalar is broadcast to all "
        "dimensions.",
    )
    parser.add_argument(
        "--max_distance",
        dest="max_distance",
        type=str,
        default="0.2",
        help="Maximum distance to the reference point. Floats are relative to "
        "the first mask dimension and ints are absolute voxel counts.",
    )

    parser.add_argument(
        "--exclude_with_size",
        dest="exclude_with_size",
        action="store_true",
        default=False,
        help="Excludes objects based on their physical size.",
    )
    parser.add_argument(
        "--minimum_size",
        dest="minimum_size",
        type=str,
        default=0.0,
        help="Minimum object size in mm^3.",
    )
    parser.add_argument(
        "--maximum_size",
        dest="maximum_size",
        type=str,
        default=inf,
        help="Maximum object size in mm^3.",
    )
    parser.add_argument(
        "--keep_n_largest",
        dest="keep_n_largest",
        type=int,
        default=None,
        help="Number of largest remaining objects to keep when excluding by "
        "size.",
    )

    parser.add_argument(
        "--exclude_touching_border",
        dest="exclude_touching_border",
        action="store_true",
        default=False,
        help="Excludes objects which touch the image border.",
    )
    parser.add_argument(
        "--fill_holes",
        dest="fill_holes",
        action="store_true",
        default=False,
        help="Fills holes in the masks.",
    )

    args = parser.parse_args(arguments)

    reference_values = [_parse_number(x) for x in args.reference]
    reference = (
        reference_values[0] if len(reference_values) == 1 else reference_values
    )
    max_distance = _parse_number(args.max_distance)
    minimum_size = _parse_number(args.minimum_size)
    maximum_size = _parse_number(args.maximum_size)

    output_path = Path(args.output_path)
    output_path.mkdir(parents=True, exist_ok=True)

    mask_paths = [
        (Path(input_path), mask_path)
        for input_path in args.input_paths
        for pattern in args.patterns
        for mask_path in Path(input_path).glob(pattern)
    ]
    if len(mask_paths) == 0:
        logger.warning(
            "No masks found matching the given input paths and patterns"
        )

    kwargs = dict(
        remove_labels=args.remove_labels,
        keep_labels=args.keep_labels,
        exclude_touching_border=args.exclude_touching_border,
        exclude_with_distance=args.exclude_with_distance,
        reference=reference,
        max_distance=max_distance,
        exclude_with_size=args.exclude_with_size,
        minimum_size=minimum_size,
        maximum_size=maximum_size,
        keep_n_largest=args.keep_n_largest,
        fill_holes=args.fill_holes,
    )
    wrapper_with_args = partial(wrapper, **kwargs)
    total = len(mask_paths)
    if args.n_workers < 2:
        map_fn = map(wrapper_with_args, mask_paths)
    else:
        pool = Pool(args.n_workers)
        map_fn = pool.imap_unordered(wrapper_with_args, mask_paths)
    with tqdm(map_fn, total=total, desc="Postprocessing masks") as pbar:
        for mask, input_path, mask_path in pbar:
            relative_path = mask_path.relative_to(input_path)
            output_path_image = output_path / input_path.name / relative_path
            output_path_image.parent.mkdir(parents=True, exist_ok=True)
            sitk.WriteImage(mask, str(output_path_image))
            pbar.set_description(f"Wrote mask to {output_path_image}")


if __name__ == "__main__":
    main()
