from dataclasses import dataclass
from typing import List, Tuple

import numpy as np
import SimpleITK as sitk
from multiprocess import Pool
from tqdm import tqdm

from adell_mri.utils.python_logging import get_logger

logger = get_logger(__name__)

DatasetDict = dict[str, dict[str, str]]


@dataclass
class ReadSpacing:
    """
    Reads the spacing for a given key in a dataset dictionary.

    Args:
        dataset_dict (DatasetDict): Dictionary containing study IDs and paths.
        image_key (str): Key for the image path value to read spacing from.
    """

    dataset_dict: DatasetDict
    image_key: str

    def __call__(self, key: str) -> Tuple[float, float, float]:
        """
        Args:
            key (str): key in `self.dataset_dict`

        Returns:
            Tuple[str, Tuple[float, float, float]]: Tuple containing the key
                and spacing for the corresponding image.

        """
        sp = sitk.ReadImage(self.dataset_dict[key][self.image_key]).GetSpacing()
        return key, sp


def spacing_values_from_dataset_json(
    dataset_dict: DatasetDict, key: str, n_workers: int = 1
) -> dict[str, Tuple[float, float, float]]:
    """
    Retrieves spacings for all elements with a given key.

    Args:
        dataset_dict (DatasetDict): dataset dictionary.
        key (str): key corresponding to image paths in dataset dictionary
            (must be SITK-readable).
        n_workers (int, optional): number of workers. Defaults to 1.

    Returns:
        dict[str, Tuple[float, float, float]]: dictionary with study IDs as
            keys and spacings as values.
    """
    all_spacings = {}
    read_spacing = ReadSpacing(dataset_dict, key)
    with tqdm(dataset_dict) as pbar:
        pbar.set_description("Inferring target spacing")
        if n_workers > 1:
            pool = Pool(n_workers)
            path_iterable = pool.imap(read_spacing, dataset_dict.keys())
        else:
            path_iterable = map(read_spacing, dataset_dict.keys())
        for key, spacing in path_iterable:
            all_spacings[key] = spacing
            pbar.update()
    return all_spacings


def get_spacing_quantile(
    spacing_dict: dict[str, Tuple[float, float, float]], quantile: float = 0.5
) -> List[float]:
    """
    Gets the spacing at a specified quantile from a dictionary of spacings.

    Args:
        spacing_dict: Dictionary with keys corresponding to studies and
            values containing the spacing for that study.
        quantile: Quantile at which to return the spacing.

    Returns:
        List[float]: The spacing at the specified quantile.
    """
    all_spacings = np.array([spacing_dict[k] for k in spacing_dict])
    output = np.quantile(all_spacings, quantile, axis=0).tolist()
    logger.info("Inferred spacing: %s", output)
    return output


def spacing_from_dataset_json(
    dataset_dict: DatasetDict,
    key: str,
    quantile: float = 0.5,
    n_workers: int = 1,
) -> List[float]:
    """
    Calculates the spacing at a given quantile using a dataset dictionary and a
    key.

    Args:
        dataset_dict (DatasetDict): dictionary containing study IDs as values
            and a dictionary of SITK-readable paths as values.
        key (str): key for the value that will be used to extract the spacing.
        quantile (float, optional): spacing quantile that will be returned.
            Defaults to 0.5.

    Returns:
        List[float]: spacing at the specified quantile.
    """
    spacing_values = spacing_values_from_dataset_json(
        dataset_dict=dataset_dict, key=key, n_workers=n_workers
    )
    output = get_spacing_quantile(spacing_values, quantile)
    return output


def resample_image(
    sitk_image: sitk.Image,
    out_spacing: List[float] = [1.0, 1.0, 1.0],
    out_size: List[int] = None,
    is_label: bool = False,
) -> sitk.Image:
    """
    Resamples a SimpleITK image to a specified output spacing and size.

    Args:
      sitk_image: The SimpleITK image to resample.
      out_spacing: The desired output spacing.
      out_size: The desired output size. If None, it will be calculated
        automatically based on out_spacing.
      is_label: Whether the image is a label mask.

    Returns:
      The resampled SimpleITK image.
    """

    spacing = sitk_image.GetSpacing()
    size = sitk_image.GetSize()

    if out_size is None:
        out_size = [
            int(np.round(size * (sin / sout)))
            for size, sin, sout in zip(size, spacing, out_spacing)
        ]

    pad_value = sitk_image.GetPixelIDValue()

    resample = sitk.ResampleImageFilter()
    resample.SetOutputSpacing(list(out_spacing))
    resample.SetSize(out_size)
    resample.SetOutputDirection(sitk_image.GetDirection())
    resample.SetOutputOrigin(sitk_image.GetOrigin())
    resample.SetTransform(sitk.Transform())
    resample.SetDefaultPixelValue(pad_value)
    if is_label:
        resample.SetInterpolator(sitk.sitkNearestNeighbor)
    else:
        resample.SetInterpolator(sitk.sitkBSpline)

    # perform resampling
    sitk_image = resample.Execute(sitk_image)

    return sitk_image


def resample_image_to_target(
    moving: sitk.Image,
    target: sitk.Image,
    is_label: bool = False,
) -> sitk.Image:
    """
    Resamples a SimpleITK image to the space of a target image.

    Args:
      moving: The SimpleITK image to resample.
      target: The target SimpleITK image to match.
      is_label (bool): whether the moving image is a label mask.

    Returns:
      The resampled SimpleITK image matching the target properties.
    """
    if is_label:
        interpolation = sitk.sitkNearestNeighbor
    else:
        interpolation = sitk.sitkBSpline

    output = sitk.Resample(moving, target, sitk.Transform(), interpolation, 0)
    return output


def crop_image(sitk_image: sitk.Image, output_size: list[int]) -> sitk.Image:
    """
    Crops a SimpleITK image to a specified output size.

    Pads the image if it is smaller than the output size before cropping to the
    center region.

    Args:
        sitk_image (sitk.Image): SITK image.
        output_size (list[int]): output size.

    Returns:
        sitk.Image: cropped/padded SITK image.
    """

    output_size = np.array(output_size)
    curr_size = np.array(sitk_image.GetSize())
    # pad in case image is too small
    if any(curr_size < output_size):
        total_padding = np.maximum((0, 0, 0), output_size - curr_size)
        lower = np.int16(total_padding // 2)
        upper = np.int16(total_padding - lower)
        sitk_image = sitk.ConstantPad(
            sitk_image, lower.tolist(), upper.tolist(), 0.0
        )
    curr_size = np.array(sitk_image.GetSize())
    total_crop = np.maximum((0, 0, 0), curr_size - output_size)
    lower = np.int16(total_crop // 2)
    upper = np.int16((total_crop - lower))

    sitk_image = sitk.Crop(sitk_image, lower.tolist(), upper.tolist())
    return sitk_image


def copy_information_nd(
    target_image: sitk.Image, source_image: sitk.Image
) -> sitk.Image:
    """
    Copies information from a source image to a target image. Unlike the
    standard CopyInformation method in SimpleITK, the source image can have
    fewer axes than the target image as long as the first n axes of each are
    identical (where n is the number of axes in the source image).

    Args:
        target_image (sitk.Image): target image.
        source_image (sitk.Image): source information for metadata.

    Raises:
        Exception: if the source image has more dimensions than the target
            image.

    Returns:
        sitk.Image: target image with metadata copied from source image.
            The metadata information for the additional axes is set to 0 in the
            case of the origin, 1.0 in the case of the spacing and to the
            identity in the case of the direction.
    """
    size_source = source_image.GetSize()
    size_target = target_image.GetSize()
    n_dim_in = len(size_source)
    n_dim_out = len(size_target)
    if n_dim_in == n_dim_out:
        target_image.CopyInformation(source_image)
        return target_image
    elif n_dim_in > n_dim_out:
        raise Exception(
            "target_image has to have the same or more dimensions than\
                source_image"
        )
    if size_target[:n_dim_in] != size_source:
        out_str = f"sizes are different (target={size_target[:n_dim_in]}"
        out_str += f" size_source={size_source})"
        return out_str
    spacing = list(source_image.GetSpacing())
    origin = list(source_image.GetOrigin())
    direction = list(source_image.GetDirection())
    while len(origin) != n_dim_out:
        spacing.append(1.0)
        origin.append(0.0)
    direction = np.reshape(direction, (n_dim_in, n_dim_in))
    direction = np.pad(
        direction, ((0, n_dim_out - n_dim_in), (0, n_dim_out - n_dim_in))
    )
    x, y = np.diag_indices(n_dim_out - n_dim_in)
    x = x + n_dim_in
    y = y + n_dim_in
    direction[(x, y)] = 1.0
    target_image.SetSpacing(spacing)
    target_image.SetOrigin(origin)
    target_image.SetDirection(direction.flatten())
    return target_image
