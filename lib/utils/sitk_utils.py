import SimpleITK as sitk
import numpy as np
from multiprocess import Pool
from tqdm import tqdm
from typing import List
from typing import Dict, Tuple
from dataclasses import dataclass
from ..custom_types import DatasetDict


@dataclass
class ReadSpacing:
    dataset_dict: DatasetDict
    image_key: str

    def __call__(self, key: str) -> Tuple[float, float, float]:
        sp = sitk.ReadImage(
            self.dataset_dict[key][self.image_key]
        ).GetSpacing()
        return key, sp


def spacing_values_from_dataset_json(
    dataset_dict: DatasetDict, key: str, n_workers: int = 1
) -> Dict[str, Tuple[float, float, float]]:
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
    spacing_dict: Dict[str, Tuple[float, float, float]], quantile: float = 0.5
):
    all_spacings = np.array([spacing_dict[k] for k in spacing_dict])
    output = np.quantile(all_spacings, quantile, axis=0).tolist()
    print("Inferred spacing:", output)
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
    moving: sitk.Image, target: sitk.Image
) -> sitk.Image:
    interpolation = sitk.sitkNearestNeighbor
    output = sitk.Resample(
        moving,
        target.GetSize(),
        sitk.Transform(),
        interpolation,
        target.GetOrigin(),
        target.GetSpacing(),
        target.GetDirection(),
        0,
        moving.GetPixelID(),
    )
    return output


def crop_image(sitk_image: sitk.Image, output_size: list[int]):
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
