import numpy as np
import monai

from ..custom_types import List, Tuple, Dict, Union
from typing import Sequence

n_dim = 3

generic_augments = [
    "gaussian_noise",
    "shift_intensity",
    "scale_intensity",
    "contrast",
    "gaussian_smooth_x",
    "gaussian_smooth_y",
    "gaussian_smooth_z",
    "gaussian_sharpen_x",
    "gaussian_sharpen_y",
    "gaussian_sharpen_z",
    "coarse_dropout",
]
mri_specific_augments = ["rbf", "gibbs_noise", "spike_noise", "rician_noise"]
spatial_augments = [
    "rotate_x",
    "rotate_y",
    "rotate_z",
    "translate_x",
    "translate_y",
    "translate_z",
    "shear_x",
    "shear_y",
    "shear_z",
    "scale_x",
    "scale_y",
    "scale_z",
]

AUG_DICT = {
    "gaussian_noise": monai.transforms.RandGaussianNoise,
    "shift_intensity": monai.transforms.RandShiftIntensity,
    "scale_intensity": monai.transforms.RandScaleIntensity,
    "rbf": monai.transforms.RandBiasField,
    "contrast": monai.transforms.RandAdjustContrast,
    "gaussian_smooth_x": monai.transforms.RandGaussianSmooth,
    "gaussian_smooth_y": monai.transforms.RandGaussianSmooth,
    "gaussian_smooth_z": monai.transforms.RandGaussianSmooth,
    "gaussian_sharpen_x": monai.transforms.RandGaussianSharpen,
    "gaussian_sharpen_y": monai.transforms.RandGaussianSharpen,
    "gaussian_sharpen_z": monai.transforms.RandGaussianSharpen,
    "coarse_dropout": monai.transforms.RandCoarseDropout,
    "gibbs_noise": monai.transforms.RandGibbsNoise,
    "spike_noise": monai.transforms.RandKSpaceSpikeNoise,
    "rician_noise": monai.transforms.RandRicianNoise,
    "rotate_x": monai.transforms.RandAffine,
    "rotate_y": monai.transforms.RandAffine,
    "rotate_z": monai.transforms.RandAffine,
    "translate_x": monai.transforms.RandAffine,
    "translate_y": monai.transforms.RandAffine,
    "translate_z": monai.transforms.RandAffine,
    "shear_x": monai.transforms.RandAffine,
    "shear_y": monai.transforms.RandAffine,
    "shear_z": monai.transforms.RandAffine,
    "scale_x": monai.transforms.RandAffine,
    "scale_y": monai.transforms.RandAffine,
    "scale_z": monai.transforms.RandAffine,
}

AUG_DICT_DICT = {
    "gaussian_noise": monai.transforms.RandGaussianNoised,
    "shift_intensity": monai.transforms.RandShiftIntensityd,
    "scale_intensity": monai.transforms.RandScaleIntensityd,
    "rbf": monai.transforms.RandBiasFieldd,
    "contrast": monai.transforms.RandAdjustContrastd,
    "gaussian_smooth_x": monai.transforms.RandGaussianSmoothd,
    "gaussian_smooth_y": monai.transforms.RandGaussianSmoothd,
    "gaussian_smooth_z": monai.transforms.RandGaussianSmoothd,
    "gaussian_sharpen_x": monai.transforms.RandGaussianSharpend,
    "gaussian_sharpen_y": monai.transforms.RandGaussianSharpend,
    "gaussian_sharpen_z": monai.transforms.RandGaussianSharpend,
    "coarse_dropout": monai.transforms.RandCoarseDropoutd,
    "gibbs_noise": monai.transforms.RandGibbsNoised,
    "spike_noise": monai.transforms.RandKSpaceSpikeNoised,
    "rician_noise": monai.transforms.RandRicianNoised,
    "rotate_x": monai.transforms.RandAffined,
    "rotate_y": monai.transforms.RandAffined,
    "rotate_z": monai.transforms.RandAffined,
    "translate_x": monai.transforms.RandAffined,
    "translate_y": monai.transforms.RandAffined,
    "translate_z": monai.transforms.RandAffined,
    "shear_x": monai.transforms.RandAffined,
    "shear_y": monai.transforms.RandAffined,
    "shear_z": monai.transforms.RandAffined,
    "scale_x": monai.transforms.RandAffined,
    "scale_y": monai.transforms.RandAffined,
    "scale_z": monai.transforms.RandAffined,
}

AUG_PARAM_DICT = {
    "gaussian_noise": {"std": 1},
    "shift_intensity": {"offsets": 0.5},
    "scale_intensity": {"factors": 0.5},
    "rbf": {"coeff_range": 0.3},
    "contrast": {"gamma": 3},
    "gaussian_smooth_x": {"sigma_x": 0.3},
    "gaussian_smooth_y": {"sigma_y": 0.3},
    "gaussian_smooth_z": {"sigma_z": 0.3},
    "gaussian_sharpen_x": {"sigma1_x": 0.3},
    "gaussian_sharpen_y": {"sigma1_y": 0.3},
    "gaussian_sharpen_z": {"sigma1_z": 0.3},
    "gibbs_noise": {"alpha": 1.0},
    "spike_noise": {"intensity_range": 0.5},
    "rician_noise": {"std": 0.2},
    "coarse_dropout": {"holes": 16},
}
for i, c in enumerate(["x", "y", "z"]):
    # these have to be updated
    t = 30 if c != "z" else 5
    a = np.pi / 6 if c != "z" else np.pi / 16
    AUG_PARAM_DICT["rotate_" + c] = {"rotate_range": a}
    AUG_PARAM_DICT["translate_" + c] = {"translate_range": t}
    AUG_PARAM_DICT["shear_" + c] = {"shear_range": 0.5}
    AUG_PARAM_DICT["scale_" + c] = {"scale_range": 0.3}

AUG_PARAM_CORRECTION = {
    # to have a single parameter for each transform I opt to transform
    # each value to the correct transform parameter
    # add 0.51
    "contrast": lambda x: x + 0.51,
    # tuple between 0 and value
    "rbf": lambda x: (0, x),
    "gaussian_smooth_x": lambda x: (0, x),
    "gaussian_smooth_y": lambda x: (0, x),
    "gaussian_smooth_z": lambda x: (0, x),
    "gaussian_sharpen_x": lambda x: (0, x),
    "gaussian_sharpen_y": lambda x: (0, x),
    "gaussian_sharpen_z": lambda x: (0, x),
    "gibbs_noise": lambda x: (0, x),
    "spike_noise": lambda x: (0, x),
    # tuple at index n, value between -value,value
    "rotate_x": lambda x: ((-x, x), 0, 0),
    "rotate_y": lambda x: (0, (-x, x), 0),
    "rotate_z": lambda x: (0, 0, (-x, x)),
    "translate_x": lambda x: ((-x, x), 0, 0),
    "translate_y": lambda x: (0, (-x, x), 0),
    "translate_z": lambda x: (0, 0, (-x, x)),
    # tuple at index n, value between 1-value,1+value
    "shear_x": lambda x: ((1 - x, 1 + x), 0, 0),
    "shear_y": lambda x: (0, (1 - x, 1 + x), 0),
    "shear_z": lambda x: (0, 0, (1 - x, 1 + x)),
    "scale_x": lambda x: ((1 - x, 1 + x), 0, 0),
    "scale_y": lambda x: (0, (1 - x, 1 + x), 0),
    "scale_z": lambda x: (0, 0, (1 - x, 1 + x)),
}


def get_transform_d(
    keys: List[str],
    transform_str: str,
    aug_param_dict: dict = AUG_PARAM_DICT,
    mask_keys: List[str] = [],
    dropout_size: Tuple[int] = (32, 32, 2),
):
    transform = AUG_DICT_DICT[transform_str]
    params = aug_param_dict[transform_str]
    if transform_str in AUG_PARAM_CORRECTION:
        for k in params:
            params[k] = AUG_PARAM_CORRECTION[transform_str](params[k])
    other_args = {}
    if transform_str in spatial_augments:
        mode = ["bilinear" if k not in mask_keys else "nearest" for k in keys]
        other_args["mode"] = mode
        other_args["padding_mode"] = "zeros"
    if transform_str == "coarse_dropout":
        if "holes" in params:
            params["holes"] = int(params["holes"])
        other_args["spatial_size"] = dropout_size
    return transform(keys, **params, **other_args, prob=1.0)


class AugmentationWorkhorsed(monai.transforms.RandomizableTransform):
    """
    Coordinates a dictionary of MONAI transforms. Based on the TrivialAugment
    method.
    """

    def __init__(
        self,
        augmentations: List[str],
        keys: List[str] = None,
        mask_keys: List[str] = [],
        max_mult: float = 1.0,
        N: int = 5,
        aug_param_dict: Dict[
            str, Dict[str, Union[int, float]]
        ] = AUG_PARAM_DICT,
        dropout_size: Sequence[int] = (32, 32, 2),
    ):
        """
        Args:
            augmentations (List[str]): list of augmentations to apply.
                Supported strings are the keys to AUG_DICT_DICT.
            keys (List[str], optional): keys to which the augmentations will be
                applied. Defaults to None.
            mask_keys (List[str], optional): keys corresponding to masks to
                which the augmentations will be applied. Defaults to [].
            max_mult (float, optional): the maximum multiplier for
                augmentations. Defaults to 1.0.
            N (int, optional): number of augmentations per image. Defaults to
                5.
            aug_param_dict (Dict[ str, Dict[str, Union[int, float]] ],
                optional): parameter dictionary. Defaults to AUG_PARAM_DICT.
            dropout_size (Sequence[int], optional): size for coarse dropout.
                Defaults to (32, 32, 2).
        """
        super().__init__()
        self.augmentations = augmentations
        self.keys = keys
        self.mask_keys = mask_keys
        self.max_mult = max_mult
        self.N = N
        self.aug_param_dict = aug_param_dict
        self.dropout_size = dropout_size

        self.param_dict = {
            k: {
                kk: self.aug_param_dict[k][kk] * self.max_mult
                for kk in self.aug_param_dict[k]
            }
            for k in self.augmentations
        }

        self.transforms = {
            k: get_transform_d(
                keys,
                k,
                self.param_dict,
                mask_keys,
                dropout_size=self.dropout_size,
            )
            for k in self.param_dict
        }

    def __call__(self, X):
        t_list = np.random.choice(self.augmentations, self.N, replace=False)
        for t in t_list:
            X = self.transforms[t](X)
        return X
