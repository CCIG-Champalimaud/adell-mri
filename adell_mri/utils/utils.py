import os
from collections import OrderedDict
from copy import deepcopy
from glob import glob

import monai
import numpy as np
import SimpleITK as sitk
import torch
import torch.nn.functional as F

from adell_mri.custom_types import (
    DatasetDict,
    FloatOrTensor,
    SizeDict,
    SpacingDict,
    TensorIterable,
    TensorList,
)
from adell_mri.modules.segmentation.losses import (
    binary_cross_entropy,
    binary_focal_loss,
    binary_focal_tversky_loss,
    binary_generalized_dice_loss,
    cat_cross_entropy,
    combo_loss,
    hybrid_focal_loss,
    mc_combo_loss,
    mc_focal_loss,
    mc_focal_tversky_loss,
    mc_generalized_dice_loss,
    mc_hybrid_focal_loss,
    mc_unified_focal_loss,
    unified_focal_loss,
    weighted_mse,
)

loss_factory = {
    "binary": {
        "cross_entropy": binary_cross_entropy,
        "focal": binary_focal_loss,
        "dice": binary_generalized_dice_loss,
        "tversky_focal": binary_focal_tversky_loss,
        "combo": combo_loss,
        "hybrid_focal": hybrid_focal_loss,
        "unified_focal": unified_focal_loss,
    },
    "categorical": {
        "cross_entropy": cat_cross_entropy,
        "focal": mc_focal_loss,
        "dice": mc_generalized_dice_loss,
        "tversky_focal": mc_focal_tversky_loss,
        "combo": mc_combo_loss,
        "hybrid_focal": mc_hybrid_focal_loss,
        "unified_focal": mc_unified_focal_loss,
    },
    "regression": {"mse": F.mse_loss, "weighted_mse": weighted_mse},
}


def split(x: torch.Tensor, n_splits: int, dim: int) -> TensorIterable:
    """
    Splits a tensor into n_splits tensors along dimension dim.

    Args:
        x: Tensor to split.
        n_splits: Number of splits.
        dim: Dimension along which to split the tensor.

    Returns:
        A tuple containing the splits.
    """
    size = int(x.shape[dim] // n_splits)
    return torch.split(x, size, dim)


def get_prostatex_path_dictionary(base_path: str) -> DatasetDict:
    """
    Builds a path dictionary (a dictionary where each key is a patient
    ID and each value is a dictionary containing a modality-to-MRI scan path
    mapping). Assumes that the folders "T2WAx", "DWI", "aggregated-labels-gland"
    and "aggregated-labels-lesion" in `base_path`.

    Args:
        base_path (str): path containing the "T2WAx", "DWI",
        "aggregated-labels-gland" and "aggregated-labels-lesion" folders.

    Returns:
        DatasetDict: a path dictionary.
    """
    paths = {
        "t2w": os.path.join(base_path, "T2WAx"),
        "dwi": os.path.join(base_path, "DWI"),
        "gland": os.path.join(base_path, "aggregated-labels-gland"),
        "lesion": os.path.join(base_path, "aggregated-labels-lesion"),
    }

    path_dictionary = {}
    for image_path in glob(os.path.join(paths["t2w"], "*T2WAx1*gz")):
        f_name = os.path.split(image_path)[-1]
        patient_id = f_name.split("_")[0]
        path_dictionary[patient_id] = {"T2WAx": image_path}

    for image_path in glob(os.path.join(paths["dwi"], "*gz")):
        f_name = os.path.split(image_path)[-1]
        patient_id = f_name.split("_")[0]
        # only interested in cases with both data types
        if patient_id in path_dictionary:
            path_dictionary[patient_id]["DWI"] = image_path

    for image_path in glob(os.path.join(paths["gland"], "*gz")):
        f_name = os.path.split(image_path)[-1]
        patient_id = f_name.split("_")[0]
        mod = f_name.split("_")[1]
        if patient_id in path_dictionary:
            m = "{}_gland_segmentations".format(mod)
            path_dictionary[patient_id][m] = image_path

    for image_path in glob(os.path.join(paths["lesion"], "*gz")):
        f_name = os.path.split(image_path)[-1]
        patient_id = f_name.split("_")[0]
        mod = f_name.split("_")[1]
        if patient_id in path_dictionary:
            m = "{}_lesion_segmentations".format(mod)
            path_dictionary[patient_id][m] = image_path

    return path_dictionary


def get_size_spacing_dict(
    path_dictionary: DatasetDict, keys: list[str]
) -> tuple[SizeDict, SpacingDict]:
    """
    Retrieves the scan sizes and pixel spacings from a path dictionary.

    Args:
        path_dictionary (DatasetDict): a path dictionary (see
        `get_prostatex_path_dictionary` for details).
        keys (list[str]): modality keys that should be considered in the
        path dictionary.

    Returns:
        size_dict (SizeDict): a dictionary with `keys` as keys and a list
        of scan sizes (2 or 3 int) as values.
        spacing_dict (SpacingDict): a dictionary with `keys` as keys and a
        of spacing sizes (2 or 3 floats) as values.
    """

    size_dict = {k: [] for k in keys}
    spacing_dict = {k: [] for k in keys}
    for pid in path_dictionary:
        for k in keys:
            if k in path_dictionary[pid]:
                X = sitk.ReadImage(path_dictionary[pid][k])
                size_dict[k].append(X.GetSize())
                spacing_dict[k].append(X.GetSpacing())
    return size_dict, spacing_dict


def get_loss_param_dict(
    loss_key: str,
    **kwargs,
) -> dict[str, dict[str, FloatOrTensor]]:
    """
    Constructs a keyword dictionary that can be used with the losses in
    `losses.py`.

    Args:
        loss_key (str): key corresponding to loss name.
        kwargs (optional): keyword arguments for the loss to which the weights
            will be appended.

    Returns:
        dict[str,dict[str,Union[float,torch.Tensor]]]: dictionary where each
        key refers to a loss function and each value is keyword dictionary for
        different losses.
    """

    def invert_weights(w: torch.Tensor) -> torch.Tensor:
        """
        Inverts weights if necessary. Used only for the Tversky focal loss.
        If W >= 1, then the inversion returns 1. If W < 0, the inversion
        returns 1 - W.

        Args:
            w (torch.Tensor): weight tensor.

        Returns:
            torch.Tensor: inverted W.
        """
        if torch.any(w >= 1):
            return torch.ones_like(w)
        else:
            return torch.ones_like(w) - w

    kwargs = {k: torch.as_tensor(kwargs[k]) for k in kwargs}
    if loss_key in ["focal", "focal_alt", "weighted_mse"]:
        if "weight" in kwargs:
            weights = kwargs["weight"]
            del kwargs["weight"]
            return {"alpha": weights, **kwargs}
        else:
            return kwargs
    elif loss_key in ["cross_entropy", "dice", "combo", "unified_focal"]:
        if "weight" in kwargs:
            weights = kwargs["weight"]
            del kwargs["weight"]
            return {"weight": weights, **kwargs}
        else:
            return kwargs
    elif loss_key in "tversky_focal":
        if "weight" in kwargs:
            inverted_weights = invert_weights(weights)
            s = weights + inverted_weights
            weights_tv = weights / s
            inverted_weights_tv = inverted_weights / s
            del kwargs["weight"]
            return {"alpha": inverted_weights_tv, "beta": weights_tv, **kwargs}
        else:
            return kwargs
    elif loss_key == "mse":
        return kwargs
    else:
        raise NotImplementedError(
            f"loss_key {loss_key} not in available loss_keys"
        )


def unpack_crops(X: list[TensorIterable]) -> TensorList:
    """
    Unpacks a list of nested tensor list into a single list.

    Args:
        X (list[TensorIterable]): Outer list of cropped tensor iterables.

    Returns:
        TensorIterable: Flattened iterable of all innermost elements.
    """
    output = []
    for x in X:
        for xx in x:
            output.append(xx)
    return output


def collate_last_slice(X: list[TensorIterable]) -> TensorIterable:
    """
    Collates the last slice of each tensor in a list of tensor interables along
    the batch dimension. So, given a TensorIterable of type list, where each
    tensor has shape [b, c, h, w, d], the output will be a TensorIterable of
    type list with shape [b * d, c, h, w].

    Args:
        X (list[TensorIterable]): list of tensor iterables, each containing
            tensors of the same shape.

    Returns:
        TensorIterable: object of the same type as TensorIterable with tensors
            with the last dimension contatenated along the batch dimension.
    """

    def swap(x):
        """
        Swaps the channel dimension of a tensor to the first position.

        Args:
            x (torch.Tensor): Input tensor of shape (N, C, H, W).

        Returns:
            torch.Tensor: Output tensor of shape (C, N, H, W).
        """
        out = x.permute(0, 3, 1, 2)
        return out

    def swap_cat(x):
        """
        Swaps the channel dimension of each tensor in x to the first
        position and concatenates them along the batch dimension.

        Args:
            x (list[torch.Tensor]): list of input tensors, each with shape
                (N, C, H, W).

        Returns:
            torch.Tensor: Output tensor with shape (N*C, H, W)
        """
        try:
            o = torch.cat([swap(y) for y in x])
            return o
        except Exception:
            pass

    example = X[0]
    if isinstance(example, list):
        output = []
        for elements in zip(*X):
            output.append(swap_cat(elements))
    elif isinstance(example, dict):
        keys = list(example.keys())
        output = {}
        for k in keys:
            elements = [x[k] if k in x else None for x in X]
            output[k] = swap_cat(elements)
    return output


def safe_collate(X: list[TensorIterable]) -> list[TensorIterable]:
    """
    Similar to the default collate but going only one level deep and
    returning a list if shapes are incompatible (helpful to return bounding
    boxes).

    Args:
        X (list[TensorIterable]): a list of lists or dicts of tensors.

    Returns:
        list[TensorIterable]: a list or dict of tensors (depending on the
        input).
    """

    def cat(x):
        """
        Concatenates a list of tensors or tensor-like objects into a single
        tensor.

        Args:
            x (list[torch.Tensor]): A list of tensors or tensor-like objects
                to concatenate.

        Returns:
            torch.Tensor: The concatenated tensor, or the original list if
                concatenation fails.
        """

        try:
            x = [torch.as_tensor(y) for y in x]
        except Exception:
            return x
        try:
            return torch.stack(x)
        except Exception:
            return x

    example = X[0]
    if isinstance(example, list):
        output = []
        for elements in zip(*X):
            output.append(cat(elements))
    elif isinstance(example, dict):
        keys = list(example.keys())
        output = {}
        for k in keys:
            elements = []
            for x in X:
                if k in x:
                    elements.append(x[k])
                else:
                    elements.append(None)
            output[k] = cat(elements)
    return output


def safe_collate_crops(X: list[list[TensorIterable]]) -> list[TensorIterable]:
    """
    Similar to safe_collate but handles output from MONAI cropping
    functions.

    Args:
        X (list[list[TensorIterable]]): a list of lists or dicts of tensors.

    Returns:
        list[TensorIterable]: a list or dict of tensors (depending on the
        input).
    """
    X = unpack_crops(X)
    return safe_collate(X)


def load_anchors(path: str) -> np.ndarray:
    """Loads anchor boxes from a CSV file.

    Args:
        path (str): Path to the CSV file containing anchor boxes.

    Returns:
        np.ndarray: A numpy array containing the anchor boxes.
    """
    with open(path, "r") as o:
        lines = o.readlines()
    lines = [[float(y) for y in x.strip().split(",")] for x in lines]
    return np.array(lines)


class ExponentialMovingAverage(torch.nn.Module):
    """
    Exponential moving average for model weights. The weight-averaged
    model is kept as `self.shadow` and each iteration of self.update leads
    to weight updating. This implementation is heavily based on that
    available in https://www.zijianhu.com/post/pytorch/ema/.

    Essentially, self.update(model) is called, a shadow version of the
    model (i.e. self.shadow) is updated using the exponential moving
    average formula such that $v'=(1-decay)*(v_{shadow}-v)$, where
    $v$ is the new parameter value, $v'$ is the updated value and
    $v_{shadow}$ is the exponential moving average value (i.e. the shadow).
    """

    def __init__(
        self, decay: float, final_decay: float | None = None, n_steps=None
    ):
        """
        Args:
            decay (float): decay for the exponential moving average.
            final_decay (float, optional): final value for decay. Defaults to
                None (same as initial decay).
            n_steps (float, optional): number of updates until `decay` becomes
                `final_decay` with linear scaling. Defaults to None.
        """
        super().__init__()
        self.decay = decay
        self.final_decay = final_decay
        self.n_steps = n_steps
        self.shadow = None
        self.step = 0

        if self.final_decay is None:
            self.slope = None
            self.intercept = None
        else:
            self.slope = (self.final_decay - self.decay) / self.n_steps
            self.intercept = self.decay

    def set_requires_grad_false(self, model: torch.nn.Module):
        """
        Sets requires_grad attribute of all parameters to False in a torch
        Module.

        Args:
            model (torch.Tensor): torch module.
        """
        for k, p in model.named_parameters():
            if p.requires_grad is True:
                p.requires_grad = False

    @torch.no_grad()
    def update(self, model: torch.nn.Module, exclude_keys: list[str] = None):
        """
        Updates the shadow version of the model using an exponential moving
        average.

        Args:
            model (torch.nn.Module): torch module. Must have same parameters as
                self.shadow.
            exclude_keys (list[str]): excludes these keys from the update.
        """
        if self.shadow is None:
            # this effectively skips the first epoch
            self.shadow = deepcopy(model)
            self.shadow.training = False
            self.set_requires_grad_false(self.shadow)
        else:
            if exclude_keys is None:
                exclude_keys = []
            model_params = OrderedDict(model.named_parameters())
            shadow_params = OrderedDict(self.shadow.named_parameters())

            sd_model_shadow = set.difference(
                set(shadow_params.keys()), set(model_params.keys())
            )
            sd_shadow_model = set.difference(
                set(model_params.keys()), set(shadow_params.keys())
            )
            sd_shadow_model = [x for x in sd_shadow_model if "shadow" not in x]

            assert len(sd_model_shadow) == 0
            assert len(sd_shadow_model) == 0

            for name, param in model_params.items():
                if name in exclude_keys:
                    continue
                if "shadow" not in name:
                    shadow_params[name].sub_(
                        (1 - self.decay) * (shadow_params[name] - param)
                    )

            if self.final_decay:
                self.decay = self.step * self.slope + self.intercept
            if self.decay > 1.0:
                self.decay = 1.0
            self.step += 1

    def forward(self, *args, **kwargs):
        """
        Wrapper for the shadow model ``forward`` function.

        Returns:
            Output for ``self.shadow.forward`` with args and kwargs.
        """
        return self.shadow.forward(*args, **kwargs)

    def state_dict(self) -> dict[str, torch.Tensor]:
        """
        Wrapper for the shadow model ``state_dict`` function.

        Returns:
            dict[str, torch.Tensor]: returns the ``shadows``'s state dict.
        """
        return self.shadow.state_dict()

    def load_state_dict(self, state_dict: dict[str, torch.Tensor]):
        """
        Wrapper for the shadow model ``load_state_dict`` function.

        Args:
            state_dict (dict[str, torch.Tensor]): state dictionary.
        """
        self.shadow.load_state_dict(state_dict)


def return_classes(paths: str | list[str]) -> dict[str | int, str]:
    """
    Returns a dictionary with the unique values in the images and their counts.

    Args:
        paths (str | list[str]): Path or list of paths to images.

    Returns:
        dict: Dictionary with unique values as keys and counts as values.
    """
    if isinstance(paths, str):
        paths = [paths]
    out = {}
    for path in paths:
        image = monai.transforms.LoadImage()(path)
        un_cl, counts = np.unique(image, return_counts=True)
        for u, c in zip(un_cl, counts):
            if u not in out:
                out[u] = 0
            out[u] += c
    return out
