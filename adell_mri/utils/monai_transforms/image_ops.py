import einops
import monai
import numpy as np
import torch
import torch.nn.functional as F

from adell_mri.custom_types import TensorDict, NDArrayOrTensor


class FastResample(monai.transforms.Transform):
    """
    Does what monai.transforms.Spacingd does but fast by getting rid of
    some unnecessary calculations.
    """

    def __init__(self, target: list[float], keys=list[str], mode=list[str]):
        """
        Args:
            target (list[float]): _description_
            keys (_type_, optional): _description_. Defaults to list[str].
            mode (_type_, optional): _description_. Defaults to list[str].
        """
        self.target = np.array(target, np.float64)
        self.keys = keys
        self.mode = mode

        self.interpolation_modes = {k: m for k, m in zip(self.keys, self.mode)}

    def ensure_tensor(self, x):
        x = torch.as_tensor(x).unsqueeze(1)
        return x

    def __call__(self, X):
        for k in self.keys:
            meta = X[k + "_meta_dict"]
            if "spacing" in meta:
                spacing = meta["spacing"]
                # redefine spacing
                meta["spacing"] = self.target
            else:
                spacing = meta["pixdim"][1:4]
                meta["pixdim"][1:4] = self.target
            spacing = np.array(spacing, np.float64)
            spacing_ratio = spacing / self.target
            output_shape = np.round(
                np.multiply(
                    spacing_ratio, np.array(X[k].shape[1:], dtype=np.float64)
                )
            ).astype(np.int64)
            intp = self.interpolation_modes[k]
            X[k] = F.interpolate(
                self.ensure_tensor(X[k]), size=output_shape.tolist(), mode=intp
            ).squeeze(1)
        return X


class SlicesToFirst(monai.transforms.Transform):
    """
    Returns the slices as the first spatial dimension.
    """

    def __init__(self, keys: list[str]):
        """
        Args:
            keys (list[str]): keys for which slices will be retrieved.
        """
        self.keys = keys

    def __call__(self, X):
        for k in self.keys:
            X[k] = X[k].swapaxes(0, -1)
        return X


class Index(monai.transforms.Transform):
    """
    Indexes tensors in a dictionary at a given dimension `axis`.
    Useful for datasets such as the MONAI Medical Decathlon, where
    arrays are composed of more than one modality and we only care about
    a specific modality.
    """

    def __init__(self, keys: list[str], idxs: list[int], axis: int):
        """
        Args:
            keys (list[str]): list of keys to tensors which will be indexed.
            idxs (list[int]): indices that will be retrieved.
            axis (int): axis at which indices will be retrieved.
        """
        self.keys = keys
        self.idxs = idxs
        self.axis = axis

    def __call__(self, X):
        for k in self.keys:
            if self.idxs is not None:
                X[k] = np.take(X[k], self.idxs, self.axis)
        return X


class Dropout(monai.transforms.Transform):
    """
    Drops specific channels from a tensor.
    """

    def __init__(self, channel: int, dim: int = 0):
        """
        Args:
            channel (int): channel to be dropped.
            dim (int, optional): dimension where channel is to be dropped.
                Defaults to 0.
        """
        self.channel = channel
        self.dim = dim

    def __call__(self, X: torch.Tensor):
        keep_idx = torch.ones(X.shape[self.dim]).to(X.device)
        keep_idx[self.channel] = 0.0
        reshape_sh = torch.ones(len(X.shape))
        reshape_sh[self.dim] = -1
        keep_idx = keep_idx.reshape(reshape_sh)
        return X * keep_idx


class Dropoutd(monai.transforms.Transform):
    """
    Dictionary version of Dropout.
    """

    def __init__(
        self,
        keys: str | list[str],
        channel: int | list[int],
        dim: int | list[int] = 0,
    ):
        """
        Args:
            keys (list[str] | str): keys of the dictionary to apply dropout.
            channel (int | list[int]): channels to be dropped (if list
                has to be one per key).
            dim (int | list[int], optional): dimension where channel is
                to be dropped (if list has to be one per key). Defaults to 0.
        """
        self.keys = keys
        self.channel = channel
        self.dim = dim

        if isinstance(self.channel, int):
            self.channel = [self.channel for _ in self.keys]

        if isinstance(self.dim, int):
            self.dim = [self.dim for _ in self.keys]

        self.transforms = {}
        for k, c, d in zip(self.keys, self.channel, self.dim):
            self.transforms[k] = Dropout(c, d)

    def __call__(self, X):
        for k in zip(self.keys):
            X = self.transforms[k](X[k])
        return X


class EinopsRearrange(monai.transforms.Transform):
    """
    Convenience MONAI transform to apply einops patterns to inputs.
    """

    def __init__(self, pattern: str):
        """
        Args:
            pattern (str): einops pattern.
        """
        self.pattern = pattern

    def __call__(self, X: torch.Tensor) -> torch.Tensor:
        return einops.rearrange(X, self.pattern)


class EinopsRearranged(monai.transforms.Transform):
    """
    Convenience MONAI dict transform to apply einops patterns to inputs.
    """

    def __init__(self, keys: list[str], pattern: str):
        """
        Args:
            keys (str): keys to apply einops patterns to.
            pattern (str): einops pattern.
        """
        self.keys = keys
        self.pattern = pattern

        if isinstance(self.keys, str):
            self.keys = [self.keys]

        if isinstance(self.pattern, str):
            self.trs = [EinopsRearrange(self.pattern) for _ in keys]
        else:
            self.trs = [EinopsRearrange(p) for p in self.pattern]

    def __call__(self, X: TensorDict) -> TensorDict:
        for k, tr in zip(self.keys, self.trs):
            X[k] = tr(X[k])
        return X


class SampleChannelDim(monai.transforms.Transform):
    """
    Randomly selects in_channels channels from a multi-channel dim.
    """

    def __init__(self, in_channels: int, channel_dim: int = 0):
        """
        Args:
            in_channels (int): number of channels to be randomly selected.
            channel_dim (int, optional): dimension for the channels. Defaults
                to 0.
        """
        self.in_channels = in_channels
        self.channel_dim = channel_dim

    def __call__(self, X):
        if X.shape[self.channel_dim] > self.in_channels:
            samples = np.random.choice(
                X.shape[self.channel_dim], self.in_channels, replace=False
            )
            X = X.index_select(self.channel_dim, torch.as_tensor(samples))
        return X


class SampleChannelDimd(monai.transforms.MapTransform):
    """
    Dictionary version of SampleChannelDim.
    """

    def __init__(self, keys: list[str], in_channels: int, channel_dim: int = 0):
        """
        Args:
            keys (list[str]): Keys to apply channel sampling to.
            in_channels (int): number of channels to sample.
            channel_dim (int, optional): dimension for the channels. Defaults
                to 0.
        """
        self.keys = keys
        self.in_channels = in_channels
        self.channel_dim = channel_dim

        self.transform = SampleChannelDim(self.in_channels, self.channel_dim)

    def __call__(self, X: dict[str, torch.Tensor]):
        for k in self.keys:
            X[k] = self.transform(X[k])
        return X


class GetAllCrops(monai.transforms.Transform):
    """
    Works similarly to RandCropByPosNegLabeld but returns all the crops in
    a volume or image. Pads the image such that most of the volume is contained
    in the crops (this skips padding dimensions where the padding is greater
    than half of the crop size).
    """

    def __init__(self, size: tuple[int, int] | tuple[int, int, int]):
        """
        Args:
            size (tuple[int, int] | tuple[int, int, int]): crop size.
        """
        self.size = size
        self.ndim = len(size)

    def get_pad_size(self, sh):
        remainder = [
            (y - (x % y)) if x > y else 0 for x, y in zip(sh, self.size)
        ]
        remainder = [
            x if x < (y // 2) else 0 for x, y in zip(remainder, self.size)
        ]
        remainder = [(x // 2, x - x // 2) for x in remainder]
        pad_size = [(0, 0), *remainder]
        return pad_size

    def pad(self, X):
        # pad
        pad_size = self.get_pad_size(X.shape)
        X = np.pad(X, pad_size, "constant", constant_values=0)
        return X

    def get_all_crops_2d(self, X: torch.Tensor) -> torch.Tensor:
        sh = X.shape[1:]
        X = self.pad(X)
        for i_1 in range(0, sh[0], self.size[0]):
            for j_1 in range(0, sh[1], self.size[1]):
                i_2 = i_1 + self.size[0]
                j_2 = j_1 + self.size[1]
                if all([i_2 < (sh[0] + 1), j_2 < (sh[1] + 1)]):
                    yield X[:, i_1:i_2, j_1:j_2]

    def get_all_crops_3d(self, X: torch.Tensor) -> torch.Tensor:
        sh = [x for x in X.shape[1:]]
        X = self.pad(X)
        for i_1 in range(0, sh[0], self.size[0]):
            for j_1 in range(0, sh[1], self.size[1]):
                for k_1 in range(0, sh[2], self.size[2]):
                    i_2 = i_1 + self.size[0]
                    j_2 = j_1 + self.size[1]
                    k_2 = k_1 + self.size[2]
                    if all(
                        [
                            i_2 < (sh[0] + 1),
                            j_2 < (sh[1] + 1),
                            k_2 < (sh[2] + 1),
                        ]
                    ):
                        yield X[:, i_1:i_2, j_1:j_2, k_1:k_2]

    def get_all_crops(self, X: torch.Tensor) -> torch.Tensor:
        if self.ndim == 2:
            yield from self.get_all_crops_2d(X)
        if self.ndim == 3:
            yield from self.get_all_crops_3d(X)

    def __call__(self, X: torch.Tensor) -> list[torch.Tensor]:
        if self.size is not None:
            X = [x for x in self.get_all_crops(X)]
        return X


class GetAllCropsd(monai.transforms.MapTransform):
    """
    Dictionary version of GetAllCrops.
    """

    def __init__(
        self,
        keys: list[str],
        size: tuple[int, int] | tuple[int, int, int],
    ):
        """
        Args:
            keys (list[str]): keys to crop.
            size (tuple[int, int] | tuple[int, int, int]): crop size.
        """
        self.keys = keys
        self.size = size

        self.tr = GetAllCrops(self.size)

    def __call__(
        self, X: dict[str, torch.Tensor]
    ) -> list[dict[str, torch.Tensor]]:
        output_crops = {k: X[k] for k in X}
        outputs = []
        for k in self.keys:
            output_crops[k] = list(self.tr(X[k]))

        for elements in zip(*[output_crops[k] for k in self.keys]):
            output = {k: element for k, element in zip(self.keys, elements)}
            # make sure dictionary contains everything else
            for k in X:
                if k not in output:
                    output[k] = X[k]
            outputs.append(output)
        return outputs


class AdjustSizesd(monai.transforms.MapTransform):
    """
    Pads a dict of tensors such that they all have the size of the largest
    dimensions. Alternatively, crops to the smallest size.
    """

    def __init__(self, keys: list[str], ndim: int = 3, mode: str = "pad"):
        """
        Args:
            keys (list[str]): keys to adjust sizes for.
            ndim (int, optional): number of dimensions. Defaults to 3.
            mode (str, optional): adjustment mode. Can be either pad or crop.
                Defaults to "pad".
        """
        self.keys = keys
        self.ndim = ndim
        self.mode = mode

    def pad_torch(self, X: torch.Tensor, out_size: list[int]) -> torch.Tensor:
        sh = X.shape
        pad_dim = [i - j for i, j in zip(out_size, sh[-self.ndim :])]
        a = [x // 2 for x in pad_dim]
        b = [x - y for x, y in zip(pad_dim, a)]
        pad_size = []
        for i, j in zip(reversed(a), reversed(b)):
            pad_size.extend([i, j])
        return torch.nn.functional.pad(X, pad_size)

    def pad_numpy(self, X: np.ndarray, out_size: list[int]) -> np.ndarray:
        sh = X.shape
        additional_dims = len(sh) - self.ndim
        pad_dim = [i - j for i, j in zip(out_size, sh[-self.ndim :])]
        a = [x // 2 for x in pad_dim]
        b = [x - y for x, y in zip(pad_dim, a)]
        pad_size = [
            *[(0, 0) for _ in range(additional_dims)],
            *[(i, j) for i, j in zip(a, b)],
        ]
        return np.pad(X, pad_size)

    def pad(self, X: NDArrayOrTensor, out_size: list[int]) -> NDArrayOrTensor:
        if isinstance(X, torch.Tensor):
            return self.pad_torch(X, out_size)
        else:
            return self.pad_numpy(X, out_size)

    def crop(self, X: NDArrayOrTensor, out_size: list[int]) -> NDArrayOrTensor:
        sh = X.shape
        crop_dim = [i - j for i, j in zip(sh[-self.ndim :], out_size)]
        a = [x // 2 for x in crop_dim]
        b = [x + y for x, y in zip(a, out_size)]
        if self.ndim == 2:
            return X[..., a[0] : b[0], a[1] : b[1]]
        if self.ndim == 3:
            return X[..., a[0] : b[0], a[1] : b[1], a[2] : b[2]]

    def __call__(self, X: dict[str, torch.Tensor]):
        if self.mode == "pad":
            spatial_size = np.max(
                np.array([X[k].shape[-self.ndim :] for k in self.keys]), 0
            )
            fn = self.pad
        else:
            spatial_size = np.min(
                np.array([X[k].shape[-self.ndim :] for k in self.keys]), 0
            )
            fn = self.crop
        for k in X:
            if k in self.keys:
                X[k] = fn(torch.as_tensor(X[k]), spatial_size)
        return X
