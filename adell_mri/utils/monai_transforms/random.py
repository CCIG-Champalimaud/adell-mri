from typing import Any, Optional

import monai
import numpy as np
import torch
import torch.nn.functional as F

from .image_ops import Dropout


class RandomAffined(monai.transforms.RandomizableTransform):
    """
    Reimplementation of the RandAffined transform in MONAI but works
    with differently sized inputs without forcing all inputs to the same
    shape.
    """

    def __init__(
        self,
        keys: list[str],
        spatial_sizes: list[tuple[int, int, int] | tuple[int, int]],
        mode: list[str],
        prob: float = 0.1,
        rotate_range: tuple[int, int, int] | tuple[int, int] = [0, 0, 0],
        shear_range: tuple[int, int, int] | tuple[int, int] = [0, 0, 0],
        translate_range: tuple[int, int, int] | tuple[int, int] = [
            0,
            0,
            0,
        ],
        scale_range: tuple[int, int, int] | tuple[int, int] = [0, 0, 0],
        device: "str" = "cpu",
        copy: bool = False,
    ):
        """
        Args:
            keys (list[str]): list of keys that will be randomly transformed.
            spatial_sizes (list[tuple[int,int,int] | tuple[int,int]]): dimension
                number for the inputs.
            mode (list[str]): interpolation modes. Must be the same size as
                keys.
            prob (float, optional): Probability that the transform will be
                applied. Defaults to 0.1.
            rotate_range (tuple[int,int,int] | tuple[int,int], optional):
                Rotation ranges. Defaults to [0,0,0].
            shear_range (tuple[int,int,int] | tuple[int,int], optional):
                Shear ranges. Defaults to [0,0,0].
            translate_range (tuple[int,int,int] | tuple[int,int], optional):
                Translation ranges. Defaults to [0,0,0].
            scale_range (tuple[int,int,int] | tuple[int,int], optional):
                Scale ranges. Defaults to [0,0,0].
            device (str, optional): device for computations. Defaults to "cpu".
            copy (bool, optional): whether dictionaries should be copied before
                applying the transforms. Defaults to False.
        """

        self.keys = keys
        self.spatial_sizes = [
            np.array(s, dtype=np.int32) for s in spatial_sizes
        ]
        self.mode = mode
        self.prob = prob
        self.rotate_range = np.array(rotate_range)
        self.shear_range = np.array(shear_range)
        self.translate_range = np.array(translate_range)
        self.scale_range = np.array(scale_range)
        self.device = device
        self.copy = copy

        self.affine_trans = {
            k: monai.transforms.Affine(
                spatial_size=s, mode=m, device=self.device
            )
            for k, s, m in zip(self.keys, self.spatial_sizes, self.mode)
        }

        self.get_translation_adjustment()

    def get_random_parameters(self):
        angle = self.R.uniform(-self.rotate_range, self.rotate_range)
        shear = self.R.uniform(-self.shear_range, self.shear_range)
        trans = self.R.uniform(-self.translate_range, self.translate_range)
        scale = self.R.uniform(1 - self.scale_range, 1 + self.scale_range)

        return angle, shear, trans, scale

    def get_translation_adjustment(self):
        # we have to adjust the translation to ensure that all inputs
        # do not become misaligned. to do this I assume that the first image
        # is the reference
        ref_size = self.spatial_sizes[0]
        self.trans_adj = {
            k: s / ref_size for k, s in zip(self.keys, self.spatial_sizes)
        }

    def randomize(self):
        angle, shear, trans, scale = self.get_random_parameters()
        for k in self.affine_trans:
            # we only need to update the affine grid
            self.affine_trans[k].affine_grid = monai.transforms.AffineGrid(
                rotate_params=list(angle),
                shear_params=list(shear),
                translate_params=list(trans * self.trans_adj[k]),
                scale_params=list(np.float32(scale)),
                device=self.device,
            )

    def __call__(self, data):
        self.randomize()
        if self.copy is True:
            data = data.copy()
        for k in self.keys:
            if self.R.uniform() < self.prob:
                transform = self.affine_trans[k]
                data[k], _ = transform(data[k])
        return data


class RandomSlices(monai.transforms.RandomizableTransform):
    """
    Randomly samples slices from a volume (assumes the slice dimension
    is the last dimension). A segmentation map (corresponding to
    `label_key`) is used to calculate the number of positive elements
    for each slice and these are used as sampling weights for the slice
    extraction.
    """

    def __init__(
        self,
        keys: list[str],
        label_key: list[str],
        n: int = 1,
        base: float = 0.001,
        seed=None,
    ):
        """
        Args:
            keys (list[str]): keys for which slices will be retrieved.
            label_key (list[str]): segmentation map that will be used to
            calculate sampling weights.
            n (int, optional): number of slices to be sampled. Defaults to 1.
            base (float, optional): minimum probability (ensures that slices
            with no positive cases are also sampled). Defaults to 0.01.
            seed (int, optional): seed for generator of slices. Makes it more
            deterministic.
        """
        self.keys = keys
        self.label_key = label_key
        self.n = n
        self.base = base
        self.seed = seed
        self.g = torch.Generator()
        if self.seed is not None:
            self.g.manual_seed(self.seed)
        self.is_multiclass = None
        self.M = 0

    def __call__(self, X):
        if self.label_key is not None:
            X_label = X[self.label_key]
            if isinstance(X_label, torch.Tensor) is False:
                X_label = torch.as_tensor(X_label)
            if self.is_multiclass is None:
                M = X_label.max()
                if M > 1:
                    X_label = F.one_hot(X_label, M + 1)
                    X_label = torch.squeeze(X_label.swapaxes(-1, 0))
                    X_label = X_label[1:]  # remove background dim
                    self.is_multiclass = True
                    self.M = M
                else:
                    self.is_multiclass = False
            elif self.is_multiclass is True:
                M = X_label.max()
                X_label = F.one_hot(X_label, M + 1)
                X_label = torch.squeeze(X_label.swapaxes(-1, 0))
                X_label = X_label[1:]  # remove background dim
            c = torch.flatten(X_label, start_dim=1, end_dim=-2)
            c_sum = c.sum(1)
            total_class = torch.unsqueeze(c_sum.sum(1), -1)
            total_class = torch.clamp(total_class, min=1)
            c_prop = c_sum / total_class + self.base
            slice_weight = c_prop.mean(0)
        else:
            slice_weight = torch.ones(X[self.keys[0]].shape[-1])
        slice_idxs = torch.multinomial(slice_weight, self.n, generator=self.g)
        for k in self.keys:
            X[k] = X[k][..., slice_idxs].swapaxes(0, -1)
        return X


class RandomDropout(monai.transforms.RandomizableTransform):
    """
    Randomly drops channels.
    """

    def __init__(self, dim: int = 0, prob: float = 0.1):
        """
        Args:
            dim (int, optional): dropout dimension. Defaults to 0.
            prob (float, optional): probability of dropout. Defaults to 0.1.
        """
        super().__init__(self, prob)
        self.prob = prob
        self.dim = dim

    def randomize(self, data: Optional[Any] = None) -> None:
        super().randomize(None)
        if not self._do_transform:
            return None
        self.channel = self.R.uniform(low=0, high=1)

    def __call__(self, img: torch.Tensor, randomize: bool = True):
        if randomize:
            self.randomize()
        if not self._do_transform:
            return img
        return Dropout(int(self.channel * img.shape[self.dim]), self.dim)(img)


class RandomDropoutd(
    monai.transforms.RandomizableTransform, monai.transforms.MapTransform
):
    """
    Dictionary version of RandomDropout.
    """

    def __init__(
        self,
        keys: str | list[str],
        dim: int | list[int] = 0,
        prob: float = 0.1,
    ):
        """
        Args:
            keys (str | list[str]): keys of the dictionary to apply
                dropout.
            dim (int | list[int], optional): dropout dimension. Defaults
                to 0.
            prob (float, optional): probability of dropout. Defaults to 0.1.
        """
        super().__init__(self, prob)
        self.keys = keys
        self.dim = dim
        self.prob = prob

        if isinstance(self.dim, int):
            self.dim = [self.dim for _ in self.keys]

        self.transforms = {}
        for k, d in zip(self.keys, self.dim):
            self.transforms[k] = RandomDropout(d, self.prob)

    def __call__(self, X):
        for k in zip(self.keys):
            X = self.transforms[k](X[k])
        return X
