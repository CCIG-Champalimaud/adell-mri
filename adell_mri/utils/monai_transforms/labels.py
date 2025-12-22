import monai
import numpy as np
import torch
from monai.data.meta_tensor import MetaTensor
from skimage.morphology import convex_hull_image
from sklearn.cluster import DBSCAN

from adell_mri.custom_types import (
    NDArrayOrTensor,
    NDArrayOrTensorDict,
    TensorDict,
)


def convex_hull_iter(x: np.ndarray):
    """
    Iterates through the last dimension of the input array `x`, replacing
    each slice with its convex hull using the `convex_hull_image` function.

    Args:
        x (np.ndarray): Input array to compute convex hulls over.

    Returns:
        np.ndarray: Array with the same shape as `x`, with the last dimension
        replaced with convex hulls.
    """
    for i in range(x.shape[-1]):
        x[..., i] = convex_hull_image(x[..., i])
    return x


class LabelOperatord(monai.transforms.Transform):
    """
    Label operator that merges labels if necessary.

    It takes as input a dictionary `data` containing a key specified by
    `keys` pointing to a segmentation mask. It converts the labels in this
    mask into a different representation specified by `mode`.

    The supported `mode` values are:

    - `'cat'` (default): Convert labels into categorical labels based on
      `possible_labels`.
    - `'binary'`: Convert labels into a binary mask based on
      `positive_labels`.

    The transformed labels are written into `data` with key `out_key`,
    which defaults to `key` if not specified in `output_keys`.
    """

    def __init__(
        self,
        keys: str,
        possible_labels: list[int],
        mode: str = "cat",
        positive_labels: list[int] = [1],
        label_groups: list[list[int]] = None,
        output_keys: dict[str, str] = {},
    ):
        """
        Args:
            keys (str): key for label
            possible_labels (list[int]): list of possible labels.
            mode (str, optional): sets the label merging mode between "cat" and
                "binary". Defaults to "cat".
            positive_labels (list[int], optional): _description_. Defaults to [1].
            label_groups (list[list[int]], optional): _description_. Defaults to None.
            output_keys (dict[str, str], optional): _description_. Defaults to {}.
        """
        self.keys = keys
        self.possible_labels = [str(x) for x in possible_labels]
        self.mode = mode
        self.positive_labels = positive_labels
        self.label_groups = label_groups
        self.output_keys = output_keys

        self.get_label_correspondence()

    def get_label_correspondence(self):
        if self.label_groups is not None:
            self.label_groups = [
                [str(x) for x in label_group]
                for label_group in self.label_groups
            ]
            self.possible_labels_match = {}
            for i, label_group in enumerate(self.label_groups):
                for label in label_group:
                    self.possible_labels_match[str(label)] = i
        elif self.positive_labels is not None:
            self.positive_labels = [str(x) for x in self.positive_labels]
            self.possible_labels_match = {}
            for label in self.possible_labels:
                if label in self.positive_labels:
                    self.possible_labels_match[label] = 1
                else:
                    self.possible_labels_match[label] = 0
        else:
            self.possible_labels_match = {
                str(label): i for i, label in enumerate(self.possible_labels)
            }

    def convert(self, x):
        if isinstance(x, (tuple, list)):
            x = max(x)
        if isinstance(x, MetaTensor):
            x = x.item()
        return self.possible_labels_match[str(x)]

    def __call__(self, data):
        for key in self.keys:
            if key in self.output_keys:
                out_key = self.output_keys[key]
            else:
                out_key = key
            data[out_key] = self.convert(data[key])
        return data


class LabelOperatorSegmentationd(monai.transforms.Transform):
    """
    Converts a segmentation mask to categorical or binary labels masks given
    a set of possible and positive labels.
    """

    def __init__(
        self,
        keys: str,
        possible_labels: list[int],
        mode: str = "cat",
        positive_labels: list[int] = [1],
        output_keys: dict[str, str] = {},
    ):
        """
        Args:
            keys (str): Key corresponding to label map in the data dictionary.
            possible_labels (list[int]): list of possible label values.
            mode (str): Label encoding mode, either 'cat' for categorical or
                'binary'.
            positive_labels (list[int], optional): Labels to map to 1 in
                binary encoding. Defaults to [1].
            output_keys (dict[str, str], optional): Dictionary mapping keys to
                output keys. Defaults to using the same keys.
        """
        self.keys = keys
        self.possible_labels = possible_labels
        self.mode = mode
        self.positive_labels = positive_labels
        self.output_keys = output_keys

        self.possible_labels = self.possible_labels
        self.possible_labels_match = {
            labels: i for i, labels in enumerate(self.possible_labels)
        }

    def binary(self, x):
        return np.isin(x, np.float32(self.positive_labels)).astype(np.float32)

    def categorical(self, x):
        output = np.zeros_like(x)
        for u in np.unique(x):
            if u in self.possible_labels_match:
                output[np.where(x == u)] = self.possible_labels_match[u]
        return output

    def __call__(self, data):
        for key in self.keys:
            if key in self.output_keys:
                out_key = self.output_keys[key]
            else:
                out_key = key
            if self.mode == "cat":
                data[out_key] = self.categorical(data[key])
            elif self.mode == "binary":
                data[out_key] = self.binary(data[key])
            else:
                data[out_key] = data[key]
        return data


class CombineBinaryLabelsd(monai.transforms.Transform):
    """
    Combines binary label maps.
    """

    def __init__(
        self, keys: list[str], mode: str = "any", output_key: str = None
    ):
        """
        Args:
            keys (list[str]): list of keys.
            mode (str, optional): how labels are combined. Defaults to
                "any".
            output_key (str, optional): name for the output key. Defaults to
                the name of the first key in keys.
        """
        self.keys = keys
        self.mode = mode
        if output_key is None:
            self.output_key = self.keys[0]
        else:
            self.output_key = output_key

    def __call__(self, X):
        tmp = [X[k] for k in self.keys]
        output = torch.stack(tmp, -1)
        if self.mode == "any":
            output = np.float32(output.sum(-1) > 0)
        elif self.mode == "majority":
            output = np.float32(output.mean(-1) > 0.5)
        X[self.output_key] = output
        return X


class ConvexHull(monai.transforms.Transform):
    """
    Calculates the convex hull of a segmentation mask.
    """

    backend = [monai.utils.TransformBackends.NUMPY]

    def __init__(self) -> None:
        super().__init__()

    def __call__(self, img: NDArrayOrTensor) -> NDArrayOrTensor:
        img = monai.utils.convert_to_tensor(
            img, track_meta=monai.data.meta_obj.get_track_meta()
        )
        img_np, *_ = monai.utils.convert_data_type(img, np.ndarray)
        if img_np.shape[0] == 1:
            out_np = convex_hull_iter(img_np[0])[None]
        else:
            out_np = convex_hull_iter(img_np[0])
        out, *_ = monai.utils.type_conversion.convert_to_dst_type(out_np, img)
        return out


class ConvexHulld(monai.transforms.MapTransform):
    """
    Dictionary version of ConvexHull.
    """

    backend = [monai.utils.TransformBackends.NUMPY]

    def __init__(self, keys: list[str]) -> None:
        super().__init__(keys=keys)

        self.transform = ConvexHull()

    def __call__(self, X: NDArrayOrTensorDict) -> NDArrayOrTensorDict:
        for k in self.keys:
            X[k] = self.transform(X[k])
        return X


class ConvertToOneHot(monai.transforms.Transform):
    """
    Convenience MONAI transform to convert a set of keys in a
    dictionary into a single one-hot format dictionary. Useful to coerce
    several binary class problems into a single multi-class problem.
    """

    def __init__(
        self, keys: str, out_key: str, priority_key: str, bg: bool = True
    ):
        """
        Args:
            keys (str): keys that willbe used to construct the one-hot
            encoding.
            out_key (str): key for the output.
            priority_key (str): key for the element that takes priority when
            more than one key is available for the same position.
            bg (bool, optional): whether a level for the "background" class
            should be included. Defaults to True.
        """
        super().__init__()
        self.keys = keys
        self.out_key = out_key
        self.priority_key = priority_key
        self.bg = bg

    def __call__(self, X: TensorDict) -> TensorDict:
        """
        Args:
            X (TensorDict)

        Returns:
            TensorDict
        """
        rel = {k: X[k] for k in self.keys}
        p = X[self.priority_key]
        dev = p.device
        p_inv = torch.ones_like(p, device=dev) - p
        for k in self.keys:
            if k != self.priority_key:
                rel[k] = rel[k] * p_inv
        out = [rel[k] for k in self.keys]
        if self.bg is True:
            bg_tensor = torch.where(
                torch.cat(out, 0).sum(0) > 0,
                torch.zeros_like(p, device=dev),
                torch.ones_like(p, device=dev),
            )
            out.insert(0, bg_tensor)
        out = torch.cat(out, 0)
        out = torch.argmax(out, 0, keepdim=True)
        X[self.out_key] = out
        return X


class DbscanAssistedSegmentSelection(monai.transforms.MapTransform):
    """
    A segment ("connected" component) selection module. It uses DBSCAN under the
    hood to get rid of spurious, small and noisy activations while maintaining
    loosely connected components as specified using min_dist. If nothing else is
    specified, then the this will just get rid of small noise specs; if
    filter_by_size==True it will filter by size and return the largest keep_n
    segments. If filter_by_dist_to_centre==True, this will return only the segment
    which is closest to the centre. If more than one channel is provided, it performs
    this separately for each channel.
    """

    def __init__(
        self,
        min_dist: int = 1,
        filter_by_size: bool = False,
        filter_by_dist_to_centre: bool = False,
        keep_n: int = 1,
    ):
        """
        Args:
            min_dist (int, optional): minimum distance in DBSCAN. Defaults to
                1.
            filter_by_size (bool, optional): filter lesions by size. Defaults
                to False.
            filter_by_dist_to_centre (bool, optional): filters lesions by
                distance to centre. Defaults to False.
            keep_n (int, optional): keeps only the keep_n largest lesions when
                filter_by_size is True. Defaults to 1.
        """
        self.min_dist = min_dist
        self.filter_by_size = filter_by_size
        self.filter_by_dist_to_centre = filter_by_dist_to_centre
        self.keep_n = keep_n

    def __call__(self, X: torch.Tensor | np.ndarray):
        sh = np.array(X.shape[1:])
        image_centre = sh / 2
        if isinstance(X, torch.Tensor):
            X = X.detach().cpu().numpy()
            is_tensor = True
        else:
            is_tensor = False
        output = []
        for i in range(X.shape[0]):
            dbscan = DBSCAN(self.min_dist)
            coords = np.stack(np.where(X[i] > 0.5), 1)
            labels = dbscan.fit(coords).labels_
            unique_labels = np.unique(labels)
            unique_labels = unique_labels[unique_labels > 0]

            output[i] = np.zeros(*sh)

            dist_to_centre = {}
            sizes = {}
            for label in unique_labels:
                idxs = labels == label
                dist_to_centre[label] = np.square(
                    np.mean(coords[idxs]) - image_centre
                )
                sizes[label] = np.sum(idxs)

            labels_to_keep = []
            if self.filter_by_size is True:
                sorted_labels = sorted(sizes.keys(), key=lambda k: -sizes[k])
                if len(sorted_labels) > self.keep_n:
                    sorted_labels = sorted_labels[: self.keep_n]
                labels_to_keep.extend(sorted_labels)

            if self.filter_by_dist_to_centre is True:
                sorted_labels = sorted(
                    dist_to_centre.keys(), key=lambda k: sizes[k]
                )
                label_to_keep = None
                idx = 0
                while label_to_keep is None:
                    curr_label = sorted_labels[idx]
                    if self.filter_by_size is True:
                        if curr_label in labels_to_keep:
                            label_to_keep = curr_label
                    else:
                        label_to_keep = curr_label
                    idx += 1
                labels_to_keep = [label_to_keep]
            for label in labels_to_keep:
                output[i][coords[labels == label]] = 1.0

        output = np.stack(output)
        if is_tensor:
            output = torch.as_tensor(output)
        return output


class CropFromMaskd(monai.transforms.MapTransform):
    """
    Crops the input image(s) from a binary mask.

    Finds the extremes of the positive class in the binary mask along each
    dimension. Uses these to crop the image(s) to the smallest box containing
    the mask.
    """

    def __init__(
        self,
        keys: list[str] | str,
        mask_key: str,
        output_size: list[int] = None,
    ):
        """
        Args:
            keys (List[str] | str): Keys of the input images.
            mask_key (str): Key of the binary mask.
            output_size (List[int], optional): output size. If provided, uses
                this to determine crop region instead of the mask extremes.
                Defaults to None.
        """
        super().__init__(keys=keys)
        self.mask_key = mask_key
        self.output_size = output_size

        if isinstance(self.keys, str):
            self.keys = [self.keys]

    def get_centre_extremes(self, mask: torch.Tensor):
        if mask.shape[0] == 1:
            mask = mask[0]
        coords = torch.where(mask)
        if len(coords[0]) > 0:
            extremes = [(c.min(), c.max()) for c in coords]
            centre = [(e[1] + e[0]) // 2 for e in extremes]
        else:
            centre = [c // 2 for c in mask.shape]
            extremes = None
        return centre, extremes

    def __call__(self, X: dict[str, torch.Tensor]):
        centres, extremes = self.get_centre_extremes(X[self.mask_key])
        min_shape = np.array([X[k].shape for k in self.keys]).min(0)[1:]
        if (self.output_size is not None) or (extremes is None):
            half_size = [x // 2 for x in self.output_size]
            extremes = [
                (c - h, c + (o - h))
                for c, h, o in zip(centres, half_size, self.output_size)
            ]
            for i in range(len(extremes)):
                if extremes[i][0] < 0:
                    extremes[i] = (0, self.output_size[i])
                if extremes[i][1] > min_shape[i]:
                    extremes[i] = (
                        min_shape[i] - self.output_size[i],
                        min_shape[i],
                    )

        for k in self.keys:
            if len(extremes) == 2:
                X[k] = X[k][
                    :,
                    extremes[0][0] : extremes[0][1],
                    extremes[1][0] : extremes[1][1],
                ]
            elif len(extremes) == 3:
                X[k] = X[k][
                    :,
                    extremes[0][0] : extremes[0][1],
                    extremes[1][0] : extremes[1][1],
                    extremes[2][0] : extremes[2][1],
                ]
            else:
                raise Exception(
                    "mask and image should have same size or \
                                output_size should have length identical to the \
                                spatial dimensions of the image"
                )
        return X
