import numpy as np
import SimpleITK as sitk
import torch
import torch.functional as F
import einops
import monai
from copy import deepcopy
from itertools import product
from skimage.morphology import convex_hull_image
from skimage import measure
from sklearn.cluster import DBSCAN
from pydicom import dcmread
from typing import List, Iterable, Tuple, Dict, Union, Any, Optional
from ..custom_types import (
    TensorDict,
    TensorOrNDarray,
    NDArrayOrTensorDict,
    Size2dOr3d,
)


def normalize_along_slice(
    X: torch.Tensor,
    min_value: float = 0.0,
    max_value: float = 1.0,
    dim: int = -1,
) -> torch.Tensor:
    """
    Performs minmax normalization along a given axis for a tensor.

    Args:
        X (torch.Tensor): tensor.
        min_value (float, optional): min value for output tensor. Defaults to
            0.0.
        max_value (float, optional): max value for output tensor. Defaults to
            1.0.
        dim (int, optional): dimension along which the minmax normalization is
            performed. Defaults to -1.

    Returns:
        torch.Tensor: minmax normalized tensor.
    """
    sh = X.shape
    assert dim < len(sh)
    assert (
        max_value > min_value
    ), "max_value {} must be larger than min_value {}".format(
        max_value, min_value
    )
    if dim < 0:
        dim = len(sh) + dim
    dims = ["c", "h", "w", "d"]
    lhs = " ".join(dims)
    rhs = "{} ({})".format(
        dims[dim], " ".join([d for d in dims if d != dims[dim]])
    )
    average_shape = [1 if i != dim else sh[dim] for i in range(len(sh))]
    flat_X = einops.rearrange(X, "{} -> {}".format(lhs, rhs))
    dim_max = flat_X.max(-1).values.reshape(average_shape)
    dim_min = flat_X.min(-1).values.reshape(average_shape)
    identical = dim_max == dim_min
    mult = torch.where(identical, 0.0, 1.0)
    denominator = torch.where(identical, 1.0, dim_max - dim_min)
    X = (X - dim_min) / denominator * mult
    X = X * (max_value - min_value) + min_value
    return X


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


class LoadIndividualDICOM(monai.transforms.Transform):
    """
    LoadIndividualDICOM reads a path to a DICOM file, converts it to an array
    using pydicom, and reshapes it to have a batch dimension.

    This is a thin wrapper around pydicom's dcmread function to load individual
    DICOM files and reshape them to NCHW format for batch processing.
    """

    def __init__(self):
        pass

    def __call__(self, X: str):
        out = dcmread(X).pixel_array
        if len(out.shape) == 2:
            out = out[None, :, :, None]
        elif len(out.shape) == 3:
            out = out[None, :, :, :]
        return out


class LoadIndividualDICOMd(monai.transforms.MapTransform):
    """LoadIndividualDICOMd applies LoadIndividualDICOM transform to multiple keys.

    LoadIndividualDICOMd takes a list of keys as input. For each key, it applies the
    LoadIndividualDICOM transform to the corresponding value in the input dictionary.

    Args:
        keys (List[str]): List of keys to apply transform to.

    """

    def __init__(self, keys: List[str]):
        self.keys = keys
        self.tr = LoadIndividualDICOM()

    def __call__(self, X: str):
        for k in self.keys:
            X[k] = self.tr(X[k])
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


class PrintShaped(monai.transforms.InvertibleTransform):
    """
    Convenience MONAI transform that prints the shape of elements in a
    dictionary of tensors. Used for debugging.
    """

    def __init__(self, prefix=""):
        self.prefix = prefix

    def __call__(self, X):
        for k in X:
            try:
                print(self.prefix, k, X[k].shape)
            except Exception:
                pass
        return X

    def inverse(self, X):
        return self(X)


class PrintSumd(monai.transforms.InvertibleTransform):
    """
    Convenience MONAI transform that prints the sum of elements in a
    dictionary of tensors. Used for debugging.
    """

    def __init__(self, prefix=""):
        self.prefix = prefix

    def __call__(self, X):
        for k in X:
            try:
                print(self.prefix, k, X[k].sum())
            except Exception:
                pass
        return X

    def inverse(self, X):
        return self(X)


class PrintRanged(monai.transforms.InvertibleTransform):
    """
    Convenience MONAI transform that prints the sum of elements in a
    dictionary of tensors. Used for debugging.
    """

    def __init__(self, prefix="", keys: List[str] = None):
        self.prefix = prefix
        self.keys = keys

    def __call__(self, X):
        if self.keys is not None:
            keys = self.keys
        else:
            keys = X.keys()
        for k in keys:
            try:
                print(self.prefix, k, X[k].min(), X[k].max())
            except Exception:
                pass
        return X

    def inverse(self, X):
        return self(X)


class PrintTyped(monai.transforms.InvertibleTransform):
    """
    Convenience MONAI transform that prints the type of elements in a
    dictionary of tensors. Used for debugging.
    """

    def __init__(self, prefix=""):
        self.prefix = prefix

    def __call__(self, X):
        for k in X:
            print(self.prefix, k, type(X[k]))
        return X

    def inverse(self, X):
        return self(X)


class Printd(monai.transforms.InvertibleTransform):
    """
    Convenience MONAI transform that prints elements. Used for debugging.
    """

    def __init__(self, prefix="", keys=None):
        self.prefix = prefix
        self.keys = keys

    def __call__(self, X):
        for k in X:
            if self.keys is not None:
                if k in self.keys:
                    print(self.prefix, k, X[k])
            else:
                print(self.prefix, k, X[k])
        return X

    def inverse(self, X):
        return self(X)


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
        keys: List[str],
        label_key: List[str],
        n: int = 1,
        base: float = 0.001,
        seed=None,
    ):
        """
        Args:
            keys (List[str]): keys for which slices will be retrieved.
            label_key (List[str]): segmentation map that will be used to
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


class SlicesToFirst(monai.transforms.Transform):
    """
    Returns the slices as the first spatial dimension.
    """

    def __init__(self, keys: List[str]):
        """
        Args:
            keys (List[str]): keys for which slices will be retrieved.
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

    def __init__(self, keys: List[str], idxs: List[int], axis: int):
        """
        Args:
            keys (List[str]): list of keys to tensors which will be indexed.
            idxs (List[int]): indices that will be retrieved.
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


class BBToAdjustedAnchors(monai.transforms.Transform):
    """
    Maps bounding boxes in corner format (x1y1z1x2y2z2) to their anchor
    representation.
    """

    def __init__(
        self,
        anchor_sizes: Union[np.ndarray, List[List]],
        input_sh: Iterable,
        output_sh: Iterable,
        iou_thresh: float,
    ):
        """
        Args:
            anchor_sizes (Union[np.ndarray,List[List]]): a two dimensional
            array or list of lists containing anchors in corner format.
            input_sh (Iterable): an iterable containing the input shape of the
            image containing the bounding boxes.
            output_sh (Iterable): an iterable containing the output shape for
            the bounding box anchor representation map
            iou_thresh (float): IoU threshold to consider a bounding box as a
            positive.
        """
        self.anchor_sizes = [np.array(x) for x in anchor_sizes]
        self.input_sh = input_sh
        self.output_sh = np.array(output_sh)
        self.iou_thresh = iou_thresh

        self.setup_long_anchors()

    def setup_long_anchors(self):
        """Sets anchors up."""
        image_sh = np.array(self.input_sh)
        self.rel_sh = image_sh / self.output_sh

        long_coords = []
        for c in product(*[np.arange(x) for x in self.output_sh]):
            long_coords.append(c)

        self.long_coords = np.array(long_coords)
        rel_anchor_sizes = [x / self.rel_sh for x in self.anchor_sizes]

        self.long_anchors = []
        for rel_anchor_size in rel_anchor_sizes:
            # adding 0.5 centres the bounding boxes in each cell
            anchor_h = rel_anchor_size / 2
            long_anchor_rel = [
                long_coords - anchor_h + 0.5,
                long_coords + anchor_h + 0.5,
            ]
            self.long_anchors.append(np.stack(long_anchor_rel, axis=-1))

    def __call__(
        self,
        bb_vertices: Iterable,
        classes: Iterable,
        shape: np.ndarray = None,
    ) -> np.ndarray:
        """Converts a set of bounding box vertices into their anchor
        representation.

        Args:
            bb_vertices (Iterable): list of lists or array of bounding box
                vertices. Shape has to be [N,6], where N is the number of
                bounding boxes.
            classes (Iterable): vector of classes, shape [N,1].
            shape (np.ndarray, optional): shape of the input image. Defaults to
                `self.input_sh`.

        Returns:
            output (np.ndarray): anchor representation of the bounding boxes.
            Shape is [1+7*A,*self.output_sh], where A is the number of anchors
            and 7 contains the objectness (1), center adjustments (3) and
            size adjustments (3) to the anchor.
        """
        bb_vertices = np.array(bb_vertices)
        if len(bb_vertices.shape) < 2:
            bb_vertices = bb_vertices[np.newaxis, :]
        bb_vertices = np.stack(
            [bb_vertices[:, :3], bb_vertices[:, 3:]], axis=-1
        )
        # bb_vertices[:,:,1]-bb_vertices[:,:,0]
        output = np.zeros([1 + 7 * len(self.long_anchors), *self.output_sh])
        # no vertices present
        if bb_vertices.shape[1] == 0:
            return output
        if shape is None:
            shape = self.input_sh
            rel_sh = self.rel_sh
            rel_bb_vert = bb_vertices / rel_sh[np.newaxis, :, np.newaxis]
        else:
            rel_sh = shape / self.output_sh
            rel_bb_vert = bb_vertices / rel_sh[np.newaxis, :, np.newaxis]
        for i in range(rel_bb_vert.shape[0]):
            hits = 0
            all_iou = []
            rel_bb_size = np.subtract(
                rel_bb_vert[i, :, 1], rel_bb_vert[i, :, 0]
            )
            center = np.mean(rel_bb_vert[i, :, :], axis=-1)
            bb_vol = np.prod(rel_bb_size + 1 / rel_sh)
            cl = classes[i]
            for I, long_anchor in enumerate(self.long_anchors):
                anchor_size = long_anchor[0, :, 1] - long_anchor[0, :, 0]
                rel_bb_size_adj = np.log(rel_bb_size / anchor_size)
                anchor_dim = long_anchor[0, :, 1] - long_anchor[0, :, 0]
                intersects = np.logical_and(
                    np.all(
                        long_anchor[:, :, 1] > rel_bb_vert[i, :, 0], axis=1
                    ),
                    np.all(
                        long_anchor[:, :, 0] < rel_bb_vert[i, :, 1], axis=1
                    ),
                )
                inter_dim = np.minimum(rel_bb_size, anchor_dim)
                inter_vol = np.prod(inter_dim + 1 / rel_sh, axis=-1)
                anchor_vol = np.prod(anchor_dim, axis=-1)
                union_vol = anchor_vol + bb_vol - inter_vol

                iou = inter_vol / union_vol
                intersection_idx = np.logical_and(
                    iou > self.iou_thresh, intersects
                )
                box_coords = self.long_coords[intersection_idx]

                all_iou.append(iou)

                center_adjustment = center - (box_coords + 0.5)
                distance_idx = np.all(np.abs(center_adjustment) < 1, axis=1)

                box_coords = box_coords[distance_idx]
                center_adjustment = center_adjustment[distance_idx]

                for j in range(box_coords.shape[0]):
                    idx = tuple(
                        [
                            tuple([1 + k + I * 7 for k in range(7)]),
                            *box_coords[j],
                        ]
                    )
                    idx_cl = tuple([0, *box_coords[j]])
                    v = np.array(
                        [iou, *center_adjustment[j], *rel_bb_size_adj]
                    )
                    if iou > output[idx][0]:
                        output[idx] = v
                        output[idx_cl] = cl
                        hits += 1

        return output

    def adjusted_anchors_to_bb_vertices(
        self, anchor_map: np.ndarray
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Converts an anchor map into the input anchors.

        Args:
            anchor_map (np.ndarray): anchor map as produced by __call__.

        Returns:
            top_left_output: top left corner for the bounding boxes.
            bottom_right_output: bottom right corner for the bounding boxes.
        """
        top_left_output = []
        bottom_right_output = []
        for i in range(len(self.anchor_sizes)):
            anchor_size = self.anchor_sizes[i]
            rel_anchor_size = np.array(anchor_size)
            sam = anchor_map[(1 + i * 7) : (1 + i * 7 + 7)]
            coords = np.where(sam[0] > 0)
            adj_anchors_long = np.zeros([len(coords[0]), 7])
            for j, coord in enumerate(zip(*coords)):
                center_idxs = tuple([tuple([k for k in range(7)]), *coord])
                v = sam[center_idxs]
                adj_anchors_long[j, :] = v
            correct_centers = np.add(
                adj_anchors_long[:, 1:4] + 0.5, np.stack(coords, axis=1)
            )
            correct_centers = correct_centers * self.rel_sh
            correct_dims = np.multiply(
                adj_anchors_long[:, 4:], rel_anchor_size
            )
            top_left = correct_centers - correct_dims / 2
            bottom_right = correct_centers + correct_dims / 2
            top_left_output.append(top_left)
            bottom_right_output.append(bottom_right)
        return top_left_output, bottom_right_output


class BBToAdjustedAnchorsd(monai.transforms.MapTransform):
    """
    Dictionary transform of the BBToAdjustedAnchors transforrm.
    """

    def __init__(
        self,
        anchor_sizes: torch.Tensor,
        input_sh: Tuple[int],
        output_sh: Tuple[int],
        iou_thresh: float,
        bb_key: str = "bb",
        class_key: str = "class",
        shape_key: str = "shape",
        output_key: Dict[str, str] = {},
    ):
        """
        Args:
            anchor_sizes (Union[np.ndarray,List[List]]): a two dimensional
                array or list of lists containing anchors in corner format.
            input_sh (Iterable): an iterable containing the input shape of the
                image containing the bounding boxes.
            output_sh (Iterable): an iterable containing the output shape for
                the bounding box anchor representation map
            iou_thresh (float): IoU threshold to consider a bounding box as a
                positive.
            bb_key (str, optional): key corresponding to the bounding boxes.
                Defaults to "bb".
            class_key (str, optional): key corresponding to the classes.
                Defaults to "class".
            shape_key (str, optional): key corresponding to the shapes.
                Defaults to "shape".
            output_key (Dict[str,str], optional): key for output. Defaults to
                self.bb_key.
        """

        self.anchor_sizes = anchor_sizes
        self.input_sh = input_sh
        self.output_sh = output_sh
        self.iou_thresh = iou_thresh
        self.mask_to_anchors = BBToAdjustedAnchors(
            self.anchor_sizes, self.input_sh, self.output_sh, self.iou_thresh
        )
        self.bb_key = bb_key
        self.class_key = class_key
        self.shape_key = shape_key
        self.output_key = output_key

    def __call__(self, data: dict) -> dict:
        if self.output_key is not None:
            out_k = self.output_key
        else:
            out_k = self.bb_key
        data[out_k] = self.mask_to_anchors(
            data[self.bb_key], data[self.class_key], data[self.shape_key]
        )
        return data


class MasksToBB(monai.transforms.Transform):
    """
    Calculates bounding boxes from masks.
    """

    def __init__(self, mask_mode: str = "mask_is_label"):
        """
        Args:
            mask_mode (str, optional): how objects in the mask are treated.
                mask_is_labels uses the mask as label maps; infer_labels infers
                connected components using skimage.measure.label; single_object assumes
                the mask represents a single, not necessarily connected, object.
                Defaults to "mask_is_label".
        """
        self.mask_mode = mask_mode

    def __call__(
        self, X: TensorOrNDarray
    ) -> Tuple[List[np.ndarray], List[int], Size2dOr3d]:
        """
        Converts a binary mask with a channel (first) dimension into a set of
        bounding boxes, classes and shape. The bounding boxes are obtained as the
        upper and lower corners of the connected components and the classes are
        obtained as the median value of the pixels/voxels in those connected
        components. The shape is the shape of the input X.

        Args:
            X (TensorOrNDarray): an array with shape [1,H,W] or [1,H,W,D].

        Returns:
            Tuple[List[np.ndarray], List[int], Size2dOr3d]: a list with bounding
                boxes (each with size 4 or 6 depending on the number of input
                dimensions), a list of classes and the shape of the input X.
        """
        X = X[0]
        if isinstance(X, torch.Tensor):
            X = X.cpu().numpy()
        if self.mask_mode == "infer_labels":
            labels = measure.label(X, background=0)
        elif self.mask_mode == "mask_is_labels":
            labels = X
        elif self.mask_mode == "single_object":
            X = np.float32(X > 0)
            labels = X
        unique_labels = np.unique(labels)
        unique_labels = unique_labels[unique_labels != 0]
        bounding_boxes = []
        classes = []
        for u in unique_labels:
            coords = np.where(labels == u)
            cl = np.median(X[coords]).round()
            upper_corner = [x.min() for x in coords]
            lower_corner = [x.max() for x in coords]
            bounding_boxes.append(np.concatenate([upper_corner, lower_corner]))
            classes.append(cl)
        return bounding_boxes, classes, X.shape


class MasksToBBd(monai.transforms.Transform):
    """
    Dictionary version of MaskstoBB.
    """

    def __init__(
        self,
        keys: List[str],
        bounding_box_key: str = "bounding_boxes",
        classes_key: str = "classes",
        shape_key: str = "shape",
        mask_mode: str = "mask_is_labels",
        replace: bool = True,
    ):
        """
        Args:
            keys (List[str]): list of keys.
            bounding_box_key (str): name of output bounding boxes key.
            classes_key (str): name of output classes key.
            shape_keys (str): name of shape key.
            replace (bool, optional): replaces the values in bounding_box_key,
                classes_key and shape_keys if these are already present in the
                data dictionary. Defaults to True.
        """
        self.keys = keys
        self.bounding_box_key = bounding_box_key
        self.classes_key = classes_key
        self.shape_key = shape_key
        self.mask_mode = mask_mode
        self.replace = replace

        self.tr = MasksToBB(mask_mode=mask_mode)

    def __call__(self, data):
        for k in list(data.keys()):
            if k in self.keys:
                if self.bounding_box_key not in data or self.replace:
                    bb, cl, sh = self.tr(data[k])
                    data[self.bounding_box_key] = bb
                    data[self.classes_key] = cl
                    data[self.shape_key] = sh
                elif self.bounding_box_key in data and self.replace == False:
                    bb, cl, sh = self.tr(data[k])
                    data[self.bounding_box_key].extend(bb)
                    data[self.classes_key].extend(cl)
                    data[self.shape_key] = sh
        return data


class RandomFlipWithBoxes(monai.transforms.Transform):
    """
    Randomly augmentatat images and bounding boxes by flipping axes.
    """

    def __init__(self, axes=[0, 1, 2], prob=0.5):
        """
        Args:
            axes (list, optional): list of axes to flip. Defaults to [0,1,2].
            prob (float, optional): rotation probability. Defaults to 0.5.
        """
        self.axes = axes
        self.prob = prob

    def flip_boxes(self, boxes, axis, center):
        boxes = boxes - center
        boxes[:, axis, :] = -boxes[:, axis, :]
        boxes = boxes + center
        return boxes

    def flip_image(self, image, axis):
        return torch.flip(image, (axis,))

    def __call__(self, images, boxes):
        center = np.expand_dims(np.array(images[0].shape[1:]), 0)
        center = center[:, :, np.newaxis]
        axes_to_flip = []
        for axis in self.axes:
            if np.random.uniform() < self.prob:
                axes_to_flip.append(axis)
                boxes = self.flip_boxes(boxes, axis, center)
                for image in images:
                    image = self.flip_image(image, axis)
        return images, boxes


class RandomFlipWithBoxesd(monai.transforms.MapTransform):
    """
    Dictionary transform for RandomFlipWithBoxes.
    """

    def __init__(
        self,
        image_keys: List[str],
        box_key: str,
        box_key_nest: str = None,
        axes: List[int] = [0, 1, 2],
        prob: float = 0.5,
    ):
        """
        Args:
            image_keys (List[str]): keys for images that will be flipped.
            box_key (str): keys for bounding boxes that will be flipped.
            box_key_nest (str): optional key that considers that bounding
            boxes are nested in dictionaries. Defaults to None (no nesting).
            axes (List[int], optional): axes where flipping will occur.
            Defaults to [0,1,2].
            prob (float, optional): probability that the transform will be
            applied. Defaults to 0.5.
        """
        self.image_keys = image_keys
        self.box_key = box_key
        self.box_key_nest = box_key_nest
        self.axes = axes
        self.prob = prob
        self.flipping_op = RandomFlipWithBoxes(axes, prob)

    def __call__(self, data):
        images = [data[k] for k in self.image_keys]
        if self.box_key_nest is not None:
            boxes = data[self.box_key][self.box_key_nest]
        else:
            boxes = data[self.box_key]

        images, boxes = self.flipping_op(images, boxes)
        for k, image in zip(self.image_keys, images):
            data[k] = image

        if self.box_key_nest is not None:
            data[self.box_key][self.box_key_nest] = boxes
        else:
            data[self.box_key] = boxes
        return data


class RandomAffined(monai.transforms.RandomizableTransform):
    """
    Reimplementation of the RandAffined transform in MONAI but works
    with differently sized inputs without forcing all inputs to the same
    shape.
    """

    def __init__(
        self,
        keys: List[str],
        spatial_sizes: List[Union[Tuple[int, int, int], Tuple[int, int]]],
        mode: List[str],
        prob: float = 0.1,
        rotate_range: Union[Tuple[int, int, int], Tuple[int, int]] = [0, 0, 0],
        shear_range: Union[Tuple[int, int, int], Tuple[int, int]] = [0, 0, 0],
        translate_range: Union[Tuple[int, int, int], Tuple[int, int]] = [
            0,
            0,
            0,
        ],
        scale_range: Union[Tuple[int, int, int], Tuple[int, int]] = [0, 0, 0],
        device: "str" = "cpu",
        copy: bool = False,
    ):
        """
        Args:
            keys (List[str]): list of keys that will be randomly transformed.
            spatial_sizes (List[Union[Tuple[int,int,int],Tuple[int,int]]]): dimension
                number for the inputs.
            mode (List[str]): interpolation modes. Must be the same size as
                keys.
            prob (float, optional): Probability that the transform will be
                applied. Defaults to 0.1.
            rotate_range (Union[Tuple[int,int,int],Tuple[int,int]], optional):
                Rotation ranges. Defaults to [0,0,0].
            shear_range (Union[Tuple[int,int,int],Tuple[int,int]], optional):
                Shear ranges. Defaults to [0,0,0].
            translate_range (Union[Tuple[int,int,int],Tuple[int,int]], optional):
                Translation ranges. Defaults to [0,0,0].
            scale_range (Union[Tuple[int,int,int],Tuple[int,int]], optional):
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
        possible_labels: List[int],
        mode: str = "cat",
        positive_labels: List[int] = [1],
        label_groups: List[List[int]] = None,
        output_keys: Dict[str, str] = {},
    ):
        """
        Args:
            keys (str): key for label
            possible_labels (List[int]): list of possible labels.
            mode (str, optional): sets the label merging mode between "cat" and
                "binary". Defaults to "cat".
            positive_labels (List[int], optional): _description_. Defaults to [1].
            label_groups (List[List[int]], optional): _description_. Defaults to None.
            output_keys (Dict[str, str], optional): _description_. Defaults to {}.
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
        possible_labels: List[int],
        mode: str = "cat",
        positive_labels: List[int] = [1],
        output_keys: Dict[str, str] = {},
    ):
        """
        Args:
            keys (str): Key corresponding to label map in the data dictionary.
            possible_labels (List[int]): List of possible label values.
            mode (str): Label encoding mode, either 'cat' for categorical or
                'binary'.
            positive_labels (List[int], optional): Labels to map to 1 in
                binary encoding. Defaults to [1].
            output_keys (Dict[str, str], optional): Dictionary mapping keys to
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
        self, keys: List[str], mode: str = "any", output_key: str = None
    ):
        """
        Args:
            keys (List[str]): list of keys.
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


class ConditionalRescaling(monai.transforms.Transform):
    """
    Rescales an array using scale if any value in the array is larger than
    max_value.
    """

    def __init__(self, max_value: float, scale: float):
        """
        Args:
            max_value (float): maximum value for condition.
            scale (float): scaling factor.
        """
        self.max_value = max_value
        self.scale = scale

    def __call__(self, X):
        if X.max() > self.max_value:
            X = X * self.scale
        return X


class ConditionalRescalingd(monai.transforms.Transform):
    """
    Dictionary version of ConditionalRescaling.
    """

    def __init__(self, keys: List[str], max_value: float, scale: float):
        """
        Args:
            keys (List[str]): keys to which conditional rescaling will be
                applied.
            max_value (float): maximum value for condition.
            scale (float): scaling factor.
        """
        self.keys = keys
        self.max_value = max_value
        self.scale = scale

        self.transforms = {
            k: ConditionalRescaling(self.max_value, self.scale)
            for k in self.keys
        }

    def __call__(self, data):
        for k in data:
            if k in self.transforms:
                data[k] = self.transforms[k](data[k])
        return data


class Offset(monai.transforms.Transform):
    """
    Offsets an array using an offset value (array = array - offset). If no
    value is provided, the minimum value for the array is used.
    """

    def __init__(self, offset: float = None):
        """
        Args:
            offset (float, optional): value for offset. Defaults to None.
        """
        self.offset = offset

    def __call__(self, data):
        offset = data.min() if self.offset is None else self.offset
        return data - offset


class Offsetd(monai.transforms.MapTransform):
    """
    Dictionary version of Offset.
    """

    def __init__(self, keys: List[str], offset: float = None):
        """
        Args:
            keys (List[str]): keys to offset.
            offset (float, optional): value for offset. Defaults to None.
        """

        self.keys = keys
        self.offset = offset
        self.tr = {k: Offset(offset) for k in self.keys}

    def __call__(self, data):
        for k in self.keys:
            data[k] = self.tr[k](data[k])
        return data


class CopyEntryd(monai.transforms.Transform):
    """
    Transform that copies an entry in a dictionary with a new key.
    """

    def __init__(self, keys: List[str], out_keys: dict[str, str]):
        """
        Args:
            keys (List[str]): list of keys.
            out_keys (dict[str]): dictionary with input_key:output_key pairs.
        """
        self.keys = keys
        self.out_keys = out_keys

    def __call__(self, data):
        for k in list(data.keys()):
            if k in self.keys:
                if k in self.out_keys:
                    data[self.out_keys[k]] = deepcopy(data[k])
        return data


class FastResample(monai.transforms.Transform):
    """
    Does what monai.transforms.Spacingd does but fast by getting rid of
    some unnecessary calculations.
    """

    def __init__(self, target: List[float], keys=List[str], mode=List[str]):
        """
        Args:
            target (List[float]): _description_
            keys (_type_, optional): _description_. Defaults to List[str].
            mode (_type_, optional): _description_. Defaults to List[str].
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


class ExposeTransformKeyd(monai.transforms.Transform):
    """
    Exposes metadata keys for a given input dictionary. Deprecated since MONAI
    moved to MetaTensors.

    This transform looks for a specified transform class in the transforms
    applied to a key in the input dictionary. It then extracts a nested value
    from that transform's metadata and adds it to the dictionary under a
    specified output key.

    This is useful for exposing internal transform metadata for downstream use.
    """

    def __init__(
        self,
        transform_key: str,
        transform_class: str,
        nested_pattern: List[str],
        output_key: str = None,
    ):
        """
        Args:
            transform_key (str): key corresponding to transform parameters.
            transform_class (str): string corresponding to the class name.
            nested_pattern (List[str]): nested pattern of strings to recover
                value from transform metadata.
            output_key (str, optional): output key for exposed value. Defaults
                to None.
        """
        self.transform_key = transform_key
        self.transform_class = transform_class
        self.nested_pattern = nested_pattern
        self.output_key = output_key

    def __call__(self, X):
        if self.output_key is None:
            self.output_key = self.nested_pattern[-1]
        for t in X[self.transform_key]:
            if t["class"] == self.transform_class:
                curr = t
                for k in self.nested_pattern:
                    curr = curr[k]
                X[self.output_key] = curr
        return X


class ExposeTransformKeyMetad(monai.transforms.Transform):
    """
    Exposes metadata keys for a given transform applied to a MetaTensor.

    This transform looks for a specified transform class in the transforms
    applied to a key in the input dictionary. It then extracts a nested value
    from that transform's metadata and adds it to the dictionary under a
    specified output key.

    This is useful for exposing internal transform metadata for downstream use.
    """

    def __init__(
        self,
        key: str,
        transform_class: str,
        nested_pattern: List[str],
        output_key: str = None,
    ):
        """
        Args:
            key (str): key corresponding to relevant MetaTensor.
            transform_class (str): string corresponding to the class name.
            nested_pattern (List[str]): nested pattern of strings to recover
                value from transform metadata.
            output_key (str, optional): output key for exposed value. Defaults
                to None.
        """
        self.key = key
        self.transform_class = transform_class
        self.nested_pattern = nested_pattern
        self.output_key = output_key

    def __call__(self, X):
        if self.output_key is None:
            output_key = "box_" + self.key
        else:
            output_key = self.output_key
        for t in X[self.key].applied_operations:
            if t["class"] == self.transform_class:
                curr = t
                for k in self.nested_pattern:
                    curr = curr[k]
                X[output_key] = curr
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
        keys: Union[str, List[str]],
        channel: Union[int, List[int]],
        dim: Union[int, List[int]] = 0,
    ):
        """
        Args:
            keys (List[str] | str): keys of the dictionary to apply dropout.
            channel (Union[int, List[int]]): channels to be dropped (if list
                has to be one per key).
            dim (Union[int, List[int]], optional): dimension where channel is
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
        keys: Union[str, List[str]],
        dim: Union[int, List[int]] = 0,
        prob: float = 0.1,
    ):
        """
        Args:
            keys (Union[str, List[str]]): keys of the dictionary to apply
                dropout.
            dim (Union[int, List[int]], optional): dropout dimension. Defaults
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


class CreateImageAndWeightsd(monai.transforms.Transform):
    """
    Replaces missing keys in the dictionary with empty tensors and creates
    weight variables (0 if the key is absent and 1 if it is present).

    The weight is defined as f"{key}_weight"
    """

    def __init__(self, keys: list[str], shape: list[int]):
        """
        Args:
            keys (list[str]): keys whose presence is assessed.
            shape (list[int]): output shape of tensors.
        """
        self.keys = keys
        self.shape = shape

    def __call__(self, X):
        for k in self.keys:
            shape = (
                self.shape
                if isinstance(self.shape, str)
                else X[self.shape].shape
            )
            weight_key = f"{k}_weight"
            if k not in X:
                X[k] = np.zeros(shape, dtype=np.uint8)
                X[weight_key] = 0
            elif X[k] == "fill_with_empty":
                X[k] = np.zeros(shape, dtype=np.uint8)
                X[weight_key] = 0
            else:
                X[weight_key] = 1
        return X


class BiasFieldCorrection(monai.transforms.Transform):
    """
    MONAI transform that automatically performs bias field correction using
    SITK.
    """

    def __init__(
        self, n_fitting_levels: int, n_iter: int, shrink_factor: float
    ):
        """
        Args:
            n_fitting_levels (int): number of fitting levels.
            n_iter (int): number of correction iterations.
            shrink_factor (float): shrink factor.
        """
        self.n_fitting_levels = n_fitting_levels
        self.n_iter = n_iter
        self.shrink_factor = shrink_factor

    def correct_bias_field(self, image):
        image_ = image
        mask_image = sitk.OtsuThreshold(image_)
        if self.shrink_factor > 1:
            image_ = sitk.Shrink(
                image_, [self.shrink_factor] * image_.GetDimension()
            )
            mask_image = sitk.Shrink(
                mask_image, [self.shrink_factor] * mask_image.GetDimension()
            )
        corrector = sitk.N4BiasFieldCorrectionImageFilter()
        corrector.SetMaximumNumberOfIterations(
            self.n_fitting_levels * [self.n_iter]
        )
        corrector.SetConvergenceThreshold(0.001)
        corrector.Execute(image_, mask_image)
        log_bf = corrector.GetLogBiasFieldAsImage(image)
        corrected_input_image = image / sitk.Exp(log_bf)
        corrected_input_image = sitk.Cast(
            corrected_input_image, sitk.sitkFloat32
        )
        corrected_input_image.CopyInformation(image)
        for k in image.GetMetaDataKeys():
            v = image.GetMetaData(k)
            corrected_input_image.SetMetaData(k, v)
        return corrected_input_image

    def correct_bias_field_from_metadata_tensor(self, X):
        X_ = sitk.GetImageFromArray(X.data.numpy())
        X_ = self.correct_bias_field(X_)
        X_ = sitk.GetArrayFromImage(X_)
        X.data = X_
        return X_

    def __call__(self, X):
        return self.correct_bias_field_from_array(X)


class BiasFieldCorrectiond(monai.transforms.MapTransform):
    """
    Dictionary version of BiasFieldCorrection
    """

    def __init__(
        self,
        keys: List[str],
        n_fitting_levels: int,
        n_iter: int,
        shrink_factor: int,
    ):
        """
        Args:
            keys (List[str]): keys to apply bias field correction to.
            n_fitting_levels (int): number of fitting levels.
            n_iter (int): number of correction iterations.
            shrink_factor (float): shrink factor.
        """
        self.keys = keys
        self.n_fitting_levels = n_fitting_levels
        self.n_iter = n_iter
        self.shrink_factor = shrink_factor

        self.transform = BiasFieldCorrection(
            self.n_fitting_levels, self.n_iter, self.shrink_factor
        )

    def __call__(self, X):
        for k in self.keys:
            X[k] = self.transform(X[k])
        return X


class ConvexHull(monai.transforms.Transform):
    """
    Calculates the convex hull of a segmentation mask.
    """

    backend = [monai.utils.TransformBackends.NUMPY]

    def __init__(self) -> None:
        super().__init__()

    def __call__(self, img: TensorOrNDarray) -> TensorOrNDarray:
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

    def __init__(self, keys: List[str]) -> None:
        super().__init__(keys=keys)

        self.transform = ConvexHull()

    def __call__(self, X: NDArrayOrTensorDict) -> NDArrayOrTensorDict:
        for k in self.keys:
            X[k] = self.transform(X[k])
        return X


class ScaleIntensityAlongDim(monai.transforms.Transform):
    """
    MONAI transform that applies normalize_along_slice to inputs. This
    normalizes individual slices along a given dimension dim.
    """

    def __init__(
        self, min_value: float = 0.0, max_value: float = 1.0, dim: int = -1
    ):
        """
        Args:
            min_value (float, optional): min value for output tensor. Defaults
                to 0.0.
            max_value (float, optional): max value for output tensor. Defaults
                to 1.0.
            dim (int, optional): dimension along which the minmax normalization
                is performed. Defaults to -1.
        """
        self.min_value = min_value
        self.max_value = max_value
        self.dim = dim

    def __call__(self, X: torch.Tensor) -> torch.Tensor:
        return normalize_along_slice(
            X, min_value=self.min_value, max_value=self.max_value, dim=self.dim
        )


class ScaleIntensityAlongDimd(monai.transforms.MapTransform):
    """
    MONAI dict transform that applies normalize_along_slice to inputs. This
    normalizes individual slices along a given dimension dim.
    """

    def __init__(
        self,
        keys: List[str],
        min_value: float = 0.0,
        max_value: float = 1.0,
        dim: int = -1,
    ):
        """
        Args:
            min_value (float, optional): min value for output tensor. Defaults to
                0.0.
            max_value (float, optional): max value for output tensor. Defaults to
                0.0.
            dim (int, optional): dimension along which the minmax normalization is
                performed. Defaults to -1.
        """

        self.keys = keys
        self.min_value = min_value
        self.max_value = max_value
        self.dim = dim

        if isinstance(self.keys, str):
            self.keys = [self.keys]

        self.tr = ScaleIntensityAlongDim(
            min_value=min_value, max_value=max_value, dim=dim
        )

    def __call__(self, X: TensorDict) -> TensorDict:
        for k in self.keys:
            X[k] = self.tr(X[k])
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

    def __init__(self, keys: List[str], pattern: str):
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


class RandAffineWithBoxesd(monai.transforms.RandomizableTransform):
    """
    EXPERIMENTAL.
    Uses MONAI's `RandAffined` to transform an image and then applies
    the same affine transform to a bounding box with format xy(z)xy(z).
    """

    def __init__(
        self, image_keys: List[str], box_keys: List[str], *args, **kwargs
    ):
        """
        Args:
            image_keys (List[str]): list of image keys.
            box_keys (List[str]): list of bounding box keys.
            args, kwargs: arguments and keyword arguments for RandAffined.
        """
        self.image_keys = image_keys
        self.box_keys = box_keys

        self.rand_affine_d = monai.transforms.RandAffined(
            image_keys, *args, **kwargs
        )

    def get_all_corners(self, tl, br, n_dim):
        # (b, number_of_corners, number_of_dimensions)
        corners = torch.zeros([tl.shape[0], 2**n_dim, n_dim])
        coord_const = tuple([i for i in range(n_dim)])
        tl_br = torch.stack([tl, br], -1)
        for i, c in enumerate(product(*[range(2) for _ in range(n_dim)])):
            corners[:, i, :] = tl_br[:, coord_const, c]
        return corners

    def coords_to_homogeneous_coords(self, coords):
        # (b, number_of_corners, number_of_dimensions) to
        # (b, number_of_corners, number_of_dimensions + 1)
        return torch.concat(
            [
                coords,
                torch.ones(
                    [coords.shape[0], coords.shape[1], 1],
                    dtype=coords.dtype,
                    device=coords.device,
                ),
            ],
            2,
        )

    def rotate_coords(self, coords, sh, affine):
        center = torch.as_tensor(sh[1:]).unsqueeze(0) / 2
        n_dim = coords.shape[1] // 2
        tl = coords[:, :n_dim]
        br = coords[:, n_dim:]
        corners = self.get_all_corners(tl, br, n_dim)
        corners = corners - center
        corners = self.coords_to_homogeneous_coords(corners)
        corners = torch.matmul(affine, corners.swapaxes(1, 2)).swapaxes(1, 2)
        corners = corners[:, :, :-1]
        corners = corners + center.unsqueeze(0)
        return corners

    def rotate_box(self, coords, sh, affine):
        affine_corners = self.rotate_coords(coords, sh, affine)
        # tl and br are (batch_size, n_dim)
        tl = affine_corners.min(1).values
        br = affine_corners.max(1).values
        # output is (batch_size, n_dim * 2)
        return torch.concat([tl, br], 1)

    def __call__(self, X):
        X = self.rand_affine_d(X)
        # retrieve rand_affine_info
        image_example = X[self.image_keys[0]]
        rand_affine_info = self.rand_affine_d.pop_transform(image_example)
        rand_affine_info = rand_affine_info["extra_info"]["rand_affine_info"]
        sh = image_example.shape
        # if affine has been applied, rotate boxes
        if "extra_info" in rand_affine_info:
            affine = rand_affine_info["extra_info"]["affine"]
            for k in self.box_keys:
                is_array = isinstance(X[k], np.ndarray)
                center = (np.array(sh) / 2)[np.newaxis, 1:]
                center = np.concatenate([center, center], 1)
                X[k] = self.rotate_box(torch.as_tensor(X[k]), sh, affine)
                if is_array:
                    X[k] = X[k].cpu().numpy()
        return X


class RandRotateWithBoxesd(monai.transforms.RandomizableTransform):
    """
    Uses MONAI's `RandAffined` to rotate an image and then applies
    the same rotation transform to a bounding box with format xy(z)xy(z).
    """

    def __init__(
        self,
        image_keys: List[str],
        box_keys: List[str],
        mode: List[str],
        rotate_range: Any = None,
        padding_mode: str = "zeros",
        prob: float = 0.1,
    ):
        """
        Args:
            image_keys (List[str]): list of image keys.
            box_keys (List[str]): list of bounding box keys.
            mode (List[str]): list of modes for RandAffined.
            rotate_range (List[str]): rotation ranges for RandAffined.
            padding_mode (str): padding mode for RandAffined.
            prob (float): probability of applying this transform.
        """

        self.image_keys = image_keys
        self.box_keys = box_keys
        self.mode = mode
        self.rotate_range = rotate_range
        self.padding_mode = padding_mode
        self.prob = prob

        self.rand_affine_d = monai.transforms.RandAffined(
            image_keys,
            mode=mode,
            rotate_range=rotate_range,
            padding_mode=padding_mode,
            prob=prob,
        )
        self.affine_box = monai.apps.detection.transforms.array.AffineBox()

    def __call__(self, X):
        X = self.rand_affine_d(X)
        # retrieve rand_affine_info
        image_example = X[self.image_keys[0]]
        rand_affine_info = self.rand_affine_d.pop_transform(image_example)
        rand_affine_info = rand_affine_info["extra_info"]["rand_affine_info"]
        if "extra_info" in rand_affine_info:
            sh = np.array(image_example.shape[1:])[np.newaxis, :]
            center = (sh - 1) / 2
            center_rep = np.concatenate([center, center], 1)
            affine = rand_affine_info["extra_info"]["affine"]
            self.last_affine = affine
            for k in self.box_keys:
                X[k] = self.affine_box(X[k] - center_rep, affine) + center_rep
        return X


class SampleChannelDim(monai.transforms.Transform):
    """
    Randomly selects n_channels channels from a multi-channel dim.
    """

    def __init__(self, n_channels: int, channel_dim: int = 0):
        """
        Args:
            n_channels (int): number of channels to be randomly selected.
            channel_dim (int, optional): dimension for the channels. Defaults
                to 0.
        """
        self.n_channels = n_channels
        self.channel_dim = channel_dim

    def __call__(self, X):
        if X.shape[self.channel_dim] > self.n_channels:
            samples = np.random.choice(
                X.shape[self.channel_dim], self.n_channels, replace=False
            )
            X = X.index_select(self.channel_dim, torch.as_tensor(samples))
        return X


class SampleChannelDimd(monai.transforms.MapTransform):
    """
    Dictionary version of SampleChannelDim.
    """

    def __init__(self, keys: List[str], n_channels: int, channel_dim: int = 0):
        """
        Args:
            keys (List[str]): Keys to apply channel sampling to.
            n_channels (int): number of channels to sample.
            channel_dim (int, optional): dimension for the channels. Defaults
                to 0.
        """
        self.keys = keys
        self.n_channels = n_channels
        self.channel_dim = channel_dim

        self.transform = SampleChannelDim(self.n_channels, self.channel_dim)

    def __call__(self, X: Dict[str, torch.Tensor]):
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

    def __init__(self, size: Union[Tuple[int, int], Tuple[int, int, int]]):
        """
        Args:
            size (Union[Tuple[int, int], Tuple[int, int, int]]): crop size.
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

    def __call__(self, X: torch.Tensor) -> List[torch.Tensor]:
        if self.size is not None:
            X = [x for x in self.get_all_crops(X)]
        return X


class GetAllCropsd(monai.transforms.MapTransform):
    """
    Dictionary version of GetAllCrops.
    """

    def __init__(
        self,
        keys: List[str],
        size: Union[Tuple[int, int], Tuple[int, int, int]],
    ):
        """
        Args:
            keys (List[str]): keys to crop.
            size (Union[Tuple[int, int], Tuple[int, int, int]]): crop size.
        """
        self.keys = keys
        self.size = size

        self.tr = GetAllCrops(self.size)

    def __call__(
        self, X: Dict[str, torch.Tensor]
    ) -> List[Dict[str, torch.Tensor]]:
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

    def __init__(self, keys: List[str], ndim: int = 3, mode: str = "pad"):
        """
        Args:
            keys (List[str]): keys to adjust sizes for.
            ndim (int, optional): number of dimensions. Defaults to 3.
            mode (str, optional): adjustment mode. Can be either pad or crop.
                Defaults to "pad".
        """
        self.keys = keys
        self.ndim = ndim
        self.mode = mode

    def pad_torch(self, X: torch.Tensor, out_size: List[int]) -> torch.Tensor:
        sh = X.shape
        pad_dim = [i - j for i, j in zip(out_size, sh[-self.ndim :])]
        a = [x // 2 for x in pad_dim]
        b = [x - y for x, y in zip(pad_dim, a)]
        pad_size = []
        for i, j in zip(reversed(a), reversed(b)):
            pad_size.extend([i, j])
        return torch.nn.functional.pad(X, pad_size)

    def pad_numpy(self, X: np.ndarray, out_size: List[int]) -> np.ndarray:
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

    def pad(self, X: TensorOrNDarray, out_size: List[int]) -> TensorOrNDarray:
        if isinstance(X, torch.Tensor):
            return self.pad_torch(X, out_size)
        else:
            return self.pad_numpy(X, out_size)

    def crop(self, X: TensorOrNDarray, out_size: List[int]) -> TensorOrNDarray:
        sh = X.shape
        crop_dim = [i - j for i, j in zip(sh[-self.ndim :], out_size)]
        a = [x // 2 for x in crop_dim]
        b = [x + y for x, y in zip(a, out_size)]
        if self.ndim == 2:
            return X[..., a[0] : b[0], a[1] : b[1]]
        if self.ndim == 3:
            return X[..., a[0] : b[0], a[1] : b[1], a[2] : b[2]]

    def __call__(self, X: Dict[str, torch.Tensor]):
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

    def __call__(self, X: Union[torch.Tensor, np.ndarray]):
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
            if self.filter_by_size == True:
                sorted_labels = sorted(sizes.keys(), key=lambda k: -sizes[k])
                if len(sorted_labels) > self.keep_n:
                    sorted_labels = sorted_labels[: self.keep_n]
                labels_to_keep.extend(sorted_labels)

            if self.filter_by_dist_to_centre == True:
                sorted_labels = sorted(
                    dist_to_centre.keys(), key=lambda k: sizes[k]
                )
                label_to_keep = None
                idx = 0
                while label_to_keep is None:
                    curr_label = sorted_labels[idx]
                    if self.filter_by_size == True:
                        if curr_label in labels_to_keep == True:
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
    """Crops the input image(s) from a binary mask.

    Finds the extremes of the positive class in the binary mask along each
    dimension. Uses these to crop the image(s) to the smallest box containing
    the mask.
    """

    def __init__(
        self,
        keys: List[str] | str,
        mask_key: str,
        output_size: List[int] = None,
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

    def __call__(self, X: Dict[str, torch.Tensor]):
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
