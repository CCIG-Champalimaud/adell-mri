from itertools import product
from typing import Any, Iterable

import monai
import numpy as np
import torch
from skimage import measure

from ...custom_types import Size2dOr3d, TensorOrNDarray


class BBToAdjustedAnchors(monai.transforms.Transform):
    """
    Maps bounding boxes in corner format (x1y1z1x2y2z2) to their anchor
    representation.
    """

    def __init__(
        self,
        anchor_sizes: np.ndarray | list[list],
        input_sh: Iterable,
        output_sh: Iterable,
        iou_thresh: float,
    ):
        """
        Args:
            anchor_sizes (Union[np.ndarray,list[list]]): a two dimensional
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
            for anchor_idx, long_anchor in enumerate(self.long_anchors):
                anchor_size = long_anchor[0, :, 1] - long_anchor[0, :, 0]
                rel_bb_size_adj = np.log(rel_bb_size / anchor_size)
                anchor_dim = long_anchor[0, :, 1] - long_anchor[0, :, 0]
                intersects = np.logical_and(
                    np.all(long_anchor[:, :, 1] > rel_bb_vert[i, :, 0], axis=1),
                    np.all(long_anchor[:, :, 0] < rel_bb_vert[i, :, 1], axis=1),
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
                            tuple([1 + k + anchor_idx * 7 for k in range(7)]),
                            *box_coords[j],
                        ]
                    )
                    idx_cl = tuple([0, *box_coords[j]])
                    v = np.array([iou, *center_adjustment[j], *rel_bb_size_adj])
                    if iou > output[idx][0]:
                        output[idx] = v
                        output[idx_cl] = cl
                        hits += 1

        return output

    def adjusted_anchors_to_bb_vertices(
        self, anchor_map: np.ndarray
    ) -> tuple[np.ndarray, np.ndarray]:
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
            correct_dims = np.multiply(adj_anchors_long[:, 4:], rel_anchor_size)
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
        input_sh: tuple[int],
        output_sh: tuple[int],
        iou_thresh: float,
        bb_key: str = "bb",
        class_key: str = "class",
        shape_key: str = "shape",
        output_key: dict[str, str] = {},
    ):
        """
        Args:
            anchor_sizes (Union[np.ndarray,list[list]]): a two dimensional
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
            output_key (dict[str,str], optional): key for output. Defaults to
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
    ) -> tuple[list[np.ndarray], list[int], Size2dOr3d]:
        """
        Converts a binary mask with a channel (first) dimension into a set of
        bounding boxes, classes and shape. The bounding boxes are obtained as the
        upper and lower corners of the connected components and the classes are
        obtained as the median value of the pixels/voxels in those connected
        components. The shape is the shape of the input X.

        Args:
            X (TensorOrNDarray): an array with shape [1,H,W] or [1,H,W,D].

        Returns:
            tuple[list[np.ndarray], list[int], Size2dOr3d]: a list with bounding
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
        keys: list[str],
        bounding_box_key: str = "bounding_boxes",
        classes_key: str = "classes",
        shape_key: str = "shape",
        mask_mode: str = "mask_is_labels",
        replace: bool = True,
    ):
        """
        Args:
            keys (list[str]): list of keys.
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
                elif self.bounding_box_key in data and self.replace is False:
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
        image_keys: list[str],
        box_key: str,
        box_key_nest: str = None,
        axes: list[int] = [0, 1, 2],
        prob: float = 0.5,
    ):
        """
        Args:
            image_keys (list[str]): keys for images that will be flipped.
            box_key (str): keys for bounding boxes that will be flipped.
            box_key_nest (str): optional key that considers that bounding
            boxes are nested in dictionaries. Defaults to None (no nesting).
            axes (list[int], optional): axes where flipping will occur.
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


class RandAffineWithBoxesd(monai.transforms.RandomizableTransform):
    """
    EXPERIMENTAL.
    Uses MONAI's `RandAffined` to transform an image and then applies
    the same affine transform to a bounding box with format xy(z)xy(z).
    """

    def __init__(
        self, image_keys: list[str], box_keys: list[str], *args, **kwargs
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
        image_keys: list[str],
        box_keys: list[str],
        mode: list[str],
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
