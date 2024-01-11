import numpy as np
import torch
from copy import deepcopy
from typing import List, Callable, Union, Dict, Sequence, Tuple

TensorOrArray = Union[np.ndarray, torch.Tensor]
MultiFormatInput = Union[
    TensorOrArray, Dict[str, TensorOrArray], Sequence[TensorOrArray]
]
Coords = Union[Sequence[Tuple[int, int]], Sequence[Tuple[int, int, int]]]
Shape = Union[Tuple[int, int], Tuple[int, int, int]]


def get_shape(X: MultiFormatInput) -> Shape:
    """
    Gets the shape from a possibly 1-level nested structure of tensors or numpy
    arrays. Assumes that, if nested, all tensors have the same shape.

    Args:
        X (MultiFormatInput): a tensor, an array or a dictionary/list/tuple of
            tensors and arrays.

    Raises:
        NotImplementedError: error if input is not tensor, array, dict,
            tuple or list.

    Returns:
        Shape: tensor/array shape.
    """
    if isinstance(X, (np.ndarray, torch.Tensor)):
        return X.shape
    elif isinstance(X, dict):
        k = list(X.keys())[0]
        return X[k].shape
    elif isinstance(X, (tuple, list)):
        return X[0].shape
    else:
        raise NotImplementedError(
            "Supported inputs are np.ndarray, tensor, dict, tuple, list"
        )


def cat_array(X: List[TensorOrArray], *args, **kwargs) -> TensorOrArray:
    """
    Concatenates a list of arrays or tensors.

    Args:
        X (List[TensorOrArray]): List of arrays or tensors to concatenate.
        *args: Additional positional arguments to pass to concatenate.
        **kwargs: Additional keyword arguments to pass to concatenate.

    Returns:
        TensorOrArray: Concatenated array or tensor.
    """

    if isinstance(X[0], np.ndarray):
        return np.concatenate(X, *args, **kwargs)
    elif isinstance(X[0], torch.Tensor):
        return torch.cat(X, *args, **kwargs)


def stack_array(X: List[TensorOrArray], *args, **kwargs) -> TensorOrArray:
    """
    Stacks a list of arrays or tensors.

    Args:
        X (List[TensorOrArray]): List of arrays or tensors to stack.
        *args: Additional positional arguments to pass to stack.
        **kwargs: Additional keyword arguments to pass to stack.

    Returns:
        TensorOrArray: Stacked array or tensor.
    """

    if isinstance(X[0], np.ndarray):
        return np.stack(X, *args, **kwargs)
    elif isinstance(X[0], torch.Tensor):
        return torch.stack(X, *args, **kwargs)


def multi_format_cat(
    X: List[MultiFormatInput], *args, **kwargs
) -> MultiFormatInput:
    """
    Concatenates a list of multi-format inputs (np.ndarray, torch.Tensor, dict,
    tuple, list).

    Supports concatenation for each input format. Dicts are concatenated by
    key, tuples/lists are concatenated by index.

    Args:
        X: List of multi-format inputs to concatenate.
        *args: Additional args to pass to cat_array or concatenate.
        **kwargs: Additional kwargs to pass to cat_array or concatenate.

    Returns:
        Concatenated multi-format input matching X's format.
    """
    if isinstance(X[0], (np.ndarray, torch.Tensor)):
        return cat_array(X, *args, **kwargs)
    elif isinstance(X[0], dict):
        output = {k: [] for k in X[0]}
        for x in X:
            for k in x:
                output[k].append(x[k])
        return {k: cat_array(output[k]) for k in output}
    elif isinstance(X[0], (tuple, list)):
        output = [[] for _ in X[0]]
        for x in X:
            for i in range(len(x)):
                output[i].append(x[i])
        return [cat_array(o) for o in output]
    else:
        raise NotImplementedError(
            "Supported inputs are np.ndarray, dict, tuple, list"
        )


def multi_format_stack(
    X: List[MultiFormatInput], *args, **kwargs
) -> MultiFormatInput:
    """
    Stacks a list of multi-format inputs (np.ndarray, torch.Tensor, dict,
    tuple, list).

    Supports stacking for each input format. Dicts are stacked by key,
    tuples/lists are concatenated by index.

    Args:
        X: List of multi-format inputs to stack.
        *args: Additional args to pass to stack_array or stack.
        **kwargs: Additional kwargs to pass to stack_array or stack.

    Returns:
        Stacked multi-format input matching X's format.
    """
    if isinstance(X[0], (np.ndarray, torch.Tensor)):
        return stack_array(X, *args, **kwargs)
    elif isinstance(X[0], dict):
        output = {k: [] for k in X[0]}
        for x in X:
            for k in x:
                output[k].append(x[k])
        return {k: stack_array(output[k]) for k in output}
    elif isinstance(X[0], (tuple, list)):
        output = [[] for _ in X[0]]
        for x in X:
            for i in range(len(x)):
                output[i].append(x[i])
        return [stack_array(o) for o in output]
    else:
        raise NotImplementedError(
            "Supported inputs are np.ndarray, dict, tuple, list"
        )


def multi_format_stack_or_cat(
    X: List[MultiFormatInput], ndim: int, *args, **kwargs
) -> MultiFormatInput:
    """
    Concatenates/stacks a list of multi-format inputs (np.ndarray,
    torch.Tensor, dict, tuple, list).

    Supports concatenation/stacking for each input format. Dicts are
    concatenated/stacked by key, tuples/lists are concatenated by index.

    Concatenation is performed when the shape of the tensor in the nested
    structure is equal to ndim - 2; stacking happens when this is equal to
    ndim - 1.

    Args:
        X: List of multi-format inputs to concatenate/stack.
        *args: Additional args to pass to multi_format_cat/multi_format_stack.
        **kwargs: Additional kwargs to pass to
            multi_format_cat/multi_format_stack.

    Returns:
        Stacked/concatenated multi-format input matching X's format.
    """
    sh = get_shape(X[0])
    if len(sh) == ndim + 2:
        return multi_format_cat(X, *args, **kwargs)
    elif len(sh) == ndim + 1:
        return multi_format_stack(X, *args, **kwargs)


class TensorListReduction:
    """
    Reduces a list of tensors to a single tensor using the given strategy.
    Applies optional pre- and post-processing functions. Default strategy
    is to take the mean.
    """

    def __init__(
        self,
        preproc_fn: callable = None,
        postproc_fn: callable = None,
        strategy: str = "mean",
    ):
        """
        Args:
            preproc_fn (callable, optional): preprocessing function (applied to
                all elements of input when called). Defaults to None.
            postproc_fn (callable, optional): postprocessing function (applied
                to reduced output). Defaults to None.
            strategy (str, optional): strategy for reduction. Defaults to
                "mean" (only has mean available).
        """
        assert strategy in ["mean"]
        self.preproc_fn = preproc_fn
        self.postproc_fn = postproc_fn
        self.strategy = strategy

    def __call__(self, X: list[torch.Tensor] | torch.Tensor):
        if isinstance(X, (list, tuple)):
            if self.preproc_fn is not None:
                X = [self.preproc_fn(x) for x in X]
            if self.strategy == "mean":
                X = torch.stack(X).mean(0)
        else:
            if self.preproc_fn is not None:
                X = self.preproc_fn(X)
        if self.postproc_fn is not None:
            X = self.postproc_fn(X)
        return X


class FlippedInference:
    """
    Flips the input and runs inference on each flip, reverting the flip after
    each trial.
    """

    def __init__(
        self,
        inference_function: Callable,
        flips: List[List[int]],
        flip_idx: List[int] = None,
        flip_keys: List[str] = None,
        ndim: int = 3,
        inference_batch_size: int = 1,
    ):
        """
        Args:
            inference_function (Callable): base inference function.
            flips (List[List[int]]): list of dimensions that should be flipped.
            flip_idx (list[int]): dimension index for flipping. Defaults to None.
            flip_keys (list[str], optional): list of keys for flipping. Defaults
                to ["image"].
            ndim (int, optional): number of spatial dimensions. Defaults to 3.
            inference_batch_size (int, optional): size of batch size for
                inference. Defaults to 1.
        """
        self.inference_function = inference_function
        self.flips = flips
        self.flip_idx = flip_idx
        self.flip_keys = flip_keys
        self.ndim = ndim
        self.inference_batch_size = inference_batch_size

    def flip_array(self, X: TensorOrArray, axis: Tuple[int]) -> TensorOrArray:
        if isinstance(X[0], np.ndarray):
            return np.flip(X, axis)
        elif isinstance(X[0], torch.Tensor):
            return torch.flip(X, axis)

    def flip(self, X: MultiFormatInput, axis: List[int]) -> Shape:
        """
        Flips tensors in a possibly 1-level nested structure of tensors or numpy
        arrays. Assumes that, if nested, all tensors have the same shape.

        Args:
            X (MultiFormatInput): a tensor, an array or a dictionary/list/tuple of
                tensors and arrays.
            axis (List[int]): list of flip dimensions.

        Raises:
            NotImplementedError: error if input is not tensor, array, dict,
                tuple or list.

        Returns:
            MultiFormatInput: input with flipped tensors.
        """
        axis = tuple(axis)
        if isinstance(X, (np.ndarray, torch.Tensor)):
            return self.flip_array(deepcopy(X), axis)
        elif isinstance(X, dict):
            X_out = deepcopy(X)
            for k in X:
                if self.flip_keys is not None:
                    if k in self.flip_keys:
                        X_out[k] = self.flip_array(X_out[k], axis)
                else:
                    X_out[k] = self.flip_array(X_out[k], axis)
            return X_out
        elif isinstance(X, (tuple, list)):
            X_out = deepcopy(X)
            for k in X:
                if self.flip_keys is not None:
                    if k in self.flip_keys:
                        X_out[k] = self.flip_array(X_out[k], axis)
                else:
                    X_out[k] = self.flip_array(X_out[k], axis)
            return X_out
        else:
            raise NotImplementedError(
                "Supported inputs are np.ndarray, dict, tuple, list"
            )

    def __call__(self, X: MultiFormatInput, *args, **kwargs) -> TensorOrArray:
        batch = []
        batch_flips = []
        output = None
        if (self.ndim + 1) == len(get_shape(X)):
            original_batch_size = 1
            shift = 1
        else:
            original_batch_size = get_shape(X)[0]
            shift = 0
        for flip in self.flips:
            flipped_X = self.flip(X, flip)
            batch.append(flipped_X)
            batch_flips.append(flip)
            if len(batch) == self.inference_batch_size:
                batch = multi_format_stack_or_cat(batch, self.ndim)
                with torch.no_grad():
                    result = self.inference_function(
                        batch, *args, **kwargs
                    ).detach()
                result = [
                    self.flip(x, tuple([ff + shift for ff in f]))
                    for x, f in zip(
                        torch.split(result, original_batch_size),
                        batch_flips,
                    )
                ]
                result = torch.stack(result, -1).sum(-1)
                # lazy output
                if output is None:
                    output = torch.zeros_like(result)
                output = output + result
                batch = []
                batch_flips = []
        if len(batch) > 0:
            batch = multi_format_stack_or_cat(batch, self.ndim)
            with torch.no_grad():
                result = self.inference_function(
                    batch, *args, **kwargs
                ).detach()
            result = [self.flip(x, f) for x, f in zip(X, batch_flips)]
            result = torch.stack(result, -1).sum(-1)
            # lazy output
            if output is None:
                output = torch.zeros_like(result)
            output = output + result
            batch = []
            batch_flips = []
        return output / len(self.flips)


class SlidingWindowSegmentation:
    """
    A sliding window inferece operator for image segmentation. The only
    specifications in the constructor are the sliding window size, the inference
    function (a forward method from torch or predict_step from Lightning, for
    example), and an optional stride (recommended as this avoids edge artifacts).

    The supported inputs are np.ndarray, torch tensors or simple strucutres
    constructed from them (lists, dictionaries, tuples). If a structure is provided,
    the inference operator expects all entries to have the same size, i.e. the same
    height, width, depth. Supports both 2D and 3D inputs.

    Assumes all inputs are batched and have a channel dimension.
    """

    def __init__(
        self,
        sliding_window_size: Shape,
        inference_function: Callable,
        n_classes: int,
        stride: Shape = None,
        inference_batch_size: int = 1,
    ):
        """
        Args:
            sliding_window_size (Shape): size of the sliding window. Should be
                a sequence of integers.
            inference_function (Callable): function that produces an inference.
            n_classes (int): number of channels in the output.
            stride (Shape, optional): stride for the sliding window. Defaults to
                None (same as sliding_window_size).
            inference_batch_size (int, optional): batch size for inference.
                Defaults to 1.
        """
        self.sliding_window_size = sliding_window_size
        self.inference_function = inference_function
        self.n_classes = n_classes
        self.stride = stride
        self.inference_batch_size = inference_batch_size

        if self.stride is None:
            self.stride = self.sliding_window_size

        self.ndim = len(sliding_window_size)

    def adjust_if_necessary(
        self, x1: int, x2: int, M: int, a: int
    ) -> Tuple[int, int]:
        """Adjusts coordinate bounds x2 and x1 if x2 is larger than M. If that is
        the case, x1 is adjusted to M-a and x2 is adjusted to M.

        Args:
            x1 (int): lower bound.
            x2 (int): upper bound.
            M (int): maximum.
            a (int): distance between bounds.

        Returns:
            Tuple[int,int]: adjusted x1 and x2.
        """
        if x2 > M:
            x1, x2 = M - a, M
        return x1, x2

    def get_device(self, X: MultiFormatInput) -> str:
        """
        Gets the device from a possibly 1-level nested structure of tensors or numpy
        arrays. Assumes that, if nested, all tensors have the same shape.

        Args:
            X (MultiFormatInput): a tensor, an array or a dictionary/list/tuple of
                tensors and arrays.

        Raises:
            NotImplementedError: error if input is not tensor, array, dict,
                tuple or list.

        Returns:
            str: device.
        """
        if isinstance(X, (np.ndarray, torch.Tensor)):
            return X.device
        elif isinstance(X, dict):
            k = list(X.keys())[0]
            return X[k].device
        elif isinstance(X, (tuple, list)):
            return X[0].device
        else:
            raise NotImplementedError(
                "Supported inputs are np.ndarray, dict, tuple, list"
            )

    def get_zeros_fn(self, X: MultiFormatInput) -> Callable:
        """
        Gets the zero function from a possibly 1-level nested structure of
        tensors or numpy arrays.

        Args:
            X (MultiFormatInput): a tensor, an array or a dictionary/list/tuple of
                tensors and arrays.

        Raises:
            NotImplementedError: error if input is not tensor, array, dict,
                tuple or list.

        Returns:
            Callable: np.zeros if inferred type is np.ndarray, torch.zeros if inferred
                type is torch.Tensor.
        """
        if isinstance(X, (np.ndarray, torch.Tensor)):
            x = X
        elif isinstance(X, dict):
            k = list(X.keys())[0]
            x = X[k]
        elif isinstance(X, (tuple, list)):
            x = X[0]
        else:
            raise NotImplementedError(
                "Supported inputs are np.ndarray, dict, tuple, list"
            )
        if isinstance(x, np.ndarray):
            return np.zeros
        elif isinstance(x, torch.Tensor):
            return torch.zeros

    def get_split_fn(self, X: MultiFormatInput) -> Shape:
        """
        Gets the split function from a possibly 1-level nested structure of
        tensors or numpy arrays.

        Args:
            X (MultiFormatInput): a tensor, an array or a dictionary/list/tuple of
                tensors and arrays.

        Raises:
            NotImplementedError: error if input is not tensor, array, dict,
                tuple or list.

        Returns:
            Callable: np.split if inferred type is np.ndarray, torch.split if inferred
                type is torch.Tensor.
        """
        if isinstance(X, (np.ndarray, torch.Tensor)):
            x = X
        elif isinstance(X, dict):
            k = list(X.keys())[0]
            x = X[k]
        elif isinstance(X, (tuple, list)):
            x = X[0]
        else:
            raise NotImplementedError(
                "Supported inputs are np.ndarray, dict, tuple, list"
            )
        if isinstance(x, np.ndarray):
            return np.split
        elif isinstance(x, torch.Tensor):
            return torch.split

    def extract_patch_from_array(
        self, X: TensorOrArray, coords: Coords
    ) -> TensorOrArray:
        """
        Extracts a patch from a tensor or array using coords. I.e., for a 2d input:

        ```
        (x1,x2),(y1,y2) = coords
        patch = X[:,:,x1:x2,y1:y2]
        ```

        If the input shape is not compliant with coords (2 + len(coords)), the
        input is returned without crops.

        Args:
            X (TensorOrArray): tensor or array.
            coords (Coords): tuple with coordinate pairs. Should have 2 or 3
                coordinate pairs for 2D and 3D inputs, respectively.

        Returns:
            TensorOrArray: patch from X.
        """
        if self.ndim == 2:
            (x1, x2), (y1, y2) = coords
            return X[..., x1:x2, y1:y2]
        elif self.ndim == 3:
            (x1, x2), (y1, y2), (z1, z2) = coords
            return X[..., x1:x2, y1:y2, z1:z2]

    def extract_patch(
        self, X: MultiFormatInput, coords: Coords
    ) -> MultiFormatInput:
        """
        Extracts patches from a possibly 1-level nested structure of tensors or
        numpy arrays.

        Args:
            X (MultiFormatInput): a tensor, an array or a dictionary/list/tuple of
                tensors and arrays.
            coords (Coords): tuple with coordinate pairs. Should have 2 or 3
                coordinate pairs for 2D and 3D inputs, respectively.

        Raises:
            NotImplementedError: error if input is not tensor, array, dict,
                tuple or list.

        Returns:
            MultiFormatInput: a possibly 1-level nested structure of tensors or
                numpy array patches.
        """
        if isinstance(X, (np.ndarray, torch.Tensor)):
            return self.extract_patch_from_array(X, coords)
        elif isinstance(X, dict):
            return {
                k: self.extract_patch_from_array(X[k], coords)
                for k in X
                if isinstance(X[k], (np.ndarray, torch.Tensor))
            }
        elif isinstance(X, (tuple, list)):
            return [
                self.extract_patch_from_array(x, coords)
                for x in X
                if isinstance(x, (np.ndarray, torch.Tensor))
            ]
        else:
            st = "Supported inputs are np.ndarray, torch.Tensor, dict, tuple, list"
            raise NotImplementedError(st)

    def get_all_crops_2d(
        self, X: MultiFormatInput
    ) -> Tuple[MultiFormatInput, Coords]:
        """
        Gets all crops in a 2D image. For a given set of crop bounds x1 and x2,
        if any of these exceeds the input size, they are adjusted such that they
        are fully contained within the image, with x2 corresponding to the image
        size on that dimension.

        Args:
            X (MultiFormatInput): a tensor, an array or a dictionary/list/tuple of
                tensors and arrays.

        Yields:
            X (MultiFormatInput): a tensor patch, an array patch or a
                dictionary/list/tuple of tensor/array patches.
        """
        sh = get_shape(X)[-self.ndim :]
        for i in range(0, sh[0], self.stride[0]):
            for j in range(0, sh[1], self.stride[1]):
                i_1, j_1 = i, j
                i_2 = i_1 + self.sliding_window_size[0]
                j_2 = j_1 + self.sliding_window_size[1]
                i_1, i_2 = self.adjust_if_necessary(
                    i_1, i_2, sh[0], self.sliding_window_size[0]
                )
                j_1, j_2 = self.adjust_if_necessary(
                    j_1, j_2, sh[1], self.sliding_window_size[1]
                )
                coords = ((i_1, i_2), (j_1, j_2))
                yield self.extract_patch(X, coords), coords

    def get_all_crops_3d(
        self, X: MultiFormatInput
    ) -> Tuple[MultiFormatInput, Coords]:
        """
        Gets all crops in a 3D image. For a given set of crop bounds x1 and x2,
        if any of these exceeds the input size, they are adjusted such that they
        are fully contained within the image, with x2 corresponding to the image
        size on that dimension.

        Args:
            X (MultiFormatInput): a tensor, an array or a dictionary/list/tuple of
                tensors and arrays.

        Yields:
            X (MultiFormatInput): a tensor patch, an array patch or a
                dictionary/list/tuple of tensor/array patches.
        """
        sh = get_shape(X)[-self.ndim :]
        for i in range(0, sh[0], self.stride[0]):
            for j in range(0, sh[1], self.stride[1]):
                for k in range(0, sh[2], self.stride[2]):
                    i_1, j_1, k_1 = i, j, k
                    i_2 = i_1 + self.sliding_window_size[0]
                    j_2 = j_1 + self.sliding_window_size[1]
                    k_2 = k_1 + self.sliding_window_size[2]
                    i_1, i_2 = self.adjust_if_necessary(
                        i_1, i_2, sh[0], self.sliding_window_size[0]
                    )
                    j_1, j_2 = self.adjust_if_necessary(
                        j_1, j_2, sh[1], self.sliding_window_size[1]
                    )
                    k_1, k_2 = self.adjust_if_necessary(
                        k_1, k_2, sh[2], self.sliding_window_size[2]
                    )
                    coords = ((i_1, i_2), (j_1, j_2), (k_1, k_2))
                    yield self.extract_patch(X, coords), coords

    def get_all_crops(self, X: TensorOrArray) -> TensorOrArray:
        """
        Convenience function articulating get_all_crops_2d and get_all_crops_3d.

        Args:
            X (MultiFormatInput): a tensor, an array or a dictionary/list/tuple of
                tensors and arrays.

        Yields:
            X (MultiFormatInput): a tensor patch, an array patch or a
                dictionary/list/tuple of tensor/array patches.
        """
        if self.ndim == 2:
            yield from self.get_all_crops_2d(X)
        if self.ndim == 3:
            yield from self.get_all_crops_3d(X)

    def adjust_edges(self, X: TensorOrArray) -> TensorOrArray:
        return X

    def update_output(
        self,
        output_array: TensorOrArray,
        output_denominator: TensorOrArray,
        tmp_out: TensorOrArray,
        coords: Shape,
    ) -> Tuple[TensorOrArray, TensorOrArray]:
        """
        Updates the output array and the output denominator given a tmp_out and
        coords (tmp_out and coords should be compatible).

        Args:
            output_array (TensorOrArray): array/tensor storing the sum of all
                tmp_out array/tensor.
            output_denominator (TensorOrArray): array/tensor storing the
                denominator for all tmp_out array/tensor.
            tmp_out (TensorOrArray): new patch to be added to output_array.
            coords (Shape): new coords where tmp_out should be added and where
                output_denominator should be updated.

        Returns:
            Tuple[TensorOrArray,TensorOrArray]: updated output_array and
                output_denominator.
        """
        if self.ndim == 2:
            (x1, x2), (y1, y2) = coords
            output_array[..., x1:x2, y1:y2] += tmp_out.squeeze(0).squeeze(0)
            output_denominator[..., x1:x2, y1:y2] += 1.0
        elif self.ndim == 3:
            (x1, x2), (y1, y2), (z1, z2) = coords
            output_array[..., x1:x2, y1:y2, z1:z2] += tmp_out.squeeze(
                0
            ).squeeze(0)
            output_denominator[..., x1:x2, y1:y2, z1:z2] += 1.0
        return output_array, output_denominator

    def __call__(self, X: MultiFormatInput, *args, **kwargs) -> TensorOrArray:
        """
        Extracts patches for a given input tensor/array X, predicts the
        segmentation outputs for all cases and aggregates all outputs. If a
        prediction is done twice for the same region (due to striding), the average
        value for these regions is used.

        Args:
            X (MultiFormatInput): a tensor, an array or a dictionary/list/tuple of
                tensors and arrays.
            args, kwargs: arguments and keyword arguments for inference_function.

        Returns:
            TensorOrArray: output prediction.
        """
        output_size = list(get_shape(X))
        device = self.get_device(X)
        zeros_fn = self.get_zeros_fn(X)
        split_fn = self.get_split_fn(X)
        # this condition checks to see if the input is unbatched
        if len(output_size) == (self.ndim + 2):
            output_size[1] = self.n_classes
        elif len(output_size) < (self.ndim + 2):
            output_size[0] = self.n_classes
        else:
            raise Exception(
                "length of input array shape should be <= self.ndim+2"
            )
        output_array = zeros_fn(output_size).to(device)
        output_denominator = zeros_fn(output_size).to(device)
        batch = []
        batch_coords = []
        for cropped_input, coords in self.get_all_crops(X):
            batch.append(cropped_input)
            batch_coords.append(coords)
            if len(batch) == self.inference_batch_size:
                original_batch_size = output_size[0]
                batch = multi_format_stack_or_cat(batch, self.ndim)
                with torch.no_grad():
                    batch_out = self.inference_function(batch, *args, **kwargs)
                batch_out = split_fn(batch_out, original_batch_size, 0)
                for out, coords in zip(batch_out, batch_coords):
                    output_array, output_denominator = self.update_output(
                        output_array, output_denominator, out, coords
                    )
                batch = []
                batch_coords = []
        if len(batch) > 0:
            original_batch_size = output_size[0]
            batch = multi_format_stack_or_cat(batch, self.ndim)
            with torch.no_grad():
                batch_out = self.inference_function(batch, *args, **kwargs)
            batch_out = split_fn(batch_out, original_batch_size, 0)
            for out, coords in zip(batch_out, batch_coords):
                output_array, output_denominator = self.update_output(
                    output_array, output_denominator, out, coords
                )
        output_array = output_array / output_denominator
        return output_array


class SegmentationInference:
    """
    Coordinates a sliding window inference operator and a flipped inference
    operator into a single convenient function.
    """

    def __init__(
        self,
        base_inference_function: list[Callable] | Callable,
        sliding_window_size: list[int] = None,
        stride: list[int] = None,
        inference_batch_size: int = 1,
        n_classes: int = 2,
        flip: bool = False,
        flip_idx: List[int] = None,
        flip_keys: List[str] = ["image"],
        ndim: int = 3,
        reduction: callable = None,
    ):
        """
        Args:
            base_inference_function (list[Callable] | Callable): base inference
                function. If this is a list, the output is averaged at the end.
            sliding_window_size (Shape): size of the sliding window. Should be
                a sequence of integers.
            n_classes (int): number of channels in the output.
            stride (Shape | float, optional): stride for the sliding window.
                Defaults to None (same as sliding_window_size). If float,
                sliding_window_size is multiplied by stride to obtain the
                actual stride size.

            flip (bool, optional): triggers flipped prediction. Defaults to False.
            flip_idx (list[int]): dimension index for flipping. Defaults to None.
            flip_keys (list[str], optional): list of keys for flipping. Defaults
                to ["image"].
            ndim (int, optional): number of spatial dimensions. Defaults to 3.

            inference_batch_size (int, optional): batch size for inference.
                Defaults to 1.

            reduction (callable, optional): sets strategy for reduction when a
                list of inference functions are provided. Defaults to None (no
                reduction)
        """
        self.base_inference_function = base_inference_function
        self.sliding_window_size = sliding_window_size
        self.n_classes = n_classes
        self.stride = stride
        self.flip = flip
        self.flip_idx = flip_idx
        self.flip_keys = flip_keys
        self.ndim = ndim
        self.inference_batch_size = inference_batch_size
        self.reduction = reduction

        self.update_base_inference_function(base_inference_function)

    def update_base_inference_function(
        self, base_inference_function: list[callable]
    ):
        if base_inference_function is None:
            return
        inference_function = base_inference_function
        if isinstance(inference_function, (list, tuple)):
            if self.sliding_window_size is not None:
                if isinstance(self.stride, float):
                    self.stride = [
                        int(x * self.stride) for x in self.sliding_window_size
                    ]
                inference_function = [
                    SlidingWindowSegmentation(
                        inference_function=fn,
                        sliding_window_size=self.sliding_window_size,
                        n_classes=self.n_classes if self.n_classes > 2 else 1,
                        stride=self.stride,
                        inference_batch_size=self.inference_batch_size,
                    )
                    for fn in inference_function
                ]
            if self.flip == True:
                flips = [(1,), (2,), (3,)]
                inference_function = [
                    FlippedInference(
                        inference_function=fn,
                        flips=flips,
                        flip_idx=self.flip_idx,
                        flip_keys=self.flip_keys,
                        ndim=self.ndim,
                        inference_batch_size=self.inference_batch_size,
                    )
                    for fn in inference_function
                ]
        else:
            if self.sliding_window_size is not None:
                if isinstance(self.stride, float):
                    self.stride = [
                        int(x * self.stride) for x in self.sliding_window_size
                    ]
                inference_function = SlidingWindowSegmentation(
                    sliding_window_size=self.sliding_window_size,
                    inference_function=inference_function,
                    n_classes=self.n_classes if self.n_classes > 2 else 1,
                    stride=self.stride,
                    inference_batch_size=self.inference_batch_size,
                )
            if self.flip == True:
                flips = [(1,), (2,), (3,)]
                inference_function = FlippedInference(
                    inference_function=inference_function,
                    flips=flips,
                    flip_idx=self.flip_idx,
                    flip_keys=self.flip_keys,
                    ndim=self.ndim,
                    inference_batch_size=len(flips),
                )
        self.inference_function = inference_function

    def __call__(self, X: MultiFormatInput, *args, **kwargs) -> TensorOrArray:
        if isinstance(self.inference_function, (list, tuple)):
            with torch.no_grad():
                output = [
                    inference_function(X, *args, **kwargs)
                    for inference_function in self.inference_function
                ]
                if self.reduction is not None:
                    output = self.reduction(output)
        else:
            with torch.no_grad():
                output = self.inference_function(X, *args, **kwargs)
                if self.reduction is not None:
                    output = self.reduction(output)

        return output
