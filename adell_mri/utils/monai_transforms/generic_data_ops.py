import numpy as np
import monai
from copy import deepcopy


class CopyEntryd(monai.transforms.Transform):
    """
    Transform that copies an entry in a dictionary with a new key.
    """

    def __init__(self, keys: list[str], out_keys: dict[str, str]):
        """
        Args:
            keys (list[str]): list of keys.
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
        nested_pattern: list[str],
        output_key: str = None,
    ):
        """
        Args:
            transform_key (str): key corresponding to transform parameters.
            transform_class (str): string corresponding to the class name.
            nested_pattern (list[str]): nested pattern of strings to recover
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
        nested_pattern: list[str],
        output_key: str = None,
    ):
        """
        Args:
            key (str): key corresponding to relevant MetaTensor.
            transform_class (str): string corresponding to the class name.
            nested_pattern (list[str]): nested pattern of strings to recover
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
