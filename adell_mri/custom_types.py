"""
Contains a set of useful custom types for development.
"""

from typing import Callable, Dict, Iterable, List, Tuple, Union

import numpy as np
import torch

DatasetDict = Dict[str, Dict[str, str]]
TensorList = List[torch.Tensor]
TensorDict = Dict[str, torch.Tensor]
TensorIterable = Iterable[torch.Tensor]
NDArrayOrTensor = Union[np.ndarray, torch.Tensor]
NDArrayOrTensorDict = Dict[str, NDArrayOrTensor]
FloatOrTensor = Union[torch.Tensor, float]
SizeDict = Dict[str, List[Union[Tuple[int, int, int], Tuple[int, int]]]]
SpacingDict = Dict[
    str, List[Union[Tuple[float, float, float], Tuple[float, float]]]
]
BBDict = Dict[str, Dict[str, Union[int, float]]]
ModuleList = Union[List[torch.nn.Module], torch.nn.ModuleList]
AveragingFunctionType = Callable[
    [torch.Tensor, torch.Tensor, torch.LongTensor], torch.FloatTensor
]
Size2dOr3d = Union[Tuple[int, int], Tuple[int, int, int]]
