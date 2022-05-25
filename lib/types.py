import numpy as np
import torch

from typing import Dict,Union,List,Tuple

PathDict = Dict[str,Dict[str,str]]
TensorList = List[torch.Tensor]
TensorDict = Dict[str,torch.Tensor]
TensorOrNDarray = Union[np.ndarray,torch.Tensor]
FloatOrTensor = Union[torch.Tensor,float]
SizeDict = Dict[str,List[Union[Tuple[int,int,int],Tuple[int,int]]]]
SpacingDict = Dict[str,List[Union[Tuple[float,float,float],Tuple[float,float]]]]
