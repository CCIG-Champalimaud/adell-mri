import numpy as np
import torch
from copy import deepcopy
from typing import List,Callable,Union,Dict,Sequence,Tuple

TensorOrArray = Union[np.ndarray,torch.Tensor]
MultiFormatInput = Union[TensorOrArray,
                         Dict[str,TensorOrArray],
                         Sequence[TensorOrArray]]
Coords = Union[Sequence[Tuple[int,int]],Sequence[Tuple[int,int,int]]]
Shape = Union[Tuple[int,int],Tuple[int,int,int]]

def get_shape(X:MultiFormatInput)->Shape:
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
    if isinstance(X,(np.ndarray,torch.Tensor)):
        return X.shape
    elif isinstance(X,dict):
        k = list(X.keys())[0]
        return X[k].shape
    elif isinstance(X,(tuple,list)):
        return X[0].shape
    else:
        raise NotImplementedError(
            "Supported inputs are np.ndarray, tensor, dict, tuple, list")

def cat_array(X:List[TensorOrArray],*args,**kwargs)->TensorOrArray:
    if isinstance(X[0],np.ndarray):
        return np.concatenate(X,*args,**kwargs)
    elif isinstance(X[0],torch.Tensor):
        return torch.cat(X,*args,**kwargs)

def stack_array(X:List[TensorOrArray],*args,**kwargs)->TensorOrArray:
    if isinstance(X[0],np.ndarray):
        return np.stack(X,*args,**kwargs)
    elif isinstance(X[0],torch.Tensor):
        return torch.stack(X,*args,**kwargs)

def multi_format_cat(X:List[MultiFormatInput],
                     *args,**kwargs)->MultiFormatInput:
    if isinstance(X[0],(np.ndarray,torch.Tensor)):
        return cat_array(X,*args,**kwargs)
    elif isinstance(X[0],dict):
        output = {k:[] for k in X[0]}
        for x in X:
            for k in x:
                output[k].append(x[k])
        return {k:cat_array(output[k]) for k in output}
    elif isinstance(X[0],(tuple,list)):
        output = [[] for _ in X[0]]
        for x in X:
            for i in range(len(x)):
                output[i].append(x[i])
        return [cat_array(o) for o in output]
    else:
        raise NotImplementedError(
            "Supported inputs are np.ndarray, dict, tuple, list")

def multi_format_stack(X:List[MultiFormatInput],
                       *args,**kwargs)->MultiFormatInput:
    if isinstance(X[0],(np.ndarray,torch.Tensor)):
        return stack_array(X,*args,**kwargs)
    elif isinstance(X[0],dict):
        output = {k:[] for k in X[0]}
        for x in X:
            for k in x:
                output[k].append(x[k])
        return {k:stack_array(output[k]) for k in output}
    elif isinstance(X[0],(tuple,list)):
        output = [[] for _ in X[0]]
        for x in X:
            for i in range(len(x)):
                output[i].append(x[i])
        return [stack_array(o) for o in output]
    else:
        raise NotImplementedError(
            "Supported inputs are np.ndarray, dict, tuple, list")

def multi_format_stack_or_cat(X:List[MultiFormatInput],
                              ndim:int,
                              *args,**kwargs)->MultiFormatInput:
    sh = get_shape(X[0])
    if len(sh) == ndim + 2:
        return multi_format_cat(X,*args,**kwargs)
    elif len(sh) == ndim + 1:
        return multi_format_stack(X,*args,**kwargs)

class FlippedInference:
    """
    """
    def __init__(self,
                 inference_function:Callable,
                 flips:List[List[int]],
                 flip_idx:List[int]=None,
                 flip_keys:List[str]=None,
                 ndim:int=3,
                 inference_batch_size:int=1):
        self.inference_function = inference_function
        self.flips = flips
        self.flip_idx = flip_idx
        self.flip_keys = flip_keys
        self.ndim = ndim
        self.inference_batch_size = inference_batch_size
    
    def flip_array(self,X:TensorOrArray,axis:Tuple[int])->TensorOrArray:
        if isinstance(X[0],np.ndarray):
            return np.flip(X,axis)
        elif isinstance(X[0],torch.Tensor):
            return torch.flip(X,axis)
    
    def flip(self,X:MultiFormatInput,axis:List[int])->Shape:
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
        if isinstance(X,(np.ndarray,torch.Tensor)):
            return self.flip_array(deepcopy(X),axis)
        elif isinstance(X,dict):
            X_out = deepcopy(X)
            for k in X:
                if self.flip_keys is not None:
                    if k in self.flip_keys:
                        X_out[k] = self.flip_array(X_out[k],axis)
                else:
                    X_out[k] = self.flip_array(X_out[k],axis)
            return X_out
        elif isinstance(X,(tuple,list)):
            X_out = deepcopy(X)
            for k in X:
                if self.flip_keys is not None:
                    if k in self.flip_keys:
                        X_out[k] = self.flip_array(X_out[k],axis)
                else:
                    X_out[k] = self.flip_array(X_out[k],axis)
            return X_out
        else:
            raise NotImplementedError(
                "Supported inputs are np.ndarray, dict, tuple, list")
    
    def __call__(self,
                 X:MultiFormatInput,
                 *args,**kwargs)->TensorOrArray:
        batch = []
        batch_flips = []
        output = None
        original_batch_size = get_shape(X)[0]
        for flip in self.flips:
            flipped_X = self.flip(X,flip)
            batch.append(flipped_X)
            batch_flips.append(flip)
            if len(batch) == self.inference_batch_size:
                batch = multi_format_stack_or_cat(batch,self.ndim)
                result = self.inference_function(batch,*args,**kwargs)
                result = [self.flip(x,f) for x,f in zip(
                    torch.split(result,original_batch_size),batch_flips)]
                result = torch.stack(result,-1).sum(-1)
                # lazy output
                if output is None:
                    output = torch.zeros_like(result)
                output = output + result
                batch = []
                batch_flips = []
        if len(batch) > 0:
            batch = multi_format_stack_or_cat(batch,self.ndim)
            with torch.no_grad():
                result = self.inference_function(batch,*args,**kwargs)
            result = [self.flip(x,f) for x,f in zip(X,batch_flips)]
            result = torch.stack(result,-1).sum(-1)
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
    def __init__(self,
                 sliding_window_size:Shape,
                 inference_function:Callable,
                 n_classes:int,
                 stride:Shape=None,
                 inference_batch_size:int=1):
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
        
    def adjust_if_necessary(self,x1:int,x2:int,M:int,a:int)->Tuple[int,int]:
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
            x1,x2 = M-a,M
        return x1,x2

    def get_device(self,X:MultiFormatInput)->str:
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
        if isinstance(X,(np.ndarray,torch.Tensor)):
            return X.device
        elif isinstance(X,dict):
            k = list(X.keys())[0]
            return X[k].device
        elif isinstance(X,(tuple,list)):
            return X[0].device
        else:
            raise NotImplementedError(
                "Supported inputs are np.ndarray, dict, tuple, list")
            
    def get_zeros_fn(self,X:MultiFormatInput)->Callable:
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
        if isinstance(X,(np.ndarray,torch.Tensor)):
            x = X
        elif isinstance(X,dict):
            k = list(X.keys())[0]
            x = X[k]
        elif isinstance(X,(tuple,list)):
            x = X[0]
        else:
            raise NotImplementedError(
                "Supported inputs are np.ndarray, dict, tuple, list")
        if isinstance(x,np.ndarray):
            return np.zeros
        elif isinstance(x,torch.Tensor):
            return torch.zeros

    def get_split_fn(self,X:MultiFormatInput)->Shape:
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
        if isinstance(X,(np.ndarray,torch.Tensor)):
            x = X
        elif isinstance(X,dict):
            k = list(X.keys())[0]
            x = X[k]
        elif isinstance(X,(tuple,list)):
            x = X[0]
        else:
            raise NotImplementedError(
                "Supported inputs are np.ndarray, dict, tuple, list")
        if isinstance(x,np.ndarray):
            return np.split
        elif isinstance(x,torch.Tensor):
            return torch.split

    def extract_patch_from_array(self,X:TensorOrArray,coords:Coords)->TensorOrArray:
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
            (x1,x2),(y1,y2) = coords
            return X[...,x1:x2,y1:y2]
        elif self.ndim == 3:
            (x1,x2),(y1,y2),(z1,z2) = coords
            return X[...,x1:x2,y1:y2,z1:z2]

    def extract_patch(self,X:MultiFormatInput,coords:Coords)->MultiFormatInput:
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
        if isinstance(X,(np.ndarray,torch.Tensor)):
            return self.extract_patch_from_array(X,coords)
        elif isinstance(X,dict):
            return {k:self.extract_patch_from_array(X[k],coords) for k in X
                    if isinstance(X[k],(np.ndarray,torch.Tensor))}
        elif isinstance(X,(tuple,list)):
            return [self.extract_patch_from_array(x,coords) for x in X
                    if isinstance(x,(np.ndarray,torch.Tensor)]
        else:
            st = "Supported inputs are np.ndarray, torch.Tensor, dict, tuple, list"
            raise NotImplementedError(st)

    def get_all_crops_2d(self,X:MultiFormatInput)->Tuple[MultiFormatInput,
                                                         Coords]:
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
        sh = get_shape(X)[-self.ndim:]
        for i in range(0,sh[0],self.stride[0]):
            for j in range(0,sh[1],self.stride[1]):
                i_1,j_1 = i,j
                i_2 = i_1 + self.sliding_window_size[0]
                j_2 = j_1 + self.sliding_window_size[1]
                i_1,i_2 = self.adjust_if_necessary(
                    i_1,i_2,sh[0],self.sliding_window_size[0])
                j_1,j_2 = self.adjust_if_necessary(
                    j_1,j_2,sh[1],self.sliding_window_size[1])
                coords = ((i_1,i_2),(j_1,j_2))
                yield self.extract_patch(X,coords),coords

    def get_all_crops_3d(self,X:MultiFormatInput)->Tuple[MultiFormatInput,
                                                         Coords]:
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
        sh = get_shape(X)[-self.ndim:]
        for i in range(0,sh[0],self.stride[0]):
            for j in range(0,sh[1],self.stride[1]):
                for k in range(0,sh[2],self.stride[2]):
                    i_1,j_1,k_1 = i,j,k
                    i_2 = i_1 + self.sliding_window_size[0]
                    j_2 = j_1 + self.sliding_window_size[1]
                    k_2 = k_1 + self.sliding_window_size[2]
                    i_1,i_2 = self.adjust_if_necessary(
                        i_1,i_2,sh[0],self.sliding_window_size[0])
                    j_1,j_2 = self.adjust_if_necessary(
                        j_1,j_2,sh[1],self.sliding_window_size[1])
                    k_1,k_2 = self.adjust_if_necessary(
                        k_1,k_2,sh[2],self.sliding_window_size[2])
                    coords = ((i_1,i_2),(j_1,j_2),(k_1,k_2))
                    yield self.extract_patch(X,coords),coords

    def get_all_crops(self,X:TensorOrArray)->TensorOrArray:
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

    def adjust_edges(self,X:TensorOrArray)->TensorOrArray:
        return X
    
    def update_output(self,
                      output_array:TensorOrArray,
                      output_denominator:TensorOrArray,
                      tmp_out:TensorOrArray,
                      coords:Shape)->Tuple[TensorOrArray,TensorOrArray]:
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
            (x1,x2),(y1,y2) = coords
            output_array[...,x1:x2,y1:y2] += tmp_out.squeeze(0).squeeze(0)
            output_denominator[...,x1:x2,y1:y2] += 1.0
        elif self.ndim == 3:
            (x1,x2),(y1,y2),(z1,z2) = coords
            output_array[...,x1:x2,y1:y2,z1:z2] += tmp_out.squeeze(0).squeeze(0)
            output_denominator[...,x1:x2,y1:y2,z1:z2] += 1.0
        return output_array,output_denominator

    def __call__(self,
                 X:MultiFormatInput,
                 *args,**kwargs)->TensorOrArray:
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
            raise Exception("length of input array shape should be <= self.ndim+2")
        output_array = zeros_fn(output_size).to(device)
        output_denominator = zeros_fn(output_size).to(device)
        batch = []
        batch_coords = []
        for cropped_input,coords in self.get_all_crops(X):
            batch.append(cropped_input)
            batch_coords.append(coords)
            if len(batch) == self.inference_batch_size:
                original_batch_size = output_size[0]
                batch = multi_format_stack_or_cat(batch,self.ndim)
                with torch.no_grad():
                   batch_out = self.inference_function(batch,*args,**kwargs)
                batch_out = split_fn(batch_out,original_batch_size,0)
                for out,coords in zip(batch_out,batch_coords):
                    output_array,output_denominator = self.update_output(
                        output_array,output_denominator,out,coords)
                batch = []
                batch_coords = []
        if len(batch) > 0:
            original_batch_size = output_size[0]
            batch = multi_format_stack_or_cat(batch,self.ndim)
            with torch.no_grad():
                batch_out = self.inference_function(batch,*args,**kwargs)
            batch_out = split_fn(batch_out,original_batch_size,0)
            for out,coords in zip(batch_out,batch_coords):
                output_array,output_denominator = self.update_output(
                    output_array,output_denominator,out,coords)            
        output_array = output_array / output_denominator
        return output_array