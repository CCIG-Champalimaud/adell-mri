import os
import numpy as np
import SimpleITK as sitk
import torch
import torch.nn.functional as F
import monai
from glob import glob

from typing import Dict,List,Tuple
from .modules.losses import *
from .types import *

activation_factory = {
    "elu": torch.nn.ELU,
    "hard_shrink": torch.nn.Hardshrink,
    "hard_tanh": torch.nn.Hardtanh,
    "leaky_relu": torch.nn.LeakyReLU,
    "logsigmoid": torch.nn.LogSigmoid,
    "prelu": torch.nn.PReLU,
    "relu": torch.nn.ReLU,
    "relu6": torch.nn.ReLU6,
    "rrelu": torch.nn.RReLU,
    "selu": torch.nn.SELU,
    "celu": torch.nn.CELU,
    "sigmoid": torch.nn.Sigmoid,
    "softplus": torch.nn.Softplus,
    "softshrink": torch.nn.Softshrink,
    "softsign": torch.nn.Softsign,
    "tanh": torch.nn.Tanh,
    "tanhshrink": torch.nn.Tanhshrink,
    "threshold": torch.nn.Threshold,
    "softmin": torch.nn.Softmin,
    "softmax": torch.nn.Softmax,
    "logsoftmax": torch.nn.LogSoftmax}

loss_factory = {
    "binary":{
        "cross_entropy":binary_cross_entropy,
        "focal":binary_focal_loss,
        "dice":generalized_dice_loss,
        "tversky_focal":binary_focal_tversky_loss,
        "combo":combo_loss,
        "hybrid_focal":hybrid_focal_loss,
        "unified_focal":unified_focal_loss},
    "categorical":{
        "cross_entropy":cat_cross_entropy,
        "focal":mc_focal_loss,
        "dice":generalized_dice_loss,
        "tversky_focal":mc_focal_tversky_loss,
        "combo":mc_combo_loss,
        "hybrid_focal":mc_hybrid_focal_loss,
        "unified_focal":mc_unified_focal_loss}}

def get_prostatex_path_dictionary(base_path:str)->PathDict:
    """Builds a path dictionary (a dictionary where each key is a patient
    ID and each value is a dictionary containing a modality-to-MRI scan path
    mapping). Assumes that the folders "T2WAx", "DWI", "aggregated-labels-gland"
    and "aggregated-labels-lesion" in `base_path`.

    Args:
        base_path (str): path containing the "T2WAx", "DWI", 
        "aggregated-labels-gland" and "aggregated-labels-lesion" folders.

    Returns:
        PathDict: a path dictionary.
    """
    paths = {
        "t2w":os.path.join(base_path,"T2WAx"),
        "dwi":os.path.join(base_path,"DWI"),
        "gland":os.path.join(base_path,"aggregated-labels-gland"),
        "lesion":os.path.join(base_path,"aggregated-labels-lesion")}

    path_dictionary = {}
    for image_path in glob(os.path.join(paths['t2w'],"*T2WAx1*gz")):
        f_name = os.path.split(image_path)[-1]
        patient_id = f_name.split('_')[0]
        path_dictionary[patient_id] = {
            "T2WAx":image_path}

    for image_path in glob(os.path.join(paths['dwi'],"*gz")):
        f_name = os.path.split(image_path)[-1]
        patient_id = f_name.split('_')[0]
        # only interested in cases with both data types
        if patient_id in path_dictionary: 
            path_dictionary[patient_id]["DWI"] = image_path

    for image_path in glob(os.path.join(paths['gland'],"*gz")):
        f_name = os.path.split(image_path)[-1]
        patient_id = f_name.split('_')[0]
        mod = f_name.split('_')[1]
        if patient_id in path_dictionary:
            m = "{}_gland_segmentations".format(mod)
            path_dictionary[patient_id][m] = image_path

    for image_path in glob(os.path.join(paths['lesion'],"*gz")):
        f_name = os.path.split(image_path)[-1]
        patient_id = f_name.split('_')[0]
        mod = f_name.split('_')[1]
        if patient_id in path_dictionary:
            m = "{}_lesion_segmentations".format(mod)
            path_dictionary[patient_id][m] = image_path
    
    return path_dictionary

def get_size_spacing_dict(
    path_dictionary:PathDict,
    keys:List[str])->Tuple[SizeDict,SpacingDict]:
    """Retrieves the scan sizes and pixel spacings from a path dictionary.

    Args:
        path_dictionary (PathDict): a path dictionary (see 
        `get_prostatex_path_dictionary` for details).
        keys (List[str]): modality keys that should be considered in the
        path dictionary.

    Returns:
        size_dict (SizeDict): a dictionary with `keys` as keys and a list
        of scan sizes (2 or 3 int) as values.
        spacing_dict (SpacingDict): a dictionary with `keys` as keys and a 
        of spacing sizes (2 or 3 floats) as values.
    """
    
    size_dict = {k:[] for k in keys}
    spacing_dict = {k:[] for k in keys}
    for pid in path_dictionary:
        for k in keys:
            if k in path_dictionary[pid]:
                X = sitk.ReadImage(path_dictionary[pid][k])
                size_dict[k].append(X.GetSize())
                spacing_dict[k].append(X.GetSpacing())
    return size_dict,spacing_dict

def get_loss_param_dict(
    weights:torch.Tensor,
    gamma:FloatOrTensor,
    comb:FloatOrTensor)->Dict[str,Dict[str,FloatOrTensor]]:
    """Constructs a keyword dictionary that can be used with the losses in 
    `losses.py`.

    Args:
        weights (torch.Tensor): weights for different classes (or for the 
        positive class).
        gamma (Union[torch.Tensor,float]): gamma for focal losses.
        comb (Union[torch.Tensor,float]): relative combination coefficient for
        combination losses.

    Returns:
        Dict[str,Dict[str,Union[float,torch.Tensor]]]: dictionary where each 
        key refers to a loss function and each value is keyword dictionary for
        different losses. 
    """
    loss_param_dict = {
        "cross_entropy":{"weight":weights},
        "focal":{"alpha":weights,"gamma":gamma},
        "dice":{"weight":weights},
        "tversky_focal":{
            "alpha":weights,"beta":1-weights,"gamma":gamma},
        "combo":{
            "alpha":comb,"beta":weights},
        "unified_focal":{
            "delta":weights,"gamma":gamma,
            "lam":comb}}
    return loss_param_dict

def collate_last_slice(X):
    def swap(x):
        return x.unsqueeze(1).swapaxes(1,-1).squeeze(-1)
    def swap_cat(x):
        try:
            o = torch.cat([swap(y) for y in x])
            return o
        except: pass

    example = X[0]
    if isinstance(example,list):
        output = []
        for elements in zip(*X):
            output.append(swap_cat(elements))
    elif isinstance(example,dict):
        keys = list(example.keys())
        output = {}
        for k in keys:
            elements = [x[k] for x in X]
            output[k] = swap_cat(elements)
    return output

class ConvertToOneHot(monai.transforms.Transform):
    def __init__(self,keys:str,out_key:str,
                 priority_key:str,bg:bool=True)->monai.transforms.Transform:
        """Convenience MONAI transform to convert a set of keys in a 
        dictionary into a single one-hot format dictionary. Useful to coerce
        several binary class problems into a single multi-class problem.

        Args:
            keys (str): keys that willbe used to construct the one-hot 
            encoding.
            out_key (str): key for the output.
            priority_key (str): key for the element that takes priority when
            more than one key is available for the same position.
            bg (bool, optional): whether a level for the "background" class 
            should be included. Defaults to True.

        Returns:
            monai.transforms.Transform
        """
        super().__init__()
        self.keys = keys
        self.out_key = out_key
        self.priority_key = priority_key
        self.bg = bg
    
    def __call__(self,X:TensorDict)->TensorDict:
        """
        Args:
            X (TensorDict)

        Returns:
            TensorDict
        """
        rel = {k:X[k] for k in self.keys}
        p = X[self.priority_key]
        dev = p.device
        p_inv = torch.ones_like(p,device=dev) - p
        for k in self.keys:
            if k != self.priority_key:
                rel[k] = rel[k] * p_inv
        out = [rel[k] for k in self.keys]
        if self.bg == True:
            bg_tensor = torch.where(
                torch.cat(out,0).sum(0)>0,
                torch.zeros_like(p,device=dev),
                torch.ones_like(p,device=dev))
            out.insert(0,bg_tensor)
        out = torch.cat(out,0)
        out = torch.argmax(out,0,keepdim=True)
        X[self.out_key] = out
        return X

class PrintShaped(monai.transforms.Transform):
    """Convenience MONAI transform that prints the shape of elements in a 
    dictionary of tensors. Used for debugging.
    """
    def __call__(self,X):
        for k in X:
            try: print(k,X[k].shape)
            except: pass
        return X

class RandomSlices(monai.transforms.RandomizableTransform):
    def __init__(self,keys:List[str],label_key:List[str],
                 n:int=1,base:float=0.001):
        """Randomly samples slices from a volume (assumes the slice dimension
        is the last dimension). A segmentation map (corresponding to 
        `label_key`) is used to calculate the number of positive elements
        for each slice and these are used as sampling weights for the slice
        extraction.

        Args:
            keys (List[str]): keys for which slices will be retrieved.
            label_key (List[str]): segmentation map that will be used to 
            calculate sampling weights.
            n (int, optional): number of slices to be sampled. Defaults to 1.
            base (float, optional): minimum probability (ensures that slices 
            with no positive cases are also sampled). Defaults to 0.01.
        """
        self.keys = keys
        self.label_key = label_key
        self.n = n
        self.base = base
        self.is_multiclass = None
        self.M = 0
    
    def __call__(self,X):
        X_label = X[self.label_key]
        if isinstance(X_label,torch.Tensor) == False:
            X_label = torch.as_tensor(X_label)
        if self.is_multiclass is None:
            M = X_label.max()
            if M > 1:
                X_label = F.one_hot(X_label,M+1)
                X_label = torch.squeeze(X_label.swapaxes(-1,0))
                X_label = X_label[1:] # remove background dim
                self.is_multiclass = True
                self.M = M
            else:
                self.is_multiclass = False
        elif self.is_multiclass == True:
            X_label = F.one_hot(X_label,self.M+1)
            X_label = torch.squeeze(X_label.swapaxes(-1,0))
            X_label = X_label[1:] # remove background dim
        c = torch.flatten(X_label,start_dim=1,end_dim=-2)
        c_sum = c.sum(1)
        total_class = torch.unsqueeze(c_sum.sum(1),-1)
        c_prop = c_sum/total_class + self.base
        slice_weight = c_prop.mean(0)
        slice_idxs = torch.multinomial(slice_weight,self.n)
        for k in self.keys:
            X[k] = np.take(X[k],slice_idxs,-1).swapaxes(0,-1)
        return X

class SlicesToFirst(monai.transforms.Transform):
    def __init__(self,keys:List[str]):
        """Returns the slices as the first spatial dimension.

        Args:
            keys (List[str]): keys for which slices will be retrieved.
        """
        self.keys = keys
    
    def __call__(self,X):
        for k in self.keys:
            X[k] = torch.unsqueeze(X[k],0).swapaxes(0,-1).squeeze(1)
        return X

class Index(monai.transforms.Transform):
    def __init__(self,keys:List[str],idxs:List[int],axis:int):
        """Indexes tensors in a dictionary at a given dimension `axis`.
        Useful for datasets such as the MONAI Medical Decathlon, where 
        arrays are composed of more than one modality and we only care about
        a specific modality.

        Args:
            keys (List[str]): list of keys to tensors which will be indexed.
            idxs (List[int]): indices that will be retrieved.
            axis (int): axis at which indices will be retrieved.
        """
        self.keys = keys
        self.idxs = idxs
        self.axis = axis
    
    def __call__(self,X):
        for k in self.keys:
            if self.idxs is not None:
                X[k] = np.take(X[k],self.idxs,self.axis)
        return X