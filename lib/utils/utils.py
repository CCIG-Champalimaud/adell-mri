import os
import json
import numpy as np
import SimpleITK as sitk
import torch
import torch.nn.functional as F
from copy import deepcopy
from glob import glob
from collections import OrderedDict

from typing import Dict,List,Tuple,Any
from ..modules.losses import *
from ..custom_types import (
    DatasetDict,SizeDict,SpacingDict,FloatOrTensor,
    TensorIterable,BBDict)

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
        "unified_focal":mc_unified_focal_loss},
    "regression":{
        "mse":F.mse_loss,
        "weighted_mse":weighted_mse
    }}

def split(x,n_splits,dim):
    size = int(x.shape[dim]//n_splits)
    return torch.split(x,size,dim)

def get_prostatex_path_dictionary(base_path:str)->DatasetDict:
    """Builds a path dictionary (a dictionary where each key is a patient
    ID and each value is a dictionary containing a modality-to-MRI scan path
    mapping). Assumes that the folders "T2WAx", "DWI", "aggregated-labels-gland"
    and "aggregated-labels-lesion" in `base_path`.

    Args:
        base_path (str): path containing the "T2WAx", "DWI", 
        "aggregated-labels-gland" and "aggregated-labels-lesion" folders.

    Returns:
        DatasetDict: a path dictionary.
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
    path_dictionary:DatasetDict,
    keys:List[str])->Tuple[SizeDict,SpacingDict]:
    """Retrieves the scan sizes and pixel spacings from a path dictionary.

    Args:
        path_dictionary (DatasetDict): a path dictionary (see 
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
    comb:FloatOrTensor,
    threshold:FloatOrTensor=0.5,
    scale:FloatOrTensor=1.0)->Dict[str,Dict[str,FloatOrTensor]]:
    """Constructs a keyword dictionary that can be used with the losses in 
    `losses.py`.

    Args:
        weights (torch.Tensor): weights for different classes (or for the 
            positive class).
        gamma (Union[torch.Tensor,float]): gamma for focal losses.
        comb (Union[torch.Tensor,float]): relative combination coefficient for
            combination losses.
        threshold (Union[torch.Tensor,float],optional): threshold for the 
            positive class in the focal loss. Helpful in cases where one is 
            trying to model the probability explictly. Defaults to 0.5.
        scale (float, optional): scaling factor for CE and focal losses.

    Returns:
        Dict[str,Dict[str,Union[float,torch.Tensor]]]: dictionary where each 
        key refers to a loss function and each value is keyword dictionary for
        different losses. 
    """
    def invert_weights(w):
        if torch.any(w >= 1):
            return torch.ones_like(w)
        else:
            return torch.ones_like(w) - w
    weights = torch.as_tensor(weights)
    gamma = torch.as_tensor(gamma)
    comb = torch.as_tensor(comb)
    scale = torch.as_tensor(scale)
    
    inverted_weights = invert_weights(weights)
    s = weights + inverted_weights
    weights_tv = weights / s
    inverted_weights_tv = inverted_weights / s
    
    loss_param_dict = {
        "cross_entropy":{"weight":weights,"scale":scale,"eps":eps},
        "focal":{"alpha":weights,"gamma":gamma,"scale":scale,"eps":eps},
        "focal_alt":{"alpha":weights,"gamma":gamma},
        "dice":{"weight":weights},
        "tversky_focal":{
            "alpha":inverted_weights_tv,"beta":weights_tv,
            "gamma":gamma,"scale":scale},
        "combo":{
            "alpha":comb,"beta":weights,"gamma":gamma,"scale":scale},
        "unified_focal":{
            "delta":weights,"gamma":gamma,
            "lam":comb,"threshold":threshold,"scale":scale},
        "weighted_mse":{"alpha":weights,"threshold":threshold},
        "mse":{}}
    return loss_param_dict

def collate_last_slice(X:List[TensorIterable])->TensorIterable:
    def swap(x):
        out = x.permute(0,3,1,2)
        return out
    def swap_cat(x):
        try:
            o = torch.cat([swap(y) for y in x])
            return o
        except:
            pass

    example = X[0]
    if isinstance(example,list):
        output = []
        for elements in zip(*X):
            output.append(swap_cat(elements))
    elif isinstance(example,dict):
        keys = list(example.keys())
        output = {}
        for k in keys:
            elements = [x[k] if k in x else None for x in X]
            output[k] = swap_cat(elements)
    return output

def safe_collate(X:List[TensorIterable])->List[TensorIterable]:
    """Similar to the default collate but going only one level deep and 
    returning a list if shapes are incompatible (helpful to return bounding
    boxes).

    Args:
        X (List[TensorIterable]): a list of lists or dicts of tensors.

    Returns:
        List[TensorIterable]: a list or dict of tensors (depending on the
        input).
    """
    def cat(x):
        try: x = [torch.as_tensor(y) for y in x]
        except: return x
        try: return torch.stack(x)
        except: return x

    example = X[0]
    # this small check unpacks the batch in case RandCropByPosNegLabeld
    # was used since it returns a list (?????)
    if len(example) == 1:
        X = [x[0] for x in X]
        example = X[0]
    if isinstance(example,list):
        output = []
        for elements in zip(*X):
            output.append(cat(elements))
    elif isinstance(example,dict):
        keys = list(example.keys())
        output = {}
        for k in keys:
            elements = []
            for x in X:
                if k in x:
                    elements.append(x[k])
                else:
                    elements.append(None)
            output[k] = cat(elements)
    return output

def load_bb(path:str)->BBDict:
    with open(path) as o:
        lines = o.readlines()
    lines = [x.strip() for x in lines]
    output = {}
    for line in lines:
        line = line.split(',')
        patient_id = line[0]
        cl = int(line[-1])
        ndim = len(line[1:-1])//3
        uc = [int(i) for i in line[1:(1+ndim)]]
        lc = [int(i) for i in line[(1+ndim):(4+ndim)]]
        sh = [int(i) for i in line[(4+ndim):(7+ndim)]]
        if patient_id in output:
            output[patient_id]["boxes"].append([uc,lc])
            output[patient_id]["labels"].append(cl)
        else:
            output[patient_id] = {
                "boxes":[[uc,lc]],
                "labels":[cl],
                "shape":np.array(sh)}
    for k in output:
        output[k]["boxes"] = np.array(output[k]["boxes"]).swapaxes(1,2)
    return output

def load_bb_json(path:str)->BBDict:
    with open(path) as o:
        data_dict = json.load(o)
    k_del = []
    for k in data_dict:
        bb = []
        for box in data_dict[k]['boxes']:
            ndim = len(box)//2
            bb.append([box[:ndim],box[ndim:]])
        if len(bb) > 0:
            data_dict[k]['boxes'] = np.array(
                bb).swapaxes(1,2)
            data_dict[k]['shape'] = np.array(
                data_dict[k]['shape'])
        else:
            k_del.append(k)
    for k in k_del:
        del data_dict[k]
    return data_dict

def load_anchors(path:str)->np.ndarray:
    with open(path,'r') as o:
        lines = o.readlines()
    lines = [[float(y) for y in x.strip().split(',')] for x in lines]
    return np.array(lines)

def resample_image(sitk_image,out_spacing=[1.0, 1.0, 1.0],
                   out_size=None,is_label=False):
    spacing = sitk_image.GetSpacing()
    size = sitk_image.GetSize()

    if out_size is None:
        out_size = [
            int(np.round(size * (sin / sout)))
            for size, sin, sout in zip(size, spacing, out_spacing)]

    pad_value = sitk_image.GetPixelIDValue()

    resample = sitk.ResampleImageFilter()
    resample.SetOutputSpacing(list(out_spacing))
    resample.SetSize(out_size)
    resample.SetOutputDirection(sitk_image.GetDirection())
    resample.SetOutputOrigin(sitk_image.GetOrigin())
    resample.SetTransform(sitk.Transform())
    resample.SetDefaultPixelValue(pad_value)
    if is_label:
        resample.SetInterpolator(sitk.sitkNearestNeighbor)
    else:
        resample.SetInterpolator(sitk.sitkBSpline)

    # perform resampling
    sitk_image = resample.Execute(sitk_image)

    return sitk_image

def set_classification_layer_bias(pos,neg,network):
    value = torch.as_tensor(np.log(pos/neg))
    for k,v in network.named_parameters():
        if "classification" in k:
            if list(v.shape) == [1]:
                with torch.no_grad():
                    v[0] = value
                    
class ExponentialMovingAverage(torch.nn.Module):
    def __init__(self,decay:float,final_decay:float=None,n_steps=None):
        """Exponential moving average for model weights. The weight-averaged
        model is kept as `self.shadow` and each iteration of self.update leads
        to weight updating. This implementation is heavily based on that 
        available in https://www.zijianhu.com/post/pytorch/ema/.

        Essentially, self.update(model) is called, a shadow version of the 
        model (i.e. self.shadow) is updated using the exponential moving 
        average formula such that $v'=(1-decay)*(v_{shadow}-v)$, where
        $v$ is the new parameter value, $v'$ is the updated value and 
        $v_{shadow}$ is the exponential moving average value (i.e. the shadow).

        Args:
            decay (float): decay for the exponential moving average.
            final_decay (float, optional): final value for decay. Defaults to
                None (same as initial decay).
            n_steps (float, optional): number of updates until `decay` becomes
                `final_decay` with linear scaling. Defaults to None.
        """
        super().__init__()
        self.decay = decay
        self.final_decay = final_decay
        self.n_steps = n_steps
        self.shadow = None
        self.step = 0

        if self.final_decay is None:
            self.final_decay = self.decay
        self.slope = (self.final_decay - self.decay)/self.n_steps
        self.intercept = self.decay

    def set_requires_grad_false(self,model):
        for k,p in model.named_parameters():
            if p.requires_grad is True:
                p.requires_grad = False

    @torch.no_grad()
    def update(self,model:torch.nn.Module):
        if self.shadow is None:
            # this effectively skips the first epoch
            self.shadow = deepcopy(model)
            self.shadow.training = False
            self.set_requires_grad_false(self.shadow)
        else:
            model_params = OrderedDict(model.named_parameters())
            shadow_params = OrderedDict(self.shadow.named_parameters())

            shadow_params_keys = list(shadow_params.keys())

            different_params = set.difference(
                set(shadow_params_keys),
                set(model_params.keys()))
            assert len(different_params) == 0

            for name, param in model_params.items():
                if 'shadow' not in name:
                    shadow_params[name].sub_(
                       (1.-self.decay) * (shadow_params[name]-param))
            
            self.decay = self.step*self.slope + self.intercept
            if self.decay > 1.0:
                self.decay = 1.0
            self.step += 1
