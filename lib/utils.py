import os
import numpy as np
import SimpleITK as sitk
import torch
import torch.nn.functional as F
import monai
from glob import glob

from typing import Dict,List,Union
from .modules.losses import *

PathDict = Dict[str,Dict[str,str]]
TensorOrNDarray = Union[np.ndarray,torch.Tensor]

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

def get_size_spacing_dict(path_dictionary:PathDict,keys:List[str]):
    size_dict = {k:[] for k in keys}
    spacing_dict = {k:[] for k in keys}
    for pid in path_dictionary:
        for k in keys:
            if k in path_dictionary[pid]:
                X = sitk.ReadImage(path_dictionary[pid][k])
                size_dict[k].append(X.GetSize())
                spacing_dict[k].append(X.GetSpacing())
    return size_dict,spacing_dict

def get_loss_param_dict(weights:torch.Tensor,
                        gamma:Union[torch.Tensor,float],
                        comb:Union[torch.Tensor,float]):
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

class ConvertToOneHot(monai.transforms.Transform):
    def __init__(self,keys:str,out_key:str,priority_key:str,bg:bool=True):
        super().__init__()
        self.keys = keys
        self.out_key = out_key
        self.priority_key = priority_key
        self.bg = bg
    
    def __call__(self,X):
        rel = {k:X[k] for k in self.keys}
        p = X[self.priority_key]
        dev = p.device
        p_inv = torch.ones_like(p).to(dev) - p
        for k in self.keys:
            if k != self.priority_key:
                rel[k] = rel[k] * p_inv
        out = [rel[k] for k in self.keys]
        if self.bg == True:
            bg_tensor = torch.where(
                torch.cat(out,0).sum(0)>0,
                torch.zeros_like(p).to(dev),
                torch.ones_like(p).to(dev))
            out.insert(0,bg_tensor)
        out = torch.cat(out,0)
        out = torch.argmax(out,0)
        X[self.out_key] = out
        return X

class PrintShaped(monai.transforms.Transform):
    def __call__(self,X):
        for k in X:
            try: print(k,X[k].shape)
            except: pass
        return X