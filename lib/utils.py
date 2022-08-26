from copy import deepcopy
import os
import json
import numpy as np
import SimpleITK as sitk
import itk
from pandas import Categorical
import torch
import torch.nn.functional as F
import monai
from glob import glob
from itertools import product

from typing import Dict,List,Tuple
from .modules.losses import *
from .modules.layers import activation_factory
from .types import *

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
    comb:FloatOrTensor,
    threshold:FloatOrTensor=0.5)->Dict[str,Dict[str,FloatOrTensor]]:
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
        dev (str, optional): device to which parameters should be mapped. 
        Defaults to "cuda".

    Returns:
        Dict[str,Dict[str,Union[float,torch.Tensor]]]: dictionary where each 
        key refers to a loss function and each value is keyword dictionary for
        different losses. 
    """
    weights = torch.as_tensor(weights)
    gamma = torch.as_tensor(gamma)
    comb = torch.as_tensor(comb)
    
    loss_param_dict = {
        "cross_entropy":{"weight":weights},
        "focal":{"alpha":weights,"gamma":gamma,"threshold":threshold},
        "focal_alt":{"alpha":weights,"gamma":gamma},
        "dice":{"weight":weights},
        "tversky_focal":{
            "alpha":weights,"beta":1-weights,"gamma":gamma},
        "combo":{
            "alpha":comb,"beta":weights},
        "unified_focal":{
            "delta":weights,"gamma":gamma,
            "lam":comb,"threshold":threshold},
        "weighted_mse":{"alpha":weights,"threshold":threshold},
        "mse":{}}
    return loss_param_dict

def collate_last_slice(X:List[TensorIterable])->TensorIterable:
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
        try:
            x = [torch.as_tensor(y) for y in x]
        except:
            return x
        try:
            return torch.stack(x)
        except:
            return x

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
    def __init__(self,prefix=""):
        self.prefix = prefix

    def __call__(self,X):
        for k in X:
            try: print(self.prefix,k,X[k].shape)
            except: pass
        return X

class PrintSumd(monai.transforms.Transform):
    """Convenience MONAI transform that prints the sum of elements in a 
    dictionary of tensors. Used for debugging.
    """
    def __call__(self,X):
        for k in X:
            try: print(k,X[k].sum())
            except: pass
        return X

class RandomSlices(monai.transforms.RandomizableTransform):
    def __init__(self,keys:List[str],label_key:List[str],
                 n:int=1,base:float=0.001,seed=None):
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
    
    def __call__(self,X):
        if self.label_key is not None:
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
                M = X_label.max()
                X_label = F.one_hot(X_label,M+1)
                X_label = torch.squeeze(X_label.swapaxes(-1,0))
                X_label = X_label[1:] # remove background dim
            c = torch.flatten(X_label,start_dim=1,end_dim=-2)
            c_sum = c.sum(1)
            total_class = torch.unsqueeze(c_sum.sum(1),-1)
            total_class = torch.clamp(total_class,min=1)
            c_prop = c_sum/total_class + self.base
            slice_weight = c_prop.mean(0)
        else:
            slice_weight = torch.ones([X[k][0]].shape[-1])
        slice_idxs = torch.multinomial(slice_weight,self.n,generator=self.g)
        for k in self.keys:
            X[k] = np.take(X[k],slice_idxs,-1).swapaxes(0,-1)
        return X

class RandomCubesWithClassd(monai.transforms.RandomizableTransform):
    def __init__(self,key:str,label_key:str,
                 output_sh,positive_proba,n,
                 cubes_out_key:str="cubes",classes_out_key:str="labels"):
        self.key = key
        self.label_key = label_key
        self.output_sh = np.array(output_sh)
        self.positive_proba = positive_proba
        self.n = n
        self.cubes_out_key = cubes_out_key
        self.classes_out_key = classes_out_key

        self.half_size = self.output_sh//2

    def get_coords_from_centers(self,centers,sh):
        for center in centers:
            uc = np.int32(center - self.half_size)
            uc = np.where(uc<0,0,uc)
            lc = uc + self.output_sh
            diff = sh - lc
            lc = np.where(diff < 0,lc-np.abs(diff),lc)
            uc = np.where(diff < 0,uc-np.abs(diff),uc)
            yield uc,lc

    def __call__(self,X):
        cubes = []
        classes = []
        labels = X[self.label_key]
        sh = np.array(labels.shape)[1:]
        positive = np.stack(np.where(labels > 0),axis=1)[:,-3:]
        is_positive = np.random.uniform(size=self.n)<self.positive_proba
        n_positive = np.sum(is_positive)
        n_negative = self.n - n_positive
        positive_idx = np.random.randint(positive.shape[0],size=n_positive)
        positive_centers = positive[positive_idx]
        negative_centers = np.random.randint(sh,size=[n_negative,3])
        for uc,lc in self.get_coords_from_centers(positive_centers,sh):
            L = labels[:,uc[0]:lc[0],uc[1]:lc[1],uc[2]:lc[2]]
            if np.count_nonzero(L) > 0: 
                C = 1
            else:
                C = 0
            cube = X[self.key][:,uc[0]:lc[0],uc[1]:lc[1],uc[2]:lc[2]]
            classes.append(C)
            cubes.append(cube)
        for uc,lc in self.get_coords_from_centers(negative_centers,sh):
            L = labels[:,uc[0]:lc[0],uc[1]:lc[1],uc[2]:lc[2]]
            if np.count_nonzero(L) > 0: 
                C = 1
            else:
                C = 0
            cube = X[self.key][:,uc[0]:lc[0],uc[1]:lc[1],uc[2]:lc[2]]
            classes.append(C)
            cubes.append(cube)
        
        X[self.cubes_out_key] = np.concatenate(cubes,0)
        X[self.classes_out_key] = np.stack(classes,0)
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
            X[k] = X[k].swapaxes(0,-1)
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

class MaskToAdjustedAnchors(monai.transforms.Transform):
    def __init__(self,
                 anchor_sizes:Union[np.ndarray,List[List]],
                 input_sh:Iterable,
                 output_sh:Iterable,
                 iou_thresh:float):
        """Maps bounding boxes in corner format (x1y1z1x2y2z2) into their 
        anchor representation.

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
        """Sets anchors up.
        """
        image_sh = np.array(self.input_sh)
        self.rel_sh = image_sh / self.output_sh

        long_coords = []
        for c in product(
            *[np.arange(x) for x in self.output_sh]):
            long_coords.append(c)

        self.long_coords = np.array(long_coords)
        rel_anchor_sizes = [x/self.rel_sh
                            for x in self.anchor_sizes]

        self.long_anchors = []
        for rel_anchor_size in rel_anchor_sizes:
            # adding 0.5 centres the bounding boxes in each cell
            anchor_h = rel_anchor_size/2
            long_anchor_rel = [
                long_coords-anchor_h + 0.5,
                long_coords+anchor_h + 0.5]
            self.long_anchors.append(
                np.stack(long_anchor_rel,axis=-1))

    def __call__(self,bb_vertices:Iterable,classes:Iterable,
                 shape:np.ndarray=None)->np.ndarray:
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
        bb_size = bb_vertices[:,:,1]-bb_vertices[:,:,0]
        if shape is None:
            shape = self.input_sh
            rel_sh = self.rel_sh
            rel_bb_vert = bb_vertices/rel_sh[np.newaxis,:,np.newaxis]
        else:
            rel_sh = shape/self.output_sh
            rel_bb_vert = bb_vertices/rel_sh[np.newaxis,:,np.newaxis]
        output = np.zeros([1+7*len(self.long_anchors),*self.output_sh])
        for i in range(rel_bb_vert.shape[0]):
            hits = 0
            all_iou = []
            rel_bb_size = np.subtract(rel_bb_vert[i,:,1],rel_bb_vert[i,:,0])
            center = np.mean(rel_bb_vert[i,:,:],axis=-1)
            bb_vol = np.prod(rel_bb_size+1/rel_sh)
            cl = classes[i]
            for I,long_anchor in enumerate(self.long_anchors):
                anchor_size = long_anchor[0,:,1] - long_anchor[0,:,0]
                rel_bb_size_adj = rel_bb_size/anchor_size
                anchor_dim = long_anchor[0,:,1]-long_anchor[0,:,0]
                intersects = np.logical_and(
                    np.all(long_anchor[:,:,1]>rel_bb_vert[i,:,0],axis=1),
                    np.all(long_anchor[:,:,0]<rel_bb_vert[i,:,1],axis=1))
                inter_dim = np.minimum(rel_bb_size,anchor_dim)
                inter_vol = np.prod(inter_dim+1/rel_sh,axis=-1)
                anchor_vol = np.prod(anchor_dim,axis=-1)
                union_vol = anchor_vol+bb_vol-inter_vol

                iou = inter_vol / union_vol
                intersection_idx = np.logical_and(
                    iou>self.iou_thresh,intersects)
                box_coords = self.long_coords[intersection_idx]
                
                all_iou.append(iou)

                center_adjustment = center-(box_coords+0.5)
                distance_idx = np.all(np.abs(center_adjustment) < 1,axis=1)

                box_coords = box_coords[distance_idx]
                center_adjustment = center_adjustment[distance_idx]

                for j in range(box_coords.shape[0]):
                    idx = tuple(
                        [tuple([1+k+I*7 for k in range(7)]),
                         *box_coords[j]])
                    idx_cl = tuple([0,*box_coords[j]])
                    v = np.array(
                        [iou,*center_adjustment[j],*rel_bb_size_adj])
                    if iou > output[idx][0]:
                        output[idx] = v
                        output[idx_cl] = cl
                        hits += 1

        return output
    
    def adjusted_anchors_to_bb_vertices(self,
        anchor_map:np.ndarray)->Tuple[np.ndarray,np.ndarray]:
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
            sam = anchor_map[(1+i*7):(1+i*7+7)]
            coords = np.where(sam[0]>0)
            adj_anchors_long = np.zeros([len(coords[0]),7])
            for j,coord in enumerate(zip(*coords)):
                center_idxs = tuple([tuple([k for k in range(7)]),*coord])
                v = sam[center_idxs]
                adj_anchors_long[j,:] = v
            correct_centers = np.add(
                adj_anchors_long[:,1:4]+0.5,np.stack(coords,axis=1))
            correct_centers = correct_centers * self.rel_sh
            correct_dims = np.multiply(
                adj_anchors_long[:,4:],rel_anchor_size)
            top_left = correct_centers - correct_dims/2
            bottom_right = correct_centers + correct_dims/2
            top_left_output.append(top_left)
            bottom_right_output.append(bottom_right)
        return top_left_output,bottom_right_output

class MaskToAdjustedAnchorsd(monai.transforms.MapTransform):
    def __init__(self,anchor_sizes:torch.Tensor,
                 input_sh:Tuple[int],output_sh:Tuple[int],iou_thresh:float,
                 bb_key:str="bb",class_key:str="class",shape_key:str="shape",
                 output_key:Dict[str,str]={}):
        """Dictionary transform of the MaskToAdjustedAnchors transforrm.

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
        self.mask_to_anchors = MaskToAdjustedAnchors(
            self.anchor_sizes,self.input_sh,self.output_sh,self.iou_thresh)
        self.bb_key = bb_key
        self.class_key = class_key
        self.shape_key = shape_key
        self.output_key = output_key
    
    def __call__(self,data:dict)->dict:
        if self.output_key is not None: out_k = self.output_key
        else: out_k = self.bb_key
        data[out_k] = self.mask_to_anchors(
            data[self.bb_key],
            data[self.class_key],
            data[self.shape_key])
        return data

class RandomFlipWithBoxes(monai.transforms.Transform):
    def __init__(self,axes=[0,1,2],prob=0.5):
        """Randomly augmentatat images and bounding boxes by flipping axes.

        Args:
            axes (list, optional): list of axes to flip. Defaults to [0,1,2].
            prob (float, optional): rotation probability. Defaults to 0.5.
        """
        self.axes = axes
        self.prob = prob
    
    def flip_boxes(self,boxes,axis,center):
        boxes = boxes - center
        boxes[:,axis,:] = -boxes[:,axis,:]
        boxes = boxes + center
        return boxes

    def flip_image(self,image,axis):
        return torch.flip(image,(axis,))

    def __call__(self,images,boxes):
        center = np.expand_dims(np.array(images[0].shape[1:]),0)
        center = center[:,:,np.newaxis]
        axes_to_flip = []
        for axis in self.axes:
            if np.random.uniform() < self.prob:
                axes_to_flip.append(axis)
                boxes = self.flip_boxes(boxes,axis,center)
                for image in images:
                    image = self.flip_image(image,axis)
        return images,boxes

class RandomFlipWithBoxesd(monai.transforms.MapTransform):
    def __init__(
        self,
        image_keys:List[str],
        box_key:str,
        box_key_nest:str=None,
        axes:List[int]=[0,1,2],prob:float=0.5):
        """Dictionary transform for RandomFlipWithBoxes.

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
        self.flipping_op = RandomFlipWithBoxes(axes,prob)
    
    def __call__(self,data):
        images = [data[k] for k in self.image_keys]
        if self.box_key_nest is not None:
            boxes = data[self.box_key][self.box_key_nest]
        else:
            boxes = data[self.box_key]

        images,boxes = self.flipping_op(images,boxes)
        for k,image in zip(self.image_keys,images):
            data[k] = image
        
        if self.box_key_nest is not None:
            data[self.box_key][self.box_key_nest] = boxes
        else:
            data[self.box_key] = boxes
        return data

class RandomAffined(monai.transforms.RandomizableTransform):
    def __init__(
        self,
        keys:List[str],
        spatial_sizes:List[Union[Tuple[int,int,int],Tuple[int,int]]],
        mode:List[str],
        prob:float=0.1,
        rotate_range:Union[Tuple[int,int,int],Tuple[int,int]]=[0,0,0],
        shear_range:Union[Tuple[int,int,int],Tuple[int,int]]=[0,0,0],
        translate_range:Union[Tuple[int,int,int],Tuple[int,int]]=[0,0,0],
        scale_range:Union[Tuple[int,int,int],Tuple[int,int]]=[0,0,0],
        device:"str"="cpu",
        copy:bool=False):
        """Reimplementation of the RandAffined transform in MONAI but works 
        with differently sized inputs without forcing all inputs to the same
        shape.

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
        self.spatial_sizes = [np.array(s,dtype=np.int32) for s in spatial_sizes]
        self.mode = mode
        self.prob = prob
        self.rotate_range = np.array(rotate_range)
        self.shear_range = np.array(shear_range)
        self.translate_range = np.array(translate_range)
        self.scale_range = np.array(scale_range)
        self.device = device
        self.copy = copy

        self.affine_trans = {
            k:monai.transforms.Affine(
                spatial_size=s,
                mode=m,
                device=self.device)
            for k,s,m in zip(self.keys,self.spatial_sizes,self.mode)}
        
        self.get_translation_adjustment()

    def get_random_parameters(self):
        angle = self.R.uniform(
            -self.rotate_range,self.rotate_range)
        shear = self.R.uniform(
            -self.shear_range,self.shear_range)
        trans = self.R.uniform(
            -self.translate_range,self.translate_range)
        scale = self.R.uniform(
            1-self.scale_range,1+self.scale_range)

        return angle,shear,trans,scale
    
    def get_translation_adjustment(self):
        # we have to adjust the translation to ensure that all inputs
        # do not become misaligned. to do this I assume that the first image
        # is the reference
        ref_size = self.spatial_sizes[0]
        self.trans_adj = {
            k:s/ref_size
            for k,s in zip(self.keys,self.spatial_sizes)}
    
    def randomize(self):
        angle,shear,trans,scale = self.get_random_parameters()
        for k in self.affine_trans:
            # we only need to update the affine grid
            self.affine_trans[k].affine_grid = monai.transforms.AffineGrid(
                rotate_params=list(angle),
                shear_params=list(shear),
                translate_params=list(trans*self.trans_adj[k]),
                scale_params=list(np.float32(scale)),
                device=self.device)

    def __call__(self,data):
        self.randomize()
        if self.copy == True:
            data = data.copy()
        for k in self.keys:
            if self.R.uniform() < self.prob:
                transform = self.affine_trans[k]
                data[k],_ = transform(data[k])
        return data

class LabelOperatord(monai.transforms.Transform):
    def __init__(self,keys:str,possible_labels:List[int],
                 mode:str="cat",positive_labels:List[int]=[1],
                 output_keys:Dict[str,str]={}):
        self.keys = keys
        self.possible_labels = possible_labels
        self.mode = mode
        self.positive_labels = positive_labels
        self.output_keys = output_keys

        self.possible_labels = self.possible_labels
        self.possible_labels_match = {
            l:i for i,l in enumerate(self.possible_labels)}

    def binary(self,x):
        if max(x) in self.positive_labels:
            x = 1
        else:
            x = 0
        return x

    def categorical(self,x):
        return self.possible_labels_match[max(x)]

    def __call__(self,data):
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

class LabelOperatorSegmentationd(monai.transforms.Transform):
    def __init__(self,keys:str,
                 possible_labels:List[int],
                 mode:str="cat",
                 positive_labels:List[int]=[1],
                 output_keys:Dict[str,str]={}):
        self.keys = keys
        self.possible_labels = possible_labels
        self.mode = mode
        self.positive_labels = positive_labels
        self.output_keys = output_keys

        self.possible_labels = self.possible_labels
        self.possible_labels_match = {
            l:i for i,l in enumerate(self.possible_labels)}

    def binary(self,x):
        return np.isin(x,self.positive_labels).astype(np.float32)

    def categorical(self,x):
        output = np.zeros_like(x)
        for u in np.unique(x):
            if u in self.possible_labels_match:
                output[np.where(x==u)] = self.possible_labels_match[u]
        return output

    def __call__(self,data):
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
    def __init__(self,keys:List[str],mode:str="any",output_key:str=None):
        """Combines binary label maps.

        Args:
            keys (List[str]): list of keys.
            mode (str, optional): how labels are combined. Defaults to 
                "majority".
            output_key (str, optional): name for the output key. Defaults to
                the name of the first key in keys.
        """
        self.keys = keys
        self.mode = mode
        if output_key is None:
            self.output_key = self.keys[0]
        else:
            self.output_key = output_key
        
    def __call__(self,X):
        tmp = [X[k] for k in self.keys]
        output = torch.stack(tmp,-1)
        if self.mode == "any":
            output = np.float32(output.sum(-1) > 0)
        elif self.mode == "majority":
            output = np.float32(output.mean(-1) > 0.5)
        X[self.output_key] = output
        return X

class ConditionalRescaling(monai.transforms.Transform):
    def __init__(self,max_value,scale):
        self.max_value = max_value
        self.scale = scale
    
    def __call__(self,X):
        if X.max() > self.max_value:
            X = X * self.scale
        return X

class ConditionalRescalingd(monai.transforms.Transform):
    def __init__(self,keys,max_value,scale):
        self.keys = keys
        self.max_value = max_value
        self.scale = scale
        
        self.transforms = {
            k:ConditionalRescaling(self.max_value,self.scale) 
            for k in self.keys}
    
    def __call__(self,data):
        for k in data:
            if k in self.transforms:
                data[k] = self.transforms[k](data[k])
        return data

class CopyEntryd(monai.transforms.Transform):
    def __init__(self,keys,out_keys):
        self.keys = keys
        self.out_keys = out_keys
    
    def __call__(self,data):
        for k in list(data.keys()):
            if k in self.keys:
                if k in self.out_keys:
                    data[self.out_keys[k]] = deepcopy(data[k])
        return data

class PartiallyRandomSampler(torch.utils.data.Sampler):
    def __init__(self, classes: Iterable, 
                 keep_classes: List[int]=[1], non_keep_ratio: float=1.0,
                 seed:int=None) -> None:
        self.classes = classes
        self.keep_classes = keep_classes
        self.non_keep_ratio = non_keep_ratio
        self.seed = seed
        
        self.keep_list = []
        self.non_keep_list = []

        for x,c in enumerate(self.classes):
            if c in self.keep_classes:
                self.keep_list.append(x)
            else:
                self.non_keep_list.append(x)
        self.n_keep = len(self.keep_list)
        self.n = len(self.keep_list) + int(self.n_keep*(self.non_keep_ratio))

        if self.seed is None:
            self.seed = np.random.randint(1e6)
        self.rng = np.random.default_rng(self.seed)

    def __iter__(self):
        rand_list = [
            *self.keep_list,*self.rng.choice(
                self.non_keep_list,int(self.n_keep*(self.non_keep_ratio)))]
        self.rng.shuffle(rand_list)
        yield from iter(rand_list)

    def __len__(self) -> int:
        return self.n

class FastResample(monai.transforms.Transform):
    def __init__(self,target:List[float],keys=List[str],mode=List[str]):
        """Does what monai.transforms.Spacingd does but fast by getting rid of
        some unnecessary calculations.

        Args:
            target (List[float]): _description_
            keys (_type_, optional): _description_. Defaults to List[str].
            mode (_type_, optional): _description_. Defaults to List[str].
        """
        self.target = np.array(target,np.float64)
        self.keys = keys
        self.mode = mode

        self.interpolation_modes = {
            k:m for k,m in zip(self.keys,self.mode)}

    def ensure_tensor(self,x):
        x = torch.as_tensor(x).unsqueeze(1)
        return x

    def __call__(self,X):
        for k in self.keys:
            meta = X[k + '_meta_dict']
            if 'spacing' in meta:
                spacing = meta['spacing']
                # redefine spacing
                meta['spacing'] = self.target
            else:
                spacing = meta['pixdim'][1:4]
                meta['pixdim'][1:4] = self.target
            spacing = np.array(spacing,np.float64)
            spacing_ratio = spacing/self.target
            output_shape = np.round(np.multiply(
                spacing_ratio,
                np.array(X[k].shape[1:],dtype=np.float64))).astype(np.int64)
            intp = self.interpolation_modes[k]
            X[k] = F.interpolate(
                self.ensure_tensor(X[k]),
                size=output_shape.tolist(),mode=intp).squeeze(1)
        return X