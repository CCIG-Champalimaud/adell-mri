import numpy as np
import re
import torch

from typing import Dict,Union,List,Any

def load_checkpoint_to_model(
        model:torch.nn.Module,
        checkpoint:Union[str,Dict[str,torch.Tensor]],
        exclude_from_state_dict:List[str])->torch.nn.Module:
    if isinstance(checkpoint,str):
        sd = torch.load(checkpoint)
    else:
        sd = checkpoint
    if "state_dict" in sd:
        sd = sd["state_dict"]

    print(f"Loading checkpoint from {checkpoint}")
    if exclude_from_state_dict is not None:
        for pattern in exclude_from_state_dict:
            sd = {k:sd[k] for k in sd
                  if re.search(pattern,k) is None}
    output = model.load_state_dict(sd,strict=False)

    if len(output.unexpected_keys) > 0:
        raise Exception("Dictionary contains more keys than it should!")
    print(f"\t{output.missing_keys}")

def get_class_weights(class_weights:List[Union[float,str]],
                      n_classes:int,
                      classes:List[Any],
                      positive_labels:List[Any],
                      possible_labels:List[Any],
                      label_groups:List[List[Any]]=None)->List[float]:
    if class_weights is not None:
        if class_weights[0] == "adaptive":
            if n_classes == 2:
                pos = len([x for x in classes 
                            if x in positive_labels])
                neg = len(classes) - pos
                weight_neg = (1 / neg) * (len(classes) / 2.0)
                weight_pos = (1 / pos) * (len(classes) / 2.0)
                class_weights = weight_pos/weight_neg
            else:
                pos = {k:0 for k in possible_labels}
                for c in classes:
                    pos[c] += 1
                if label_groups is not None:
                    new_pos = {i:0 for i in range(len(label_groups))}
                    for i in range(len(label_groups)):
                        label_group = label_groups[i]
                        for label in label_group:
                            new_pos[i] += pos[label]
                    pos = new_pos
                pos = np.array([pos[k] for k in pos])
                class_weights = (1 / pos) * (len(classes) / 2.0)
        else:
            class_weights = [float(x) for x in class_weights]

    return class_weights

def conditional_parameter_freezing(network:torch.nn.Module,
                                   freeze_regex:List[str]=None,
                                   do_not_freeze_regex:List[str]=None,
                                   state_dict:Dict[str,torch.Tensor]=None):
    """Freezes (or not) parameters according to a list of regex and loads an 
    optional state dict if frozen keys match dictionary.

    Args:
        network (torch.nn.Module): torch module with a named_parameters 
            attribute.
        freeze_regex (List[str], optional): regex for parameter names that 
            should be frozen. Defaults to None.
        do_not_freeze_regex (List[str], optional): regex for parameter names 
            that should not be frozen (overrides freeze_regex). Defaults to 
            None.
        state_dict (Dict[str,torch.Tensor], optional): state dict that replaces
            frozen values. Defaults to None.
    """
    keys_to_load = []
    freeze_regex_list = []
    do_not_freeze_regex_list = []

    if freeze_regex is not None:
        freeze_regex_list = [
            re.compile(fr) for fr in freeze_regex]
    if do_not_freeze_regex is not None:
        do_not_freeze_regex_list = [
            re.compile(dnfr) for dnfr in do_not_freeze_regex]
    
    for key,param in network.named_parameters():
        freeze = False
        if any([fr.search(key) is not None 
                for fr in freeze_regex_list]):
            freeze = True
        if any([dnfr.search(key) is not None 
                for dnfr in do_not_freeze_regex_list]):
            freeze = False
        if freeze is True:
            param.requires_grad = False
            if state_dict is not None:
                if key in state_dict:
                    keys_to_load.append(key)
    if state_dict is not None:
        with torch.no_grad():
            network.load_state_dict(
                {k:state_dict[k] for k in keys_to_load})

def set_classification_layer_bias(pos:float,
                                  neg:float,
                                  network:torch.nn.Module,
                                  class_substr:str="classification"):
    """Sets the classification layer bias according to class prevalence in the
    binary classification setting.

    Args:
        pos (float): number of positive cases.
        neg (float): number of negative cases.
        network (torch.nn.Module): network.
        class_substr (str, optional): class substring corresponding to bias.
            Defaults to "classification".
    """
    value = torch.as_tensor(np.log(pos/neg))
    for k,v in network.named_parameters():
        if class_substr in k:
            if list(v.shape) == [1]:
                with torch.no_grad():
                    v[0] = value
