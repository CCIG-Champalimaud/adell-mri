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
                      possible_labels:List[Any])->List[float]:
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
                pos = np.array([pos[k] for k in pos])
                class_weights = (1 / pos) * (len(classes) / 2.0)
        else:
            class_weights = [float(x) for x in class_weights]

    return class_weights
