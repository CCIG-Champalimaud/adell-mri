import torch
import numpy as np

from types import List,Dict,Union

ParamGroupDict = Dict[
    str,Union[List[torch.nn.Parameter],float,int,bool,str,bool]]

def create_param_groups(model:torch.nn.Module,
                        var_groups:List[Union[List[str],str]],
                        kwargs_groups:List[Dict[str,Union[str,int,float,bool]]]=None,
                        include_leftovers:bool=True) -> ParamGroupDict:
    """Given a model `model` and a set of groups of variable names `var_groups`
    or substrings creates a torch.optim.Optimizer compatible list of 
    dictionaries.

    Args:
        model (torch.nn.Module): a PyTorch model or module.
        var_groups (List[Union[List[str],str]]): a list containing either: 
            i) a list, where each element is a string that should be an exact 
            match with a given parameter name (no error is raised if not) or 
            ii) A string, which will be used as a substring for pattern 
            detection (if a parameter name contains this string, then it will
            be added to this parameter group)
        kwargs_groups (List[Dict[str,Union[str,int,float,bool]]], optional):
            a list with the same length as `var_groups`, containing other
            keyword arguments (as a dictionary) for the optimizer. Defaults to
            None (default keyword arguments).
        include_leftovers (bool, optional): whether the last group should 
            contain parameters whose names did not match any of the provided
            parameter names in var_groups. Defaults to True.

    Returns:
        ParamGroupDict: a dictionary of values that can be set as `params` in
            torch.optim.Optimizer objects.
    """
    if kwargs_groups is None: kwargs_groups = [{} for _ in var_groups]
    if include_leftovers == True: leftovers = {"params":[]}
    assert len(kwargs_groups) == len(var_groups), \
        "len(kwargs_groups) should be the same as len(var_groups)"
    params = []
    for kw in kwargs_groups:
        tmp_dict = {"params":[]}
        for k in kw:
            tmp_dict[k] = kw[k]
        params.append(tmp_dict)
    for k,param in model.named_parameters():
        included = False
        for i,var_group in enumerate(var_groups):
            if isinstance(var_group,list):
                if k in var_group:
                    params[i]["params"].append(param)
                    included = True
            if isinstance(var_group,str):
                if var_group in k:
                    params[i]["params"].append(param)
                    included = True
        if included == False and include_leftovers == True:
            leftovers["params"].append(param)
    if include_leftovers == True:
        params.append(leftovers)
    return params

class EarlyStopper:
    """Early stopping of training if the validation loss does not improve
    after `patience` epochs.
    
    Based on https://stackoverflow.com/a/73704579
    """
    def __init__(self,patience:int=1,min_delta:float=0.0):
        """
        Args:
            patience (int, optional): number of epochs with no improvement.
                Defaults to 1.
            min_delta (float, optional): minimum difference between current loss
                and minimum loss to consider that the loss is not improving. 
                Defaults to 0.
        """
        self.patience = patience
        self.min_delta = min_delta
        self.counter = 0
        self.min_validation_loss = np.inf

    def __call__(self, validation_loss:torch.Tensor)->bool:
        """
        Args:
            validation_loss (torch.Tensor): value for the validation loss.

        Returns:
            bool: whether or not the training should be stopped.
        """
        try:
            validation_loss = validation_loss.numpy()
        except:
            validation_loss = float(validation_loss)
        if validation_loss < self.min_validation_loss:
            self.min_validation_loss = validation_loss
            self.counter = 0
        elif validation_loss > (self.min_validation_loss + self.min_delta):
            self.counter += 1
            if self.counter >= self.patience:
                return True
        return False
