"""
Set of functions to make argparse compatible with YAML config files. In 
essence, these functions replace arguments in a given argparse.Namespace using
a given YAML configuration file.
"""

import os
import sys
import argparse
import yaml
import json
import importlib
from hydra import compose as hydra_compose
from hydra import initialize_config_dir
from omegaconf import OmegaConf

from typing import Dict, Any, List

def get_dvc_params(path:str=None)->Dict[str,any]:
    """Retrieves dvc parameters recursively. For example, if the path is
    `training`, then the actual parameters are assumed to be in the `training`
    field of the dvc parameters.

    Args:
        path (str): params structure, containing none or more ":" characters
            corresponding to nested fields in the dvc config.

    Returns:
        Dict[str,Any]: parameter dictionary.
    """
    if importlib.util.find_spec("dvc") is None:
        raise ImportError(
            "please install dvc (`pip install dvc`) if you wish to use dvc\
                params")
    
    import dvc.api
    if ":" in path:
        keys = ":".split(path)
    else:
        keys = []
    params = dvc.api.params_show()
    for k in keys:
        params = params[k]
    return params

def read_param_file(path:str)->Dict[str,Any]:
    """Reads parameters from a YAML file. If any ":" is present in path, then 
    it returns the field after ":" from the YAML config file. For example, if
    the path is `config.yaml:training`, then the parameters are assumed to be
    in the `training` field of `config.yaml`. This works recursively.

    Args:
        path (str): path to YAML file, containing none or more ":" characters
            corresponding to nested fields in the YAML file.

    Returns:
        Dict[str,Any]: parameter dictionary.
    """
    if ":" in path:
        out = ":".split(path)
        path,keys = out[0],out[1:]
    else:
        keys = []
    params = yaml.load(path)
    for k in keys:
        params = params[k]
    return params

def get_params(path:str)->Dict[str,Any]:
    """Wrapper around `get_dvc_params` and `read_param_file`. If the first
    element of `path.split(":")` is `"dvc"`, then `get_dvc_params` is called
    for the rest of `path`. Otherwise, `read_param_file` is used.

    Args:
        path (str): path to config.

    Returns:
        Dict[str,Any]: parameter dictionary.
    """
    path_ = path.split(":")
    if len(path_) > 1:
        keys = ":".join(path_[1:])
    else:
        keys = ""
    if path_[0] == "dvc":
        params = get_dvc_params(keys)
    else:
        params = read_param_file(path)
    return params

def merge_args(args:argparse.Namespace,
               param_dict:Dict[str,Any],
               sys_arg:List[str]=None)->argparse.Namespace:
    """Replaces the key-value pairs in a given argparse.Namespace `args` by 
    the values in param_dict if they are not defined in sys_arg. In essence,
    the priority is: default arguments < param_dict < command line arguments.
    This function works in place but also returns `args`.

    Args:
        args (argparse.Namespace): argparse.Namespace as returned by 
            `ArgumentParser().parse_args()`.
        param_dict (Dict[str,Any]): dictionary containing parameters that will
            be used to replace arguments in args.
        sys_arg (List[str], optional): command line arguments as returned by
            `sys.argv`, used to check if an argument has been defined in the
            command line (checks all strings with a leading "--"). Defaults to
            None (sys.argv[1:]).

    Returns:
        argparse.Namespace: merged argparse.Namespace
    """
    if sys_arg is None:
        sys_arg = sys.argv[1:]
    defined_args = [x.replace("--","") for x in sys_arg if x[:2] == "--"]
    for k in param_dict:
        if hasattr(args,k):
            # skips if argument has already been defined
            if k not in defined_args:
                setattr(args,k,param_dict[k])
        else:
            raise KeyError("{} is not an ArgumentParser argument".format(k))
    return args

def compose(path: str, job_name: str, overrides: List[str]=None):
    config_path, config_name = os.path.dirname(path), os.path.basename(path)
    config_path = os.path.abspath(config_path)
    if overrides is None:
        overrides = []
    with initialize_config_dir(version_base=None, 
                               config_dir=config_path, 
                               job_name=job_name):
        cfg = hydra_compose(config_name=config_name, overrides=overrides)
        yaml_config = OmegaConf.to_yaml(cfg)
        params = yaml.safe_load(yaml_config)
    return params

def parse_ids(id_list:List[str],
              output_format:str="nested_list"):
    def parse_id_file(id_file:str):
        if ":" in id_file:
            id_file,id_set = id_file.split(":")
            id_set = id_set.split(",")
        else:
            id_set = None
        term = id_file.split(".")[-1]
        if term == "csv" or term == "folds":
            with open(id_file) as o:
                out = [x.strip().split(",") for x in o.readlines()]
            out = {x[0]:x[1:] for x in out}
        elif term == "json":
            with open(id_file) as o:
                out = json.load(o)
        else:
            with open(id_file) as o:
                out = {"id_set":[x.strip() for x in o.readlines()]}
        if id_set is None:
            id_set = list(out.keys())
        return {k:out[k] for k in id_set}
    
    def merge_dictionary(nested_list:Dict[str,list]):
        output_list = []
        for list_tmp in nested_list.values():
            output_list.extend(list_tmp)
        return output_list
    
    output = {}
    for element in id_list:
        if os.path.exists(element.split(":")[0]) is True:
            tmp = parse_id_file(element)
            for k in tmp:
                if k not in output:
                    output[k] = []
                output[k].extend(tmp[k])
        else:
            if "cli" not in output:
                output["cli"] = []
            output["cli"].extend([element.split(",")])
    if output_format == "list":
        output = merge_dictionary(output)
    else:
        output = [output[k] for k in output]
    return output