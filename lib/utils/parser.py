"""
Set of functions to make argparse compatible with YAML config files. In 
essence, these functions replace arguments in a given argparse.Namespace using
a given YAML configuration file.
"""

import sys
import argparse
import yaml
import importlib 

from typing import Dict,Any,List

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
