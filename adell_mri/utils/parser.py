"""
Set of functions to make argparse compatible with YAML config files. In 
essence, these functions replace arguments in a given argparse.Namespace using
a given YAML configuration file.
"""

import argparse
import importlib
import json
import os
import sys
from typing import Any, Dict, List

import yaml
from hydra import compose as hydra_compose
from hydra import initialize_config_dir
from omegaconf import OmegaConf


def get_dvc_params(path: str = None) -> Dict[str, any]:
    """
    Retrieves dvc parameters recursively. For example, if the path is
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
                params"
        )

    import dvc.api

    if ":" in path:
        keys = ":".split(path)
    else:
        keys = []
    params = dvc.api.params_show()
    for k in keys:
        params = params[k]
    return params


def read_param_file(path: str) -> Dict[str, Any]:
    """
    Reads parameters from a YAML file. If any ":" is present in path, then
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
        path, keys = out[0], out[1:]
    else:
        keys = []
    params = yaml.load(path)
    for k in keys:
        params = params[k]
    return params


def get_params(path: str) -> Dict[str, Any]:
    """
    Wrapper around `get_dvc_params` and `read_param_file`. If the first
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


def merge_args(
    args: argparse.Namespace,
    param_dict: Dict[str, Any],
    sys_arg: List[str] = None,
) -> argparse.Namespace:
    """
    Replaces the key-value pairs in a given argparse.Namespace `args` by
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
    defined_args = [x.replace("--", "") for x in sys_arg if x[:2] == "--"]
    for k in param_dict:
        if hasattr(args, k):
            # skips if argument has already been defined
            if k not in defined_args:
                setattr(args, k, param_dict[k])
        else:
            raise KeyError("{} is not an ArgumentParser argument".format(k))
    return args


def compose(
    path: str, job_name: str, overrides: List[str] = None
) -> dict[str, Any]:
    """
    Loads a Hydra config from a given path, applies optional overrides,
    and returns the config as a dict.

    The path is split into the config directory and config name. The config
    directory is made absolute. Overrides are applied to the config using
    Hydra's compose functionality. The resulting config is converted to YAML,
    then loaded into a dict and returned.

    Args:
        path (str): path to config file.
        job_name (str): name of job/experiment.
        overrides (List[str]): list of overrides for config file.
    """
    config_path, config_name = os.path.dirname(path), os.path.basename(path)
    config_path = os.path.abspath(config_path)
    if overrides is None:
        overrides = []
    with initialize_config_dir(
        version_base=None, config_dir=config_path, job_name=job_name
    ):
        cfg = hydra_compose(config_name=config_name, overrides=overrides)
        yaml_config = OmegaConf.to_yaml(cfg)
        params = yaml.safe_load(yaml_config)
    return params


def parse_ids(
    id_list: list[str],
    output_format: str = "nested_list",
) -> list[list[str]] | list[str]:
    """
    Parses a list of ID files and returns IDs in the specified format.

    This function accepts a list of ID file paths and parses each one into a
    nested dictionary mapping ID sets to lists of IDs. It can merge these
    dictionaries into a flat list, or keep them nested, based on the
    output_format argument.

    Supports CSV (.csv or .folds) or JSON (.json). Can also take
    comma-separated arguments.

    `id_list` must always be a list of strings:
        1) If it is a list of file paths ending in .csv or .folds, assumes the
        first column in these files is the set identifier and the remamining
        columns are ids.
        2) If it is a list of file paths ending in .json, assumes the file is
        in JSON format and the set identifiers are keys corresponding to lists
        of ids.
        - If it is a comma-separated string, splits on comma and uses each
        element as a set.

    To specify specific sets for 1) and 2), a colon can be used to separate the
    file name from the list of comma-separated sets, i.e.
    `"dataset.json:cv1,cv2"` selects the sets corresponding to keys `"cv1"`
    and `"cv2"`.

    Args:
        id_list (List[str]): can be either a CSV (first column set identifier,
            remaining columns are case identifiers), a JSON (set keys
            corresponding to lists with case identifiers) or a list of
            comma-separated strings.
        output_format (str, optional): returns a nested list (`"nested_list"`,
            one for each specified set) or a list (`"list"`, a flattened
            version of the former). Defaults to `"nested_list"`
    """

    def parse_id_file(id_file: str):
        if ":" in id_file:
            id_file, id_set = id_file.split(":")
            id_set = id_set.split(",")
        else:
            id_set = None
        term = id_file.split(".")[-1]
        if term == "csv" or term == "folds":
            with open(id_file) as o:
                out = [x.strip().split(",") for x in o.readlines()]
            out = {x[0]: x[1:] for x in out}
        elif term == "parquet":
            try:
                import pandas as pd
            except Exception:
                raise ImportError(
                    "Pandas is required to parse parquet files. ",
                    "Please install it with `pip install pandas`.",
                )
            out = pd.read_parquet(id_file).to_dict("list")
        elif term == "json":
            with open(id_file) as o:
                out = json.load(o)
        else:
            with open(id_file) as o:
                out = {"id_set": [x.strip() for x in o.readlines()]}
        if id_set is None:
            id_set = list(out.keys())
        return {k: out[k] for k in id_set}

    def merge_dictionary(nested_list: Dict[str, list]):
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
            output["cli"].extend(element.split(","))
    if output_format == "list":
        output = merge_dictionary(output)
    else:
        output = [output[k] for k in output]
    return output
