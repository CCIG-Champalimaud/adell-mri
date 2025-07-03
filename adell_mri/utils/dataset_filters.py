"""
Functions for dataset filtering.
"""

import os
from copy import deepcopy
from typing import List

from adell_mri.custom_types import DatasetDict


def print_verbose(*args, verbose: bool = False, **kwargs):
    """
    Prints the arguments if verbose is True.
    """
    if verbose:
        print(*args, **kwargs)


def fill_missing_with_value(
    D: DatasetDict, filters: List[str], verbose: bool = False
) -> DatasetDict:
    """
    Imputes missing values with a given value, both present in filters as a
    list of strings specified as key:value pairs.

    Args:
        D (DatasetDict): dataset dictionary.
        filters (List[str]): list of with key:value pairs.
        verbose (bool, optional): verbosity. Defaults to False.

    Returns:
        DatasetDict: imputed dataset dictionary.
    """
    print_verbose(f"Filling keys: {filters}", verbose=verbose)
    n = 0
    filters = [k.split(":") for k in filters]
    filters = {k[0]: k[1] for k in filters}
    for key in D:
        for filter_key in filters:
            if filter_key not in D[key]:
                D[key][filter_key] = filters[filter_key]
                n += 1
    print_verbose(f"\tFilled keys: {n}", verbose=verbose)
    return D


def fill_conditional(
    D: DatasetDict, filters: List[str], verbose: bool = False
) -> DatasetDict:
    """
    Imputes missing values with a given value, both present in filters as a
    list of strings specified as key_to_fill:value_to_fill pairs if
    key_to_check:value_to_check evaluates to True.

    Args:
        D (DatasetDict): dataset dictionary.
        filters (List[str]): list of with
            key_to_fill:value_to_fill^key_to_check:value_to_check pairs.
        verbose (bool, optional): verbosity. Defaults to False.

    Returns:
        DatasetDict: imputed dataset dictionary.
    """
    print_verbose(f"Filling keys conditionally: {filters}", verbose=verbose)
    n = 0
    filters = [k.split("^") for k in filters]
    filters = [(k[0].split(":"), k[1].split(":")) for k in filters]
    for key in D:
        for filter in filters:
            new_key = filter[0][0]
            new_value = filter[0][1]
            filter_key = filter[1][0]
            filter_value = filter[1][1]
            if filter_key in D[key]:
                if str(D[key][filter_key]) == str(filter_value):
                    if new_key not in D[key]:
                        D[key][new_key] = new_value
                        n += 1
    print_verbose(f"\tFilled keys: {n}", verbose=verbose)
    return D


def filter_dictionary_with_presence(
    D: DatasetDict, filters: List[str], verbose: bool = False
) -> DatasetDict:
    """Filters a dictionary based on whether a nested dictionary has the keys
    specified in filters.

    Args:
        D (DatasetDict): dataset dictionary.
        filters (List[str]): list of strings.
        verbose (bool, optional): verbosity. Defaults to False.

    Returns:
        DatasetDict: filtered dataset dictionary.
    """
    print_verbose("Filtering on: {} presence".format(filters), verbose=verbose)
    print_verbose("\tInput size: {}".format(len(D)), verbose=verbose)
    out_dict = {}
    for pid in D:
        check = True
        for k in filters:
            if k not in D[pid]:
                check = False
        if check is True:
            out_dict[pid] = D[pid]
    print_verbose("\tOutput size: {}".format(len(out_dict)), verbose=verbose)
    return out_dict


def filter_dictionary_with_existence(
    D: DatasetDict, filters: List[str], verbose: bool = False
) -> DatasetDict:
    """Filters a dictionary based on whether files with a given key exist.

    Args:
        D (DatasetDict): dataset dictionary.
        filters (List[str]): list of strings.
        verbose (bool, optional): verbosity. Defaults to False.

    Returns:
        DatasetDict: filtered dataset dictionary.
    """
    print_verbose("Filtering on: {} existence".format(filters), verbose=verbose)
    print_verbose("\tInput size: {}".format(len(D)), verbose=verbose)
    out_dict = {}
    for pid in D:
        check = True
        for k in filters:
            if k not in D[pid]:
                check = False
            elif os.path.exists(D[pid][k]) is False:
                check = False
        if check is True:
            out_dict[pid] = D[pid]
    print_verbose("\tOutput size: {}".format(len(out_dict)), verbose=verbose)
    return out_dict


def filter_dictionary_with_possible_labels(
    D: DatasetDict,
    possible_labels: List[str],
    label_key: str,
    verbose: bool = False,
) -> DatasetDict:
    """Filters a dictionary by checking whether the possible_labels are
    included in DatasetDict[patient_id][label_key].

    Args:
        D (DatasetDict): dataset dictionary.
        possible_labels (List[str]): list of possible labels.
        label_key (str): key corresponding to the label field.
        verbose (bool, optional): verbosity. Defaults to False.

    Returns:
        DatasetDict: filtered dataset dictionary.
    """
    print_verbose(
        "Filtering on possible labels: {}".format(possible_labels),
        verbose=verbose,
    )
    print_verbose("\tInput size: {}".format(len(D)), verbose=verbose)
    out_dict = {}
    for pid in D:
        check = True
        if label_key not in D[pid]:
            check = False
        else:
            if str(D[pid][label_key]) not in possible_labels:
                check = False
        if check is True:
            out_dict[pid] = D[pid]
    print_verbose("\tOutput size: {}".format(len(out_dict)), verbose=verbose)
    return out_dict


def filter_dictionary_with_filters(
    D: DatasetDict,
    filters: List[str],
    filter_is_optional: bool = False,
    verbose: bool = True,
) -> DatasetDict:
    """Filters a dataset dictionary with custom filters:
    * If "key=value": tests if field key is equal/contains value
    * If "key>value": tests if field key is larger than value
    * If "key<value": tests if field key is smaller than value

    For inequalities, the value is converted to float.

    Args:
        D (DatasetDict): dataset dictionary.
        filters (List[str]): filters.
        filter_is_optional (bool, optional): considers the filters to be
            optional. Defaults to False.
        verbose (bool, optional): verbosity. Defaults to False.

    Returns:
        DatasetDict: filtered dataset dictionary.
    """
    print_verbose("Filtering on: {}".format(filters), verbose=verbose)
    print_verbose("\tInput size: {}".format(len(D)), verbose=verbose)
    processed_filters = {
        "eq": [],
        "gt": [],
        "lt": [],
        "neq": [],
        "in": [],
        "match": [],
        "not_match": [],
    }
    for f in filters:
        if "!=" in f:
            processed_filters["neq"].append(f.split("!="))
        elif "=" in f:
            processed_filters["eq"].append(f.split("="))
        elif ">" in f:
            processed_filters["gt"].append(f.split(">"))
        elif "<" in f:
            processed_filters["lt"].append(f.split("<"))
        elif "(in)" in f:
            k, v = f.split("(in)")
            v = v.split(",")
            processed_filters["in"].append([k, v])
        elif "(match)" in f:
            k, v = f.split("(match)")
            processed_filters["match"].append([k, v])
        elif "(!match)" in f:
            k, v = f.split("(!match)")
            processed_filters["not_match"].append([k, v])
        else:
            err = (
                "filter {} must have one of ['=','<','>','!=','(in)'].".format(
                    f
                )
            )
            err += " For example: age>50 or clinical_variable!=true"
            raise NotImplementedError(err)
    out_dict = {}
    for pid in D:
        check = True
        for k in processed_filters:
            for kk, v in processed_filters[k]:
                if kk in D[pid]:
                    if k == "eq":
                        if "[" in str(D[pid][kk]) or isinstance(
                            D[pid][kk], list
                        ):
                            tmp = [str(x) for x in D[pid][kk]]
                            if v not in tmp:
                                check = False
                        else:
                            if str(D[pid][kk]) != v:
                                check = False
                    elif k == "gt":
                        if float(D[pid][kk]) <= float(v):
                            check = False
                    elif k == "lt":
                        if float(D[pid][kk]) >= float(v):
                            check = False
                    elif k == "neq":
                        if str(D[pid][kk]) == v:
                            check = False
                    elif k == "in":
                        if str(D[pid][kk]) not in v:
                            check = False
                    elif k == "match":
                        if v not in str(D[pid][kk]):
                            check = False
                    elif k == "not_match":
                        if v in str(D[pid][kk]):
                            check = False
                elif filter_is_optional is False:
                    check = False
        if check is True:
            out_dict[pid] = D[pid]
    print_verbose("\tOutput size: {}".format(len(out_dict)), verbose=verbose)
    return out_dict


def filter_dictionary(
    D: DatasetDict,
    filters_presence: List[str] = None,
    filters_existence: List[str] = None,
    possible_labels: List[str] = None,
    label_key: str = None,
    filters: List[str] = None,
    filter_is_optional: bool = False,
    verbose: bool = False,
) -> DatasetDict:
    """
    Wraps all dataset filters in a more convenient function.

    Args:
        D (DatasetDict): dataset dictionary
        filters_presence (List[str], optional): list of filters for
            :func:`filter_dictionary_with_presence`. Defaults to None.
        filters_existence (List[str], optional): list of filters for
            :func:`filter_dictionary_with_existence`. Defaults to None.
        possible_labels (List[str], optional): list of possible labels.
            Defaults to None.
        label_key (str, optional): label key. Defaults to None.
        filters (List[str], optional): list of filters for
            :func:`filter_dictionary_with_filters`. Defaults to None.
        filter_is_optional (bool, optional): considers the filters to be
            optional. Defaults to False.
        verbose (bool, optional): verbosity. Defaults to False.

    Returns:
        DatasetDict: filtered dictionary.
    """
    D = deepcopy(D)
    if filters_presence is not None:
        D = filter_dictionary_with_presence(
            D, filters_presence, verbose=verbose
        )
    if filters_existence is not None:
        D = filter_dictionary_with_existence(
            D, filters_existence, verbose=verbose
        )
    if (possible_labels is not None) and (label_key is not None):
        D = filter_dictionary_with_possible_labels(
            D, possible_labels, label_key, verbose=verbose
        )
    if filters is not None:
        D = filter_dictionary_with_filters(
            D, filters, filter_is_optional=filter_is_optional, verbose=verbose
        )
    return D
