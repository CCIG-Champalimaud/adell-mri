from typing import Dict,List
from ..custom_types import DatasetDict

def filter_dictionary_with_presence(D:DatasetDict,
                                    filters:List[str])->DatasetDict:
    """Filters a dictionary based on whether a nested dictionary has the keys
    specified in filters.

    Args:
        D (DatasetDict): dataset dictionary.
        filters (List[str]): list of strings.

    Returns:
        DatasetDict: filtered dataset dictionary.
    """
    print("Filtering on: {} presence".format(filters))
    print("\tInput size: {}".format(len(D)))
    out_dict = {}
    for pid in D:
        check = True
        for k in filters:
            if k not in D[pid]:
                check = False
        if check == True:
            out_dict[pid] = D[pid]
    print("\tOutput size: {}".format(len(out_dict)))
    return out_dict

def filter_dictionary_with_possible_labels(D:DatasetDict,
                                           possible_labels:List[str],
                                           label_key:str)->DatasetDict:
    """Filters a dictionary by checking whether the possible_labels are
    included in DatasetDict[patient_id][label_key].

    Args:
        D (DatasetDict): dataset dictionary.
        possible_labels (List[str]): list of possible labels.
        label_key (str): key corresponding to the label field.

    Returns:
        DatasetDict: filtered dataset dictionary.
    """
    print("Filtering on possible labels: {}".format(possible_labels))
    print("\tInput size: {}".format(len(D)))
    out_dict = {}
    for pid in D:
        check = True
        if label_key not in D[pid]:
            check = False
        else:
            if str(D[pid][label_key]) not in possible_labels:
                check = False
        if check == True:
            out_dict[pid] = D[pid]
    print("\tOutput size: {}".format(len(out_dict)))
    return out_dict

def filter_dictionary_with_filters(D:DatasetDict,
                                   filters:List[str])->DatasetDict:
    """Filters a dataset dictionary with custom filters:
    * If "key=value": tests if field key is equal/contains value
    * If "key>value": tests if field key is larger than value
    * If "key>value": tests if field key is smaller than value
    
    For inequalities, the value is converted to float.

    Args:
        D (DatasetDict): dataset dictionary.
        filters (List[str]): filters.

    Returns:
        DatasetDict: filtered dataset dictionary.
    """
    print("Filtering on: {}".format(filters))
    print("\tInput size: {}".format(len(D)))
    processed_filters = {
        "eq":[],"gt":[],"lt":[]}
    for f in filters:
        if "=" in f:
            processed_filters["eq"].append(f.split("="))
        elif ">" in f:
            processed_filters["gt"].append(f.split(">"))
        elif "=" in f:
            processed_filters["lt"].append(f.split("<"))
        else:
            err = "filter {} must have one of ['=','<','>'].".format(f)
            err += " For example: age>50 or clinical_variable=true"
            raise NotImplementedError(err)
    out_dict = {}
    for pid in D:
        check = True
        for k in processed_filters:
            for kk,v in processed_filters[k]:
                if kk in D[pid]:
                    if k == "eq":
                        # if there is a 
                        if "[" in D[pid][kk] or isinstance(D[pid][kk],list):
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
                else:
                    check = False
        if check == True:
            out_dict[pid] = D[pid]
    print("\tOutput size: {}".format(len(out_dict)))
    return out_dict
