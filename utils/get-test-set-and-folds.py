import argparse
import json
import numpy as np
import os
from sklearn.model_selection import StratifiedKFold,KFold,train_test_split

from typing import List, Dict

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

if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    # data
    parser.add_argument(
        '--dataset_json',dest='dataset_json',type=str,
        help="JSON containing dataset information",required=True)
    parser.add_argument(
        '--stratify',dest='stratify',type=str,nargs="+",
        help="Stratify on this variable",default=None)
    parser.add_argument(
        '--all_keys',dest='all_keys',type=str,nargs="+",
        help="all keys",default=None)
    parser.add_argument(
        '--n_folds',dest="n_folds",
        help="Number of validation folds",default=5,type=int)
    parser.add_argument(
        '--fraction_test',dest="fraction_test",
        help="Fraction of samples to be considered for test",
        default=0.30,type=float)
    parser.add_argument(
        '--seed',dest="seed",
        help="random seed",default=42,type=int)
    parser.add_argument(
        '--excluded_ids',dest='excluded_ids',nargs="+",
        help='ids that will be excluded',default=None,type=str)

    args = parser.parse_args()

    if args.excluded_ids is not None:
        excluded_ids = parse_ids(args.excluded_ids,"list")
    else:
        excluded_ids = []

    strata = []
    data_dict = json.load(open(args.dataset_json,'r'))
    data_dict = {k:data_dict[k] for k in data_dict if k not in excluded_ids}
    if args.all_keys is not None:
        new_dict = {}
        for k in data_dict:
            check = True
            for kk in args.all_keys:
                if kk not in data_dict[k]:
                    check = False
            if check == True:
                new_dict[k] = data_dict[k]
        data_dict = new_dict
    
    if args.stratify is not None:
        for s in args.stratify:
            for k in data_dict:
                if s not in data_dict[k]:
                    data_dict[k][s] = None
        strata = ["".join([str(data_dict[k][kk]) for kk in args.stratify])
                  for k in data_dict]
        for u,c in zip(*np.unique(strata,return_counts=True)):
            if c < args.n_folds * 2:
                strata = ["nan" if s == u else s for s in strata]
    else:
        strata = None
    
    all_pids = [k for k in data_dict]
    if args.fraction_test > 0.0:
        all_train_pids,test_pids = train_test_split(
            range(len(all_pids)),test_size=args.fraction_test,
            random_state=args.seed,shuffle=True,stratify=strata)
    else:
        all_train_pids = range(len(all_pids))
        test_pids = []

    if args.n_folds > 1:
        if strata is not None:
            fold_generator = StratifiedKFold(
                args.n_folds,shuffle=True,random_state=args.seed).split(
                    all_train_pids,[strata[i] for i in all_train_pids])
        else:
            fold_generator = KFold(
                args.n_folds,shuffle=True,random_state=args.seed).split(all_train_pids)
    else:
        if strata is not None:
            fold_generator = iter(
                [train_test_split(
                    range(len(all_train_pids)),test_size=0.2,
                    random_state=args.seed,stratify=[strata[i] for i in all_train_pids])])
        else:
            fold_generator = iter(
                [train_test_split(
                    range(len(all_train_pids)),test_size=0.2,
                    random_state=args.seed)])
    
    print("test," + ",".join([all_pids[i] for i in test_pids]))
    
    for i,(train_idxs,val_idxs) in enumerate(fold_generator):
        print(f"cv{i+1}," + ','.join(
            [all_pids[all_train_pids[i]] for i in val_idxs]))