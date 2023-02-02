import argparse
import json
import numpy as np
from sklearn.model_selection import StratifiedKFold,KFold,train_test_split

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

    args = parser.parse_args()

    strata = []
    data_dict = json.load(open(args.dataset_json,'r'))
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
            data_dict = {k:data_dict[k] for k in data_dict if s in data_dict[k]}
        strata = ["".join([str(data_dict[k][kk]) for kk in args.stratify])
                  for k in data_dict]
        for u,c in zip(*np.unique(strata,return_counts=True)):
            if c < args.n_folds * 2:
                strata = ["nan" if s == u else s for s in strata]
    else:
        strata = None
    
    all_pids = [k for k in data_dict]
    all_train_pids,test_pids = train_test_split(
        range(len(all_pids)),test_size=args.fraction_test,
        random_state=args.seed,shuffle=True,stratify=strata)

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