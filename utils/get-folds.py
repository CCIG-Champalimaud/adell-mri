import argparse
import json
from sklearn.model_selection import KFold,train_test_split

if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    # data
    parser.add_argument(
        '--dataset_json',dest='dataset_json',type=str,
        help="JSON containing dataset information",required=True)
    parser.add_argument(
        '--all_keys',dest='all_keys',type=str,nargs="+",
        help="all keys",required=True)
    parser.add_argument(
        '--n_folds',dest="n_folds",
        help="Number of validation folds",default=5,type=int)
    parser.add_argument(
        '--seed',dest="seed",
        help="random seed",default=42,type=int)

    args = parser.parse_args()

    data_dict = json.load(open(args.dataset_json,'r'))
    for k in args.all_keys:
        nd = {}
        for kk in data_dict:
            if k in data_dict[kk]:
                nd[kk] = data_dict[kk]
        data_dict = nd

    all_pids = [k for k in data_dict]

    if args.n_folds > 1:
        fold_generator = KFold(
            args.n_folds,shuffle=True,random_state=args.seed).split(all_pids)
    else:
        fold_generator = iter(
            [train_test_split(range(len(all_pids)),test_size=0.2)])
    
    for train_idxs,val_idxs in fold_generator:
        print(','.join([all_pids[i] for i in val_idxs]))