import argparse
import json

import numpy as np
from sklearn.model_selection import KFold, StratifiedKFold, train_test_split

desc = "Splits JSON dataset entries into prospective test set (according to a \
    date key) and folds"


def main(arguments):
    parser = argparse.ArgumentParser(description=desc)

    # data
    parser.add_argument(
        "--dataset_json",
        dest="dataset_json",
        type=str,
        help="JSON containing dataset information",
        required=True,
    )
    parser.add_argument(
        "--stratify",
        dest="stratify",
        type=str,
        nargs="+",
        help="Stratify on this variable",
        default=None,
    )
    parser.add_argument(
        "--all_keys",
        dest="all_keys",
        type=str,
        nargs="+",
        help="all keys",
        default=None,
    )
    parser.add_argument(
        "--n_folds",
        dest="n_folds",
        help="Number of validation folds",
        default=5,
        type=int,
    )
    parser.add_argument(
        "--fraction_test",
        dest="fraction_test",
        help="Fraction of samples to be considered for test",
        default=0.30,
        type=float,
    )
    parser.add_argument("--seed", dest="seed", help="random seed", default=42, type=int)

    args = parser.parse_args(arguments)

    data_dict = json.load(open(args.dataset_json, "r"))

    if args.all_keys is not None:
        new_dict = {}
        for k in data_dict:
            check = True
            for kk in args.all_keys:
                if kk not in data_dict[k]:
                    check = False
            if check is True:
                new_dict[k] = data_dict[k]
        data_dict = new_dict

    data_dict = {k: data_dict[k] for k in data_dict if "study_date" in data_dict[k]}

    if args.stratify is not None:
        for s in args.stratify:
            for k in data_dict:
                if s not in data_dict[k]:
                    data_dict[k][s] = None
        strata = [
            "".join([str(data_dict[k][kk]) for kk in args.stratify]) for k in data_dict
        ]
        for u, c in zip(*np.unique(strata, return_counts=True)):
            if c < args.n_folds * 2:
                strata = ["nan" if s == u else s for s in strata]
    else:
        strata = None

    data_dict = {
        k: data_dict[k]
        for k in sorted(data_dict, key=lambda k: data_dict[k]["study_date"])
    }
    all_pids = [k for k in data_dict]
    n = int((1 - args.fraction_test) * len(all_pids))
    all_idxs = [i for i in range(len(all_pids))]
    all_train_pids, test_pids = all_idxs[:n], all_idxs[n:]

    if args.n_folds > 1:
        if strata is not None:
            fold_generator = StratifiedKFold(
                args.n_folds, shuffle=True, random_state=args.seed
            ).split(all_train_pids, [strata[i] for i in all_train_pids])
        else:
            fold_generator = KFold(
                args.n_folds, shuffle=True, random_state=args.seed
            ).split(all_train_pids)
    else:
        if strata is not None:
            fold_generator = iter(
                [
                    train_test_split(
                        range(len(all_train_pids)),
                        test_size=0.2,
                        random_state=args.seed,
                        stratify=[strata[i] for i in all_train_pids],
                    )
                ]
            )
        else:
            fold_generator = iter(
                [
                    train_test_split(
                        range(len(all_train_pids)),
                        test_size=0.2,
                        random_state=args.seed,
                    )
                ]
            )

    print("test," + ",".join([all_pids[i] for i in test_pids]))

    for i, (train_idxs, val_idxs) in enumerate(fold_generator):
        print(f"cv{i+1}," + ",".join([all_pids[all_train_pids[i]] for i in val_idxs]))
