import argparse
import json
import re

desc = "Merges two JSON datasets. Solves conflicts using suffixes"


def main(arguments):
    parser = argparse.ArgumentParser(description=desc)

    parser.add_argument(
        "--input_paths",
        dest="input_paths",
        action="store",
        nargs="+",
        help="Paths to JSON files.",
        required=True,
    )
    parser.add_argument(
        "--suffixes",
        dest="suffixes",
        action="store",
        nargs="+",
        help="When solving conflicts uses these suffixes.",
        default=None,
    )
    parser.add_argument(
        "--rename",
        dest="rename",
        action="store",
        help="Renames old fields to new names (format: old_name,new_name).",
        default=None,
        nargs="+",
    )
    parser.add_argument(
        "--keep_all",
        dest="keep_all",
        action="store_true",
        help="Keeps all entries, even if they are exactly identical between \
            records (suffixes/indices will be added).",
    )

    args = parser.parse_args(arguments)

    dict_list = [json.load(open(x)) for x in args.input_paths]
    if args.suffixes is None:
        args.suffixes = [None for _ in dict_list]
    if args.rename is None:
        args.rename = {}
    else:
        args.rename = dict([tuple(k.split(",")) for k in args.rename])

    all_keys = set.intersection(*[set(x.keys()) for x in dict_list])
    all_keys = list(all_keys)

    output = {}
    for k in all_keys:
        output[k] = {}
        for D, suffix in zip(dict_list, args.suffixes):
            for kk in D[k]:
                updated_kk = kk
                if kk in output[k]:
                    if D[k][kk] != output[k][kk] or args.keep_all is True:
                        new_kk = False
                        if suffix is None:
                            while new_kk is False:
                                index = re.search(r"[0-9]+$", updated_kk)
                                if index is not None:
                                    index = index.group()
                                    kk_root = updated_kk.replace("_" + index, "")
                                    updated_kk = "{}_{}".format(kk_root, int(index) + 1)
                                else:
                                    index = 0
                                    kk_root = updated_kk
                                    updated_kk = "{}_{}".format(kk_root, int(index) + 1)
                                if updated_kk not in output[k]:
                                    new_kk = True
                        else:
                            updated_kk = "{}_{}".format(kk, suffix)
                output[k][updated_kk] = D[k][kk]

    for k in sorted(list(output.keys())):
        r = {}
        for kk in sorted(list(output[k].keys())):
            if kk in args.rename:
                new_kk = args.rename[kk]
            else:
                new_kk = kk
            r[new_kk] = output[k][kk]
        output[k] = r
    print(json.dumps(output, indent=2))
