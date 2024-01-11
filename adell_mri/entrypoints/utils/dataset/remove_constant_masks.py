import argparse
import json
import numpy as np
import SimpleITK as sitk
from tqdm import tqdm

desc = "Removes empty masks from dataset JSON"


def main(arguments):
    parser = argparse.ArgumentParser(description=desc)

    parser.add_argument(
        "--input_json",
        dest="input_json",
        required=True,
        help="Path to input JSON file",
    )
    parser.add_argument(
        "--mask_keys",
        dest="mask_keys",
        required=True,
        nargs="+",
        help="Mask keys (entries with no mask keys or where the mask has only \
            one value are removed).",
    )

    args = parser.parse_args(arguments)

    data_dict = json.load(open(args.input_json, "r"))
    all_keys = list(data_dict.keys())
    nc_keys = []
    for k in tqdm(all_keys):
        constant = True
        for kk in args.mask_keys:
            if kk in data_dict[k]:
                mask = sitk.GetArrayFromImage(sitk.ReadImage(data_dict[k][kk]))
                if len(np.unique(mask)) > 1:
                    constant = False
        if constant == False:
            nc_keys.append(k)

    data_dict = {k: data_dict[k] for k in nc_keys}
    print(json.dumps(data_dict, indent=2))
