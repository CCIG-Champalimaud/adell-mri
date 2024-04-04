import argparse
import json

desc = "Creates individual entries from a hierarchical JSON"


def main(arguments):
    parser = argparse.ArgumentParser(description=desc)

    parser.add_argument(
        "--input_json",
        dest="input_json",
        required=True,
        help="Path to input JSON file",
    )
    parser.add_argument(
        "--image_keys",
        dest="image_keys",
        required=True,
        nargs="+",
        help="Keys that are to be kept as individual entries.",
    )

    args = parser.parse_args(arguments)

    data_dict = json.load(open(args.input_json, "r"))
    all_keys = list(data_dict.keys())
    image_keys = args.image_keys
    output = {}
    for k in all_keys:
        for kk in data_dict[k]:
            if kk in image_keys:
                new_k = k + "_" + kk
                output[new_k] = {"image": data_dict[k][kk]}

    print(json.dumps(output, indent=2))
