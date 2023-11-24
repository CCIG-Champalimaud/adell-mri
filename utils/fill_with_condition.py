import argparse
import os
import json
import SimpleITK as sitk
from tqdm import tqdm

if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument("--json_path")
    parser.add_argument("--output_json_path")
    parser.add_argument("--source_key")
    parser.add_argument("--output_key")
    parser.add_argument("--condition")

    args = parser.parse_args()

    df = json.load(open(args.json_path))

    condition_key, condition_value = args.condition.split("==")

    for key in tqdm(df):
        if args.source_key in df[key]:
            if args.output_key not in df[key]:
                if condition_key in df[key]:
                    if str(df[key][condition_key]) == condition_value:
                        path = df[key][args.source_key]
                        image = sitk.ReadImage(path) * 0
                        dir = os.sep.join(os.path.split(path)[:-1])
                        output_path = f"{dir}/mask_filled.nii.gz"
                        sitk.WriteImage(image, output_path)
                        df[key][args.output_key] = output_path

    with open(args.output_json_path, "w") as o:
        json.dump(df, o, indent=2)
