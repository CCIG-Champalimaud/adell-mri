desc = "Creates empty masks for a given image key if a condition is fulfilled"


def main(arguments):
    import argparse
    import json
    import os

    import SimpleITK as sitk
    from tqdm import tqdm

    parser = argparse.ArgumentParser(description=desc)

    parser.add_argument(
        "--json_path", required=True, help="Path to dataset JSON"
    )
    parser.add_argument(
        "--output_json_path", required=True, help="Path to output dataset JSON"
    )
    parser.add_argument(
        "--source_key", required=True, help="Key for the source image"
    )
    parser.add_argument(
        "--output_key", required=True, help="key for the output image"
    )
    parser.add_argument(
        "--condition",
        required=True,
        help="Condition used to create empty images. Only supports equality \
            (i.e. key==value)",
    )

    args = parser.parse_args(arguments)

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
