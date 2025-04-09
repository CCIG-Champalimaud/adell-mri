import numpy as np

desc = "Prints entries with nan/infinite in DICOM dataset"


def calculate_parameters(x: np.ndarray):
    return {
        "min": x.min(),
        "nan count": np.sum(np.isnan(x)),
        "inf count": np.sum(np.isinf(x)),
        "max": x.max(),
    }


def main(arguments):
    import argparse
    import json

    from pydicom import dcmread
    from tqdm import tqdm

    parser = argparse.ArgumentParser(description=desc)

    parser.add_argument(
        "--json_path", dest="json_path", help="Path to DICOM dataset JSON"
    )

    args = parser.parse_args(arguments)

    all_data = json.load(open(args.input_path))

    for k in tqdm(all_data):
        for kk in all_data[k]:
            for element in all_data[k][kk]:
                image_path = element["image"]
                image = dcmread(image_path).pixel_array

                p = calculate_parameters(image)

                if p["nan count"] > 0:
                    print(image_path, p, "nan")
                if p["inf count"] > 0:
                    print(image_path, p, "inf")
