import argparse
import json
import numpy as np
from pydicom import dcmread
from tqdm import tqdm


def calculate_parameters(x: np.ndarray):
    return {
        "min": x.min(),
        "nan count": np.sum(np.isnan(x)),
        "inf count": np.sum(np.isinf(x)),
        "max": x.max(),
    }


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument("--input_path", dest="input_path")

    args = parser.parse_args()

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
