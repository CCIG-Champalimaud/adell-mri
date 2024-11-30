import argparse
import json
import os
from multiprocessing import Pool
from pathlib import Path

from tqdm import tqdm

ImageInformation = tuple[str, str, str]
ImageDictionary = dict[str, dict[str, dict[str, str]]]

desc = "Creates JSON file with image paths."


def process_image(path: str) -> ImageInformation:
    image_name = str(path).split(os.sep)[-1]
    study_uid = image_name
    series_uid = image_name
    path = str(path)
    return path, study_uid, series_uid


def update_dict(
    d: ImageDictionary, study_uid: str, series_uid: str, dcm: str
) -> ImageDictionary:
    if study_uid not in d:
        d[study_uid] = {}
    if series_uid not in d[study_uid]:
        d[study_uid][series_uid] = []
    d[study_uid][series_uid].append({"image": dcm})
    return d


def main(arguments):
    parser = argparse.ArgumentParser(description=desc)
    parser.add_argument(
        "--input_path",
        dest="input_path",
        required=True,
        help="Path to folder containing nibabel compatible files",
    )
    parser.add_argument(
        "--patterns",
        dest="patterns",
        default=["*png"],
        nargs="+",
        help="Pattern to match for inputs (assumes each pattern corresponds to\
            a modality).",
    )
    parser.add_argument(
        "--output_json",
        dest="output_json",
        required=True,
        help="Path for output JSON file",
    )
    parser.add_argument(
        "--n_workers",
        dest="n_workers",
        default=0,
        type=int,
        help="Number of parallel processes for Pool",
    )

    args = parser.parse_args(arguments)

    path = Path(args.input_path)

    image_dict = {}
    for pattern in args.patterns:
        all_dicoms = list(path.rglob(pattern))
        with tqdm(all_dicoms) as pbar:
            if args.n_workers in [0, 1]:
                for dcm in all_dicoms:
                    out = process_image(dcm)
                    if out is not None:
                        dcm, study_uid, series_uid = out
                        update_dict(image_dict, study_uid, series_uid, dcm)
                    pbar.update()
            else:
                pool = Pool(args.n_workers)
                for out in pool.imap(process_image, all_dicoms):
                    if out is not None:
                        dcm, study_uid, series_uid = out
                        update_dict(image_dict, study_uid, series_uid, dcm)
                    pbar.update()

    pretty_dict = json.dumps(image_dict, indent=2)
    Path(args.output_json).parent.mkdir(exist_ok=True)
    with open(args.output_json, "w") as o:
        o.write(pretty_dict + "\n")
