import os
import json
import argparse
from multiprocessing import Pool
from pathlib import Path
from pydicom import dcmread
from pydicom import errors
from tqdm import tqdm

DICOMInformation = tuple[str, str, str, tuple[float, float, float]]
DICOMDictionary = dict[
    str, dict[str, dict[str, str | tuple[float, float, float]]]
]

desc = "Creates JSON file with DICOM paths."


def process_dicom(dcm: str) -> DICOMInformation:
    # dicom_dict defined outside of function scope
    study_uid, series_uid, dcm_root = str(dcm).split(os.sep)[-3:]
    dcm = str(dcm)
    try:
        dcm_file = dcmread(dcm)
    except errors.InvalidDicomError:
        return None
    # removes cases of segmentation
    if dcm_file[0x0008, 0x0060].value == "SEG":
        return None
    # in some cases, pixel array data is corrupted, which causes
    # failures when accessing pixel_array; this makes the dataset
    # construction skip these files
    try:
        dcm_file.pixel_array
    except:
        return None
    # signals poorly specified orientation
    if (0x0020, 0x0037) in dcm_file:
        orientation = dcm_file[0x0020, 0x0037].value
        orientation = [x for x in orientation]
    else:
        orientation = None
    return dcm, study_uid, series_uid, orientation


def process_dicom_directory(series_dir: str) -> list[DICOMInformation]:
    return [process_dicom(str(x)) for x in Path(series_dir).rglob("*dcm")]


def update_dict(
    d: DICOMDictionary,
    study_uid: str,
    series_uid: str,
    orientation: tuple[float, float, float],
    dcm: str,
    included_ids: list[str] = [],
) -> DICOMDictionary:
    add = False
    if included_ids is not None:
        if study_uid in included_ids:
            add = True
    else:
        add = True

    if add == True:
        if study_uid not in d:
            d[study_uid] = {}
        if series_uid not in d[study_uid]:
            d[study_uid][series_uid] = []
        d[study_uid][series_uid].append(
            {"image": dcm, "orientation": orientation}
        )
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
        "--included_ids",
        dest="included_ids",
        default=None,
        help="Study UIDs to be excluded",
    )
    parser.add_argument(
        "--patterns",
        dest="patterns",
        default=["*/"],
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

    if args.included_ids is not None:
        with open(args.included_ids) as o:
            included_ids = {x.strip(): "" for x in o.readlines()}
    else:
        included_ids = None

    dicom_dict = {}
    for pattern in args.patterns:
        print("Locating all studies/series...")
        all_dicoms = [str(x) for x in path.glob(pattern)]
        print("Iterating studies/series...")
        with tqdm(all_dicoms) as pbar:
            if args.n_workers in [0, 1]:
                for dcm in all_dicoms:
                    outs = process_dicom_directory(dcm)
                    for out in outs:
                        if out is not None:
                            dcm, study_uid, series_uid, orientation = out
                            update_dict(
                                dicom_dict,
                                study_uid,
                                series_uid,
                                orientation,
                                dcm,
                                included_ids,
                            )
                    pbar.update()
            else:
                pool = Pool(args.n_workers)
                for outs in pool.imap(process_dicom_directory, all_dicoms):
                    for out in outs:
                        if out is not None:
                            dcm, study_uid, series_uid, orientation = out
                            update_dict(
                                dicom_dict,
                                study_uid,
                                series_uid,
                                orientation,
                                dcm,
                                included_ids,
                            )
                    pbar.update()

    pretty_dict = json.dumps(dicom_dict, indent=2)
    Path(args.output_json).parent.mkdir(exist_ok=True)
    with open(args.output_json, "w") as o:
        o.write(pretty_dict + "\n")
