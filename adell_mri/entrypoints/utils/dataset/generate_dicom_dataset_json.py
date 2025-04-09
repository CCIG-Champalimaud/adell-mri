import os
from multiprocessing import Pool
from pathlib import Path

from pydicom import dcmread, errors
from pydicom.dataset import FileDataset
from pydicom.multival import MultiValue

DICOMInformation = tuple[str, str, str, tuple[float, float, float]]
DICOMDictionary = dict[
    str, dict[str, dict[str, str | tuple[float, float, float]]]
]

desc = "Creates JSON file with DICOM paths."

SEG_SOP_TAG = (0x0008, 0x0060)
BVALUE_TAG = (0x0018, 0x9087)
SIEMENS_BVALUE_TAG = (0x0019, 0x100C)
GE_BVALUE_TAG = (0x0043, 0x1039)
PATIENT_POSITION_TAG = (0x0020, 0x0032)
PATIENT_ORIENTATION_TAG = (0x0020, 0x0037)
SERIES_DESCRIPTION_TAG = (0x0008, 0x103E)


def retrieve_bvalue(f: FileDataset) -> float | None:
    bval = None
    if BVALUE_TAG in f:
        bval = f[BVALUE_TAG].value
        if isinstance(bval, bytes):
            bval = int.from_bytes(bval, byteorder="big")
            if bval > 5000:
                bval = f[BVALUE_TAG].value[0]
    elif SIEMENS_BVALUE_TAG in f:
        bval = f[SIEMENS_BVALUE_TAG].value
    elif GE_BVALUE_TAG in f:
        bval = f[GE_BVALUE_TAG].value
        if isinstance(bval, bytes):
            bval = bval.decode().split("\\")[0]
        elif isinstance(bval, MultiValue):
            bval = bval[0]
        if len(str(bval)) > 5:
            bval = str(bval)[-4:].lstrip("0")
    if bval is None:
        return None
    bval = float(bval)
    return bval




def process_dicom(dcm: str) -> DICOMInformation:
    def get_float_list(f: FileDataset, tag: tuple[int, int]) -> list[float]:
        if tag in f:
            value = f[tag].value
            value = [float(x) for x in value]
        else:
            value = None
        return value

    # dicom_dict defined outside of function scope
    study_uid, series_uid, dcm_root = str(dcm).split(os.sep)[-3:]
    dcm = str(dcm)
    try:
        dcm_file = dcmread(dcm)
    except errors.InvalidDicomError:
        return None
    # removes cases of segmentation
    if dcm_file[SEG_SOP_TAG].value == "SEG":
        return None
    # in some cases, pixel array data is corrupted, which causes
    # failures when accessing pixel_array; this makes the dataset
    # construction skip these files
    try:
        dcm_file.pixel_array
    except ValueError:
        return None
    # signals poorly specified orientation
    orientation = get_float_list(dcm_file, PATIENT_ORIENTATION_TAG)
    position = get_float_list(dcm_file, PATIENT_POSITION_TAG)
    sd = (
        dcm_file[SERIES_DESCRIPTION_TAG].value
        if SERIES_DESCRIPTION_TAG in dcm_file
        else None
    )

    metadata = {
        "orientation": orientation,
        "bvalue": retrieve_bvalue(dcm_file),
        "series_description": sd,
        "patient_position": position,
    }
    return dcm, study_uid, series_uid, metadata


def process_dicom_directory(series_dir: str) -> list[DICOMInformation]:
    return [process_dicom(str(x)) for x in Path(series_dir).rglob("*dcm")]


def update_dict(
    d: DICOMDictionary,
    study_uid: str,
    series_uid: str,
    metadata: tuple[float, float, float],
    dcm: str,
    included_ids: list[str] = [],
    max_bvalue: int = None,
) -> DICOMDictionary:
    add = False
    if included_ids is not None:
        if study_uid in included_ids:
            add = True
    else:
        add = True

    if max_bvalue is not None:
        if metadata["bvalue"] > max_bvalue:
            return d

    if add is True:
        if study_uid not in d:
            d[study_uid] = {}
        if series_uid not in d[study_uid]:
            d[study_uid][series_uid] = []
        d[study_uid][series_uid].append({"image": dcm, **metadata})
    return d


def main(arguments):
    import argparse
    import json
    from multiprocessing import Pool
    from pathlib import Path

    from tqdm import tqdm

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
    parser.add_argument(
        "--max_bvalue",
        type=int,
        default=None,
        help="Maximum bvalue. Helps exclude some aberrant b-values",
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
            if args.n_workers < 2:
                for dcm in pbar:
                    outs = process_dicom_directory(dcm)
                    for out in outs:
                        if out is not None:
                            dcm, study_uid, series_uid, metadata = out
                            update_dict(
                                dicom_dict,
                                study_uid,
                                series_uid,
                                metadata,
                                dcm,
                                included_ids,
                                max_bvalue=args.max_bvalue,
                            )
                    pbar.update()
            else:
                with Pool(args.n_workers) as p:
                    for outs in p.imap(process_dicom_directory, pbar):
                        for out in outs:
                            if out is not None:
                                dcm, study_uid, series_uid, metadata = out
                                update_dict(
                                    dicom_dict,
                                    study_uid,
                                    series_uid,
                                    metadata,
                                    dcm,
                                    included_ids,
                                    max_bvalue=args.max_bvalue,
                                )
                        pbar.update()

    pretty_dict = json.dumps(dicom_dict, indent=2)
    Path(args.output_json).parent.mkdir(exist_ok=True)
    with open(args.output_json, "w") as o:
        o.write(pretty_dict + "\n")
