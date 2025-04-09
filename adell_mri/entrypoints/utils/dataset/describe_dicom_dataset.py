import os
from pydicom import dcmread
from pydicom.multival import MultiValue
from pydicom.valuerep import IS, DSfloat

desc = "Describes DICOM datasets with general statistics."

TAGS_TO_RETRIEVE = {
    (0x0018, 0x0024): "Sequence Name",
    (0x0018, 0x0080): "Repetition Time",
    (0x0018, 0x0081): "Echo Time",
    (0x0018, 0x0087): "Magnetic Field Strength",
    (0x0018, 0x1020): "Software Versions",
    (0x0018, 0x1030): "Protocol Name",
    (0x0018, 0x1041): "Contrast/Bolus Volume",
    (0x0018, 0x1044): "Contrast/Bolus Total Dose",
    (0x0018, 0x1048): "Contrast/Bolus Ingredient",
    (0x0018, 0x1049): "Contrast/Bolus Ingredient Concentra",
    (0x0018, 0x1251): "Transmit Coil Name",
    (0x0028, 0x0030): "Pixel Spacing",
    (0x0008, 0x103E): "Series Description",
    (0x0008, 0x0070): "Manufacturer",
    (0x0008, 0x1090): "Manufacturer's Model Name",
    (0x0008, 0x0060): "Modality",
    (0x0018, 0x0050): "Slice Thickness",
    (0x0008, 0x0030): "Study Time",
    (0x0008, 0x0031): "Series Time",
    (0x0008, 0x0032): "Acquisition Time",
    (0x0008, 0x0033): "Content Time",
    (0x0008, 0x0008): "Image Type",
    (0x0020, 0x0013): "Instance Number",
    (0x0020, 0x000D): "Study Instance UID",
    (0x0020, 0x000E): "Series Instance UID",
}

ADDITIONAL_FIELDS = ["file_name", "dir_name"]


def format_value_for_csv(
    v: list | tuple | str | float | int | DSfloat | IS,
) -> str:
    if isinstance(v, (list, tuple)):
        v = "_".join(v)
    elif isinstance(v, (DSfloat, IS, float, int)):
        v = str(v)
    elif v is None:
        v = str(None)
    return v


def read_retrieve_dicom(dcm_file: str) -> tuple[dict, str, str]:
    dcm = dcmread(
        dcm_file,
        stop_before_pixels=True,
        specific_tags=TAGS_TO_RETRIEVE.keys(),
    )
    study_uid, series_uid = dcm_file.split(os.sep)[-3:-1]
    dcm_dict = {}
    for tag, name in TAGS_TO_RETRIEVE.items():
        v = dcm.get(tag, None)
        if v is not None:
            v = v.value
            if isinstance(v, MultiValue):
                v = [str(x) for x in v]
        dcm_dict[name] = v
    dcm_dict["file_name"] = os.path.basename(dcm_file)
    dcm_dict["dir_name"] = os.path.dirname(dcm_file)
    return dcm_dict, study_uid, series_uid


def add_to_dict(d: dict, study_uid: str, series_uid: str, value: dict) -> dict:
    if study_uid not in d:
        d[study_uid] = {}
    if series_uid not in d[study_uid]:
        d[study_uid][series_uid] = []
    d[study_uid][series_uid].append(value)


def main(arguments):
    import argparse
    import json
    from itertools import chain
    from multiprocessing import Pool
    from pathlib import Path
    from tqdm import tqdm
    
    parser = argparse.ArgumentParser(description=desc)

    parser.add_argument(
        "--path",
        required=True,
        help="Path to DICOM dataset.",
        type=str,
    )
    parser.add_argument(
        "--n_workers",
        default=0,
        help="Number of processes (concurrent workers)",
        type=int,
    )
    parser.add_argument(
        "--output",
        default=None,
        help="Path to output file.",
        type=str,
    )
    parser.add_argument(
        "--output_format",
        default="json",
        help="Output format.",
        type=str,
        choices=["json", "csv"],
    )

    args = parser.parse_args(arguments)

    all_dcm_paths = [str(x) for x in Path(args.path).rglob("*dcm")]

    organized_dataset = {}

    if args.n_workers > 1:
        with Pool(args.n_workers) as p:
            iterator = p.imap(read_retrieve_dicom, all_dcm_paths)
            for out, st_id, se_id in tqdm(iterator, total=len(all_dcm_paths)):
                add_to_dict(
                    organized_dataset,
                    study_uid=st_id,
                    series_uid=se_id,
                    value=out,
                )

    else:
        iterator = map(read_retrieve_dicom, all_dcm_paths)
        for out, st_id, se_id in tqdm(iterator, total=len(all_dcm_paths)):
            add_to_dict(
                organized_dataset, study_uid=st_id, series_uid=se_id, value=out
            )

    suv = {}
    for study_uid in organized_dataset:
        suv[study_uid] = {}
        for series_uid in organized_dataset[study_uid]:
            suv[study_uid][series_uid] = {
                k: []
                for k in chain(TAGS_TO_RETRIEVE.values(), ADDITIONAL_FIELDS)
            }
            for instance in organized_dataset[study_uid][series_uid]:
                for k in instance:
                    if instance[k] not in suv[study_uid][series_uid][k]:
                        v = instance[k]
                        suv[study_uid][series_uid][k].append(v)
            suv[study_uid][series_uid]["n_images"] = [
                len(organized_dataset[study_uid][series_uid])
            ]

    if args.output_format == "json":
        out = json.dumps(suv, indent=4)

    elif args.output_format == "csv":
        header = list(TAGS_TO_RETRIEVE.values())
        header = ["study_uid", "series_uid", *header]
        out = ",".join(header) + "\n"
        for study_uid in suv:
            for series_uid in suv[study_uid]:
                out += f"{study_uid},{series_uid}"
                unique_values = suv[study_uid][series_uid]
                for k in unique_values:
                    out += "," + "|".join(
                        [format_value_for_csv(x) for x in unique_values[k]]
                    )
                out += "\n"

    if args.output is not None:
        with open(args.output, "w") as f:
            f.write(out)
    else:
        print(out)
