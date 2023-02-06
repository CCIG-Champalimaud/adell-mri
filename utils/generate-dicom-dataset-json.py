import os
import json
import argparse
from pathlib import Path
from pydicom import dcmread
from tqdm import tqdm

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Creates JSON file with DICOM paths.")
    parser.add_argument(
        '--input_path',dest="input_path",required=True,
        help="Path to folder containing nibabel compatible files")
    parser.add_argument(
        '--patterns',dest="patterns",default=["*dcm"],nargs="+",
        help="Pattern to match for inputs (assumes each pattern corresponds to\
            a modality).")
    parser.add_argument(
        '--output_json',dest="output_json",required=True,
        help="Path for output JSON file")
    
    args = parser.parse_args()

    path = Path(args.input_path)
        
    dicom_dict = {}
    for pattern in args.patterns:
        all_dicoms = list(path.rglob(pattern))
        for dcm in tqdm(all_dicoms):
            study_uid,series_uid,dcm_root = str(dcm).split(os.sep)[-3:]
            if study_uid not in dicom_dict:
                dicom_dict[study_uid] = {}
            if series_uid not in dicom_dict[study_uid]:
                dicom_dict[study_uid][series_uid] = []
            dcm = str(dcm)
            dcm_file = dcmread(dcm)
            # removes cases of segmentation 
            if dcm_file[0x0008,0x0060].value == "SEG":
                continue
            # signals poorly specified orientation
            if (0x0020, 0x0037) in dcm_file:
                orientation = dcm_file[0x0020, 0x0037].value
                orientation = [x for x in orientation]
            else:
                orientation = None
            dicom_dict[study_uid][series_uid].append({
                "image":dcm,
                "orientation":orientation})

    pretty_dict = json.dumps(dicom_dict,indent=2)
    Path(args.output_json).parent.mkdir(exist_ok=True)
    with open(args.output_json,'w') as o:
        o.write(pretty_dict+"\n")