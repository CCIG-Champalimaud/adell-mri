import os
import json
import argparse
from multiprocessing import Pool
from pathlib import Path
from pydicom import dcmread
from tqdm import tqdm

def process_dicom(dcm):
    study_uid,series_uid,dcm_root = str(dcm).split(os.sep)[-3:]
    if study_uid not in dicom_dict:
        dicom_dict[study_uid] = {}
    if series_uid not in dicom_dict[study_uid]:
        dicom_dict[study_uid][series_uid] = []
    dcm = str(dcm)
    dcm_file = dcmread(dcm)
    # removes cases of segmentation 
    if dcm_file[0x0008,0x0060].value == "SEG":
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
    return dcm,study_uid,series_uid,orientation

def update_dict(d,study_uid,series_uid,orientation,dcm):
    if study_uid not in d:
        d[study_uid] = {}
    if series_uid not in d[study_uid]:
        d[study_uid][series_uid] = []
    d[study_uid][series_uid].append({
        "image":dcm,
        "orientation":orientation})
    return d

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
    parser.add_argument(
        "--n_workers",dest="n_workers",default=0,type=int,
        help="Number of parallel processes for Pool")
    
    args = parser.parse_args()

    path = Path(args.input_path)
        
    dicom_dict = {}
    for pattern in args.patterns:
        all_dicoms = list(path.rglob(pattern))
        with tqdm(all_dicoms) as pbar:
            if args.n_workers in [0,1]:
                for dcm in all_dicoms:
                    out = process_dicom(dcm)
                    if out is not None:
                        dcm,study_uid,series_uid,orientation = out
                        update_dict(
                            dicom_dict,study_uid,series_uid,orientation,dcm)
                    pbar.update()
            else:
                pool = Pool(args.n_workers)
                for out in pool.imap(process_dicom,all_dicoms):
                    if out is not None:
                        dcm,study_uid,series_uid,orientation = out
                        update_dict(
                            dicom_dict,study_uid,series_uid,orientation,dcm)
                    pbar.update()

    pretty_dict = json.dumps(dicom_dict,indent=2)
    Path(args.output_json).parent.mkdir(exist_ok=True)
    with open(args.output_json,'w') as o:
        o.write(pretty_dict+"\n")