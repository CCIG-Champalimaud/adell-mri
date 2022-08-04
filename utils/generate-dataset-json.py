import os
import json
import re
import argparse
import numpy as np
import monai
from skimage import measure
from glob import glob
from tqdm import tqdm

def value_range(x):
    return x.min(),x.max()

def mask_to_bb(img:np.ndarray)->list:
    img = np.round(img)
    labelled_image = measure.label(img,connectivity=3)
    uniq = np.unique(labelled_image)
    uniq = uniq[uniq != 0]
    
    bb_vertices = []
    c = []
    for u in uniq:
        C = np.where(labelled_image == u)
        bb = np.array([value_range(c) for c in C])
        if np.all(bb[:,1] == bb[:,0]) == False:
            bb_vertices.append(bb)
            c.append(np.median(img[C]))

    return bb_vertices,c

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Creates JSON file with paths and bounding boxes.")
    parser.add_argument(
        '--input_path',dest="input_path",required=True,
        help="Path to folder containing nibabel compatible files")
    parser.add_argument(
        '--mask_path',dest="mask_path",required=True,
        help="Path to folder containing nibabel compatible masks")
    parser.add_argument(
        '--mask_key',dest="mask_key",default="mask",
        help="Custom mask for the key. Helpful if later merging json files")
    parser.add_argument(
        '--class_csv_path',dest="class_csv_path",default=None,
        help="Path to CSV with classes. Assumes first column is study ID and \
            last column is class.")
    parser.add_argument(
        '--patterns',dest="patterns",default="*nii.gz",nargs="+",
        help="Pattern to match for inputs (assumes each pattern corresponds to\
            a modality).")
    parser.add_argument(
        '--mask_pattern',dest="mask_pattern",default="*nii.gz",
        help="Pattern to match  for mask")
    parser.add_argument(
        '--id_pattern',dest="id_pattern",default="*",type=str,
        help="Pattern to extract IDs from image files")
    parser.add_argument(
        '--output_json',dest="output_json",required=True,
        help="Path for output JSON file")
    
    args = parser.parse_args()

    t = monai.transforms.Compose([
        monai.transforms.LoadImaged(['image']),
        monai.transforms.AddChanneld(['image']),
        monai.transforms.Orientationd(['image'],"RAS")])

    bb_dict = {}
    all_paths = {
        p:glob(os.path.join(args.input_path,p))
        for p in args.patterns}
    class_dict_csv = {}
    if args.class_csv_path:
        with open(args.class_csv_path,'r') as o:
            for l in o:
                l = l.strip().split(',')
                identifier,cl = l[0],l[-1]
                class_dict_csv[identifier] = cl
    mask_paths = glob(os.path.join(args.mask_path,args.mask_pattern))
    for file_path in tqdm(all_paths[args.patterns[0]]):
        image_id = file_path.split(os.sep)[-1]
        image_id = re.match(args.id_pattern,image_id).group()
        alt_file_paths = []
        for k in args.patterns[1:]:
            paths = all_paths[k]
            for alt_path in paths:
                if image_id in alt_path:
                    alt_file_paths.append(alt_path)
        mask_path = [p for p in mask_paths if image_id in p]
        if len(mask_path) > 0:
            mask_path = mask_path[0]
            bb_dict[image_id] = {
                "image":file_path,
                args.mask_key:mask_path}
            if len(alt_file_paths) > 0:
                for i,p in enumerate(alt_file_paths):
                    bb_dict[image_id]["image_"+str(i+1)] = p
            bb_dict[image_id]["boxes"] = []
            bb_dict[image_id]["shape"] = ""
            bb_dict[image_id]["labels"] = []
            fdata = t({'image':mask_path})['image'][0]
            sh = np.array(fdata.shape)
            unique_labels = []
            for bb,c in zip(*mask_to_bb(fdata)):
                c = int(c)
                bb = [int(x) for x in bb.flatten("F")]
                bb_dict[image_id]["boxes"].append(bb)
                bb_dict[image_id]["labels"].append(c)
                if c not in unique_labels:
                    unique_labels.append(c)
            bb_dict[image_id]["shape"] = [int(x) for x in sh]
            if len(unique_labels) == 0:
                unique_labels = [0]
            bb_dict[image_id]["image_labels"] = unique_labels
        elif image_id in class_dict_csv:
            bb_dict[image_id] = {
                "image":file_path,
                "image_labels":[int(class_dict_csv[image_id])]}
            if len(alt_file_paths) > 0:
                for i,p in enumerate(alt_file_paths):
                    bb_dict[image_id]["image_"+str(i+1)] = p

    pretty_dict = json.dumps(bb_dict,indent=2)
    with open(args.output_json,'w') as o:
        o.write(pretty_dict+"\n")