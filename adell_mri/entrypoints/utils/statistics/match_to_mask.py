import argparse
import os
import nibabel as nib
import SimpleITK as sitk
import re
import numpy as np
from glob import glob
from tqdm import tqdm

desc = "Script with very specific utility - given a set of MRI sequences and a \
    set of masks, determine which sequence is the most likely to have been used \
    as a  template for the mask. Assumes that the size/spacing of the mask \
    will be  similar to that of the input."


def main(arguments):
    parser = argparse.ArgumentParser(description=desc)

    parser.add_argument(
        "--input_patterns",
        dest="input_patterns",
        nargs="+",
        type=str,
        help="Patterns for glob yielding files.",
        required=True,
    )
    parser.add_argument(
        "--pattern_id",
        dest="pattern_id",
        type=str,
        default="[A-Za-z0-9]+_[A-Za-z0-9]+",
        help="Regex pattern for the patient id.",
    )
    parser.add_argument(
        "--pattern_sequence",
        dest="pattern_sequence",
        type=str,
        default="[A-Za-z0-9]+_[A-Za-z0-9]+",
        help="Regex pattern for the sequence.",
    )
    parser.add_argument(
        "--mask_pattern",
        dest="mask_pattern",
        type=str,
        help="Pattern to folder containing labels",
    )

    args = parser.parse_args(arguments)

    path_dictionary = {}
    all_sequences = []
    for input_pattern in args.input_patterns:
        for path in glob(input_pattern):
            pid = path.split(os.sep)[-1]
            sequence = re.search(args.pattern_sequence, pid).group()
            pid = re.search(args.pattern_id, pid).group()
            if pid in path_dictionary:
                path_dictionary[pid][sequence] = path
            else:
                path_dictionary[pid] = {sequence: path}
            if sequence not in all_sequences:
                all_sequences.append(sequence)

    for path in glob(args.mask_pattern):
        pid = path.split(os.sep)[-1]
        sequence = re.search(args.pattern_sequence, pid).group()
        pid = re.search(args.pattern_id, pid).group()
        if pid in path_dictionary:
            path_dictionary[pid]["mask"] = path

    t = len(all_sequences) + 1
    matches = {}
    has_non_zero = {}
    good_pids = []
    for pid in path_dictionary:
        if "mask" in path_dictionary[pid]:
            good_pids.append(pid)
    for pid in tqdm(good_pids):
        match = []
        paths = path_dictionary[pid]
        mask = nib.load(paths["mask"])
        fdata = mask.get_fdata()
        mask_has_nonzero = np.count_nonzero(fdata) > 0
        shapes_cur = {}
        space_cur = {}
        for seq_id in all_sequences:
            seq = sitk.ReadImage(paths[seq_id])
            shapes_cur[seq_id] = seq.GetSize()
            space_cur[seq_id] = seq.GetSpacing()
        for seq_id in shapes_cur:
            seq_shape = shapes_cur[seq_id]
            if np.all(seq_shape == mask.shape):
                match.append(seq_id)

        match = "_".join(sorted(match))
        if match in matches:
            matches[match].append(pid)
            has_non_zero[match].append(mask_has_nonzero)
        else:
            matches[match] = [pid]
            has_non_zero[match] = [mask_has_nonzero]

    for k in matches:
        print(k, len(matches[k]), np.sum(has_non_zero[k]))
