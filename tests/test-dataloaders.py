import argparse
import random
import numpy as np
import torch
import monai
from sklearn.model_selection import train_test_split
from tqdm import tqdm

import time
import os
import sys
sys.path.append(os.path.join(os.path.dirname(os.path.abspath(__file__)), '..'))

import torch
from lib.modules.segmentation import *

from lib.utils import (
    get_prostatex_path_dictionary,
    get_size_spacing_dict,
    collate_last_slice,
    ConvertToOneHot,
    RandomSlices,
    SlicesToFirst)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    # data
    parser.add_argument(
        '--base_path',dest='base_path',type=str,
        help="Path to ProstateX dataset",required=True)
    parser.add_argument(
        '--mod',dest='mod',type=str,choices=["T2WAx","DWI"],
        help="Key to be used",required=True)
    parser.add_argument(
        '--classes',dest='classes',type=str,nargs='+',
        choices=["gland","lesion"],
        help="Classes to be considered",required=True)
    
    # training
    parser.add_argument(
        '--seed',dest='seed',help="Random seed",default=42,type=int)
    parser.add_argument(
        '--n_workers',dest='n_workers',
        help="No. of workers",default=1,type=int)
    parser.add_argument(
        '--augment',dest='augment',action="store_true",
        help="Use data augmentations",default=False)
    parser.add_argument(
        '--downsample_rate',dest='downsample_rate',type=float,
        help="Resizes images with downsample rate")
    parser.add_argument(
        '--batch_size',dest='batch_size',type=int)
    parser.add_argument(
        '--spatial_dim',dest='spatial_dim',type=int)

    args = parser.parse_args()

    torch.manual_seed(args.seed)
    random.seed(args.seed)
    np.random.seed(args.seed)
    g = torch.Generator()
    g.manual_seed(args.seed)

    args.classes = sorted(args.classes)
    if len(args.classes) == 1:
        n_classes = 2
    elif len(args.classes) == 2:
        n_classes = 3

    keys = [args.mod]
    label_keys = ["{}_{}_segmentations".format(args.mod,c)
                  for c in args.classes]
    all_keys = [args.mod,*label_keys]

    path_dictionary = get_prostatex_path_dictionary(args.base_path)
    fpd = {}
    for pid in path_dictionary:
        if all([k in path_dictionary[pid] for k in all_keys]):
            fpd[pid] = {k:path_dictionary[pid][k] for k in all_keys}
    path_dictionary = fpd
    size_dict,spacing_dict = get_size_spacing_dict(path_dictionary,[args.mod])

    for k in size_dict:
        mm = np.median(np.array(size_dict[k]),axis=0)
        # downsample only non-depth dimensions
        mm[0] = mm[0]*args.downsample_rate
        mm[1] = mm[1]*args.downsample_rate
        size_dict[k] = np.int32(np.round(mm))
        spacing_dict[k] = np.median(np.array(spacing_dict[k]),axis=0)
    
    print("Setting up transforms...")
    if args.augment == True:
        augments = [
            monai.transforms.RandBiasFieldd([args.mod]),
            monai.transforms.RandGibbsNoised([args.mod]),
            monai.transforms.RandAdjustContrastd([args.mod])]
    else:
        augments = []

    intp = ["area","nearest","nearest"][:n_classes]
    transforms_train = [
        monai.transforms.LoadImaged(all_keys),
        monai.transforms.AddChanneld(all_keys),
        monai.transforms.Orientationd(all_keys,"RAS"),
        monai.transforms.EnsureTyped(all_keys),
        # in case images should be downsampled
        monai.transforms.Resized(
            all_keys,tuple(size_dict[args.mod]),mode=intp),
        *augments,
        monai.transforms.ScaleIntensityd([args.mod],0,1),
        monai.transforms.EnsureTyped(all_keys)]

    transforms_val = [
        monai.transforms.LoadImaged(all_keys),
        monai.transforms.AddChanneld(all_keys),
        monai.transforms.Orientationd(all_keys,"RAS"),
        monai.transforms.EnsureTyped(all_keys),
        monai.transforms.Resized(
            all_keys,tuple(size_dict[args.mod]),mode=intp),
        monai.transforms.ScaleIntensityd([args.mod],0,1),
        monai.transforms.EnsureTyped(all_keys)]

    if n_classes > 2:
        lesion_key = "{}_{}_segmentations".format(args.mod,"lesion")
        label_key = "label"
        transforms_train.append(
            ConvertToOneHot(label_keys,label_key,lesion_key,bg=True))
        transforms_val.append(
            ConvertToOneHot(label_keys,label_key,lesion_key,bg=True))
    else:
        label_key = "{}_{}_segmentations".format(args.mod,args.classes[0])

    if args.spatial_dim == 2:
        transforms_train.append(
            RandomSlices([args.mod,label_key],label_key,4,base=0.01))
        transforms_val.append(
            SlicesToFirst([args.mod,label_key]))
        collate_fn = collate_last_slice
    else:
        collate_fn = None

    all_pids = [k for k in path_dictionary]

    train_idxs,val_idxs = train_test_split(range(len(all_pids)),test_size=0.2)
    train_pids = [all_pids[i] for i in train_idxs]
    val_pids = [all_pids[i] for i in val_idxs]
    path_list_train = [path_dictionary[pid] for pid in train_pids]
    path_list_val = [path_dictionary[pid] for pid in val_pids]

    train_dataset = monai.data.CacheDataset(
        path_list_train,monai.transforms.Compose(transforms_train))
    train_dataset_val = monai.data.CacheDataset(
        path_list_train,monai.transforms.Compose(transforms_val))
    validation_dataset = monai.data.CacheDataset(
        path_list_val,monai.transforms.Compose(transforms_val))

    train_loader_call = lambda: torch.utils.data.DataLoader(
        train_dataset,batch_size=args.batch_size,
        shuffle=True,num_workers=args.n_workers,generator=g,
        collate_fn=collate_fn)

    train_loader = train_loader_call()
    train_val_loader = torch.utils.data.DataLoader(
        train_dataset_val,batch_size=1,
        shuffle=False,num_workers=args.n_workers,generator=g,
        collate_fn=collate_fn)
    validation_loader = torch.utils.data.DataLoader(
        validation_dataset,batch_size=1,
        shuffle=False,num_workers=args.n_workers,generator=g,
        collate_fn=collate_fn)

    it = iter(train_loader)
    times = []
    go = True
    u = tqdm()
    while go == True:
        a = time.time()
        try: next(it)
        except Exception as e: 
            go = False
            print(e)
        b = time.time()
        times.append(b-a)
        u.update()
    u.close()

    print("Average time: {} ({} iterations)".format(np.mean(times),len(times)))
    print("Average time: {} ({} iterations, without first iteration)".format(np.mean(times[1:]),len(times)-1))
    print("Min={}, Q25%={},Median={},Q75%={},Max={}".format(
        *np.quantile(times,[0,0.25,0.5,0.75,1])))