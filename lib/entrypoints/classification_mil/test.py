import os
import argparse
import random
import json
import numpy as np
import torch
import monai
from pathlib import Path

from lightning.pytorch import Trainer
import sys
from ...utils import (
    safe_collate,ScaleIntensityAlongDimd,EinopsRearranged)
from ...utils.pl_utils import get_devices
from ...utils.dataset_filters import (
    filter_dictionary_with_filters,
    filter_dictionary_with_possible_labels,
    filter_dictionary_with_presence)
from ...monai_transforms import get_transforms_classification as get_transforms
from ...modules.classification.pl import (
    TransformableTransformerPL,MultipleInstanceClassifierPL)
from ...modules.config_parsing import parse_config_2d_classifier_3d
from ...utils.parser import parse_ids
from ...utils.parser import get_params,merge_args

def main(arguments):
    parser = argparse.ArgumentParser()

    # params
    parser.add_argument(
        '--params_from',dest='params_from',type=str,default=None,
        help="Parameter path used to retrieve values for the CLI (can be a path\
            to a YAML file or 'dvc' to retrieve dvc params)")

    # data
    parser.add_argument(
        '--dataset_json',dest='dataset_json',type=str,
        help="JSON containing dataset information",required=True)
    parser.add_argument(
        '--metric_path',dest='metric_path',type=str,required=True,
        help='Path to metrics file.')
    parser.add_argument(
        '--test_ids',dest='test_ids',type=str,nargs="+",
        help="List of IDs for testing",required=True)
    parser.add_argument(
        '--excluded_ids',dest='excluded_ids',type=str,nargs="+",
        help="List of IDs to exclude",default=None)
    parser.add_argument(
        '--one_to_one',dest="one_to_one",action="store_true",
        help="Tests the checkpoint only on the corresponding test_ids set")
    parser.add_argument(
        '--image_keys',dest='image_keys',type=str,nargs='+',
        help="Image keys in the dataset JSON.",
        required=True)
    parser.add_argument(
        '--t2_keys',dest='t2_keys',type=str,nargs='+',
        help="Image keys corresponding to T2.",default=None)
    parser.add_argument(
        '--label_keys',dest='label_keys',type=str,default="image_labels",
        help="Label keys in the dataset JSON.")
    parser.add_argument(
        '--filter_on_keys',dest='filter_on_keys',type=str,default=[],nargs="+",
        help="Filters the dataset based on a set of specific key:value pairs.")
    parser.add_argument(
        '--possible_labels',dest='possible_labels',type=str,nargs='+',
        help="All the possible labels in the data.",
        required=True)
    parser.add_argument(
        '--positive_labels',dest='positive_labels',type=str,nargs='+',
        help="Labels that should be considered positive (binarizes labels)",
        default=None)
    parser.add_argument(
        '--cache_rate',dest='cache_rate',type=float,
        help="Rate of samples to be cached",
        default=1.0)
    parser.add_argument(
        '--target_spacing',dest='target_spacing',action="store",default=None,
        help="Resamples all images to target spacing",nargs='+',type=float)
    parser.add_argument(
        '--target_size',dest='target_size',action="store",default=None,
        help="Resizes all images to target size",nargs='+',type=int)
    parser.add_argument(
        '--pad_size',dest='pad_size',action="store",
        default=None,type=float,nargs='+',
        help="Size of central padded image after resizing (if none is specified\
            then no padding is performed).")
    parser.add_argument(
        '--crop_size',dest='crop_size',action="store",
        default=None,type=float,nargs='+',
        help="Size of central crop after resizing (if none is specified then\
            no cropping is performed).")
    parser.add_argument(
        '--subsample_size',dest='subsample_size',type=int,
        help="Subsamples data to a given size",
        default=None)
    parser.add_argument(
        '--batch_size',dest='batch_size',type=int,default=None,
        help="Overrides batch size in config file")

    # network + training
    parser.add_argument(
        '--config_file',dest="config_file",
        help="Path to network configuration file (yaml)",
        required=True)
    parser.add_argument(
        '--mil_method',dest="mil_method",
        help="Multiple instance learning method name.",
        choices=["standard","transformer"],required=True)
    parser.add_argument(
        '--module_path',dest="module_path",
        help="Path to torchscript module",
        required=True)
    
    # training
    parser.add_argument(
        '--dev',dest='dev',default="cpu",
        help="Device for PyTorch training",type=str)
    parser.add_argument(
        '--seed',dest='seed',help="Random seed",default=42,type=int)
    parser.add_argument(
        '--n_workers',dest='n_workers',
        help="No. of workers",default=0,type=int)
    parser.add_argument(
        '--checkpoints',dest='checkpoints',type=str,default=None,
        nargs="+",help='List of checkpoints for testing.')

    args = parser.parse_args(arguments)

    if args.params_from is not None:
        param_dict = get_params(args.params_from)
        args = merge_args(args,param_dict,sys.argv[1:])

    torch.manual_seed(args.seed)
    random.seed(args.seed)
    np.random.seed(args.seed)
    g = torch.Generator()
    g.manual_seed(args.seed)
    rng = np.random.default_rng(args.seed)

    accelerator,devices,strategy = get_devices(args.dev)
    
    data_dict = json.load(open(args.dataset_json,'r'))
    all_test_pids = parse_ids(args.test_ids)
    if args.excluded_ids is not None:
        excluded_ids = parse_ids(args.excluded_ids,output_format="list")
        a = len(data_dict)
        data_dict = {k:data_dict[k] for k in data_dict
                     if k not in excluded_ids}
        print("Excluded {} cases with --excluded_ids".format(a - len(data_dict)))
    data_dict = filter_dictionary_with_possible_labels(
        data_dict,args.possible_labels,args.label_keys)
    if len(args.filter_on_keys) > 0:
        data_dict = filter_dictionary_with_filters(
            data_dict,args.filter_on_keys)
    data_dict = filter_dictionary_with_presence(
        data_dict,args.image_keys + [args.label_keys])
    if args.subsample_size is not None:
        strata = {}
        for k in data_dict:
            label = data_dict[k][args.label_keys]
            if label not in strata:
                strata[label] = []
            strata[label].append(k)
        p = [len(strata[k]) / len(data_dict) for k in strata]
        split = rng.multinomial(args.subsample_size,p)
        ss = []
        for k,s in zip(strata,split):
            ss.extend(rng.choice(strata[k],size=s,replace=False,shuffle=False))
        data_dict = {k:data_dict[k] for k in ss}
    all_classes = []
    for k in data_dict:
        C = data_dict[k][args.label_keys]
        if isinstance(C,list):
            C = max(C)
        all_classes.append(str(C))
    if args.positive_labels is None:
        n_classes = len(args.possible_labels)
    else:
        n_classes = 2

    if len(data_dict) == 0:
        raise Exception(
            "No data available for training \
                (dataset={}; keys={}; labels={})".format(
                    args.dataset_json,
                    args.image_keys,
                    args.label_keys()))
    
    keys = args.image_keys
    t2_keys = args.t2_keys if args.t2_keys is not None else []
    adc_keys = []
    t2_keys = [k for k in t2_keys if k in keys]

    network_config,_ = parse_config_2d_classifier_3d(
        args.config_file,0.0)
    
    if n_classes == 2:
        network_config["loss_fn"] = torch.nn.BCEWithLogitsLoss(
            torch.ones([]))
    else:
        network_config["loss_fn"] = torch.nn.CrossEntropy(
            torch.ones([n_classes]))
    
    if args.batch_size is not None:
        network_config["batch_size"] = args.batch_size

    if "batch_size" not in network_config:
        network_config["batch_size"] = 1
    
    all_pids = [k for k in data_dict]

    print("Setting up transforms...")
    label_mode = "binary" if n_classes == 2 else "cat"
    transform_arguments = {
        "keys":keys,
        "adc_keys":adc_keys,
        "target_spacing":args.target_spacing,
        "target_size":args.target_size,
        "crop_size":args.crop_size,
        "pad_size":args.pad_size,
        "possible_labels":args.possible_labels,
        "positive_labels":args.positive_labels,
        "label_key":args.label_keys,
        "clinical_feature_keys":[],
        "label_mode":label_mode}

    transforms = monai.transforms.Compose([
        *get_transforms("pre",**transform_arguments),
        *get_transforms("post",**transform_arguments),
        EinopsRearranged("image","c h w d -> 1 h w (d c)"),
        ScaleIntensityAlongDimd("image",dim=-1)])
    
    all_metrics = []
    for iteration,test_pids in enumerate(all_test_pids):
        test_list = [data_dict[pid] for pid in test_pids
                     if pid in data_dict]
        test_dataset = monai.data.CacheDataset(
            test_list,transforms,
            cache_rate=args.cache_rate,
            num_workers=args.n_workers)
        
        # PL sometimes needs a little hint to detect GPUs.
        torch.ones([1]).to("cuda" if "cuda" in args.dev else "cpu")

        test_loader = monai.data.ThreadDataLoader(
            test_dataset,batch_size=network_config["batch_size"],
            shuffle=False,num_workers=args.n_workers,
            collate_fn=safe_collate)

        if args.one_to_one is True:
            checkpoint_list = [args.checkpoints[iteration]]
        else:
            checkpoint_list = args.checkpoints
        for checkpoint in checkpoint_list:
            n_slices = int(len(keys) * args.crop_size[-1])
            boilerplate_args = {
                "n_classes":n_classes,
                "training_dataloader_call":None,
                "image_key":"image",
                "label_key":"label",
                "n_epochs":0,
                "warmup_steps":0,
                "training_batch_preproc":None,
                "start_decay":0,
                "n_slices":n_slices}

            network_config["module"] = torch.jit.load(args.module_path).to(args.dev)
            network_config["module"].requires_grad = False
            network_config["module"] = network_config["module"].eval()
            network_config["module"] = torch.jit.freeze(network_config["module"])
            if "module_out_dim" not in network_config:
                print("2D module output size not specified, inferring...")
                input_example = torch.rand(
                    1,1,*[int(x) for x in args.crop_size][:2]).to(
                        args.dev.split(":")[0])
                output = network_config["module"](input_example)
                network_config["module_out_dim"] = int(output.shape[1])
                print("2D module output size={}".format(
                    network_config["module_out_dim"]))
            if args.mil_method == "transformer":
                network = TransformableTransformerPL(
                    **boilerplate_args,
                    **network_config)
            elif args.mil_method == "standard":
                network = MultipleInstanceClassifierPL(
                    **boilerplate_args,
                    **network_config)

            train_loader_call = None
            state_dict = torch.load(checkpoint)["state_dict"]
            network.load_state_dict(state_dict)
            network = network.eval().to(args.dev)
            trainer = Trainer(accelerator=accelerator,devices=devices)
            test_metrics = trainer.test(network,test_loader)[0]
            test_metrics["checkpoint"] = checkpoint
            test_metrics["pids"] = test_pids
            all_metrics.append(test_metrics)
    
    Path(args.metric_path).parent.mkdir(exist_ok=True,parents=True)
    with open(args.metric_path,"w") as o:
        json.dump(all_metrics,o)
