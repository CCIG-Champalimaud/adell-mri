import os
import argparse
import random
import json
import numpy as np
import torch
import monai
from pathlib import Path

import sys
from lightning.pytorch import Trainer
from ...entrypoints.assemble_args import Parser
from ...utils import (
    safe_collate,ScaleIntensityAlongDimd,EinopsRearranged,
    subsample_dataset)
from ...utils.pl_utils import get_devices
from ...utils.dataset_filters import filter_dictionary
from ...monai_transforms import get_transforms_classification as get_transforms
from ...modules.classification.pl import (
    TransformableTransformerPL,MultipleInstanceClassifierPL)
from ...modules.config_parsing import parse_config_2d_classifier_3d
from ...utils.parser import parse_ids
from ...utils.parser import get_params,merge_args

def main(arguments):
    parser = Parser()

    parser.add_argument_by_key([
        "params_from",
        "dataset_json",
        "image_keys", "clinical_feature_keys", "adc_keys", "label_keys",
        "filter_on_keys",
        "possible_labels", "positive_labels", "label_groups",
        "target_spacing", "pad_size", "crop_size",
        "resize_size",
        "subsample_size", "batch_size",
        "config_file", "mil_method", "module_path",
        "dev", "seed", "n_workers",
        "metric_path",
        "test_ids",
        "one_to_one",
        "cache_rate",
        "excluded_ids",
        ("test_checkpoints","checkpoints")
    ])
    
    args = parser.parse_args(arguments)
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

    data_dict = filter_dictionary(
        data_dict,
        filters_presence=args.image_keys + [args.label_keys],
        possible_labels=args.possible_labels,
        label_key=args.label_keys,
        filters=args.filter_on_keys)
    data_dict = subsample_dataset(
        data_dict=data_dict,
        subsample_size=args.subsample_size,
        rng=rng,
        strata_key=args.label_keys)

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
    adc_keys = []

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
        "target_size":args.resize_size,
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
