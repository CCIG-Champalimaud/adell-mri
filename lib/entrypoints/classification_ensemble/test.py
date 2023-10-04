import argparse
import json
import numpy as np
import torch
import monai

from lightning.pytorch import Trainer

import sys
from ...entrypoints.assemble_args import Parser
from ...utils import (safe_collate)
from ...utils.pl_utils import (get_devices)
from ...utils.torch_utils import load_checkpoint_to_model
from ...utils.dataset_filters import (
    filter_dictionary_with_filters,
    filter_dictionary_with_possible_labels,
    filter_dictionary_with_presence)
from ...monai_transforms import get_transforms_classification as get_transforms
from ...modules.losses import OrdinalSigmoidalLoss
from ...modules.config_parsing import (
    parse_config_unet,parse_config_cat,parse_config_ensemble)
from ...modules.classification.pl import GenericEnsemblePL
from ...utils.network_factories import get_classification_network
from ...utils.parser import get_params,merge_args,parse_ids

def main(arguments):
    parser = Parser()

    parser.add_argument_by_key([
        "params_from",
        "dataset_json",
        "image_keys", "clinical_feature_keys", "adc_keys", "label_keys",
        "possible_labels", "positive_labels", "label_groups",
        "filter_on_keys", "excluded_ids",
        "cache_rate",
        "target_spacing", "pad_size", "crop_size",
        "config_files", "ensemble_config_file", 
        ("classification_net_types","net_types"),
        "test_ids", "one_to_one",
        "dev","n_workers",
        ("test_checkpoints","checkpoints"),
        "metric_path",
        "batch_size"
    ])
    
    args = parser.parse_args(arguments)

    if args.params_from is not None:
        param_dict = get_params(args.params_from)
        args = merge_args(args,param_dict,sys.argv[1:])

    if len(args.config_files) == 1:
        config_files = [args.config_files[0] for _ in args.net_types]
    else:
        config_files = args.config_files

    accelerator,devices,strategy = get_devices(args.dev)
    n_devices = len(devices) if isinstance(devices,list) else devices
    n_devices = 1 if isinstance(devices,str) else n_devices

    output_file = open(args.metric_path,'w')

    data_dict = json.load(open(args.dataset_json,'r'))
    
    if args.clinical_feature_keys is None:
        clinical_feature_keys = []
    else:
        clinical_feature_keys = args.clinical_feature_keys
    
    if args.excluded_ids is not None:
        args.excluded_ids = parse_ids(args.excluded_ids,
                                      output_format="list")
        print("Removing IDs specified in --excluded_ids")
        prev_len = len(data_dict)
        data_dict = {k:data_dict[k] for k in data_dict
                     if k not in args.excluded_ids}
        print("\tRemoved {} IDs".format(prev_len - len(data_dict)))
    data_dict = filter_dictionary_with_possible_labels(
        data_dict,args.possible_labels,args.label_keys)
    if len(args.filter_on_keys) > 0:
        data_dict = filter_dictionary_with_filters(
            data_dict,args.filter_on_keys)
    data_dict = filter_dictionary_with_presence(
        data_dict,args.image_keys + [args.label_keys] + clinical_feature_keys)
    if len(clinical_feature_keys) > 0:
        data_dict = filter_dictionary_with_filters(
            data_dict,[f"{k}!=nan" for k in clinical_feature_keys])
    all_classes = []
    for k in data_dict:
        C = data_dict[k][args.label_keys]
        if isinstance(C,list):
            C = max(C)
        all_classes.append(str(C))
    label_groups = None
    if args.label_groups is not None:
        n_classes = len(args.label_groups)
        label_groups = [label_group.split(",")
                        for label_group in args.label_groups]
    elif args.positive_labels is None:
        n_classes = len(args.possible_labels)
    else:
        n_classes = 2


    if len(data_dict) == 0:
        raise Exception(
            "No data available for training \
                (dataset={}; keys={}; labels={})".format(
                    args.dataset_json,
                    args.image_keys,
                    args.label_keys))
    
    keys = args.image_keys
    adc_keys = args.adc_keys if args.adc_keys is not None else []
    adc_keys = [k for k in adc_keys if k in keys]

    ensemble_config = parse_config_ensemble(
        args.ensemble_config_file,n_classes)
    
    network_configs = [
        parse_config_unet(config_file,len(keys),n_classes) 
        if net_type == "unet"
        else parse_config_cat(config_file)
        for config_file,net_type in zip(config_files,args.net_types)]
    
    if args.batch_size is not None:
        ensemble_config["batch_size"] = args.batch_size
    if "batch_size" not in ensemble_config:
        ensemble_config["batch_size"] = 1
    
    all_pids = [k for k in data_dict]

    print("Setting up transforms...")
    label_mode = "binary" if n_classes == 2 and label_groups is None else "cat"
    transform_arguments = {
        "keys":keys,
        "clinical_feature_keys":clinical_feature_keys,
        "adc_keys":adc_keys,
        "target_spacing":args.target_spacing,
        "crop_size":args.crop_size,
        "pad_size":args.pad_size,
        "possible_labels":args.possible_labels,
        "positive_labels":args.positive_labels,
        "label_groups":label_groups,
        "label_key":args.label_keys,
        "label_mode":label_mode}

    transforms_testing = monai.transforms.Compose([
        *get_transforms("pre",**transform_arguments),
        *get_transforms("post",**transform_arguments)])
    
    all_test_ids = parse_ids(args.test_ids)
    for iteration in range(len(all_test_ids)):
        test_ids = all_test_ids[iteration]
        test_list = [data_dict[pid] for pid in test_ids if pid in data_dict]

        print("Testing fold",iteration)
        for u,c in zip(*np.unique([x[args.label_keys] for x in test_list],
                                  return_counts=True)):
            print(f"\tCases({u}) = {c}")
        
        test_dataset = monai.data.Dataset(test_list,transforms_testing)
        
        # PL sometimes needs a little hint to detect GPUs.
        torch.ones([1]).to("cuda" if "cuda" in args.dev else "cpu")
        
        if n_classes == 2:
            ensemble_config["loss_fn"] = torch.nn.BCEWithLogitsLoss()
        elif args.net_types[0] == "ord":
            ensemble_config["loss_fn"] = OrdinalSigmoidalLoss(
                n_classes=n_classes)
        else:
            ensemble_config["loss_fn"] = torch.nn.CrossEntropyLoss()

        test_loader = monai.data.ThreadDataLoader(
            test_dataset,batch_size=ensemble_config["batch_size"],
            shuffle=False,num_workers=args.n_workers,
            collate_fn=safe_collate)

        print("Setting up testing...")
        batch_preprocessing = None

        if args.one_to_one is True:
            checkpoint_list = [args.checkpoints[iteration]]
        else:
            checkpoint_list = args.checkpoints
        for checkpoint_idx,checkpoint in enumerate(checkpoint_list):
            networks = [get_classification_network(
                net_type=net_type,
                network_config=network_config,
                dropout_param=0.0,
                seed=42,
                n_classes=n_classes,
                keys=keys,
                clinical_feature_keys=clinical_feature_keys,
                train_loader_call=None,
                max_epochs=None,
                warmup_steps=None,
                start_decay=None,
                crop_size=args.crop_size,
                clinical_feature_means=None,
                clinical_feature_stds=None,
                label_smoothing=None,
                mixup_alpha=None,
                partial_mixup=None)
                for net_type,network_config in zip(args.net_types,network_configs)]
            
            ensemble = GenericEnsemblePL(
                image_keys=["image"],
                label_key="label",
                networks=networks,
                n_classes=n_classes,
                training_dataloader_call=None,
                n_epochs=None,
                warmup_steps=None,
                start_decay=None,
                **ensemble_config)

            load_checkpoint_to_model(
                ensemble,checkpoint,exclude_from_state_dict=["loss_fn.weight"])

            trainer = Trainer(accelerator=accelerator,devices=devices)
            test_metrics = trainer.test(ensemble,test_loader)[0]
            for k in test_metrics:
                out = test_metrics[k]
                try:
                    value = float(out.detach().numpy())
                except Exception:
                    value = float(out)
                if n_classes > 2:
                    k = k.split("_")
                    if k[-1].isdigit():
                        k,idx = "_".join(k[:-1]),k[-1]
                    else:
                        k,idx = "_".join(k),0
                else:
                    idx = 0
                x = "{},{},{},{},{}".format(k,checkpoint,iteration,idx,value)
                output_file.write(x+'\n')
