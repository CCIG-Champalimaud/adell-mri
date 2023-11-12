import argparse
import json
import torch
import monai
from tqdm import tqdm
from pathlib import Path

import sys
from ...entrypoints.assemble_args import Parser
from ...utils.torch_utils import load_checkpoint_to_model
from ...utils.dataset_filters import (
    filter_dictionary_with_filters,
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
        "image_keys", "clinical_feature_keys", "adc_keys", "n_classes",
        "filter_on_keys", "excluded_ids",
        "cache_rate",
        "target_spacing", "pad_size", "crop_size",
        "config_files", "ensemble_config_file", 
        ("classification_net_types","net_types"),
        "module_paths",
        "prediction_ids", "one_to_one",
        "dev","n_workers",
        ("prediction_type","type"),
        ("prediction_checkpoints", "checkpoints"),
        "batch_size",
        "output_path"
    ])

    args = parser.parse_args(arguments)

    if args.params_from is not None:
        param_dict = get_params(args.params_from)
        args = merge_args(args,param_dict,sys.argv[1:])

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
    if len(args.filter_on_keys) > 0:
        data_dict = filter_dictionary_with_filters(
            data_dict,args.filter_on_keys)
    data_dict = filter_dictionary_with_presence(
        data_dict,args.image_keys + clinical_feature_keys)
    if len(clinical_feature_keys) > 0:
        data_dict = filter_dictionary_with_filters(
            data_dict,[f"{k}!=nan" for k in clinical_feature_keys])

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
        args.ensemble_config_file,args.n_classes)
    
    if args.module_paths is not None:
        config_files = None
        module_paths = args.module_paths
        network_configs = None
    else:
        network_configs = [
            parse_config_unet(config_file,len(keys),args.n_classes) 
            if net_type == "unet"
            else parse_config_cat(config_file)
            for config_file,net_type in zip(config_files,args.net_types)]
        if len(args.config_files) == 1:
            config_files = [args.config_files[0] for _ in args.net_types]
        else:
            config_files = args.config_files

    if args.batch_size is not None:
        ensemble_config["batch_size"] = args.batch_size
    if "batch_size" not in ensemble_config:
        ensemble_config["batch_size"] = 1
    
    all_pids = [k for k in data_dict]

    print("Setting up transforms...")
    label_mode = "binary" if args.n_classes == 2 else "cat"
    transform_arguments = {
        "keys":keys,
        "clinical_feature_keys":clinical_feature_keys,
        "adc_keys":adc_keys,
        "target_spacing":args.target_spacing,
        "crop_size":args.crop_size,
        "pad_size":args.pad_size,
        "possible_labels":None,
        "positive_labels":None,
        "label_groups":None,
        "label_key":None,
        "label_mode":label_mode}

    transforms_prediction = monai.transforms.Compose([
        *get_transforms("pre",**transform_arguments),
        *get_transforms("post",**transform_arguments)])
    
    global_output = []
    extra_args = {}

    if args.type == "probability":
        if args.n_classes > 2:
            post_proc_fn = torch.nn.Softmax(-1)
        else:
            post_proc_fn = torch.nn.Sigmoid()
    else:
        post_proc_fn = torch.nn.Identity()

    if args.prediction_ids:
        prediction_ids = parse_ids(args.prediction_ids,"list")
    else:
        prediction_ids = [[k for k in data_dict]]
    for iteration in range(len(prediction_ids)):
        curr_prediction_ids = [pid for pid in prediction_ids[iteration]
                               if pid in data_dict]
        prediction_list = [data_dict[pid] for pid in curr_prediction_ids]
        
        prediction_dataset = monai.data.CacheDataset(
            prediction_list,transforms_prediction,num_workers=args.n_workers,
            cache_rate=args.cache_rate)
        
        # PL sometimes needs a little hint to detect GPUs.
        torch.ones([1]).to("cuda" if "cuda" in args.dev else "cpu")
        
        if args.n_classes == 2:
            ensemble_config["loss_fn"] = torch.nn.BCEWithLogitsLoss()
        elif args.net_types[0] == "ord":
            ensemble_config["loss_fn"] = OrdinalSigmoidalLoss(
                n_classes=args.n_classes)
        else:
            ensemble_config["loss_fn"] = torch.nn.CrossEntropyLoss()

        batch_preprocessing = None
        if args.one_to_one is True:
            checkpoint_list = [args.checkpoints[iteration]]
        else:
            checkpoint_list = args.checkpoints
        for checkpoint_idx,checkpoint in enumerate(checkpoint_list):
            if network_configs is not None:
                networks = [get_classification_network(
                    net_type=net_type,
                    network_config=network_config,
                    dropout_param=0.0,
                    seed=42,
                    n_classes=args.n_classes,
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
            else:
                networks = []
                for module_path in module_paths:
                    network = torch.jit.load(module_path)
                    network.requires_grad = False
                    network.eval()
                    network = torch.jit.freeze(network)
                    networks.append(network)

            ensemble = GenericEnsemblePL(
                image_keys=["image"],
                label_key="label",
                networks=networks,
                n_classes=args.n_classes,
                training_dataloader_call=None,
                n_epochs=None,
                warmup_steps=None,
                start_decay=None,
                **ensemble_config).to(args.dev).eval()

            load_checkpoint_to_model(
                ensemble,checkpoint,
                exclude_from_state_dict=["loss_fn.weight"])

            output_dict = {
                "iteration":iteration,
                "prediction_ids":curr_prediction_ids,
                "checkpoint":checkpoint,
                "predictions":{}
            }

            with tqdm(total=len(curr_prediction_ids)) as pbar:
                for identifier,element in zip(curr_prediction_ids,
                                              prediction_dataset):
                    pbar.set_description("Predicting {}".format(identifier))
                    output = ensemble.forward(
                        element["image"].unsqueeze(0).to(args.dev),
                        **extra_args).detach()
                    output = output.cpu()
                    output = post_proc_fn(output)
                    output = output.numpy()[0].tolist()
                    output_dict["predictions"][identifier] = output
                    pbar.update()
            global_output.append(output_dict)
    
    Path(args.output_path).parent.mkdir(exist_ok=True,parents=True)
    with open(args.output_path,"w") as o:
        o.write(json.dumps(global_output))