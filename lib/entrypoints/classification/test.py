import argparse
import random
import json
import numpy as np
import torch
import monai

from lightning.pytorch import Trainer

import sys
from ...utils import safe_collate
from ...utils.pl_utils import get_devices
from ...monai_transforms import get_transforms_classification as get_transforms
from ...modules.classification.losses import OrdinalSigmoidalLoss
from ...modules.config_parsing import parse_config_unet,parse_config_cat
from ...utils.dataset_filters import (
    filter_dictionary_with_filters,filter_dictionary_with_possible_labels,
    filter_dictionary_with_presence)
from ...utils.network_factories import get_classification_network
from ...utils.parser import get_params,merge_args,parse_ids

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
        '--image_keys',dest='image_keys',type=str,nargs='+',
        help="Image keys in the dataset JSON.",
        required=True)
    parser.add_argument(
        '--clinical_feature_keys',dest='clinical_feature_keys',type=str,
        nargs='+',help="Tabular clinical feature keys in the dataset JSON.",
        default=None)
    parser.add_argument(
        '--adc_keys',dest='adc_keys',type=str,nargs='+',
        help="Image keys corresponding to ADC.",default=None)
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
        '--target_spacing',dest='target_spacing',action="store",default=None,
        help="Resamples all images to target spacing",nargs='+',type=float)
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

    # network
    parser.add_argument(
        '--config_file',dest="config_file",
        help="Path to network configuration file (yaml)",
        required=True)
    parser.add_argument(
        '--net_type',dest='net_type',
        help="Classification type. Can be categorical (cat) or ordinal (ord)",
        choices=["cat","ord","unet","vit","factorized_vit","vgg"],default="cat")
    
    # testing
    parser.add_argument(
        '--dev',dest='dev',default="cpu",
        help="Device for PyTorch testing",type=str)
    parser.add_argument(
        '--seed',dest='seed',help="Random seed",default=42,type=int)
    parser.add_argument(
        '--n_workers',dest='n_workers',
        help="No. of workers",default=0,type=int)
    parser.add_argument(
        '--test_ids',dest="test_ids",type=str,default=None,nargs="+",
        help="Comma-separated IDs to be used in each test")
    parser.add_argument(
        '--one_to_one',dest="one_to_one",action="store_true",
        help="Tests the checkpoint only on the corresponding test_ids set")
    parser.add_argument(
        '--checkpoints',dest='checkpoints',type=str,default=None,
        nargs="+",help='Test using these checkpoints.')
    parser.add_argument(
        '--metric_path',dest='metric_path',type=str,default="metrics.csv",
        help='Path to file with CV metrics + information.')

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

    output_file = open(args.metric_path,'w')
    
    if args.clinical_feature_keys is None:
        clinical_feature_keys = []
    else:
        clinical_feature_keys = args.clinical_feature_keys

    data_dict = json.load(open(args.dataset_json,'r'))
    data_dict = filter_dictionary_with_possible_labels(
        data_dict,args.possible_labels,args.label_keys)
    if len(args.filter_on_keys) > 0:
        data_dict = filter_dictionary_with_filters(
            data_dict,args.filter_on_keys)
    data_dict = filter_dictionary_with_presence(
        data_dict,args.image_keys + [args.label_keys] + clinical_feature_keys)
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
            "No data available for testing \
                (dataset={}; keys={}; labels={})".format(
                    args.dataset_json,
                    args.image_keys,
                    args.label_keys))
    
    keys = args.image_keys
    adc_keys = args.adc_keys if args.adc_keys is not None else []
    adc_keys = [k for k in adc_keys if k in keys]

    if args.net_type == "unet":
        network_config,_ = parse_config_unet(args.config_file,
                                             len(keys),n_classes)
    else:
        network_config = parse_config_cat(args.config_file)
    
    if args.batch_size is not None:
        network_config["batch_size"] = args.batch_size

    if "batch_size" not in network_config:
        network_config["batch_size"] = 1
    
    all_pids = [k for k in data_dict]

    print("Setting up transforms...")
    label_mode = "binary" if n_classes == 2 else "cat"
    transform_arguments = {
        "keys":keys,
        "clinical_feature_keys":clinical_feature_keys,
        "adc_keys":adc_keys,
        "target_spacing":args.target_spacing,
        "crop_size":args.crop_size,
        "pad_size":args.pad_size,
        "possible_labels":args.possible_labels,
        "positive_labels":args.positive_labels,
        "label_key":args.label_keys,
        "label_mode":label_mode}

    transforms_val = monai.transforms.Compose([
        *get_transforms("pre",**transform_arguments),
        *get_transforms("post",**transform_arguments)])
    transforms_val.set_random_state(args.seed)

    all_test_ids = parse_ids(args.test_ids)
    for iteration in range(len(all_test_ids)):
        test_ids = all_test_ids[iteration]
        test_list = [data_dict[pid] for pid in test_ids if pid in data_dict]

        print("Testing fold",iteration)
        for u,c in zip(*np.unique([x[args.label_keys] for x in test_list],
                                  return_counts=True)):
            print(f"\tCases({u}) = {c}")
        
        test_dataset = monai.data.Dataset(test_list,transforms_val)
        
        # PL sometimes needs a little hint to detect GPUs.
        torch.ones([1]).to("cuda" if "cuda" in args.dev else "cpu")
        
        if n_classes == 2:
            network_config["loss_fn"] = torch.nn.BCEWithLogitsLoss()
        elif args.net_type == "ord":
            network_config["loss_fn"] = OrdinalSigmoidalLoss(
                n_classes=n_classes)
        else:
            network_config["loss_fn"] = torch.nn.CrossEntropy()

        test_loader = monai.data.ThreadDataLoader(
            test_dataset,batch_size=network_config["batch_size"],
            shuffle=False,num_workers=args.n_workers,
            collate_fn=safe_collate)

        print("Setting up testing...")
        if args.net_type == "unet":
            act_fn = network_config["activation_fn"]
        else:
            act_fn = "swish"
        batch_preprocessing = None

        if args.one_to_one is True:
            checkpoint_list = [args.checkpoints[iteration]]
        else:
            checkpoint_list = args.checkpoints
        for checkpoint in checkpoint_list:
            network = get_classification_network(
                net_type=args.net_type,
                network_config=network_config,
                dropout_param=0,
                seed=None,
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

            state_dict = torch.load(checkpoint)["state_dict"]
            state_dict = {k:state_dict[k] for k in state_dict
                          if "loss_fn.weight" not in k}
            network.load_state_dict(state_dict)
            trainer = Trainer(accelerator=accelerator,devices=devices)
            test_metrics = trainer.test(network,test_loader)[0]
            for k in test_metrics:
                out = test_metrics[k]
                if n_classes == 2:
                    try:
                        value = float(out.detach().numpy())
                    except Exception:
                        value = float(out)
                    x = "{},{},{},{},{}".format(k,checkpoint,iteration,0,value)
                    output_file.write(x+'\n')
                    print(x)
                else:
                    for i,v in enumerate(out):
                        x = "{},{},{},{},{}".format(k,checkpoint,iteration,i,v)
                        output_file.write(x+'\n')
                        print(x)
