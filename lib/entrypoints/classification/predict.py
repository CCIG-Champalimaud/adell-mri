import argparse
import random
import json
import numpy as np
import torch
import monai
from pathlib import Path
from tqdm import tqdm

import sys
from lib.utils.utils import subsample_dataset
from ...monai_transforms import get_transforms_classification as get_transforms
from ...modules.classification.losses import OrdinalSigmoidalLoss
from ...modules.config_parsing import parse_config_unet,parse_config_cat
from ...utils.dataset_filters import filter_dictionary
from ...utils.network_factories import get_classification_network
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
        '--filter_on_keys',dest='filter_on_keys',type=str,default=[],nargs="+",
        help="Filters the dataset based on a set of specific key:value pairs.")
    parser.add_argument(
        '--n_classes',dest='n_classes',type=int,
        help="Number of classes.",required=True)
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
    parser.add_argument(
        '--cache_rate',dest='cache_rate',type=float,default=1.0,
        help="Fraction of data that will be cached using CacheDataset")

    # network
    parser.add_argument(
        '--config_file',dest="config_file",
        help="Path to network configuration file (yaml)",
        required=True)
    parser.add_argument(
        '--net_type',dest='net_type',
        help="Classification type. Can be categorical (cat) or ordinal (ord)",
        choices=["cat","ord","unet","vit","factorized_vit","vgg"],default="cat")
    
    # prediction
    parser.add_argument(
        '--dev',dest='dev',default="cpu",
        help="Device for PyTorch testing",type=str)
    parser.add_argument(
        '--seed',dest='seed',help="Random seed",default=42,type=int)
    parser.add_argument(
        '--n_workers',dest='n_workers',
        help="No. of workers",default=0,type=int)
    parser.add_argument(
        '--prediction_ids',dest="prediction_ids",type=str,default=None,nargs="+",
        help="Comma-separated IDs to be used in each test")
    parser.add_argument(
        '--one_to_one',dest="one_to_one",action="store_true",
        help="Predicts for a checkpoint using the corresponding prediction_ids")
    parser.add_argument(
        '--type',dest='type',action="store",default="probability",
        help="Returns either probability the classification probability or the\
            features in the last layer.",
        choices=["probability","logit","features"])
    parser.add_argument(
        '--checkpoints',dest='checkpoints',type=str,default=None,
        nargs="+",help='Test using these checkpoints.')
    parser.add_argument(
        '--output_path',dest='output_path',type=str,default="output.csv",
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

    if args.clinical_feature_keys is None:
        clinical_feature_keys = []
    else:
        clinical_feature_keys = args.clinical_feature_keys

    data_dict = json.load(open(args.dataset_json,'r'))
    
    data_dict = filter_dictionary(
        data_dict,
        filters_presence=args.image_keys + clinical_feature_keys,
        filters=args.filter_on_keys)
    data_dict = subsample_dataset(
        data_dict=data_dict,
        subsample_size=args.subsample_size,
        rng=rng)

    if len(data_dict) == 0:
        raise Exception(
            "No data available for prediction \
                (dataset={}; keys={})".format(
                    args.dataset_json,
                    args.image_keys))
    
    keys = args.image_keys
    adc_keys = args.adc_keys if args.adc_keys is not None else []
    adc_keys = [k for k in adc_keys if k in keys]

    if args.net_type == "unet":
        network_config,_ = parse_config_unet(args.config_file,
                                             len(keys),args.n_classes)
    else:
        network_config = parse_config_cat(args.config_file)
    
    if args.batch_size is not None:
        network_config["batch_size"] = args.batch_size

    if "batch_size" not in network_config:
        network_config["batch_size"] = 1

    transform_arguments = {
        "keys":keys,
        "clinical_feature_keys":clinical_feature_keys,
        "adc_keys":adc_keys,
        "target_spacing":args.target_spacing,
        "crop_size":args.crop_size,
        "pad_size":args.pad_size}

    transforms_prediction = monai.transforms.Compose([
        *get_transforms("pre",**transform_arguments),
        *get_transforms("post",**transform_arguments)])

    global_output = []
    if args.type in ["probability","logit"]:
        extra_args = {}
    else:
        extra_args = {"return_features":True}

    if args.type == "probability":
        if args.n_classes > 2:
            post_proc_fn = torch.nn.Softmax(-1)
        else:
            post_proc_fn = torch.nn.Sigmoid()
    else:
        post_proc_fn = torch.nn.Identity()
    
    if args.prediction_ids:
        prediction_ids = parse_ids(args.prediction_ids)
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
            network_config["loss_fn"] = torch.nn.BCEWithLogitsLoss()
        elif args.net_type == "ord":
            network_config["loss_fn"] = OrdinalSigmoidalLoss(
                n_classes=args.n_classes)
        else:
            network_config["loss_fn"] = torch.nn.CrossEntropy()

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
            print(f"Predicting for {checkpoint}")
            network = get_classification_network(
                net_type=args.net_type,
                network_config=network_config,
                dropout_param=0,
                seed=None,
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

            state_dict = torch.load(checkpoint)["state_dict"]
            state_dict = {k:state_dict[k] for k in state_dict
                          if "loss_fn.weight" not in k}
            network.load_state_dict(state_dict)
            network = network.eval().to(args.dev)
            
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
                    output = network.forward(
                        element["image"].unsqueeze(0).to(args.dev),
                        **extra_args).detach()
                    if args.type == "features":
                        output = output.flatten(start_dim=2).max(-1).values.cpu()
                    else:
                        output = output.cpu()
                    output = post_proc_fn(output)
                    output = output.numpy()[0].tolist()
                    output_dict["predictions"][identifier] = output
                    pbar.update()
            global_output.append(output_dict)
    
    Path(args.output_path).parent.mkdir(exist_ok=True,parents=True)
    with open(args.output_path,"w") as o:
        o.write(json.dumps(global_output))
