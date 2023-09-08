import argparse
import random
import json
import numpy as np
import torch
import monai
from pathlib import Path
from tqdm import tqdm

import sys
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
        '--prediction_ids',dest='prediction_ids',type=str,nargs="+",
        help="List of IDs for prediction",required=True)
    parser.add_argument(
        '--excluded_ids',dest='excluded_ids',type=str,nargs="+",
        help="List of IDs to exclude",default=None)
    parser.add_argument(
        '--one_to_one',dest="one_to_one",action="store_true",
        help="Tests the checkpoint only on the corresponding prediction_ids set")
    parser.add_argument(
        '--image_keys',dest='image_keys',type=str,nargs='+',
        help="Image keys in the dataset JSON.",
        required=True)
    parser.add_argument(
        '--n_classes',dest='n_classes',type=int,required=True,
        help="Number of classes.")
    parser.add_argument(
        '--t2_keys',dest='t2_keys',type=str,nargs='+',
        help="Image keys corresponding to T2.",default=None)
    parser.add_argument(
        '--filter_on_keys',dest='filter_on_keys',type=str,default=[],nargs="+",
        help="Filters the dataset based on a set of specific key:value pairs.")
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

    # network + prediction
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
    parser.add_argument(
        '--type',dest='type',action="store",default="probability",
        help="Prediction type.",
        choices=["probability","logit","attention"])
    
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
        nargs="+",help='List of checkpoints for prediction.')
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

    accelerator,devices,strategy = get_devices(args.dev)
    
    data_dict = json.load(open(args.dataset_json,'r'))
    all_prediction_pids = parse_ids(args.prediction_ids)
    if args.excluded_ids is not None:
        excluded_ids = parse_ids(args.excluded_ids,output_format="list")
        a = len(data_dict)
        data_dict = {k:data_dict[k] for k in data_dict
                     if k not in excluded_ids}
        print("Excluded {} cases with --excluded_ids".format(a - len(data_dict)))
    
    data_dict = filter_dictionary(
        data_dict,
        filters_presence=args.image_keys,
        filters=args.filter_on_keys)
    data_dict = subsample_dataset(
        data_dict=data_dict,
        subsample_size=args.subsample_size,
        rng=rng)

    if len(data_dict) == 0:
        raise Exception(
            "No data available for training \
                (dataset={}; keys={})".format(
                    args.dataset_json,
                    args.image_keys))
    
    keys = args.image_keys
    t2_keys = args.t2_keys if args.t2_keys is not None else []
    adc_keys = []
    t2_keys = [k for k in t2_keys if k in keys]

    network_config,_ = parse_config_2d_classifier_3d(
        args.config_file,0.0)
    
    if args.n_classes == 2:
        network_config["loss_fn"] = torch.nn.BCEWithLogitsLoss(
            torch.ones([]))
    else:
        network_config["loss_fn"] = torch.nn.CrossEntropy(
            torch.ones([args.n_classes]))
    
    if args.batch_size is not None:
        network_config["batch_size"] = args.batch_size

    if "batch_size" not in network_config:
        network_config["batch_size"] = 1
    
    all_pids = [k for k in data_dict]

    print("Setting up transforms...")
    label_mode = "binary" if args.n_classes == 2 else "cat"
    transform_arguments = {
        "keys":keys,
        "adc_keys":adc_keys,
        "target_spacing":args.target_spacing,
        "target_size":args.target_size,
        "crop_size":args.crop_size,
        "pad_size":args.pad_size,
        "possible_labels":None,
        "positive_labels":None,
        "label_key":None,
        "clinical_feature_keys":[],
        "label_mode":label_mode}

    transforms = monai.transforms.Compose([
        *get_transforms("pre",**transform_arguments),
        *get_transforms("post",**transform_arguments),
        EinopsRearranged("image","c h w d -> 1 h w (d c)"),
        ScaleIntensityAlongDimd("image",dim=-1)])
    
    all_metrics = []
    for iteration,prediction_pids in enumerate(all_prediction_pids):
        prediction_pids = [pid for pid in prediction_pids
                           if pid in data_dict]
        prediction_list = [data_dict[pid] for pid in prediction_pids]
        prediction_dataset = monai.data.CacheDataset(
            prediction_list,transforms,
            cache_rate=args.cache_rate,
            num_workers=args.n_workers)
        
        # PL sometimes needs a little hint to detect GPUs.
        torch.ones([1]).to("cuda" if "cuda" in args.dev else "cpu")

        prediction_loader = monai.data.ThreadDataLoader(
            prediction_dataset,batch_size=network_config["batch_size"],
            shuffle=False,num_workers=args.n_workers,
            collate_fn=safe_collate)

        if args.one_to_one is True:
            checkpoint_list = [args.checkpoints[iteration]]
        else:
            checkpoint_list = args.checkpoints
        for checkpoint in checkpoint_list:
            n_slices = int(len(keys) * args.crop_size[-1])
            boilerplate_args = {
                "n_classes":args.n_classes,
                "training_dataloader_call":None,
                "image_key":"image",
                "label_key":None,
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

            state_dict = torch.load(checkpoint)["state_dict"]
            network.load_state_dict(state_dict)
            network = network.eval().to(args.dev)
            kwargs = {}
            if args.type == "attention":
                kwargs["return_attention"] = True
            prediction_output = []
            with tqdm(prediction_loader,total=len(prediction_loader)) as pbar:
                for idx,batch in enumerate(pbar):
                    batch = {k:batch[k].to(args.dev) for k in batch
                             if isinstance(batch[k],torch.Tensor)}
                    prediction_output.append(network.predict_step(batch,idx,**kwargs))
            attention = None
            if args.type == "probability":
                if args.n_classes == 2:
                    prediction_output = [torch.nn.functional.sigmoid(x)[0]
                                         for x in prediction_output]
                else:
                    prediction_output = [torch.nn.functional.softmax(x,axis=-1)[0]
                                         for x in prediction_output]
            elif args.type == "logit":
                prediction_output = [x[0] for x in prediction_output]
            elif args.type == "attention":
                attention = [x[1][0] for x in prediction_output]
                prediction_output = [x[0][0] for x in prediction_output]
            prediction_output = {
                k:x.detach().cpu().numpy().tolist() 
                for x,k in zip(prediction_output,prediction_pids)}
            prediction_output = {"prediction":prediction_output}
            if attention is not None:
                prediction_output["attention"] = {
                    k:x.detach().cpu().numpy().tolist() 
                    for x,k in zip(attention,prediction_pids)}
            prediction_output["checkpoint"] = checkpoint
            all_metrics.append(prediction_output)
    
    Path(args.output_path).parent.mkdir(exist_ok=True,parents=True)
    with open(args.output_path,"w") as o:
        json.dump(all_metrics,o)
