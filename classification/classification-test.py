import argparse
import random
import json
import numpy as np
import torch
import monai

from pytorch_lightning import Trainer

import sys
sys.path.append(r"..")
from lib.utils import safe_collate
from lib.utils.pl_utils import get_devices
from lib.monai_transforms import get_transforms_classification as get_transforms
from lib.modules.classification.pl import (
    ClassNetPL,UNetEncoderPL,ViTClassifierPL,
    FactorizedViTClassifierPL)
from lib.modules.layers.adn_fn import get_adn_fn
from lib.modules.losses import OrdinalSigmoidalLoss
from lib.modules.config_parsing import parse_config_unet,parse_config_cat
from lib.utils.dataset_filters import (
    filter_dictionary_with_filters,filter_dictionary_with_possible_labels,
    filter_dictionary_with_presence)
from lib.utils.parser import get_params,merge_args

if __name__ == "__main__":
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
        choices=["cat","ord","unet","vit","factorized_vit"],default="cat")
    
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

    args = parser.parse_args()

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
            "No data available for testing \
                (dataset={}; keys={}; labels={})".format(
                    args.dataset_json,
                    args.image_keys,
                    args.label_keys()))
    
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

    for iteration in range(len(args.test_ids)):
        test_ids = args.test_ids[iteration].split(",")
        test_list = [data_dict[pid] for pid in test_ids
                     if pid in data_dict]
        
        test_dataset = monai.data.CacheDataset(
            test_list,transforms_val,num_workers=args.n_workers)
        
        # PL needs a little hint to detect GPUs.
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
        adn_fn = get_adn_fn(3,"identity",act_fn=act_fn,dropout_param=0)
        batch_preprocessing = None
        boilerplate_args = {
            "n_channels":len(keys),
            "n_classes":n_classes,
            "image_key":"image",
            "label_key":"label"}

        if args.one_to_one is True:
            checkpoint_list = [args.checkpoints[iteration]]
        else:
            checkpoint_list = args.checkpoints
        for checkpoint in checkpoint_list:
            if args.net_type == "unet":
                network = UNetEncoderPL(
                    head_structure=[
                        network_config["depth"][-1] for _ in range(3)],
                    head_adn_fn=get_adn_fn(
                        1,"batch",act_fn="gelu",dropout_param=0),
                    **boilerplate_args,
                    **network_config)
            elif "vit" in args.net_type:
                image_size = [int(x) for x in args.crop_size]
                network_config["image_size"] = image_size
                if args.net_type == "vit":
                    network = ViTClassifierPL(
                        adn_fn=get_adn_fn(
                            1,"identity",act_fn="gelu",dropout_param=0),
                        **boilerplate_args,
                        **network_config)
                elif args.net_type == "factorized_vit":
                    for k in ["embed_method"]:
                        if k in network_config:
                            del network_config[k]
                    network = FactorizedViTClassifierPL(
                        adn_fn=get_adn_fn(
                            1,"identity",act_fn="gelu",dropout_param=0),
                        **boilerplate_args,
                        **network_config)                    
                
            else:
                network = ClassNetPL(
                    net_type=args.net_type,n_channels=len(keys),
                    n_classes=n_classes,
                    image_key="image",label_key="label",
                    adn_fn=adn_fn,n_epochs=args.max_epochs,
                    warmup_steps=args.warmup_steps,
                    **network_config)

            state_dict = torch.load(checkpoint)["state_dict"]
            state_dict = {k:state_dict[k] for k in state_dict
                          if k != "loss_fn.weight"}
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
