import random
import json
import numpy as np
import torch
import monai
from ..assemble_args import Parser

from lightning.pytorch import Trainer

import sys
from ...entrypoints.assemble_args import Parser
from ...utils import (
    safe_collate,
    RandomSlices,
    collate_last_slice)
from ...utils.pl_utils import get_devices
from ...utils.torch_utils import load_checkpoint_to_model
from ...utils.dataset_filters import filter_dictionary
from ...monai_transforms import (
    get_pre_transforms_generation as get_pre_transforms,
    get_post_transforms_generation as get_post_transforms)
from ...utils.network_factories import get_generative_network
from ...utils.parser import get_params,merge_args,parse_ids,compose

def get_conditional_specification(d: dict, cond_key: str):
    possible_values = []
    for k in d:
        if cond_key in d[k]:
            v = d[k][cond_key]
            if v not in possible_values:
                possible_values.append(d[k][cond_key])
    return possible_values

def return_first_not_none(*size_list):
    for size in size_list:
        if size is not None:
            return size

def main(arguments):
    parser = Parser()

    parser.add_argument_by_key([
        ('dataset_json','dataset_json',{"required": False}), 
        'params_from', 
        'image_keys', 'cat_condition_keys', 'num_condition_keys', 
        'filter_on_keys', 'excluded_ids', 
        'cache_rate', 
        'subsample_size',
        'target_spacing', 'pad_size', 'crop_size', 
        'config_file', 'overrides', 
        'dev', 'n_workers', 
        'seed', 
        'precision',
        'checkpoint', 
        'batch_size', 'learning_rate', 
        'diffusion_steps',
        'n_samples_gen'
    ])

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

    ckpt = torch.load(args.checkpoint)

    accelerator,devices,strategy = get_devices(args.dev)
    n_devices = len(devices) if isinstance(devices,list) else devices
    n_devices = 1 if isinstance(devices,str) else n_devices

    categorical_specification = ckpt["metadata"]["categorical_specification"]
    numerical_specification = ckpt["metadata"]["numerical_specification"]

    presence_keys = [*args.image_keys]
    with_conditioning = False
    if args.cat_condition_keys is not None:
        presence_keys.extend(args.cat_condition_keys)
        with_conditioning = True
    if args.num_condition_keys is not None:
        presence_keys.extend(args.num_condition_keys)
        with_conditioning = True
    
    keys = args.image_keys

    network_config = compose(args.config_file, "diffusion", args.overrides)
    network_config["batch_size"] = return_first_not_none(
        args.batch_size,network_config.get("batch_size"))
    network_config["learning_rate"] = return_first_not_none(
        args.learning_rate,network_config.get("learning_rate"))
    network_config["with_conditioning"] = with_conditioning
    network_config["cross_attention_dim"] = 256 if with_conditioning else None
    
    n_workers = args.n_workers // n_devices
    bs = network_config["batch_size"]
    real_bs = bs * n_devices
    if len(dataset) < real_bs:
        new_bs = len(dataset) // n_devices
        print(
            f"Batch size changed from {bs} to {new_bs} (dataset too small)")
        bs = new_bs
        real_bs = bs * n_devices

    loader = monai.data.ThreadDataLoader(
        dataset,batch_size=bs,
        shuffle=True,num_workers=n_workers,generator=g,
        collate_fn=collate_fn,pin_memory=True,
        persistent_workers=args.n_workers>0,
        drop_last=True)
    
    network = get_generative_network(
        network_config=network_config,
        categorical_specification=categorical_specification,
        numerical_specification=numerical_specification,
        train_loader_call=None,
        max_epochs=args.max_epochs,
        warmup_steps=args.warmup_steps,
        start_decay=args.start_decay,
        diffusion_steps=1000)

    load_checkpoint_to_model(network,args.checkpoint,[])

    trainer = Trainer(
        accelerator=accelerator,
        devices=devices,
        precision=args.precision,
        deterministic="warn")

    print("Setting up transforms...")
    transform_pre_arguments = {
        "keys":keys,
        "target_spacing":args.target_spacing,
        "crop_size":args.crop_size,
        "pad_size":args.pad_size}
    transform_post_arguments = {
        "image_keys":keys,
        "cat_keys": args.cat_condition_keys,
        "num_keys": args.num_condition_keys}

    transforms = [
        *get_pre_transforms(**transform_pre_arguments),
        *get_post_transforms(**transform_post_arguments)]
    
    # PL needs a little hint to detect GPUs.
    torch.ones([1]).to("cuda" if "cuda" in args.dev else "cpu")

    if network_config["spatial_dims"] == 2:
        transforms.append(
            RandomSlices(["image"],None,n=2,base=0.05))
        collate_fn = collate_last_slice
    else:
        collate_fn = safe_collate
    transforms = monai.transforms.Compose(transforms)
    transforms.set_random_state(args.seed)

    if args.dataset_json is not None:
        data_dict = json.load(open(args.dataset_json,'r'))
        if args.excluded_ids is not None:
            args.excluded_ids = parse_ids(args.excluded_ids,
                                        output_format="list")
            print("Removing IDs specified in --excluded_ids")
            prev_len = len(data_dict)
            data_dict = {k:data_dict[k] for k in data_dict
                        if k not in args.excluded_ids}
            print("\tRemoved {} IDs".format(prev_len - len(data_dict)))
        data_dict = filter_dictionary(data_dict, 
                                    filters_presence=presence_keys,
                                    filters=args.filter_on_keys)
        if args.subsample_size is not None and len(data_dict) > args.subsample_size:
            ss = rng.choice(list(data_dict.keys()),size=args.subsample_size)
            data_dict = {k:data_dict[k] for k in ss}

        if len(data_dict) == 0:
            raise Exception(
                "No data available for prediction \
                    (dataset={}; keys={}; labels={})".format(
                        args.dataset_json,
                        args.image_keys,
                        args.label_keys))

        all_pids = [k for k in data_dict]

        pred_list = [data_dict[pid] for pid in all_pids]
        
        print("\tPrediction set size={}".format(len(pred_list)))

        print(f"Number of cases: {len(pred_list)}")
        
        dataset = monai.data.CacheDataset(
            pred_list,transforms,
            cache_rate=args.cache_rate,
            num_workers=args.n_workers)
        
        for data in dataset:
            pass

    elif args.n_samples_gen is not None:
        size = return_first_not_none(args.crop_size,args.pad_size)
        for i in range(args.n_samples_gen):
            output = network.generate_image(size,1)

    else:
        raise Exception(
            "one of dataset_json, n_samples_gen should be specified")