import argparse
import random
import yaml
import json
import numpy as np
import torch
import monai
from ..assemble_args import Parser
from hydra import compose

from lightning.pytorch import Trainer
from lightning.pytorch.callbacks import RichProgressBar

import sys
from ...entrypoints.assemble_args import Parser
from ...utils import (
    safe_collate,
    conditional_parameter_freezing,
    RandomSlices,
    collate_last_slice)
from ...utils.pl_utils import (
    get_ckpt_callback,
    get_logger,
    get_devices,
    LogImageFromDiffusionProcess)
from ...utils.torch_utils import load_checkpoint_to_model
from ...utils.dataset_filters import filter_dictionary
from ...monai_transforms import (
    get_pre_transforms_generation as get_pre_transforms,
    get_post_transforms_generation as get_post_transforms)
from ...utils.network_factories import get_generative_network
from ...utils.parser import get_params,merge_args,parse_ids

def get_conditional_specification(d: dict, cond_key: str):
    possible_values = []
    for k in d:
        if cond_key in d[k]:
            v = d[k][cond_key]
            if v not in possible_values:
                possible_values.append(d[k][cond_key])
    return possible_values

def get_size(*size_list):
    for size in size_list:
        if size is not None:
            return size

def main(arguments):
    parser = Parser()

    parser.add_argument_by_key([
        'dataset_json', 
        'params_from', 
        'image_keys', 'cat_condition_keys', 'num_condition_keys', 
        'filter_on_keys', 'excluded_ids', 
        'cache_rate', 
        'subsample_size','val_from_train', 
        'target_spacing', 'pad_size', 'crop_size', 
        'config_file', 'overrides', 
        'warmup_steps', 'start_decay', 
        'dev', 'n_workers', 
        'seed', 
        'max_epochs', 
        'precision', 'check_val_every_n_epoch', 
        'gradient_clip_val', 'accumulate_grad_batches', 
        'checkpoint_dir', 'checkpoint_name', 'checkpoint', 'resume_from_last', 
        'exclude_from_state_dict', 
        'freeze_regex', 'not_freeze_regex', 
        'project_name', 'monitor', 'summary_dir', 'summary_name', 
        'metric_path', 'resume', 
        'dropout_param', 
        'batch_size', 'learning_rate', 
        'diffusion_steps'
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

    accelerator,devices,strategy = get_devices(args.dev)
    n_devices = len(devices) if isinstance(devices,list) else devices
    n_devices = 1 if isinstance(devices,str) else n_devices

    output_file = open(args.metric_path,'w')

    data_dict = json.load(open(args.dataset_json,'r'))
    presence_keys = [*args.image_keys]
    categorical_specification = None
    numerical_specification = None
    with_conditioning = False
    if args.cat_condition_keys is not None:
        categorical_specification = [
            get_conditional_specification(data_dict, k)
            for k in args.cat_condition_keys]
        presence_keys.extend(args.cat_condition_keys)
        with_conditioning = True
    if args.num_condition_keys is not None:
        numerical_specification = len(args.num_condition_keys)
        presence_keys.extend(args.num_condition_keys)
        with_conditioning = True
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
            "No data available for training \
                (dataset={}; keys={}; labels={})".format(
                    args.dataset_json,
                    args.image_keys,
                    args.label_keys))
    
    keys = args.image_keys

    network_config = yaml.load(args.config_file)
    network_config["batch_size"] = args.batch_size
    network_config["learning_rate"] = args.learning_rate
    network_config["with_conditioning"] = with_conditioning
    network_config["cross_attention_dim"] = 256 if with_conditioning else None
    
    all_pids = [k for k in data_dict]

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

    transforms_train = [
        *get_pre_transforms(**transform_pre_arguments),
        *get_post_transforms(**transform_post_arguments)]
    
    train_list = [data_dict[pid] for pid in all_pids]
    
    print("\tTrain set size={}".format(len(train_list)))

    ckpt_callback,ckpt_path,status = get_ckpt_callback(
        checkpoint_dir=args.checkpoint_dir,
        checkpoint_name=args.checkpoint_name,
        max_epochs=args.max_epochs,
        resume_from_last=args.resume_from_last,
        val_fold=None,
        monitor=args.monitor,
        metadata={"train_pids":all_pids,
                  "transform_arguments":{**transform_pre_arguments,
                                         **transform_post_arguments},
                  "categorical_specification": categorical_specification,
                  "numerical_specification": numerical_specification})
    ckpt = ckpt_callback is not None
    if status == "finished":
        exit()

    print(f"Number of cases: {len(train_list)}")
    
    # PL needs a little hint to detect GPUs.
    torch.ones([1]).to("cuda" if "cuda" in args.dev else "cpu")

    if network_config["spatial_dims"] == 2:
        transforms_train.append(
            RandomSlices(["image"],None,n=2,base=0.05))
        collate_fn = collate_last_slice
    else:
        collate_fn = safe_collate
    transforms_train = monai.transforms.Compose(transforms_train)
    transforms_train.set_random_state(args.seed)

    train_dataset = monai.data.CacheDataset(
        train_list,transforms_train,
        cache_rate=args.cache_rate,
        num_workers=args.n_workers)
        
    n_workers = args.n_workers // n_devices
    bs = network_config["batch_size"]
    real_bs = bs * n_devices
    if len(train_dataset) < real_bs:
        new_bs = len(train_dataset) // n_devices
        print(
            f"Batch size changed from {bs} to {new_bs} (dataset too small)")
        bs = new_bs
        real_bs = bs * n_devices

    def train_loader_call():
        return monai.data.ThreadDataLoader(
            train_dataset,batch_size=bs,
            shuffle=True,num_workers=n_workers,generator=g,
            collate_fn=collate_fn,pin_memory=True,
            persistent_workers=args.n_workers>0,
            drop_last=True)

    train_loader = train_loader_call()
    
    network = get_generative_network(
        network_config=network_config,
        categorical_specification=categorical_specification,
        numerical_specification=numerical_specification,
        train_loader_call=train_loader_call,
        max_epochs=args.max_epochs,
        warmup_steps=args.warmup_steps,
        start_decay=args.start_decay,
        diffusion_steps=1000)

    if args.checkpoint is not None:
        checkpoint = args.checkpoint
        load_checkpoint_to_model(network,checkpoint,
                                    args.exclude_from_state_dict)

    conditional_parameter_freezing(
        network,args.freeze_regex,args.not_freeze_regex)

    # instantiate callbacks and loggers
    callbacks = [RichProgressBar()]
        
    if ckpt_callback is not None:   
        callbacks.append(ckpt_callback)

    logger = get_logger(args.summary_name,args.summary_dir,
                        args.project_name,args.resume,
                        fold=None)
    
    if logger is not None:
        size = get_size(args.pad_size,args.crop_size)
        callbacks.append(
            LogImageFromDiffusionProcess(
                n_images=2,
                size=[int(x) for x in size][:network_config["spatial_dims"]]))
            
    trainer = Trainer(
        accelerator=accelerator,
        devices=devices,logger=logger,callbacks=callbacks,
        max_epochs=args.max_epochs,
        enable_checkpointing=ckpt,
        gradient_clip_val=args.gradient_clip_val,
        strategy=strategy,
        accumulate_grad_batches=args.accumulate_grad_batches,
        check_val_every_n_epoch=args.check_val_every_n_epoch,
        precision=args.precision,
        deterministic="warn")

    trainer.fit(network,train_loader,train_loader,ckpt_path=ckpt_path)

    # assessing performance on validation set
    print("Validating...")
    
    if ckpt is True:
        ckpt_list = ["last","best"]
    else:
        ckpt_list = ["last"]
    for ckpt_key in ckpt_list:
        test_metrics = trainer.test(
            network,train_loader,ckpt_path=ckpt_key)[0]
        for k in test_metrics:
            out = test_metrics[k]
            if isinstance(out,float) is False:
                value = float(out.detach().numpy())
            else:
                value = out
            output_file.write(f'{value}\n')
