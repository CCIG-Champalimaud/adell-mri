import argparse
import random
import json
import numpy as np
import torch
import monai

from lightning.pytorch import Trainer
from lightning.pytorch.callbacks import RichProgressBar

import sys
sys.path.append(r"..")
from lib.utils import (
    safe_collate,
    conditional_parameter_freezing,
    RandomSlices,
    collate_last_slice)
from lib.utils.pl_utils import (
    get_ckpt_callback,
    get_logger,
    get_devices,
    delete_checkpoints,
    LogImageFromDiffusionProcess)
from lib.utils.torch_utils import load_checkpoint_to_model
from lib.utils.dataset_filters import (
    filter_dictionary_with_filters,
    filter_dictionary_with_possible_labels,
    filter_dictionary_with_presence)
from lib.monai_transforms import (
    get_pre_transforms_generation as get_pre_transforms,
    get_post_transforms_generation as get_post_transforms)
from lib.modules.config_parsing import parse_config_diffusion_unet
from lib.utils.network_factories import get_generative_network
from lib.utils.parser import get_params,merge_args,parse_ids

def get_size(*size_list):
    for size in size_list:
        if size is not None:
            return size

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
        '--label_keys',dest='label_keys',type=str,default=None,
        help="Label keys in the dataset JSON for class guidance.")
    parser.add_argument(
        '--filter_on_keys',dest='filter_on_keys',type=str,default=[],nargs="+",
        help="Filters the dataset based on a set of specific key:value pairs.")
    parser.add_argument(
        '--possible_labels',dest='possible_labels',type=str,nargs='+',default=[],
        help="All the possible labels in the data.")
    parser.add_argument(
        '--positive_labels',dest='positive_labels',type=str,nargs='+',
        help="Labels that should be considered positive (binarizes labels)",
        default=None)
    parser.add_argument(
        '--label_groups',dest='label_groups',type=str,nargs='+',
        help="Label groups for classification.",
        default=None)
    parser.add_argument(
        '--cache_rate',dest='cache_rate',type=float,
        help="Rate of samples to be cached",default=1.0)
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
        '--subsample_training_data',dest='subsample_training_data',type=float,
        help="Subsamples training data by this fraction (for learning curves)",
        default=None)
    parser.add_argument(
        '--val_from_train',dest='val_from_train',default=None,type=float,
        help="Uses this fraction of training data as a validation set \
            during training")

    # network + training
    parser.add_argument(
        '--config_file',dest="config_file",
        help="Path to network configuration file (yaml)",
        required=True)
    parser.add_argument(
        '--net_type',dest='net_type',
        help="Classification type.",
        choices=["cat","ord","unet","vit","factorized_vit", "vgg"],default="cat")
    
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
        '--max_epochs',dest="max_epochs",
        help="Maximum number of training epochs",default=100,type=int)
    parser.add_argument(
        '--check_val_every_n_epoch',dest="check_val_every_n_epoch",
        help="Validation check frequency",default=5,type=int)
    parser.add_argument(
        '--n_folds',dest="n_folds",
        help="Number of validation folds",default=5,type=int)
    parser.add_argument(
        '--folds',dest="folds",type=str,default=None,nargs="+",
        help="Comma-separated IDs to be used in each space-separated fold")
    parser.add_argument(
        '--excluded_ids',dest='excluded_ids',type=str,default=None,nargs="+",
        help="Comma separated list of IDs to exclude.")
    parser.add_argument(
        '--checkpoint_dir',dest='checkpoint_dir',type=str,default=None,
        help='Path to directory where checkpoints will be saved.')
    parser.add_argument(
        '--checkpoint_name',dest='checkpoint_name',type=str,default=None,
        help='Checkpoint ID.')
    parser.add_argument(
        '--checkpoint',dest='checkpoint',type=str,default=None,
        nargs="+",help='Resumes training from this checkpoint.')
    parser.add_argument(
        '--freeze_regex',dest='freeze_regex',type=str,default=None,nargs="+",
        help='Matches parameter names and freezes them.')
    parser.add_argument(
        '--not_freeze_regex',dest='not_freeze_regex',type=str,default=None,
        nargs="+",help='Matches parameter names and skips freezing them (\
            overrides --freeze_regex)')
    parser.add_argument(
        '--exclude_from_state_dict',dest='exclude_from_state_dict',type=str,
        default=None,nargs="+",
            help='Regex to exclude parameters from state dict in --checkpoint')
    parser.add_argument(
        '--delete_checkpoints',dest='delete_checkpoints',action="store_true",
        help='Deletes checkpoints after training (keeps only metrics).')
    parser.add_argument(
        '--monitor',dest='monitor',type=str,default="val_loss",
        help="Metric that is monitored to determine the best checkpoint.")
    parser.add_argument(
        '--resume_from_last',dest='resume_from_last',action="store_true",
        help="Resumes from the last checkpoint stored for a given fold.")
    parser.add_argument(
        '--summary_dir',dest='summary_dir',type=str,default="summaries",
        help='Path to summary directory (for wandb).')
    parser.add_argument(
        '--summary_name',dest='summary_name',type=str,default="model_x",
        help='Summary name.')
    parser.add_argument(
        '--metric_path',dest='metric_path',type=str,default="metrics.csv",
        help='Path to file with CV metrics + information.')
    parser.add_argument(
        '--project_name',dest='project_name',type=str,default=None,
        help='Wandb project name.')
    parser.add_argument(
        '--resume',dest='resume',type=str,default="allow",
        choices=["allow","must","never","auto","none"],
        help='Whether wandb project should be resumed (check \
            https://docs.wandb.ai/ref/python/init for more details).')
    parser.add_argument(
        '--warmup_steps',dest='warmup_steps',type=float,default=0.0,
        help="Number of warmup steps (if SWA is triggered it starts after\
            this number of steps).")
    parser.add_argument(
        '--start_decay',dest='start_decay',type=float,default=None,
        help="Step at which decay starts. Defaults to starting right after \
            warmup ends.")
    parser.add_argument(
        '--gradient_clip_val',dest="gradient_clip_val",
        help="Value for gradient clipping",
        default=0.0,type=float)
    parser.add_argument(
        '--dropout_param',dest='dropout_param',type=float,
        help="Parameter for dropout.",default=0.1)
    parser.add_argument(
        '--accumulate_grad_batches',dest="accumulate_grad_batches",
        help="Number batches to accumulate before backpropgating gradient",
        default=1,type=int)
    parser.add_argument(
        '--batch_size',dest='batch_size',type=int,default=None,
        help="Overrides batch size in config file")
    parser.add_argument(
        '--learning_rate',dest='learning_rate',type=float,default=None,
        help="Overrides learning rate in config file")

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
    n_devices = len(devices) if isinstance(devices,list) else devices
    n_devices = 1 if isinstance(devices,str) else n_devices

    output_file = open(args.metric_path,'w')

    data_dict = json.load(open(args.dataset_json,'r'))
    label_keys = args.label_keys
    if args.label_keys is not None:
        presence_keys = [*args.image_keys,label_keys]
        label_groups = None
        if args.label_groups is not None:
            n_classes = len(args.label_groups)
            label_groups = [label_group.split(",")
                            for label_group in args.label_groups]
        elif args.positive_labels is None:
            n_classes = len(args.possible_labels)
        else:
            n_classes = 2
    else:
        presence_keys = args.image_keys
        n_classes = 0
    if args.excluded_ids is not None:
        args.excluded_ids = parse_ids(args.excluded_ids,
                                      output_format="list")
        print("Removing IDs specified in --excluded_ids")
        prev_len = len(data_dict)
        data_dict = {k:data_dict[k] for k in data_dict
                     if k not in args.excluded_ids}
        print("\tRemoved {} IDs".format(prev_len - len(data_dict)))
    if label_keys is not None:
        data_dict = filter_dictionary_with_possible_labels(
            data_dict,args.possible_labels,args.label_keys)
    if len(args.filter_on_keys) > 0:
        data_dict = filter_dictionary_with_filters(
            data_dict,args.filter_on_keys)
    data_dict = filter_dictionary_with_presence(
        data_dict,presence_keys)
    if args.subsample_size is not None and len(data_dict) > args.subsample_size:
        if label_keys is not None:
            strata = {}
            for k in data_dict:
                label = data_dict[k][args.label_keys]
                if label not in strata:
                    strata[label] = []
                strata[label].append(k)
            strata = {k:strata[k] for k in sorted(strata.keys())}
            ps = [len(strata[k]) / len(data_dict) for k in strata]
            split = [int(p * args.subsample_size) for p in ps]
            ss = []
            for k,s in zip(strata,split):
                ss.extend(
                    rng.choice(strata[k],size=s,
                               replace=False,shuffle=False))
        else:
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

    network_config = parse_config_diffusion_unet(args.config_file,
                                                 len(keys),
                                                 n_classes=n_classes)
    
    if args.batch_size is not None:
        network_config["batch_size"] = args.batch_size
    if args.learning_rate is not None:
        network_config["learning_rate"] = args.learning_rate

    if "batch_size" not in network_config:
        network_config["batch_size"] = 1
    
    all_pids = [k for k in data_dict]

    print("Setting up transforms...")
    label_mode = "binary" if n_classes == 2 and label_groups is None else "cat"
    transform_pre_arguments = {
        "keys":keys,
        "target_spacing":args.target_spacing,
        "crop_size":args.crop_size,
        "pad_size":args.pad_size}
    transform_post_arguments = {
        "keys":keys,
        "possible_labels":args.possible_labels,
        "positive_labels":args.positive_labels,
        "label_groups":args.label_groups,
        "label_key":label_keys,
        "label_mode":label_mode}

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
        val_fold=0,
        monitor=args.monitor,
        metadata={"train_pids":all_pids,
                  "transform_arguments":{**transform_pre_arguments,
                                         **transform_post_arguments}})
    ckpt = ckpt_callback is not None
    if status == "finished":
        exit()

    if label_keys is not None:
        classes = []
        for p in train_list:
            P = str(p[args.label_keys])
            if isinstance(P,list) or isinstance(P,tuple):
                P = max(P)
            classes.append(P)
        U,C = np.unique(classes,return_counts=True)
        for u,c in zip(U,C):
            print(f"Number of {u} cases: {c}")
    else:
        print(f"Number of cases: {len(train_list)}")
    
    # PL needs a little hint to detect GPUs.
    torch.ones([1]).to("cuda" if "cuda" in args.dev else "cpu")

    if network_config["spatial_dimensions"] == 2:
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
        label_key="label" if label_keys else None,
        train_loader_call=train_loader_call,
        max_epochs=args.max_epochs,
        warmup_steps=args.warmup_steps,
        start_decay=args.start_decay,
        size=[int(x) for x in get_size(args.pad_size,args.crop_size)][:network_config["spatial_dimensions"]])

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
                        fold=0)
    
    if logger is not None:
        callbacks.append(
            LogImageFromDiffusionProcess(n_images=8))
            
    trainer = Trainer(
        accelerator=accelerator,
        devices=devices,logger=logger,callbacks=callbacks,
        max_epochs=args.max_epochs,
        enable_checkpointing=ckpt,
        gradient_clip_val=args.gradient_clip_val,
        strategy=strategy,
        accumulate_grad_batches=args.accumulate_grad_batches,
        check_val_every_n_epoch=args.check_val_every_n_epoch,
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
            if n_classes == 2:
                try:
                    value = float(out.detach().numpy())
                except Exception:
                    value = float(out)
                x = "{},{},{},{},{}".format(k,ckpt_key,0,0,value)
                output_file.write(x+'\n')
                print(x)
            else:
                for i,v in enumerate(out):
                    x = "{},{},{},{},{}".format(k,ckpt_key,0,i,v)
                    output_file.write(x+'\n')
                    print(x)
    
    if args.delete_checkpoints == True:
        delete_checkpoints(trainer)