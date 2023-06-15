import argparse
import random
import json
import numpy as np
import torch
import monai
from sklearn.model_selection import train_test_split,StratifiedKFold

from lightning.pytorch import Trainer
from lightning.pytorch.callbacks import (
    EarlyStopping,StochasticWeightAveraging,RichProgressBar)

import sys
sys.path.append(r"..")
from lib.utils import (
    safe_collate,set_classification_layer_bias)
from lib.utils.pl_utils import get_ckpt_callback,get_logger,get_devices
from lib.utils.dataset_filters import (
    filter_dictionary_with_filters,
    filter_dictionary_with_possible_labels,
    filter_dictionary_with_presence)
from lib.monai_transforms import get_transforms_classification as get_transforms
from lib.monai_transforms import get_augmentations_class as get_augmentations
from lib.modules.losses import OrdinalSigmoidalLoss
from lib.modules.config_parsing import parse_config_unet,parse_config_cat
from lib.utils.network_factories import get_classification_network
from lib.utils.parser import get_params,merge_args,parse_ids

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
        '--t2_keys',dest='t2_keys',type=str,nargs='+',
        help="Image keys corresponding to T2.",default=None)
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
        '--cache_rate',dest='cache_rate',type=float,
        help="Rate of samples to be cached",
        default=1.0)
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
        '--augment',dest='augment',type=str,nargs="+",
        help="Use data augmentations",default=[])
    parser.add_argument(
        '--label_smoothing',dest="label_smoothing",
        help="Label smoothing value",default=None,type=float)
    parser.add_argument(
        '--mixup_alpha',dest="mixup_alpha",
        help="Alpha for mixup",default=None,type=float)
    parser.add_argument(
        '--partial_mixup',dest="partial_mixup",
        help="Applies mixup only to this fraction of the batch",
        default=None,type=float)
    parser.add_argument(
        '--max_epochs',dest="max_epochs",
        help="Maximum number of training epochs",default=100,type=int)
    parser.add_argument(
        '--n_folds',dest="n_folds",
        help="Number of validation folds",default=5,type=int)
    parser.add_argument(
        '--folds',dest="folds",type=str,default=None,nargs="+",
        help="Comma-separated IDs to be used in each space-separated fold")
    parser.add_argument(
        '--excluded_ids',dest='excluded_ids',type=str,default=None,
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
        '--early_stopping',dest='early_stopping',type=int,default=None,
        help="No. of checks before early stop (defaults to no early stop).")
    parser.add_argument(
        '--warmup_steps',dest='warmup_steps',type=float,default=0.0,
        help="Number of warmup steps (if SWA is triggered it starts after\
            this number of steps).")
    parser.add_argument(
        '--start_decay',dest='start_decay',type=float,default=None,
        help="Step at which decay starts. Defaults to starting right after \
            warmup ends.")
    parser.add_argument(
        '--swa',dest='swa',action="store_true",
        help="Use stochastic gradient averaging.",default=False)
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
    # strategies to handle imbalanced classes
    parser.add_argument(
        '--weighted_sampling',dest='weighted_sampling',action="store_true",
        help="Samples according to class proportions.",default=False)
    parser.add_argument(
        '--correct_classification_bias',dest='correct_classification_bias',
        action="store_true",default=False,
        help="Sets the final classification bias to log(pos/neg).")
    parser.add_argument(
        '--class_weights',dest='class_weights',type=str,nargs='+',
        help="Class weights (by alphanumeric order).",default=None)
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
    if args.subsample_size is not None and len(data_dict) > args.subsample_size:
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
            "No data available for training \
                (dataset={}; keys={}; labels={})".format(
                    args.dataset_json,
                    args.image_keys,
                    args.label_keys))
    
    keys = args.image_keys
    adc_keys = args.adc_keys if args.adc_keys is not None else []
    t2_keys = args.t2_keys if args.t2_keys is not None else []
    adc_keys = [k for k in adc_keys if k in keys]
    t2_keys = [k for k in t2_keys if k in keys]

    if args.net_type == "unet":
        network_config,_ = parse_config_unet(args.config_file,
                                             len(keys),n_classes)
    else:
        network_config = parse_config_cat(args.config_file)
    
    if args.batch_size is not None:
        network_config["batch_size"] = args.batch_size
    if args.learning_rate is not None:
        network_config["learning_rate"] = args.learning_rate

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
    augment_arguments = {
        "augment":args.augment,
        "t2_keys":t2_keys,
        "all_keys":keys,
        "image_keys":keys,
        "intp_resampling_augmentations":["bilinear" for _ in keys]}

    transforms_train = monai.transforms.Compose([
        *get_transforms("pre",**transform_arguments),
        get_augmentations(**augment_arguments),
        *get_transforms("post",**transform_arguments)])
    transforms_train.set_random_state(args.seed)

    transforms_val = monai.transforms.Compose([
        *get_transforms("pre",**transform_arguments),
        *get_transforms("post",**transform_arguments)])
    
    if args.folds is None:
        if args.n_folds > 1:
            fold_generator = StratifiedKFold(
                args.n_folds,shuffle=True,random_state=args.seed).split(
                    all_pids,all_classes)
        else:
            fold_generator = iter(
                [train_test_split(range(len(all_pids)),test_size=0.2)])
    else:
        args.folds = parse_ids(args.folds)
        folds = []
        for fold_idx,val_ids in enumerate(args.folds):
            train_idxs = [i for i,x in enumerate(all_pids) if x not in val_ids]
            val_idxs = [i for i,x in enumerate(all_pids) if x in val_ids]
            if len(train_idxs) == 0:
                print("No train samples in fold {}".format(fold_idx))
                continue
            if len(val_idxs) == 0:
                print("No val samples in fold {}".format(fold_idx))
                continue
            folds.append([train_idxs,val_idxs])
        args.n_folds = len(folds)
        fold_generator = iter(folds)

    for val_fold in range(args.n_folds):
        train_idxs,val_idxs = next(fold_generator)
        train_pids = [all_pids[i] for i in train_idxs]
        if args.subsample_training_data is not None:
            train_pids = rng.choice(
                train_pids,
                size=int(len(train_pids)*args.subsample_training_data),
                replace=False)
        val_pids = [all_pids[i] for i in val_idxs]
        if args.val_from_train is not None:
            n_train_val = int(len(train_pids)*args.val_from_train)
            train_val_pids = rng.choice(
                train_pids,n_train_val,replace=False)
            train_pids = [pid for pid in train_pids 
                          if pid not in train_val_pids]
        else:
            train_val_pids = val_pids
        train_list = [data_dict[pid] for pid in train_pids]
        train_val_list = [data_dict[pid] for pid in train_val_pids]
        val_list = [data_dict[pid] for pid in val_pids]
        
        print("Current fold={}".format(val_fold))
        print("\tTrain set size={}".format(len(train_idxs)))
        print("\tTrain validation set size={}".format(len(train_val_pids)))
        print("\tValidation set size={}".format(len(val_idxs)))
        
        if len(clinical_feature_keys) > 0:
            clinical_feature_values = [[train_sample[k]
                                        for train_sample in train_list]
                                       for k in clinical_feature_keys]
            clinical_feature_values = np.array(clinical_feature_values,
                                               dtype=np.float32)
            clinical_feature_means = np.mean(clinical_feature_values,axis=1)
            clinical_feature_stds = np.std(clinical_feature_values,axis=1)
        else:
            clinical_feature_means = None
            clinical_feature_stds = None
        
        ckpt_callback,ckpt_path,status = get_ckpt_callback(
            checkpoint_dir=args.checkpoint_dir,
            checkpoint_name=args.checkpoint_name,
            max_epochs=args.max_epochs,
            resume_from_last=args.resume_from_last,
            val_fold=val_fold,
            monitor=args.monitor,
            metadata={"train_pids":train_pids,
                      "transform_arguments":transform_arguments})
        ckpt = ckpt_callback is not None
        if status == "finished":
            continue
        
        train_dataset = monai.data.CacheDataset(
            train_list,transforms_train,
            cache_rate=args.cache_rate,
            num_workers=args.n_workers)
        train_dataset_val = monai.data.CacheDataset(
            train_val_list,transforms_val,
            cache_rate=args.cache_rate,
            num_workers=args.n_workers)
        validation_dataset = monai.data.Dataset(
            val_list,transforms_val)

        classes = []
        for p in train_list:
            P = str(p[args.label_keys])
            if isinstance(P,list) or isinstance(P,tuple):
                P = max(P)
            classes.append(P)
        U,C = np.unique(classes,return_counts=True)
        for u,c in zip(U,C):
            print("Number of {} cases: {}".format(u,c))
        if args.weighted_sampling is True:
            weights = {k:0 for k in args.possible_labels}
            for c in classes:
                if c in weights:
                    weights[c] += 1
            weight_sum = np.sum([weights[c] for c in args.possible_labels])
            weights = {
                k:weight_sum/(1+weights[k]*len(weights)) for k in weights}
            weight_vector = np.array([weights[k] for k in classes])
            weight_vector = np.where(weight_vector < 0.25,0.25,weight_vector)
            weight_vector = np.where(weight_vector > 4,4,weight_vector)
            sampler = torch.utils.data.WeightedRandomSampler(
                weight_vector,len(weight_vector),generator=g)
        else:
            sampler = None
        
        # PL needs a little hint to detect GPUs.
        torch.ones([1]).to("cuda" if "cuda" in args.dev else "cpu")
        
        # get class weights if necessary  
        class_weights = None
        if args.class_weights is not None:
            if args.class_weights[0] == "adaptive":
                if n_classes == 2:
                    pos = len([x for x in classes 
                               if x in args.positive_labels])
                    neg = len(classes) - pos
                    weight_neg = (1 / neg) * (len(classes) / 2.0)
                    weight_pos = (1 / pos) * (len(classes) / 2.0)
                    class_weights = weight_pos/weight_neg
                    class_weights = torch.as_tensor(
                        np.array(class_weights),device=args.dev.split(":")[0],
                        dtype=torch.float32)
                else:
                    pos = {k:0 for k in args.possible_labels}
                    for c in classes:
                        pos[c] += 1
                    pos = np.array([pos[k] for k in pos])
                    class_weights = (1 / pos) * (len(classes) / 2.0)
                    class_weights = torch.as_tensor(
                        np.array(class_weights),device=args.dev.split(":")[0],
                        dtype=torch.float32)
            else:
                class_weights = [float(x) for x in args.class_weights]
                if class_weights is not None:
                    class_weights = torch.as_tensor(
                        np.array(class_weights),device=args.dev.split(":")[0],
                        dtype=torch.float32)
                
        print("Initializing loss with class_weights: {}".format(class_weights))
        if n_classes == 2:
            network_config["loss_fn"] = torch.nn.BCEWithLogitsLoss(class_weights)
        elif args.net_type == "ord":
            network_config["loss_fn"] = OrdinalSigmoidalLoss(class_weights,n_classes)
        else:
            network_config["loss_fn"] = torch.nn.CrossEntropy(class_weights)
        
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
                shuffle=sampler is None,num_workers=n_workers,generator=g,
                collate_fn=safe_collate,pin_memory=True,
                sampler=sampler,persistent_workers=args.n_workers>0,
                drop_last=True)

        train_loader = train_loader_call()
        train_val_loader = monai.data.ThreadDataLoader(
            train_dataset_val,batch_size=network_config["batch_size"],
            shuffle=False,num_workers=n_workers,
            collate_fn=safe_collate)
        validation_loader = monai.data.ThreadDataLoader(
            validation_dataset,batch_size=network_config["batch_size"],
            shuffle=False,num_workers=n_workers,
            collate_fn=safe_collate)

        network = get_classification_network(
            net_type=args.net_type,
            network_config=network_config,
            dropout_param=args.dropout_param,
            seed=args.seed,
            n_classes=n_classes,
            keys=keys,
            clinical_feature_keys=clinical_feature_keys,
            train_loader_call=train_loader_call,
            max_epochs=args.max_epochs,
            warmup_steps=args.warmup_steps,
            start_decay=args.start_decay,
            crop_size=args.crop_size,
            clinical_feature_means=clinical_feature_means,
            clinical_feature_stds=clinical_feature_stds,
            label_smoothing=args.label_smoothing,
            mixup_alpha=args.mixup_alpha,
            partial_mixup=args.partial_mixup)

        # instantiate callbacks and loggers
        callbacks = [RichProgressBar()]
        if args.early_stopping is not None:
            early_stopping = EarlyStopping(
                'val_loss',patience=args.early_stopping,
                strict=True,mode="min")
            callbacks.append(early_stopping)
            
        if ckpt_callback is not None:   
            callbacks.append(ckpt_callback)

        if args.swa is True:
            swa_callback = StochasticWeightAveraging(
                network_config["learning_rate"],swa_epoch_start=args.warmup_steps)
            callbacks.append(swa_callback)

        logger = get_logger(args.summary_name,args.summary_dir,
                            args.project_name,args.resume,
                            fold=val_fold)
        
        if args.correct_classification_bias is True and n_classes == 2:
            pos = len([x for x in classes if x in args.positive_labels])
            neg = len(classes) - pos
            set_classification_layer_bias(pos,neg,network)
        
        trainer = Trainer(
            accelerator=accelerator,
            devices=devices,logger=logger,callbacks=callbacks,
            max_epochs=args.max_epochs,
            enable_checkpointing=ckpt,
            gradient_clip_val=args.gradient_clip_val,
            strategy=strategy,
            accumulate_grad_batches=args.accumulate_grad_batches,
            check_val_every_n_epoch=1,
            deterministic="warn")

        trainer.fit(network,train_loader,train_val_loader,ckpt_path=ckpt_path)

        # assessing performance on validation set
        print("Validating...")
        
        if ckpt is True:
            ckpt_list = ["last","best"]
        else:
            ckpt_list = ["last"]
        for ckpt_key in ckpt_list:
            test_metrics = trainer.test(
                network,validation_loader,ckpt_path=ckpt_key)[0]
            for k in test_metrics:
                out = test_metrics[k]
                if n_classes == 2:
                    try:
                        value = float(out.detach().numpy())
                    except Exception:
                        value = float(out)
                    x = "{},{},{},{},{}".format(k,ckpt_key,val_fold,0,value)
                    output_file.write(x+'\n')
                    print(x)
                else:
                    for i,v in enumerate(out):
                        x = "{},{},{},{},{}".format(k,ckpt_key,val_fold,i,v)
                        output_file.write(x+'\n')
                        print(x)
