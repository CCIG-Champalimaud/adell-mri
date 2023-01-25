import argparse
import random
import json
import numpy as np
import torch
import monai
from copy import deepcopy
from sklearn.model_selection import train_test_split,StratifiedKFold

from pytorch_lightning import Trainer
from pytorch_lightning.callbacks import (
    EarlyStopping,StochasticWeightAveraging,RichProgressBar)

import sys
sys.path.append(r"..")
from lib.utils import (
    safe_collate,set_classification_layer_bias)
from lib.pl_utils import get_ckpt_callback,get_logger,get_devices
from lib.batch_preprocessing import BatchPreprocessing
from lib.monai_transforms import get_transforms_classification as get_transforms
from lib.monai_transforms import get_augmentations_class as get_augmentations
from lib.modules.classification.classification import (UNetEncoder)
from lib.modules.classification.pl import (
    ClassNetPL,UNetEncoderPL,GenericEnsemblePL,ViTClassifierPL,
    FactorizedViTClassifierPL)
from lib.modules.layers.adn_fn import get_adn_fn
from lib.modules.losses import OrdinalSigmoidalLoss
from lib.modules.config_parsing import parse_config_unet,parse_config_cat

def filter_dictionary_with_presence(D,filters):
    print("Filtering on: {} presence".format(filters))
    print("\tInput size: {}".format(len(D)))
    out_dict = {}
    for pid in D:
        check = True
        for k in filters:
            if k not in D[pid]:
                check = False
        if check == True:
            out_dict[pid] = D[pid]
    print("\tOutput size: {}".format(len(out_dict)))
    return out_dict

def filter_dictionary_with_filters(D,filters):
    print("Filtering on: {}".format(filters))
    print("\tInput size: {}".format(len(D)))
    filters = [f.split(":") for f in filters]
    out_dict = {}
    for pid in D:
        check = True
        for k,v in filters:
            if k in D[pid]:
                if D[pid][k] != v:
                    check = False
            else:
                check = False
        if check == True:
            out_dict[pid] = D[pid]
    print("\tOutput size: {}".format(len(out_dict)))
    return out_dict

if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    # data
    parser.add_argument(
        '--dataset_json',dest='dataset_json',type=str,
        help="JSON containing dataset information",required=True)
    parser.add_argument(
        '--image_keys',dest='image_keys',type=str,nargs='+',
        help="Image keys in the dataset JSON.",
        required=True)
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
        '--branched',dest='branched',action="store_true",
        help="Uses one encoder for each image key.",default=False)
    parser.add_argument(
        '--batch_size',dest='batch_size',type=int,default=None,
        help="Overrides batch size in config file")
    parser.add_argument(
        '--learning_rate',dest='learning_rate',type=float,default=None,
        help="Overrides learning rate in config file")

    # network + training
    parser.add_argument(
        '--config_file',dest="config_file",
        help="Path to network configuration file (yaml)",
        required=True)
    parser.add_argument(
        '--net_type',dest='net_type',
        help="Classification type. Can be categorical (cat) or ordinal (ord)",
        choices=["cat","ord","unet","vit","factorized_vit"],default="cat")
    
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
        '--warmup_steps',dest='warmup_steps',type=int,default=0,
        help="Number of warmup steps (if SWA is triggered it starts after\
            this number of steps).")
    parser.add_argument(
        '--start_decay',dest='start_decay',type=int,default=None,
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

    args = parser.parse_args()

    torch.manual_seed(args.seed)
    random.seed(args.seed)
    np.random.seed(args.seed)
    g = torch.Generator()
    g.manual_seed(args.seed)
    rng = np.random.default_rng(args.seed)

    accelerator,devices,strategy = get_devices(args.dev)
    
    output_file = open(args.metric_path,'w')

    data_dict = json.load(open(args.dataset_json,'r'))
    if len(args.filter_on_keys) > 0:
        data_dict = filter_dictionary_with_filters(
            data_dict,args.filter_on_keys)
    data_dict = filter_dictionary_with_presence(
        data_dict,args.image_keys + [args.label_keys])
    if args.subsample_size is not None:
        strata = {}
        for k in data_dict:
            l = data_dict[k][args.label_keys]
            if l not in strata:
                strata[l] = []
            strata[l].append(k)
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
            "No data available for training \
                (dataset={}; keys={}; labels={})".format(
                    args.dataset_json,
                    args.image_keys,
                    args.label_keys()))
    
    keys = args.image_keys
    adc_keys = args.adc_keys if args.adc_keys is not None else []

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
        "adc_keys":adc_keys,
        "target_spacing":args.target_spacing,
        "crop_size":args.crop_size,
        "pad_size":args.pad_size,
        "branched":args.branched,
        "possible_labels":args.possible_labels,
        "positive_labels":args.positive_labels,
        "label_key":args.label_keys,
        "label_mode":label_mode}
    augment_arguments = {
        "augment":args.augment,
        "all_keys":keys,
        "image_keys":keys,
        "intp_resampling_augmentations":["bilinear" for _ in keys]}

    transforms_train = monai.transforms.Compose([
        *get_transforms("pre",**transform_arguments),
        *get_augmentations(**augment_arguments),
        *get_transforms("post",**transform_arguments)])
    transforms_train.set_random_state(args.seed)

    transforms_val = monai.transforms.Compose([
        *get_transforms("pre",**transform_arguments),
        *get_transforms("post",**transform_arguments)])
    
    transforms_train.set_random_state(args.seed)

    if args.folds is None:
        if args.n_folds > 1:
            fold_generator = StratifiedKFold(
                args.n_folds,shuffle=True,random_state=args.seed).split(
                    all_pids,all_classes)
        else:
            fold_generator = iter(
                [train_test_split(range(len(all_pids)),test_size=0.2)])
    else:
        args.n_folds = len(args.folds)
        folds = []
        for val_ids in args.folds:
            val_ids = val_ids.split(',')
            train_idxs = [i for i,x in enumerate(all_pids) if x not in val_ids]
            val_idxs = [i for i,x in enumerate(all_pids) if x in val_ids]
            folds.append([train_idxs,val_idxs])
        fold_generator = iter(folds)

    for val_fold in range(args.n_folds):
        train_idxs,val_idxs = next(fold_generator)
        train_pids = [all_pids[i] for i in train_idxs]
        val_pids = [all_pids[i] for i in val_idxs]
        train_list = [data_dict[pid] for pid in train_pids]
        val_list = [data_dict[pid] for pid in val_pids]
        
        train_dataset = monai.data.CacheDataset(
            train_list,transforms_train,
            cache_rate=args.cache_rate,
            num_workers=args.n_workers)
        train_dataset_val = monai.data.CacheDataset(
            val_list,transforms_val,
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
        if args.weighted_sampling == True:
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

        def train_loader_call(): 
            return monai.data.ThreadDataLoader(
                train_dataset,batch_size=network_config["batch_size"],
                shuffle=sampler is None,num_workers=args.n_workers,generator=g,
                collate_fn=safe_collate,pin_memory=True,sampler=sampler,
                persistent_workers=args.n_workers>0,drop_last=True)

        train_loader = train_loader_call()
        train_val_loader = monai.data.ThreadDataLoader(
            train_dataset_val,batch_size=network_config["batch_size"],
            shuffle=False,num_workers=args.n_workers,
            collate_fn=safe_collate)
        validation_loader = monai.data.ThreadDataLoader(
            validation_dataset,batch_size=network_config["batch_size"],
            shuffle=False,num_workers=args.n_workers,
            collate_fn=safe_collate)

        print("Setting up training...")
        if args.net_type == "unet":
            act_fn = network_config["activation_fn"]
        else:
            act_fn = "swish"
        adn_fn = get_adn_fn(3,"identity",act_fn=act_fn,
                            dropout_param=args.dropout_param)
        batch_preprocessing = BatchPreprocessing(
            args.label_smoothing,args.mixup_alpha,args.partial_mixup,args.seed)
        boilerplate_args = {
            "n_channels":len(keys),
            "n_classes":n_classes,
            "training_dataloader_call":train_loader_call,
            "image_key":"image",
            "label_key":"label",
            "n_epochs":args.max_epochs,
            "warmup_steps":args.warmup_steps,
            "training_batch_preproc":batch_preprocessing,
            "start_decay":args.start_decay}
        if args.branched == False:
            if args.net_type == "unet":
                network = UNetEncoderPL(
                    head_structure=[
                        network_config["depth"][-1] for _ in range(3)],
                    head_adn_fn=get_adn_fn(
                        1,"batch",act_fn="gelu",
                        dropout_param=args.dropout_param),
                    **boilerplate_args,
                    **network_config)
            elif "vit" in args.net_type:
                image_size = [int(x) for x in args.crop_size]
                network_config["image_size"] = image_size
                if args.net_type == "vit":
                    network = ViTClassifierPL(
                        adn_fn=get_adn_fn(
                            1,"identity",act_fn="gelu",
                            dropout_param=args.dropout_param),
                        **boilerplate_args,
                        **network_config)
                elif args.net_type == "factorized_vit":
                    for k in ["embed_method"]:
                        if k in network_config:
                            del network_config[k]
                    network = FactorizedViTClassifierPL(
                        adn_fn=get_adn_fn(
                            1,"identity",act_fn="gelu",
                            dropout_param=args.dropout_param),
                        **boilerplate_args,
                        **network_config)                    
                
            else:
                network = ClassNetPL(
                    net_type=args.net_type,n_channels=len(keys),
                    n_classes=n_classes,
                    training_dataloader_call=train_loader_call,
                    image_key="image",label_key="label",
                    adn_fn=adn_fn,n_epochs=args.max_epochs,
                    warmup_steps=args.warmup_steps,
                    **network_config)
        else:
            n_channels = 1
            if args.net_type == "unet":
                nc = deepcopy(network_config)
                if "n_channels" in nc:
                    del nc["n_channels"]
                networks = [
                    UNetEncoder(n_channels=n_channels,**nc)
                    for _ in args.image_keys]
            else:
                raise NotImplementedError("branched only works with net_type=='unet'")
            config = {}
            for k in network_config:
                if k in ["learning_rate","batch_size",
                         "spatial_dimensions","weight_decay"]:
                    config[k] = network_config[k]
            network = GenericEnsemblePL(
                image_keys=args.image_keys,label_key="label",
                n_classes=n_classes,training_dataloader_call=train_loader_call,
                head_structure=[network_config["depth"][-1] for _ in range(3)],
                head_adn_fn=get_adn_fn(
                    1,"batch",act_fn="swish",
                    dropout_param=args.dropout_param),
                n_epochs=args.max_epochs,
                **config)

        # instantiate callbacks and loggers
        callbacks = [RichProgressBar()]
        if args.early_stopping is not None:
            early_stopping = EarlyStopping(
                'val_loss',patience=args.early_stopping,
                strict=True,mode="min")
            callbacks.append(early_stopping)
        if args.swa == True:
            swa_callback = StochasticWeightAveraging(
                network_config["learning_rate"],swa_epoch_start=args.warmup_steps)
            callbacks.append(swa_callback)
        ckpt_callback,ckpt_path,status = get_ckpt_callback(
            args.checkpoint_dir,args.checkpoint_name,
            val_fold,args.max_epochs,args.resume_from_last,
            monitor=args.monitor)
        if ckpt_callback is not None:   
            callbacks.append(ckpt_callback)
        ckpt = ckpt_callback is not None
        if status == "finished":
            continue
        logger = get_logger(args.summary_name,args.summary_dir,
                            args.project_name,args.resume,
                            fold=val_fold)
        
        if args.correct_classification_bias == True and n_classes == 2:
            pos = len([x for x in classes if x in args.positive_labels])
            neg = len(classes) - pos
            set_classification_layer_bias(pos,neg,network)
        
        trainer = Trainer(
            accelerator=accelerator,
            devices=devices,logger=logger,callbacks=callbacks,
            max_epochs=args.max_epochs,
            enable_checkpointing=ckpt,
            resume_from_checkpoint=ckpt_path,
            gradient_clip_val=args.gradient_clip_val,
            strategy=strategy,
            accumulate_grad_batches=args.accumulate_grad_batches,
            check_val_every_n_epoch=1)

        trainer.fit(network,train_loader,train_val_loader)

        # assessing performance on validation set
        print("Validating...")
        
        if ckpt == True:
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
                    except:
                        value = float(out)
                    x = "{},{},{},{},{}".format(k,ckpt_key,val_fold,0,value)
                    output_file.write(x+'\n')
                    print(x)
                else:
                    for i,v in enumerate(out):
                        x = "{},{},{},{},{}".format(k,ckpt_key,val_fold,i,v)
                        output_file.write(x+'\n')
                        print(x)
