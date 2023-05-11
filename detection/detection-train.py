import argparse
import random
import yaml
import numpy as np
import torch
import monai
import json
from sklearn.model_selection import KFold,train_test_split

from pytorch_lightning import Trainer
from pytorch_lightning.callbacks import RichProgressBar

import sys
sys.path.append(r"..")
from lib.utils import (
    safe_collate,
    load_anchors)
from lib.monai_transforms import (
    get_transforms_detection_pre,
    get_transforms_detection_post)
from lib.modules.object_detection import YOLONet3d
from lib.utils.pl_utils import get_ckpt_callback,get_devices,get_logger
from lib.utils.dataset_filters import (
    filter_dictionary_with_filters,
    filter_dictionary_with_presence)
from lib.utils.sitk_utils import spacing_from_dataset_json
from lib.utils.detection import anchors_from_nested_list
from lib.utils.network_factories import get_detection_network

torch.backends.cudnn.benchmark = True

if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    # data
    parser.add_argument(
        '--dataset_json',dest='dataset_json',type=str,
        help="JSON containing dataset information",required=True)
    parser.add_argument(
        '--image_keys',dest='image_keys',type=str,nargs="+",
        help="Image keys in dataset JSON",required=True)
    parser.add_argument(
        '--box_key',dest='box_key',type=str,default="boxes",
        help="Box key in dataset JSON")
    parser.add_argument(
        '--box_class_key',dest='box_class_key',type=str,default="box_classes",
        help="Box class key in dataset JSON")
    parser.add_argument(
        '--shape_key',dest='shape_key',type=str,default="shape",
        help="Shape key in dataset JSON")
    parser.add_argument(
        '--mask_key',dest='mask_key',type=str,default=None,
        help="Mask key in dataset JSON")
    parser.add_argument(
        '--mask_mode',dest='mask_mode',type=str,default="mask_is_labels",
        choices=["mask_is_labels","infer_labels","single_object"],
        help="If using mask_key, defines how boxes are inferred. mask_is_labels\
            uses the mask as labels; infer_labels infers connected components \
            using skimage.measure.label; single_object assumes the mask represents\
            a single, not necessarily connected, object.")
    parser.add_argument(
        '--target_spacing',dest='target_spacing',type=str,nargs="+",
        help="Target spacing (if 'infer' then this is inferred from the training\
            set)")
    parser.add_argument(
        '--input_size',dest='input_size',type=float,nargs='+',
        help="Input size for network (for padding and cropping)",required=True)
    parser.add_argument(
        '--n_classes',dest='n_classes',type=int,default=2,
        help="Number of classes")
    parser.add_argument(
        '--anchor_csv',dest='anchor_csv',type=str,
        help="Path to CSV file containing anchors",required=True)
    parser.add_argument(
        '--min_anchor_area',dest='min_anchor_area',type=float,
        help="Minimum anchor area (filters anchors)",default=None)
    parser.add_argument(
        '--filter_on_keys',dest='filter_on_keys',type=str,default=[],nargs="+",
        help="Filters the dataset based on a set of specific key:value pairs.")
    parser.add_argument(
        '--adc_keys',dest='adc_keys',type=str,nargs="+",
        help="Image keys corresponding to ADC in dataset JSON")
    parser.add_argument(
        '--t2_keys',dest='t2_keys',type=str,nargs="+",
        help="Image keys corresponding to T2 in dataset JSON")

    # network
    parser.add_argument(
        '--config_file',dest="config_file",
        help="Path to network configuration file (yaml)",
        required=True)
    parser.add_argument(
        '--net_type',dest="net_type",
        help="Network type",choices=["yolo"],required=True)
    
    # training
    parser.add_argument('--dev',dest='dev',default="cpu",
        help="Device for PyTorch training",type=str)
    parser.add_argument(
        '--seed',dest='seed',help="Random seed",default=42,type=int)
    parser.add_argument(
        '--n_workers',dest='n_workers',
        help="No. of workers",default=0,type=int)
    parser.add_argument(
        '--augment',dest='augment',type=str,nargs="+",
        default=[],choices=["intensity","noise","rbf","rotate","trivial"],
        help="Use data augmentations (use rotate with care, not very stable \
            when objects are close to the border)")
    parser.add_argument(
        '--loss_gamma',dest="loss_gamma",
        help="Gamma for focal loss",default=2.0,type=float)
    parser.add_argument(
        '--loss_comb',dest="loss_comb",
        help="Relative weight for combined losses",default=0.5,type=float)
    parser.add_argument(
        '--max_epochs',dest="max_epochs",
        help="Maximum number of training epochs",default=100,type=int)
    parser.add_argument(
        '--warmup_steps',dest='warmup_steps',type=float,default=0.0,
        help="Number of warmup steps (if SWA is triggered it starts after\
            this number of steps).")
    parser.add_argument(
        '--n_folds',dest="n_folds",
        help="Number of validation folds",default=5,type=int)
    parser.add_argument(
        '--checkpoint_dir',dest='checkpoint_dir',type=str,default=None,
        help='Path to directory where checkpoints will be saved.')
    parser.add_argument(
        '--checkpoint_name',dest='checkpoint_name',type=str,default=None,
        help='Checkpoint ID.')
    parser.add_argument(
        '--resume_from_last',dest='resume_from_last',action="store_true",
        help="Resumes from the last checkpoint stored for a given fold.")
    parser.add_argument(
        '--monitor',dest='monitor',type=str,default="val_loss",
        help="Metric that is monitored to determine the best checkpoint.")
    parser.add_argument(
        '--project_name',dest='project_name',type=str,default=None,
        help='Wandb project name.')
    parser.add_argument(
        '--summary_dir',dest='summary_dir',type=str,default="summaries",
        help='Path to summary directory (for tensorboard).')
    parser.add_argument(
        '--summary_name',dest='summary_name',type=str,default="model_x",
        help='Summary name.')
    parser.add_argument(
        '--resume',dest='resume',type=str,default="allow",
        choices=["allow","must","never","auto","none"],
        help='Whether wandb project should be resumed (check \
            https://docs.wandb.ai/ref/python/init for more details).')
    parser.add_argument(
        '--metric_path',dest='metric_path',type=str,default="metrics.csv",
        help='Path to file with CV metrics + information.')
    parser.add_argument(
        '--class_weights',dest='class_weights',type=float,nargs='+',
        help="Class weights (by alphanumeric order).",default=1.)
    parser.add_argument(
        '--iou_threshold',dest='iou_threshold',type=float,
        help="IoU threshold for pred-gt overlaps.",default=0.5)
    parser.add_argument(
        '--dropout_param',dest='dropout_param',type=float,
        help="Parameter for dropout.",default=0.0)
    parser.add_argument(
        '--subsample_size',dest='subsample_size',type=int,
        help="Subsamples data to a given size",default=None)

    args = parser.parse_args()

    torch.manual_seed(args.seed)
    random.seed(args.seed)
    np.random.seed(args.seed)
    g = torch.Generator()
    g.manual_seed(args.seed)
    
    if args.mask_key is not None:
        mode = "mask"
    else:
        mode = "boxes"
    
    accelerator,devices,strategy = get_devices(args.dev)

    output_file = open(args.metric_path,'w')

    with open(args.dataset_json,"r") as o:
        data_dict = json.load(o)
    if mode == "boxes":
        data_dict = filter_dictionary_with_presence(
            data_dict,args.image_keys + [args.box_key,args.box_class_key])
    else:
        data_dict = filter_dictionary_with_presence(
            data_dict,args.image_keys + [args.mask_key])        
    if len(args.filter_on_keys) > 0:
        data_dict = filter_dictionary_with_filters(
            data_dict,args.filter_on_keys)
    if args.subsample_size is not None and len(data_dict) > args.subsample_size:
        data_dict = {
            k:data_dict[k] for k in np.random.choice(list(data_dict.keys()),
                                                     args.subsample_size,
                                                     replace=False)}
    
    input_size = [int(i) for i in args.input_size]
    
    keys = args.image_keys
    adc_keys = args.adc_keys if args.adc_keys else []
    t2_keys = args.t2_keys if args.t2_keys else []
    box_key = args.box_key
    box_class_key = args.box_class_key
    shape_key = args.shape_key
    mask_key = args.mask_key
    target_spacing = args.target_spacing

    with open(args.config_file,'r') as o:
        network_config = yaml.safe_load(o)

    all_pids = [k for k in data_dict]
    if args.n_folds > 1:
        fold_generator = KFold(
            args.n_folds,shuffle=True,random_state=args.seed).split(all_pids)
    else:
        fold_generator = iter(
            [train_test_split(range(len(all_pids)),test_size=0.2)])

    for val_fold in range(args.n_folds):
        train_idxs,val_idxs = next(fold_generator)
        train_idxs,train_val_idxs = train_test_split(train_idxs,test_size=0.2)
        train_pids = [all_pids[i] for i in train_idxs]
        train_val_pids = [all_pids[i] for i in train_val_idxs]
        val_pids = [all_pids[i] for i in val_idxs]
        path_list_train = [data_dict[pid] for pid in train_pids]
        path_list_train_val = [data_dict[pid] for pid in train_val_pids]
        path_list_val = [data_dict[pid] for pid in val_pids]

        if target_spacing[0] == "infer":
            target_spacing = spacing_from_dataset_json(
                dataset_dict={k:data_dict[k] for k in train_pids},
                key=keys[0],
                quantile=0.5,
                n_workers=args.n_workers)
        else:
            target_spacing = [float(x) for x in target_spacing]

        transform_arguments_pre = {
            "keys":keys,
            "adc_keys":adc_keys,
            "input_size":input_size,
            "target_spacing":target_spacing,
            "box_class_key":box_class_key,
            "shape_key":shape_key,
            "box_key":box_key,
            "mask_key":mask_key,
            "mask_mode":args.mask_mode}

        transforms_train = get_transforms_detection_pre(
            **transform_arguments_pre)

        train_dataset = monai.data.CacheDataset(
            path_list_train,
            monai.transforms.Compose(transforms_train))

        if args.anchor_csv == "infer":
            anchor_array = anchors_from_nested_list(
                train_dataset,box_key,shape_key,
                args.iou_threshold)
        else:
            anchor_array = load_anchors(args.anchor_csv)
        if args.min_anchor_area is not None:
            print("Filtering anchor area (minimum area: {})".format(
                args.min_anchor_area))
            anchor_array = anchor_array[
                np.prod(anchor_array,1) > args.min_anchor_area]
            
        output_example = YOLONet3d(
            n_channels=1,n_classes=args.n_classes,adn_fn=torch.nn.Identity,
            anchor_sizes=anchor_array,dev=args.dev)(
                torch.ones([1,1,*input_size]))
        output_size = output_example[0].shape[2:]
            
        transform_arguments_post = {
            "keys":keys,
            "t2_keys":t2_keys,
            "anchor_array":anchor_array,
            "input_size":input_size,
            "output_size":output_size,
            "iou_threshold":args.iou_threshold,
            "box_class_key":box_class_key,
            "shape_key":shape_key,
            "box_key":box_key}
                
        transforms_train_val = [
            *get_transforms_detection_pre(**transform_arguments_pre),
            *get_transforms_detection_post(**transform_arguments_post,
                                           augments=[])]
        transforms_val = [
            *get_transforms_detection_pre(**transform_arguments_pre),
            *get_transforms_detection_post(**transform_arguments_post,
                                           augments=[])]
        
        train_dataset = monai.data.CacheDataset(
            [x for x in train_dataset],
            monai.transforms.Compose(
                get_transforms_detection_post(
                    **transform_arguments_post,augments=args.augment)))
        train_dataset_val = monai.data.CacheDataset(
            path_list_train_val,
            monai.transforms.Compose(transforms_train_val))
        validation_dataset = monai.data.CacheDataset(
            path_list_val,
            monai.transforms.Compose(transforms_val))
        
        class_weights = torch.as_tensor(args.class_weights)
        class_weights = class_weights.to(args.dev)

        train_loader_call = lambda: monai.data.ThreadDataLoader(
            train_dataset,batch_size=network_config["batch_size"],
            shuffle=True,num_workers=args.n_workers,generator=g,
            collate_fn=safe_collate,pin_memory=True,
            persistent_workers=args.n_workers>0)

        train_loader = train_loader_call()
        train_val_loader = monai.data.ThreadDataLoader(
            train_dataset_val,batch_size=network_config["batch_size"],
            shuffle=False,num_workers=args.n_workers,generator=g,
            collate_fn=safe_collate,
            persistent_workers=args.n_workers>0)
        validation_loader = monai.data.ThreadDataLoader(
            validation_dataset,batch_size=1,
            shuffle=False,num_workers=args.n_workers,generator=g,
            collate_fn=safe_collate,
            persistent_workers=args.n_workers>0)

        print("Setting up training...")
        yolo = get_detection_network(
            net_type=args.net_type,
            network_config=network_config,
            dropout_param=args.dropout_param,
            loss_gamma=args.loss_gamma,
            loss_comb=args.loss_comb,
            class_weights=class_weights,
            train_loader_call=train_loader,
            iou_threshold=args.iou_threshold,
            anchor_array=anchor_array,
            n_epochs=args.max_epochs,
            warmup_steps=args.warmup_steps,
            boxes_key=box_key,
            box_class_key=box_class_key,
            n_classes=args.n_classes,
            dev=devices)
        
        callbacks = [RichProgressBar()]
        ckpt_callback,ckpt_path,status = get_ckpt_callback(
            checkpoint_dir=args.checkpoint_dir,
            checkpoint_name=args.checkpoint_name,
            max_epochs=args.max_epochs,
            resume_from_last=args.resume_from_last,
            val_fold=val_fold,
            monitor=args.monitor,
            metadata={"transform_arguments_pre":transform_arguments_pre,
                      "transform_arguments_post":transform_arguments_post})
        ckpt = ckpt_callback is not None
        if status == "finished":
            continue
        if ckpt_callback is not None:   
            callbacks.append(ckpt_callback)
            
        logger = get_logger(args.summary_name,args.summary_dir,
                            args.project_name,args.resume,
                            fold=val_fold)

        trainer = Trainer(
            accelerator="gpu" if "cuda" in args.dev else "cpu",
            devices=devices,logger=logger,callbacks=callbacks,
            max_epochs=args.max_epochs,check_val_every_n_epoch=1,
            log_every_n_steps=10)

        trainer.fit(yolo, train_loader, train_val_loader)

        # assessing performance on validation set
        print("Validating...")
        
        trainer.test(yolo,validation_loader)
        for k in yolo.test_metrics:
            out = yolo.test_metrics[k].compute()
            try:
                value = float(out.detach().numpy())
            except:
                value = float(out)
            print("{},{},{},{}".format(k,val_fold,0,value))
        
        torch.cuda.empty_cache()