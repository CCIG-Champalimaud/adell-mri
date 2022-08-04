import argparse
import random
import yaml
import numpy as np
import torch
import monai
import torchio
from sklearn.model_selection import KFold,train_test_split

from pytorch_lightning import Trainer
from pytorch_lightning.callbacks import EarlyStopping,StochasticWeightAveraging
from pytorch_lightning.loggers import TensorBoardLogger
from pytorch_lightning.callbacks.model_checkpoint import ModelCheckpoint

from lib.utils import (
    loss_factory,
    activation_factory,
    get_loss_param_dict,
    safe_collate,
    load_bb_json,
    load_anchors,
    MaskToAdjustedAnchorsd)
from lib.modules.losses import complete_iou_loss
from lib.modules.object_detection import YOLONet3d
from lib.modules.object_detection_pl import YOLONet3dPL
from lib.modules.layers import get_adn_fn

torch.backends.cudnn.benchmark = True

if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    # data
    parser.add_argument(
        '--dataset_json',dest='dataset_json',type=str,
        help="JSON containing dataset information",required=True)
    parser.add_argument(
        '--input_size',dest='input_size',type=float,nargs='+',
        help="Input size for network (for resize options",required=True)
    parser.add_argument(
        '--anchor_csv',dest='anchor_csv',type=str,
        help="Path to CSV file containing anchors",required=True)

    # network + training
    parser.add_argument(
        '--config_file',dest="config_file",
        help="Path to network configuration file (yaml)",
        required=True)
    
    # training
    parser.add_argument('--dev',dest='dev',default="cpu",
        help="Device for PyTorch training",
        choices=["cuda","cuda:0","cuda:1","cpu"],type=str)
    parser.add_argument(
        '--seed',dest='seed',help="Random seed",default=42,type=int)
    parser.add_argument(
        '--n_workers',dest='n_workers',
        help="No. of workers",default=0,type=int)
    parser.add_argument(
        '--augment',dest='augment',action="store_true",
        help="Use data augmentations",default=False)
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
        '--n_folds',dest="n_folds",
        help="Number of validation folds",default=5,type=int)
    parser.add_argument(
        '--checkpoint_dir',dest='checkpoint_dir',type=str,default="models",
        help='Path to directory where checkpoints will be saved.')
    parser.add_argument(
        '--checkpoint_name',dest='checkpoint_name',type=str,default="model_x",
        help='Checkpoint ID.')
    parser.add_argument(
        '--summary_dir',dest='summary_dir',type=str,default="summaries",
        help='Path to summary directory (for tensorboard).')
    parser.add_argument(
        '--summary_name',dest='summary_name',type=str,default="model_x",
        help='Summary name.')
    parser.add_argument(
        '--metric_path',dest='metric_path',type=str,default="metrics.csv",
        help='Path to file with CV metrics + information.')
    parser.add_argument(
        '--early_stopping',dest='early_stopping',type=int,default=None,
        help="No. of checks before early stop (defaults to no early stop).")
    parser.add_argument(
        '--swa',dest='swa',action="store_true",
        help="Use stochastic gradient averaging.",default=False)
    parser.add_argument(
        '--class_weights',dest='class_weights',type=float,nargs='+',
        help="Class weights (by alphanumeric order).",default=1.)
    parser.add_argument(
        '--iou_threshold',dest='iou_threshold',type=float,
        help="IoU threshold for pred-gt overlaps.",default=0.5)
    parser.add_argument(
        '--dropout_param',dest='dropout_param',type=float,
        help="Parameter for dropout.",default=0.1)

    args = parser.parse_args()

    torch.manual_seed(args.seed)
    random.seed(args.seed)
    np.random.seed(args.seed)
    g = torch.Generator()
    g.manual_seed(args.seed)

    output_file = open(args.metric_path,'w')

    bb_dict = load_bb_json(args.dataset_json)
    anchor_array = load_anchors(args.anchor_csv)
    all_classes = []
    for k in bb_dict:
        all_classes.extend(bb_dict[k]['labels'])
    n_classes = np.unique(all_classes).size+1
    n_anchors = anchor_array.shape[0]
    input_size = [int(i) for i in args.input_size]
    
    keys = ['image']

    path_dictionary = bb_dict

    with open(args.config_file,'r') as o:
        network_config = yaml.safe_load(o)

    if "activation_fn" in network_config:
        network_config["activation_fn"] = activation_factory[
            network_config["activation_fn"]]

    if "classification_loss_fn" in network_config:
        class_loss_key = network_config["classification_loss_fn"]
        k = "binary" if n_classes == 2 else "categorical"
        network_config["classification_loss_fn"] = loss_factory[k][
            network_config["classification_loss_fn"]]
    
    if "object_loss_fn" in network_config:
        object_loss_key = network_config["object_loss_fn"]
        network_config["object_loss_fn"] = loss_factory["binary"][
            network_config["object_loss_fn"]]

    if "batch_size" not in network_config:
        network_config["batch_size"] = 1

    output_example = YOLONet3d(
        n_channels=1,n_c=n_classes,activation_fn=torch.nn.Identity,
        anchor_sizes=anchor_array,dev=args.dev)(
            torch.ones([1,1,*input_size]))
    output_sh = output_example[0].shape[2:]

    print("Setting up transforms...")
    if args.augment == True:
        augments = [
            torchio.transforms.RandomBiasField(include=keys),
            torchio.transforms.RandomNoise(include=keys),
            torchio.transforms.RandomGamma(include=keys),
            #RandomFlipWithBoxesd(
            #    image_keys=keys,
            #    box_key="bounding_boxes",
            #    box_key_nest="boxes",
            #    axes=[0,1,2],prob=0.2)
                ]
    else:
        augments = []

    intp = ["area"]
    transforms_train = [
        monai.transforms.LoadImaged(keys),
        monai.transforms.AddChanneld(keys),
        monai.transforms.Orientationd(keys,"RAS"),
        # in case images should be downsampled
        monai.transforms.Resized(
            keys,tuple(input_size),mode=intp),
        *augments,
        MaskToAdjustedAnchorsd(
            anchor_sizes=anchor_array,input_sh=input_size,
            output_sh=output_sh,iou_thresh=args.iou_threshold,
            bb_key="boxes",class_key="labels",shape_key="shape",
            output_key="bb_map"),
        monai.transforms.ScaleIntensityd(keys,0,1),
        monai.transforms.EnsureTyped([*keys,"bb_map"])]

    transforms_train_val = [
        monai.transforms.LoadImaged(keys),
        monai.transforms.AddChanneld(keys),
        monai.transforms.Orientationd(keys,"RAS"),
        # in case images should be downsampled
        monai.transforms.Resized(
            keys,tuple(input_size),mode=intp),
        MaskToAdjustedAnchorsd(
            anchor_sizes=anchor_array,input_sh=input_size,
            output_sh=output_sh,iou_thresh=args.iou_threshold,
            bb_key="boxes",class_key="labels",shape_key="shape",
            output_key="bb_map"),
        monai.transforms.ScaleIntensityd(keys,0,1),
        monai.transforms.EnsureTyped([*keys,"bb_map"])]

    transforms_val = [
        monai.transforms.LoadImaged(keys),
        monai.transforms.AddChanneld(keys),
        monai.transforms.Orientationd(keys,"RAS"),
        # in case images should be downsampled
        monai.transforms.Resized(
            keys,tuple(input_size),mode=intp),
        MaskToAdjustedAnchorsd(
            anchor_sizes=anchor_array,input_sh=input_size,
            output_sh=output_sh,iou_thresh=args.iou_threshold,
            bb_key="boxes",class_key="labels",shape_key="shape",
            output_key="bb_map"),
        monai.transforms.ScaleIntensityd(keys,0,1),
        monai.transforms.EnsureTyped([*keys,"bb_map"])]

    all_pids = [k for k in path_dictionary]
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
        path_list_train = [path_dictionary[pid] for pid in train_pids]
        path_list_train_val = [path_dictionary[pid] for pid in train_val_pids]
        path_list_val = [path_dictionary[pid] for pid in val_pids]

        train_dataset = monai.data.CacheDataset(
            path_list_train,
            monai.transforms.Compose(transforms_train))
        train_dataset_val = monai.data.CacheDataset(
            path_list_train_val,
            monai.transforms.Compose(transforms_train_val))
        validation_dataset = monai.data.CacheDataset(
            path_list_val,
            monai.transforms.Compose(transforms_val))

        print("Calculating weights...")
        positive = np.sum([len(bb_dict[k]['labels']) for k in bb_dict])
        total = len(bb_dict) * np.prod(output_sh)
        object_weights = 1-positive/total
        object_weights = torch.as_tensor(object_weights)
        
        class_weights = torch.as_tensor(args.class_weights)
        class_weights = class_weights.to(args.dev)

        object_loss_params = get_loss_param_dict(
            1.0,args.loss_gamma,
            args.loss_comb,0.,
            args.dev)[object_loss_key]
        classification_loss_params = get_loss_param_dict(
            class_weights,args.loss_gamma,
            args.loss_comb,0.5,args.dev)[class_loss_key]

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
        adn_fn = get_adn_fn(
            3,norm_fn="batch",
            act_fn="swish",dropout_param=args.dropout_param)
        yolo = YOLONet3dPL(
            training_dataloader_call=train_loader_call,
            image_key="image",label_key="bb_map",boxes_key="boxes",
            box_label_key="labels",
            anchor_sizes=anchor_array,dev=args.dev,
            n_c=n_classes,adn_fn=adn_fn,iou_threshold=args.iou_threshold,
            reg_loss_fn=complete_iou_loss,
            classification_loss_params=classification_loss_params,
            object_loss_params=object_loss_params,
            **network_config)

        callbacks = []

        if args.early_stopping is not None:
            early_stopping = EarlyStopping(
                'val_loss',patience=args.early_stopping,
                strict=True,mode="min")
            callbacks.append(early_stopping)
        if args.swa == True:
            swa_callback = StochasticWeightAveraging()
            callbacks.append(swa_callback)

        ckpt_name = args.checkpoint_name + "_fold" + str(val_fold)
        ckpt_name = ckpt_name + "_{epoch:03d}"
        ckpt_name = ckpt_name + "_{val_loss:.3f}"
        ckpt_callback = ModelCheckpoint(
            dirpath=args.checkpoint_dir,
            filename=ckpt_name,monitor="val_loss",
            save_last=True,save_top_k=2,mode="min")
        ckpt_callback.CHECKPOINT_NAME_LAST = \
            args.checkpoint_name + "_fold" + str(val_fold) + "_last"
        callbacks.append(ckpt_callback)
        logger = TensorBoardLogger(
                save_dir=args.summary_dir, 
                name="{}_fold{}".format(args.summary_name,val_fold))

        if ':' in args.dev:
            devices = [int(args.dev.strip(':')[-1])]
        else:
            devices = 1
        trainer = Trainer(
            accelerator="gpu" if "cuda" in args.dev else "cpu",
            devices=devices,logger=logger,callbacks=callbacks,
            max_epochs=args.max_epochs,#overfit_batches=2,
            check_val_every_n_epoch=1,log_every_n_steps=10)

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