import argparse
import random
import yaml
import numpy as np
import nibabel as nib
import torch
import monai
from sklearn.model_selection import KFold
from torchmetrics import JaccardIndex,Precision

from pytorch_lightning import Trainer
from pytorch_lightning.callbacks import EarlyStopping
from pytorch_lightning.loggers import TensorBoardLogger
from pytorch_lightning.callbacks.model_checkpoint import ModelCheckpoint

from lib.utils import (
    loss_factory,
    activation_factory,
    get_prostatex_path_dictionary,
    get_size_spacing_dict,
    get_loss_param_dict,
    collate_last_slice,
    ConvertToOneHot,
    RandomSlices,
    SlicesToFirst,
    PrintShaped)
from lib.modules.segmentation_pl import UNetPL
from lib.dataoperations.lib.data_functions import MONAIDataset

if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    # data
    parser.add_argument(
        '--base_path',dest='base_path',type=str,
        help="Path to ProstateX dataset",required=True)
    parser.add_argument(
        '--mod',dest='mod',type=str,choices=["T2WAx","DWI"],
        help="Key to be used",required=True)
    parser.add_argument(
        '--classes',dest='classes',type=str,nargs='+',
        choices=["gland","lesion"],
        help="Classes to be considered",required=True)
    # network + training
    parser.add_argument(
        '--config_file',dest="config_file",
        help="Path to network configuration file (yaml)",
        required=True)
    
    # training
    parser.add_argument('--dev',dest='dev',
        help="Device for PyTorch training",choices=["cuda","cpu"],type=str)
    parser.add_argument(
        '--seed',dest='seed',help="Random seed",default=42,type=int)
    parser.add_argument(
        '--n_workers',dest='n_workers',
        help="No. of workers",default=1,type=int)
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
        
    args = parser.parse_args()

    torch.manual_seed(args.seed)
    random.seed(args.seed)
    np.random.seed(args.seed)
    g = torch.Generator()
    g.manual_seed(args.seed)

    output_file = open(args.metric_path,'w')

    args.classes = sorted(args.classes)
    if len(args.classes) == 1:
        n_classes = 2
    elif len(args.classes) == 2:
        n_classes = 3

    keys = [args.mod]
    label_keys = ["{}_{}_segmentations".format(args.mod,c)
                  for c in args.classes]
    all_keys = [args.mod,*label_keys]

    path_dictionary = get_prostatex_path_dictionary(args.base_path)
    fpd = {}
    for pid in path_dictionary:
        if all([k in path_dictionary[pid] for k in all_keys]):
            fpd[pid] = {k:path_dictionary[pid][k] for k in all_keys}
    path_dictionary = fpd
    size_dict,spacing_dict = get_size_spacing_dict(path_dictionary,[args.mod])

    for k in size_dict:
        size_dict[k] = np.int32(
            np.median(np.array(size_dict[k]),axis=0))
        spacing_dict[k] = np.median(np.array(spacing_dict[k]),axis=0)

    with open(args.config_file,'r') as o:
        network_config = yaml.safe_load(o)

    if "activation_fn" in network_config:
        network_config["activation_fn"] = activation_factory[
            network_config["activation_fn"]]
    
    if "loss_fn" in network_config:
        loss_key = network_config["loss_fn"]
        k = "binary" if n_classes == 2 else "categorical"
        network_config["loss_fn"] = loss_factory[k][
            network_config["loss_fn"]]

    if "spatial_dimensions" not in network_config:
        network_config["spatial_dimensions"] = 3

    if "batch_size" not in network_config:
        network_config["batch_size"] = 1

    print("Setting up transforms...")
    if args.augment == True:
        augments = [
            monai.transforms.RandBiasFieldd([args.mod]),
            monai.transforms.RandGibbsNoised([args.mod]),
            monai.transforms.RandAdjustContrastd([args.mod]),
            monai.transforms.RandStdShiftIntensityd([args.mod])]
    else:
        augments = []

    intp = ["bilinear","nearest","nearest"]*len(args.classes)
    transforms_train = [
        monai.transforms.LoadImaged(all_keys),
        monai.transforms.AddChanneld(all_keys),
        monai.transforms.Orientationd(all_keys,"RAS"),
        monai.transforms.EnsureTyped(all_keys),
        # just in case some minor mismatches exist
        monai.transforms.Resized(all_keys,size_dict[args.mod]),
        *augments,
        monai.transforms.ScaleIntensityd([args.mod],0,1)]

    transforms_val = [
        monai.transforms.LoadImaged(all_keys),
        monai.transforms.AddChanneld(all_keys),
        monai.transforms.Orientationd(all_keys,"RAS"),
        monai.transforms.EnsureTyped(all_keys),
        monai.transforms.Resized(all_keys,size_dict[args.mod]),
        monai.transforms.ScaleIntensityd([args.mod],0,1)]

    if n_classes > 2:
        lesion_key = "{}_{}_segmentations".format(args.mod,"lesion")
        label_key = "label"
        transforms_train.append(
            ConvertToOneHot(label_keys,label_key,lesion_key,bg=True))
        transforms_val.append(
            ConvertToOneHot(label_keys,label_key,lesion_key,bg=True))
    else:
        label_key = "{}_{}_segmentations".format(args.mod,args.classes[0])

    if network_config["spatial_dimensions"] == 2:
        transforms_train.append(
            RandomSlices(all_keys,label_key,1,base=0.01))
        transforms_val.append(
            SlicesToFirst(all_keys))

        collate_fn = collate_last_slice
    else:
        collate_fn = None

    all_pids = [k for k in path_dictionary]
    fold_generator = KFold(
        args.n_folds,shuffle=True,random_state=args.seed).split(all_pids)

    for val_fold in range(args.n_folds):
        train_idxs,val_idxs = next(fold_generator)
        train_pids = [all_pids[i] for i in train_idxs]
        val_pids = [all_pids[i] for i in val_idxs]
        path_list_train = [path_dictionary[pid] for pid in train_pids]
        path_list_val = [path_dictionary[pid] for pid in val_pids]

        train_dataset = monai.data.CacheDataset(
            path_list_train,monai.transforms.Compose(transforms_train))
        train_dataset_val = monai.data.CacheDataset(
            path_list_train,monai.transforms.Compose(transforms_val))
        validation_dataset = monai.data.CacheDataset(
            path_list_val,monai.transforms.Compose(transforms_val))

        print("Calculating class weights...")
        if n_classes == 2:
            label_key = label_keys[0]
            positive = 0.
            total = 0.
            for element in path_list_train:
                if label_key in element:
                    path = element[label_key]
                    x = nib.load(path).get_fdata()
                    positive += x.sum()
                    total += x.size
            
            print("Class proportions: {:.3f}% ({})".format(
                positive/total*100,args.classes[0]))
            weights = total/(2*positive)
            weights = torch.Tensor([weights])

        else:
            counts = {"positive":[0 for _ in args.classes],
                    "total":0}
            N = 0
            for element in path_list_train:
                for i,label_key in enumerate(label_keys):
                    if label_key in element:
                        N += 1
                        path = element[label_key]
                        x = nib.load(path).get_fdata()
                        counts["positive"][i] += x.sum()
                        size = x.size
                counts["total"] += size
                    
            weights =  [
                counts["total"]/(3*(counts["total"]-np.sum(counts["positive"]))),
                counts["total"]/(3*counts['positive'][0]),
                counts["total"]/(3*counts['positive'][1])]
            print("Class proportions: {:.3f}% (gland), {:.3f}% (lesion)".format(
                counts["positive"][0]/counts["total"]*100,
                counts["positive"][1]/counts["total"]*100))
            weights = torch.Tensor(weights)
        
        weights = weights.to(args.dev)
        weights = weights/torch.sum(weights)

        loss_params = get_loss_param_dict(
            weights=weights,gamma=args.loss_gamma,
            comb=args.loss_comb)[loss_key]

        train_loader_call = lambda: torch.utils.data.DataLoader(
            train_dataset,batch_size=network_config["batch_size"],
            shuffle=True,num_workers=args.n_workers,generator=g,
            collate_fn=collate_fn)
        train_loader = train_loader_call()
        train_val_loader = train_loader_call()

        print("Setting up training...")
        unet = UNetPL(training_dataloader_call=train_loader_call,
                    image_key=keys[0],label_key=label_key,
                    loss_params=loss_params,n_classes=n_classes,
                    **network_config)
        
        callbacks = []

        if args.early_stopping is not None:
            early_stopping = EarlyStopping(
                'val_loss',patience=args.early_stopping,
                strict=True,mode="min")
            callbacks.append(early_stopping)

        ckpt_callback = ModelCheckpoint(
            dirpath=args.checkpoint_dir,
            filename="{}_fold{}".format(args.checkpoint_name,val_fold),
            monitor="val_loss",save_last=True,save_top_k=2,mode="min")
        callbacks.append(ckpt_callback)
        logger = TensorBoardLogger(
                save_dir=args.summary_dir, 
                name="{}_fold{}".format(args.summary_name,val_fold))

        trainer = Trainer(
            accelerator="gpu" if args.dev=="cuda" else "cpu",devices=1,
            logger=logger,callbacks=callbacks,
            auto_lr_find=True,max_epochs=args.max_epochs,
            check_val_every_n_epoch=1,detect_anomaly=True)
        trainer.fit(unet, train_loader, train_val_loader)

        # assessing performance on validation set
        unet.eval()
        if n_classes > 2:
            jaccard_validation = JaccardIndex(
                num_classes=n_classes,reduction="none")
            precision_validation = Precision(
                num_classes=n_classes,mdmc_average="samplewise",
                average="none")
        else:
            jaccard_validation = JaccardIndex(
                num_classes=n_classes)
            precision_validation = Precision(
                num_classes=None)

        validation_loader = torch.utils.data.DataLoader(
            validation_dataset,batch_size=1,
            shuffle=False,num_workers=args.n_workers,generator=g,
            collate_fn=collate_fn)

        for s in validation_loader:
            pred = unet(s[args.mod])
            y = torch.squeeze(s[label_key],1)
            try: y = torch.round(y).int()
            except: pass

            jaccard_validation.update(pred,y)
            precision_validation.update(pred,y)
        
        iou = jaccard_validation.compute()
        prec = precision_validation.compute()
        if n_classes > 2:
            # multi class scenario
            for i,v in enumerate(iou):
                S = "{},{},{},{}".format(
                    "iou",val_fold,i,float(v.numpy()))
                print(S)
                output_file.write(S+'\n')
            for i,v in enumerate(prec):
                S = "{},{},{},{}".format(
                    "prec",val_fold,i,float(v.numpy()))
                print(S)
                output_file.write(S+'\n')
                
        else:
            S = "{},{},{},{}".format(
                "iou",val_fold,0,float(iou.numpy()))
            print(S)
            output_file.write(S+'\n')
            S = "{},{},{},{}".format(
                "prec",val_fold,0,float(prec.numpy()))
            print(S)
            output_file.write(S+'\n')
        
        output_file.write("{},{},{},{}\n".format(
            "train_ids",val_fold,0,":".join(train_pids)))
        output_file.write("{},{},{},{}\n".format(
            "val_ids",val_fold,0,":".join(val_pids)))