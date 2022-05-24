import argparse
import random
import yaml
import numpy as np
import nibabel as nib
import torch
import torchio
import monai

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
    ConvertToOneHot,
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
        '--loss_gamma',dest="loss_gamma",
        help="Gamma for focal loss",default=2.0,type=float)
    parser.add_argument(
        '--loss_comb',dest="loss_comb",
        help="Relative weight for combined losses",default=0.5,type=float)
    parser.add_argument(
        '--max_epochs',dest="max_epochs",
        help="Maximum number of training epochs",default=100,type=int)
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
        
    args = parser.parse_args()

    torch.manual_seed(args.seed)
    random.seed(args.seed)
    np.random.seed(args.seed)
    g = torch.Generator()
    g.manual_seed(args.seed)

    args.classes = sorted(args.classes)
    if len(args.classes) == 1:
        n_classes = 2
    elif len(args.classes) == 2:
        n_classes = 3

    path_dictionary = get_prostatex_path_dictionary(args.base_path)
    size_dict,spacing_dict = get_size_spacing_dict(path_dictionary,[args.mod])
    for k in size_dict:
        size_dict[k] = np.int32(
            np.median(np.array(size_dict[k]),axis=0))

    for k in spacing_dict:
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

    if "batch_size" not in network_config:
        network_config["batch_size"] = 1

    print("Calculating class weights...")
    keys = [args.mod]
    label_keys = ["{}_{}_segmentations".format(args.mod,c)
                  for c in args.classes]
    if n_classes == 2:
        label_key = label_keys[0]
        positive = 0.
        total = 0.
        for pid in path_dictionary:
            if label_key in path_dictionary[pid]:
                path = path_dictionary[pid][label_key]
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
        for pid in path_dictionary:
            for i,label_key in enumerate(label_keys):
                if label_key in path_dictionary[pid]:
                    N += 1
                    path = path_dictionary[pid][label_key]
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

        loss_params = get_loss_param_dict(
            weights=weights,gamma=args.loss_gamma,
            comb=args.loss_comb)[loss_key]

    print("Setting up dataloaders...")
    transforms = [
        torchio.transforms.Resample(
            include=keys+label_keys,label_keys=label_keys,
            target=spacing_dict[args.mod])]
    transforms.append(
        torchio.transforms.Resize(
            include=keys,target_shape=size_dict[args.mod]))
    for label_key in label_keys:
        transforms.append(
            torchio.transforms.Resize(
                include=[label_key],label_keys=[label_key],
                target_shape=size_dict[args.mod]))
    if n_classes > 2:
        lesion_key = "{}_{}_segmentations".format(args.mod,"lesion")
        label_key = "label"
        transforms.append(
            ConvertToOneHot(label_keys,label_key,lesion_key,bg=True))
    else:
        label_key = "{}_{}_segmentations".format(args.mod,args.classes[0])

    train_dataset = MONAIDataset(
        path_dictionary,"nifti",orientation="RAS",
        image_keys=keys+label_keys,transforms=transforms)

    train_loader_call = lambda: torch.utils.data.DataLoader(
        train_dataset,batch_size=network_config["batch_size"],
        shuffle=True,num_workers=args.n_workers,generator=g)
    train_loader = train_loader_call()

    print("Setting up training...")
    unet = UNetPL(training_dataloader_call=train_loader_call,
                  image_key=keys[0],label_key=label_key,
                  loss_params=loss_params,n_classes=n_classes,
                  **network_config)
    
    ckpt_callback = ModelCheckpoint(
        dirpath=args.checkpoint_dir,filename=args.checkpoint_name,
        monitor="val_loss",save_last=True,save_top_k=2,mode="min")
    callbacks = []
    callbacks.append(ckpt_callback)
    logger = TensorBoardLogger(
            save_dir=args.summary_dir, name=args.summary_name)

    trainer = Trainer(
        accelerator="gpu" if args.dev=="cuda" else "cpu",devices=1,
        logger=logger,callbacks=callbacks,
        auto_lr_find=True,max_epochs=args.max_epochs,
        check_val_every_n_epoch=1)
    #trainer.fit(unet, train_loader, train_val_loader)
    trainer.fit(unet,train_loader)