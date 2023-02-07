"""
Functions that return correctly configured callbacks and objects for PyTorch
Lightning. 
"""

import os
import torch
import wandb
from pytorch_lightning.loggers import WandbLogger
from pytorch_lightning.callbacks.model_checkpoint import ModelCheckpoint
from pytorch_lightning.strategies import DDPStrategy

from typing import List,Union,Tuple

def get_ckpt_callback(checkpoint_dir:str,checkpoint_name:str,
                      max_epochs:int,resume_from_last:bool,
                      val_fold:int=None,monitor="val_loss")->ModelCheckpoint:
    """Gets a checkpoint callback for PyTorch Lightning. The format for 
    for the last and 2 best checkpoints, respectively is:
    1. "{name}_fold{fold}_last.ckpt"
    2. "{name}_fold{fold}_best_{epoch}_{monitor:.3f}.ckpt"

    Args:
        checkpoint_dir (str): directory where checkpoints will be stored.
        checkpoint_name (str): root name for checkpoint.
        max_epochs (int): maximum number of training epochs (used to check if
            training has finished when resume_from_last==True).
        resume_from_last (bool): whether training should be resumed in case a
            checkpoint is detected.
        val_fold (int, optional): ID for the validation fold. Defaults to None.
        monitor (str, optional): metric which should be monitored when defining
            the best checkpoints. Defaults to "val_loss".

    Returns:
        ModelCheckpoint: PyTorch Lightning checkpoint callback.
    """
    ckpt_path = None
    ckpt_callback = None
    status = None
    
    if (checkpoint_dir is not None) and (checkpoint_name is not None):
        if val_fold is not None:
            ckpt_name = checkpoint_name + "_fold" + str(val_fold)
            ckpt_last = checkpoint_name + "_fold" + str(val_fold)
        else:
            ckpt_name = checkpoint_name
            ckpt_last = checkpoint_name
        ckpt_name = ckpt_name + "_best_{epoch}_{" + monitor + ":.3f}"
        if "loss" in monitor:
            mode = "min"
        else:
            mode = "max"
        ckpt_callback = ModelCheckpoint(
            dirpath=checkpoint_dir,
            filename=ckpt_name,monitor=monitor,
            save_last=True,save_top_k=2,mode=mode)
        
        ckpt_last = ckpt_last + "_last"
        ckpt_callback.CHECKPOINT_NAME_LAST = ckpt_last
        ckpt_last_full = os.path.join(
            checkpoint_dir,ckpt_last+'.ckpt')
        if os.path.exists(ckpt_last_full) and resume_from_last is True:
            ckpt_path = ckpt_last_full
            epoch = torch.load(ckpt_path)["epoch"]
            if epoch >= (max_epochs-1):
                print("Training has finished for this fold, skipping")
                status = "finished"
            else:
                print("Resuming training from checkpoint in {} (epoch={})".format(
                    ckpt_path,epoch))
    return ckpt_callback,ckpt_path,status

def get_logger(summary_name:str,summary_dir:str,
               project_name:str,resume:str,fold:int=None)->WandbLogger:
    """Defines a Wandb logger for PyTorch Lightning. Each run is configured
    as "{project_name}/{summary_name}_fold{fold}".

    Args:
        summary_name (str): name of the Wandb run.
        summary_dir (str): directory where summaries are stored.
        project_name (str): name of the Wandb project.
        resume (str): how the metric registry in Wandb should be resumed.
            Details in https://docs.wandb.ai/guides/track/advanced/resuming.
        fold (int, optional): ID for the validation fold. Defaults to None.

    Returns:
        WandbLogger: _description_
    """
    if (summary_name is not None) and (project_name is not None):
        wandb.finish()
        wandb_resume = resume
        if wandb_resume == "none":
            wandb_resume = None
        run_name = summary_name.replace(':','_')
        if fold is not None:
            run_name = run_name + "_fold{}".format(fold)
        logger = WandbLogger(
            save_dir=summary_dir,project=project_name,
            name=run_name,version=run_name,reinit=True,resume=wandb_resume)
    else:
        logger = None
    return logger

def get_devices(device_str:str)->Tuple[str,Union[List[int],int],str]:
    """Takes a string with form "{device}:{device_ids}" where device_ids is a
    comma separated list of device IDs (i.e. cuda:0,1).

    Args:
        device_str (str): device string. Can be "cpu" or "cuda" if no 
            parallelization is necessary or "cuda:0,1" if training is to be
            distributed across GPUs 0 and 1, for instance.
        kwargs: keyword arguments for DDPStrategy

    Returns:
        Tuple[str,Union[List[int],int],str]: a tuple containing the accelerator
            ("cpu" or "gpu") the devices (None or a list of devices as 
            specified after the ":" in the device_str) and the parallelization
            strategy ("ddp" if len(devices) > 0, None otherwise)
    """
    strategy = None
    if ":" in device_str:
        accelerator = "gpu" if "cuda" in device_str else "cpu"
        devices = [int(i) for i in device_str.split(":")[-1].split(",")]
        if len(devices) > 1:
            strategy = "ddp"
    else:
        accelerator = "gpu" if "cuda" in device_str else "cpu"
        devices = 1
    return accelerator,devices,strategy
