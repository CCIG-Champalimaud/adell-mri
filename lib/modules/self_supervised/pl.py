import torch
import torchmetrics
import pytorch_lightning as pl
import warnings
from abc import ABC

from ...custom_types import *

from ..layers import ResNet
from .self_supervised import BarlowTwinsLoss, VICRegLocalLoss,byol_loss,simsiam_loss,VICRegLoss
from ..segmentation.unet import UNet

class BarlowTwinsPL(ResNet,pl.LightningModule):
    def __init__(
        self,
        image_key: str="image",
        augmented_image_key: str="augmented_image",
        learning_rate: float=0.001,
        batch_size: int=4,
        weight_decay: float=0.005,
        training_dataloader_call: Callable=None,
        loss_lam: float=0.02,
        *args,**kwargs) -> torch.nn.Module:        
        super().__init__(*args,**kwargs)

        self.image_key = image_key
        self.augmented_image_key = augmented_image_key
        self.learning_rate = learning_rate
        self.batch_size = batch_size
        self.weight_decay = weight_decay
        self.train_dataloader_call = training_dataloader_call
        self.loss_lam = loss_lam

        self.loss = BarlowTwinsLoss(moving=True,lam=self.loss_lam)

    def calculate_loss(self,y1,y2,update=True):
        loss = self.loss(y1,y2,update=True)
        return loss.mean()

    def update_metrics(self,y1,y2,metrics,log=True):
        for k in metrics:
            metrics[k].update(y1,y2)
            if log == True:
                self.log(
                    k,metrics[k],on_epoch=True,
                    on_step=False,prog_bar=True,sync_dist=True)

    def training_step(self,batch,batch_idx):
        x1,x2 = batch[self.image_key],batch[self.augmented_image_key]
        y1,y2 = self.forward(x1),self.forward(x2)

        loss = self.calculate_loss(y1,y2)
        
        self.log("train_loss",loss,prob_bar=True)
        self.update_metrics(y1,y2,self.train_metrics)
        return loss
    
    def validation_step(self,batch,batch_idx):
        x1,x2 = batch[self.image_key],batch[self.augmented_image_key]
        y1,y2 = self.forward(x1),self.forward(x2)

        loss = self.calculate_loss(y1,y2,False)
        
        self.log("val_loss", loss,prob_bar=True,sync_dist=True)
        self.update_metrics(y1,y2,self.val_metrics)
        return loss

    def test_step(self,batch,batch_idx):
        x1,x2 = batch[self.image_key],batch[self.augmented_image_key]
        y1,y2 = self.forward(x1),self.forward(x2)

        loss = self.calculate_loss(y1,y2,False)
        
        self.log("test_loss", loss,prob_bar=True)
        self.update_metrics(y1,y2,self.test_metrics)
        return loss

    def train_dataloader(self) -> torch.utils.data.DataLoader:
        return self.training_dataloader_call(self.batch_size)

    def configure_optimizers(self):
        optimizer = torch.optim.AdamW(
            self.parameters(),lr=self.learning_rate,
            weight_decay=self.weight_decay,amsgrad=True)
        lr_schedulers = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer,patience=5,cooldown=5,min_lr=1e-6,factor=0.25,
            verbose=True)

        return {"optimizer":optimizer,
                "lr_scheduler":lr_schedulers,
                "monitor":"val_loss"}
    
    def on_validation_epoch_end(self):
        sch = self.lr_schedulers().state_dict()
        lr = self.learning_rate
        last_lr = sch['_last_lr'][0] if '_last_lr' in sch else lr
        self.log("lr",last_lr)

    def setup_metrics(self):
        metric_dict = {
            "MSE":torchmetrics.MeanSquaredError,
            "R":torchmetrics.PearsonCorrCoef,
            "CS":torchmetrics.CosineSimilarity}
        self.train_metrics = torch.nn.ModuleDict({})
        self.val_metrics = torch.nn.ModuleDict({})
        self.test_metrics = torch.nn.ModuleDict({})
        for k in metric_dict:
            self.train_metrics[k] = metric_dict[k]()
            self.val_metrics["V"+k] = metric_dict[k]()
            self.test_metrics["T"+k] = metric_dict[k]()

class NonContrastiveBasePL(pl.LightningModule,ABC):
    """Abstract method for non-contrastive PL modules. Features some very
    basic but helpful functions which I use consistently when training 
    non-contrastive self-supervised models.
    """
    def __init__(self):
        super().__init__()

    def train_dataloader(self) -> torch.utils.data.DataLoader:
        return self.training_dataloader_call(self.batch_size)

    def update_metrics(self,y1,y2,metrics,log=True):
        # torchmetrics only allows for 1d vectors for regression (?????)
        y1 = y1.flatten()
        y2 = y2.flatten()
        for k in metrics:
            metrics[k].update(y1,y2)
            if log == True:
                self.log(
                    k,metrics[k],on_epoch=True,batch_size=y1.shape[0],
                    on_step=False,prog_bar=True,sync_dist=True)

    def init_loss(self):
        self.loss = simsiam_loss
        if self.ema is not None:
            self.loss = byol_loss
        if self.vic_reg == True:
            self.loss = VICRegLoss(**self.vic_reg_loss_params)
        if self.vic_reg_local == True:
            self.loss = VICRegLocalLoss(**self.vic_reg_loss_params)

    def calculate_loss(self,y1,y2,*args):
        if self.stop_gradient == False:
            # no need to stop gradients with VICReg or VICRegL.
            l = self.loss(y1,y2,*args)
        else:
            # the famous stop gradient operation just implies detaching the
            # output tensor from the computation graph to prevent gradients
            # from being automatically propagated.
            l = self.loss(y1,y2.detach(),*args) 
        return l

    def safe_sum(self,X):
        if isinstance(X,torch.Tensor):
            return X.sum()
        else:
            return sum(X)

    def configure_optimizers(self):
        params_no_decay = []
        params_decay = []
        for k,p in self.named_parameters():
            if 'normalization' in k:
                params_no_decay.append(p)
            else:
                params_decay.append(p)
        optimizer = torch.optim.SGD(
            [{"params":params_no_decay,"weight_decay":0},
             {"params":params_decay}],
            momentum=0.9,
            lr=self.learning_rate,weight_decay=self.weight_decay)
        lr_schedulers = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer,self.n_epochs)

        return {"optimizer":optimizer,
                "lr_scheduler":lr_schedulers,
                "monitor":"val_loss"}
    
    def on_validation_epoch_end(self):
        sch = self.lr_schedulers().state_dict()
        lr = self.learning_rate
        last_lr = sch['_last_lr'][0] if '_last_lr' in sch else lr
        self.log("lr",last_lr,sync_dist=True)

    def setup_metrics(self):
        metric_dict = {
            #"MSE":torchmetrics.MeanSquaredError
            }
        self.train_metrics = torch.nn.ModuleDict({})
        self.val_metrics = torch.nn.ModuleDict({})
        self.test_metrics = torch.nn.ModuleDict({})
        for k in metric_dict:
            self.train_metrics[k] = metric_dict[k]()
            self.val_metrics["V"+k] = metric_dict[k]()
            self.test_metrics["T"+k] = metric_dict[k]()

class NonContrastiveSelfSLPL(ResNet,NonContrastiveBasePL):
    """Operates a number of non-contrastive self-supervised learning 
    methods, all of which use non-contrastive approaches to self-supervised
    learning. Included here are:
    * SimSiam (the vanilla version of this class)
    * BYOL (when an ema module is specified - this should be an 
    ExponentialMovingAverage module)
    * VICReg (when vic_reg==True)
    * VICRegL (when vic_reg_local==True)
    
    Uses a ResNet backbone, to enable the transfer of this network to a 
    standard ResNet.
    """
    def __init__(
        self,
        aug_image_key_1: str="aug_image_1",
        aug_image_key_2: str="aug_image_2",
        box_key_1: str="box_1",
        box_key_2: str="box_2",
        learning_rate: float=0.001,
        batch_size: int=4,
        weight_decay: float=0.005,
        training_dataloader_call: Callable=None,
        n_epochs: int=1000,
        ema: torch.nn.Module=None,
        vic_reg: bool=False,
        vic_reg_local: bool=False,
        vic_reg_loss_params: dict={},
        stop_gradient: bool=True,
        channels_to_batch: bool=False,
        *args,**kwargs):
        """
        Args:
            aug_image_key_1 (str, optional): key for augmented image 1. 
                Defaults to "aug_image_1".
            aug_image_key_2 (str, optional): key for augmented image 2. 
                Defaults to "aug_image_2".
            box_key_1 (str, optional): key for bounding box mapping 
                aug_image_key_1 to its original, uncropped image. (used only
                when vic_reg_local == True)
            box_key_2 (str, optional): key for bounding box mapping 
                aug_image_key_2 to its original, uncropped image. (used only
                when vic_reg_local == True)
            learning_rate (float, optional): learning rate. Defaults to 0.2.
            batch_size (int, optional): batch size. Defaults to 4.
            weight_decay (float, optional): weight decay for optimizer. 
                Defaults to 0.005.
            training_dataloader_call (Callable, optional): function that, when
                called, returns the training dataloader. Defaults to None.
            n_epochs (int, optional): number of training epochs. Defaults to 
                1000.
            ema (float, torch.nn.Module): exponential moving decay module 
                (EMA). Must have an update method that takes model as input 
                and updates the weights based on this. Defaults to None.
            vic_reg (bool, optional): uses the VIC (variance, invariance,
                covariance) regularization method from Bardes et al. (2022).
                Defaults to False.
            vic_reg_local (bool, optional): uses the VICRegL method from 
                Bardes et al. (2022). Overrides vic_reg. Defaults to False.
            vic_reg_loss_params (dict, optional): parameters for the VICRegLoss
                module. Defaults to {} (the default parameters).
            stop_gradient (bool, optional): stops gradients when calculating
                losses. Useful for VICReg. Defaults to True.
            channels_to_batch (bool, optional): resizes the input such that 
                each channel becomes an element of the batch. Defaults to 
                False.
        """
        self.aug_image_key_1 = aug_image_key_1
        self.aug_image_key_2 = aug_image_key_2
        self.box_key_1 = box_key_1
        self.box_key_2 = box_key_2
        self.learning_rate = learning_rate
        self.batch_size = batch_size
        self.weight_decay = weight_decay
        self.training_dataloader_call = training_dataloader_call
        self.n_epochs = n_epochs
        self.ema = ema
        self.vic_reg = vic_reg
        self.vic_reg_local = vic_reg_local
        self.vic_reg_loss_params = vic_reg_loss_params
        self.stop_gradient = stop_gradient
        self.channels_to_batch = channels_to_batch
        
        if channels_to_batch == True:
            kwargs["backbone_args"]["in_channels"] = 1
        
        super().__init__(*args,**kwargs)

        if self.stop_gradient == False and self.vic_reg == False:
            warnings.warn("stop_gradient=False should not (in theory) be used\
                with vic_reg=False or vic_reg_local=False")

        self.init_loss()
        if self.ema is not None:
            self.ema.update(self)
        else:
            self.ema = None
        
        self.loss_str_dict = {
            "standard":[None],
            "vicreg":["inv","var","cov"],
            "vicregl":["inv","var","cov","local"]
        }
        
        self.setup_metrics()

    def forward_ema_stop_grad(self,x,ret):
        if self.ema is not None:
            op = self.ema.shadow.forward
        else:
            op = self.forward
        if self.stop_gradient == True:
            with torch.no_grad():
                return op(x,ret)
        else:
            return op(x,ret)

    def step(self,batch,loss_str:str,metrics:dict,train=False):
        if self.vic_reg_local == False:
            ret_string_1 = "prediction"
            ret_string_2 = "projection"
            other_args = []
        else:
            ret_string_1 = "representation"
            ret_string_2 = "representation"
            box_1 = batch[self.box_key_1]
            box_2 = batch[self.box_key_2]
            other_args = [box_1,box_2]
        
        x1,x2 = batch[self.aug_image_key_1],batch[self.aug_image_key_2]
        if self.channels_to_batch == True:
            x1 = x1.reshape(-1,1,*x1.shape[2:])
            x2 = x2.reshape(-1,1,*x1.shape[2:])
        y1 = self.forward(x1,ret=ret_string_1)
        y2 = self.forward_ema_stop_grad(x2,ret=ret_string_2)
                    
        losses = self.calculate_loss(y1,y2,*other_args)
        self.update_metrics(y1,y2,metrics)
        
        # loss is already symmetrised for VICReg and VICRegL
        if (self.vic_reg == False) and (self.vic_reg_local == False):
            y1_ = self.forward_ema_stop_grad(x1,ret=ret_string_1)
            y2_ = self.forward(x2,ret=ret_string_2)
            losses = losses + self.calculate_loss(y2_,y1_,*other_args)
            self.update_metrics(y2_,y1_,metrics)
        
        if self.ema is not None and train == True:
            self.ema.update(self)

        loss = self.safe_sum(losses)
        self.log(loss_str,loss,batch_size=x1.shape[0],
                 on_epoch=True,on_step=False,prog_bar=True,
                 sync_dist=True)
        
        if self.vic_reg_local == True:
            loss_str_list = self.loss_str_dict["vicregl"]
        elif self.vic_reg == True:
            loss_str_list = self.loss_str_dict["vicreg"]
        else:
            return loss
        for s,l in zip(loss_str_list,losses):
            if s is not None:
                sub_loss_str = "{}:{}".format(loss_str,s)
            else:
                sub_loss_str = loss_str
            self.log(sub_loss_str,l,batch_size=x1.shape[0],
                     on_epoch=True,on_step=False,prog_bar=True,
                     sync_dist=True)
        return loss

    def training_step(self,batch,batch_idx):
        loss = self.step(batch,"loss",self.train_metrics,train=True)
        return loss
    
    def validation_step(self,batch,batch_idx):
        loss = self.step(batch,"val_loss",self.val_metrics)
        return loss

    def test_step(self,batch,batch_idx):
        loss = self.step(batch,"test_loss",self.test_metrics)
        return loss

class NonContrastiveSelfSLUNetPL(UNet,NonContrastiveBasePL):
    """Operates a number of non-contrastive self-supervised learning 
    methods, all of which use non-contrastive approaches to self-supervised
    learning. Included here are:
    * SimSiam (the vanilla version of this class)
    * BYOL (when an ema module is specified - this should be an 
    ExponentialMovingAverage module)
    * VICReg (when vic_reg==True)
    * VICRegL (when vic_reg_local==True)
    
    This enables the training of a UNet backbone that can then be easily
    transferred to a UNet model.
    """
    def __init__(
        self,
        aug_image_key_1: str="aug_image_1",
        aug_image_key_2: str="aug_image_2",
        box_key_1: str="box_1",
        box_key_2: str="box_2",
        learning_rate: float=0.001,
        batch_size: int=4,
        weight_decay: float=0.005,
        training_dataloader_call: Callable=None,
        n_epochs: int=1000,
        ema: torch.nn.Module=None,
        vic_reg: bool=False,
        vic_reg_local: bool=False,
        vic_reg_loss_params: dict={},
        stop_gradient: bool=True,
        channels_to_batch: bool=False,
        *args,**kwargs):
        """
        Args:
            aug_image_key_1 (str, optional): key for augmented image 1. 
                Defaults to "aug_image_1".
            aug_image_key_2 (str, optional): key for augmented image 2. 
                Defaults to "aug_image_2".
            box_key_1 (str, optional): key for bounding box mapping 
                aug_image_key_1 to its original, uncropped image. (used only
                when vic_reg_local == True)
            box_key_2 (str, optional): key for bounding box mapping 
                aug_image_key_2 to its original, uncropped image. (used only
                when vic_reg_local == True)
            learning_rate (float, optional): learning rate. Defaults to 0.2.
            batch_size (int, optional): batch size. Defaults to 4.
            weight_decay (float, optional): weight decay for optimizer. 
                Defaults to 0.005.
            training_dataloader_call (Callable, optional): function that, when
                called, returns the training dataloader. Defaults to None.
            ema (float, torch.nn.Module): exponential moving decay module 
                (EMA). Must have an update method that takes model as input 
                and updates the weights based on this. Defaults to None.
            vic_reg (bool, optional): uses the VIC (variance, invariance,
                covariance) regularization method from Bardes et al. (2022).
                Defaults to False.
            vic_reg_local (bool, optional): uses the VICRegL method from 
                Bardes et al. (2022). Overrides vic_reg. Defaults to False.
            vic_reg_loss_params (dict, optional): parameters for the VICRegLoss
                module. Defaults to {} (the default parameters).
            stop_gradient (bool, optional): stops gradients when calculating
                losses. Useful for VICReg. Defaults to True.
            channels_to_batch (bool, optional): resizes the input such that 
                each channel becomes an element of the batch. Defaults to 
                False.
        """
        self.aug_image_key_1 = aug_image_key_1
        self.aug_image_key_2 = aug_image_key_2
        self.box_key_1 = box_key_1
        self.box_key_2 = box_key_2
        self.learning_rate = learning_rate
        self.batch_size = batch_size
        self.weight_decay = weight_decay
        self.training_dataloader_call = training_dataloader_call
        self.n_epochs = n_epochs
        self.ema = ema
        self.vic_reg = vic_reg
        self.vic_reg_local = vic_reg_local
        self.vic_reg_loss_params = vic_reg_loss_params
        self.stop_gradient = stop_gradient
        self.channels_to_batch = channels_to_batch
        
        if channels_to_batch == True:
            kwargs["n_channels"] = 1

        kwargs["encoder_only"] = True
        super().__init__(*args,**kwargs)

        if self.stop_gradient == False and self.vic_reg == False:
            warnings.warn("stop_gradient=False should not (in theory) be used\
                with vic_reg=False or vic_reg_local=False")

        self.init_loss()
        if self.ema is not None:
            self.ema.update(self)
        else:
            self.ema = None
        
        self.loss_str_dict = {
            "standard":[None],
            "vicreg":["inv","var","cov"],
            "vicregl":["inv","var","cov","local"]
        }
        
        self.setup_metrics()

    def forward_ema_stop_grad(self,x):
        if self.ema is not None:
            op = self.ema.shadow.forward
        else:
            op = self.forward
        if self.stop_gradient == True:
            with torch.no_grad():
                return op(x)
        else:
            return op(x)

    def step(self,batch,loss_str:str,metrics:dict,train=False):
        if self.vic_reg_local == False:
            other_args = []
        else:
            box_1 = batch[self.box_key_1]
            box_2 = batch[self.box_key_2]
            other_args = [box_1,box_2]

        x1,x2 = batch[self.aug_image_key_1],batch[self.aug_image_key_2]
        if self.channels_to_batch == True:
            x1 = x1.reshape(-1,1,*x1.shape[2:])
            x2 = x2.reshape(-1,1,*x1.shape[2:])
        y1 = self.forward(x1)
        y2 = self.forward_ema_stop_grad(x2)

        losses = self.calculate_loss(y1,y2,*other_args)
        self.update_metrics(y1,y2,metrics)
        
        # loss is already symmetrised for VICReg and VICRegL
        if self.vic_reg == False and self.vic_reg_local == False:
            y1_ = self.forward_ema_stop_grad(x1)
            y2_ = self.forward(x2)
            losses = losses + self.calculate_loss(y2_,y1_,*other_args)
            self.update_metrics(y2_,y1_,metrics)
        
        if self.ema is not None and train == True:
            self.ema.update(self)

        loss = self.safe_sum(losses)
        self.log(loss_str,loss,batch_size=x1.shape[0],
                 on_epoch=True,on_step=False,prog_bar=True,
                 sync_dist=True)
        
        if self.vic_reg_local == True:
            loss_str_list = self.loss_str_dict["vicregl"]
        elif self.vic_reg == True:
            loss_str_list = self.loss_str_dict["vicreg"]
        else:
            return loss

        for s,l in zip(loss_str_list,losses):
            if s is not None:
                sub_loss_str = "{}:{}".format(loss_str,s)
            else:
                sub_loss_str = loss_str
            self.log(sub_loss_str,l,batch_size=x1.shape[0],
                     on_epoch=True,on_step=False,prog_bar=True,
                     sync_dist=True)

        return loss

    def training_step(self,batch,batch_idx):
        loss = self.step(batch,"loss",self.train_metrics,train=True)
        return loss
    
    def validation_step(self,batch,batch_idx):
        loss = self.step(batch,"val_loss",self.val_metrics)
        return loss

    def test_step(self,batch,batch_idx):
        loss = self.step(batch,"test_loss",self.test_metrics)
        return loss
