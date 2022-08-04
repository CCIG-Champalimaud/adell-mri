import torch
import torchmetrics
import pytorch_lightning as pl

from ..types import *

from .layers import ResNet
from .self_supervised import BarlowTwinsLoss,byol_loss, simsiam_loss

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
                    on_step=False,prog_bar=True)

    def training_step(self,batch,batch_idx):
        x1,x2 = batch[self.image_key],batch[self.augmented_image_key]
        y1,y2 = self.forward(x1),self.forward(x2)

        loss = self.calculate_loss(y1,y2)
        
        self.log("train_loss", loss)
        self.update_metrics(y1,y2,self.train_metrics)
        return loss
    
    def validation_step(self,batch,batch_idx):
        x1,x2 = batch[self.image_key],batch[self.augmented_image_key]
        y1,y2 = self.forward(x1),self.forward(x2)

        loss = self.calculate_loss(y1,y2,False)
        
        print(loss)
        self.loss_accumulator += loss.detach().cpu().numpy()
        self.loss_accumulator_d += 1

        self.log("val_loss", loss)
        self.update_metrics(y1,y2,self.val_metrics)
        return loss

    def test_step(self,batch,batch_idx):
        x1,x2 = batch[self.image_key],batch[self.augmented_image_key]
        y1,y2 = self.forward(x1),self.forward(x2)

        loss = self.calculate_loss(y1,y2,False)
        
        self.log("test_loss", loss)
        self.update_metrics(y1,y2,self.test_metrics)
        return loss

    def train_dataloader(self) -> torch.utils.data.DataLoader:
        return self.training_dataloader_call()

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
        for k in self.val_metrics: 
            val = self.val_metrics[k].compute()
            self.log(
                k,val,prog_bar=True)
            self.val_metrics[k].reset()
        val_loss = self.loss_accumulator/self.loss_accumulator_d
        sch = self.lr_schedulers().state_dict()
        lr = self.learning_rate
        last_lr = sch['_last_lr'][0] if '_last_lr' in sch else lr
        self.log("lr",last_lr)
        self.log("val_loss",val_loss,prog_bar=True)
        self.loss_accumulator = 0.
        self.loss_accumulator_d = 0.

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
            self.val_metrics["val"+k] = metric_dict[k]()
            self.test_metrics["test"+k] = metric_dict[k]()

class BootstrapYourOwnLatentPL(ResNet,pl.LightningModule):
    def __init__(
        self,
        aug_image_key_1: str="aug_image_1",
        aug_image_key_2: str="aug_image_2",
        learning_rate: float=0.001,
        batch_size: int=4,
        weight_decay: float=0.005,
        training_dataloader_call: Callable=None,
        n_epochs: int=1000,
        ema: torch.nn.Module=None,
        *args,**kwargs):
        """Class coordinating BYOL and SimSiam training (SimSiam can be seen
        as a special case of BYOL with no EMA teacher).

        Args:
            aug_image_key_1 (str, optional): key for augmented image 1. 
                Defaults to "aug_image_1".
            aug_image_key_2 (str, optional): key for augmented image 2. 
                Defaults to "aug_image_2".
            learning_rate (float, optional): learning rate. Defaults to 0.2.
            batch_size (int, optional): batch size. Defaults to 4.
            weight_decay (float, optional): weight decay for optimizer. 
                Defaults to 0.005.
            training_dataloader_call (Callable, optional): function that, when
                called, returns the training dataloader. Defaults to None.
            ema (float, optional): exponential moving decay module (EMA). Must
                have an update method that takes model as input and updates the
                weights based on this.
        """
        super().__init__(*args,**kwargs)

        self.aug_image_key_1 = aug_image_key_1
        self.aug_image_key_2 = aug_image_key_2
        self.learning_rate = learning_rate
        self.batch_size = batch_size
        self.weight_decay = weight_decay
        self.train_dataloader_call = training_dataloader_call
        self.n_epochs = n_epochs
        self.ema = ema

        self.loss = byol_loss if self.ema is not None else simsiam_loss
        if self.ema is not None:
            self.ema.update(self)
        else:
            self.ema = None
        
        self.setup_metrics()
        self.loss_accumulator = 0.
        self.loss_accumulator_d = 0.

    def calculate_loss(self,y1,y2):
        loss = self.loss(y1,y2.detach()) # the famous stop gradient operation
        return loss

    def update_metrics(self,y1,y2,metrics,log=True):
        # torchmetrics only allows for 1d vectors for regression (?????)
        y1 = y1.flatten()
        y2 = y2.flatten()
        for k in metrics:
            metrics[k].update(y1,y2)
            if log == True:
                self.log(
                    k,metrics[k],on_epoch=True,
                    on_step=False,prog_bar=True)

    def training_step(self,batch,batch_idx):
        x1,x2 = batch[self.aug_image_key_1],batch[self.aug_image_key_2]
        y1 = self.forward(x1,ret="prediction")
        y2 = self.forward(x2,ret="prediction")
        if self.ema is not None:
            with torch.no_grad():
                y1_ = self.ema.shadow.forward(x1)
                y2_ = self.ema.shadow.forward(x2)
        else:
            with torch.no_grad():
                y1_ = self.forward(x1)
                y2_ = self.forward(x2)
        loss = self.calculate_loss(y1,y2_)
        loss = loss + self.calculate_loss(y2,y1_)
        
        self.log("train_loss", loss, batch_size=x1.shape[0])
        self.update_metrics(y1,y2_,self.train_metrics)
        self.update_metrics(y2,y1_,self.train_metrics)

        if self.ema is not None:
            self.ema.update(self)
        
        return loss
    
    def validation_step(self,batch,batch_idx):
        x1,x2 = batch[self.aug_image_key_1],batch[self.aug_image_key_2]
        y1 = self.forward(x1,ret="prediction")
        y2 = self.forward(x2,ret="prediction")
        if self.ema is not None:
            y1_ = self.ema.shadow.forward(x1)
            y2_ = self.ema.shadow.forward(x2)
        else:
            y1_ = self.forward(x1)
            y2_ = self.forward(x2)
        loss = self.calculate_loss(y1,y2_)
        loss = loss + self.calculate_loss(y2,y1_)

        self.loss_accumulator += loss.detach().cpu().numpy()
        self.loss_accumulator_d += 1
        
        self.update_metrics(y1,y2_,self.val_metrics)
        self.update_metrics(y2,y1_,self.val_metrics)
        return loss

    def test_step(self,batch,batch_idx):
        x1,x2 = batch[self.aug_image_key_1],batch[self.aug_image_key_2]
        y1 = self.forward(x1,ret="prediction")
        y2 = self.forward(x2,ret="prediction")
        if self.ema is not None:
            y1_ = self.ema.shadow.forward(x1)
            y2_ = self.ema.shadow.forward(x2)
        else:
            y1_ = self.forward(x1)
            y2_ = self.forward(x2)
        loss = self.calculate_loss(y1,y2_)
        loss = loss + self.calculate_loss(y2,y1_)
        
        self.log("test_loss", loss)
        self.update_metrics(y1,y2_,self.test_metrics)
        self.update_metrics(y2,y1_,self.test_metrics)
        return loss

    def train_dataloader(self) -> torch.utils.data.DataLoader:
        return self.training_dataloader_call()

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
        for k in self.val_metrics: 
            val = self.val_metrics[k].compute()
            self.log(
                k,val,prog_bar=True)
            self.val_metrics[k].reset()
        if self.loss_accumulator_d > 0:
            val_loss = self.loss_accumulator/self.loss_accumulator_d
        else:
            val_loss = self.loss_accumulator
        sch = self.lr_schedulers().state_dict()
        lr = self.learning_rate
        last_lr = sch['_last_lr'][0] if '_last_lr' in sch else lr
        self.log("lr",last_lr)
        self.log("val_loss",val_loss,prog_bar=True)
        self.loss_accumulator = 0.
        self.loss_accumulator_d = 0.

    def setup_metrics(self):
        metric_dict = {
            #"MSE":torchmetrics.MeanSquaredError,
            #"R":torchmetrics.PearsonCorrCoef,
            }
        self.train_metrics = torch.nn.ModuleDict({})
        self.val_metrics = torch.nn.ModuleDict({})
        self.test_metrics = torch.nn.ModuleDict({})
        for k in metric_dict:
            self.train_metrics[k] = metric_dict[k]()
            self.val_metrics["V"+k] = metric_dict[k]()
            self.test_metrics["T"+k] = metric_dict[k]()
