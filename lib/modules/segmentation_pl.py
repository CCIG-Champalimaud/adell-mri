import torch
import torchmetrics
import pytorch_lightning as pl
from typing import Callable

from .segmentation import UNet,AHNet

class UNetPL(pl.LightningModule,UNet):
    def __init__(
        self,
        image_key: str="image",
        label_key: str="label",
        learning_rate: float=0.001,
        batch_size: int=4,
        weight_decay: float=0.005,
        training_dataloader_call: Callable=None,
        loss_fn: Callable=torch.nn.functional.binary_cross_entropy,
        loss_params: dict={},*args,**kwargs) -> torch.nn.Module:
        """Standard U-Net [1] implementation for Pytorch Lightning.

        Args:
            image_key (str): key corresponding to the key from the train
            dataloader.
            learning_rate (float, optional): learning rate. Defaults to 0.001.
            batch_size (int, optional): batch size. Defaults to 4.
            weight_decay (float, optional): weight decay for optimizer. Defaults 
            to 0.005.
            training_dataloader_call (Callable, optional): call for the 
            training dataloader. Defaults to None.
            loss_fn (Callable, optional): function to calculate the loss. 
            Defaults to torch.nn.functional.binary_cross_entropy.
            loss_params (dict, optional): additional parameters for the loss
            function. Defaults to {}.
            args: arguments for UNet class.
            kwargs: keyword arguments for UNet class.

        [1] https://www.nature.com/articles/s41592-018-0261-2

        Returns:
            pl.LightningModule: a U-Net module.
        """
        
        pl.LightningModule.__init__(self)
        UNet.__init__(self,*args,**kwargs)
        
        self.image_key = image_key
        self.label_key = label_key
        self.learning_rate = learning_rate
        self.batch_size = batch_size
        self.weight_decay = weight_decay
        self.training_dataloader_call = training_dataloader_call
        self.loss_fn = loss_fn
        self.loss_params = loss_params
        
        self.setup_metrics()
        
        self.loss_accumulator = 0.
        self.loss_accumulator_d = 0.
   
    def calculate_loss(self,prediction,y,weights=None):
        loss = self.loss_fn(prediction,y,**self.loss_params)
        return loss.mean()
    
    def training_step(self,batch,batch_idx):
        x, y = batch[self.image_key],batch[self.label_key]
        prediction = self.forward(x)
        prediction = torch.squeeze(prediction,1)
        y = torch.squeeze(y,1)
        batch_size = int(prediction.shape[0])
        if batch_size == 1:
            y = torch.unsqueeze(y,0)
        y = torch.squeeze(y,1)

        loss = self.calculate_loss(prediction,y)
            
        self.log("train_loss", loss)
        prediction = prediction
        try: y = torch.round(y).int()
        except: pass
        for k in self.train_metrics:
            self.train_metrics[k](prediction,y)
            self.log(
                k,self.train_metrics[k],on_epoch=True,
                on_step=False,prog_bar=True)
        return loss
    
    def validation_step(self,batch,batch_idx):
        x, y = batch[self.image_key],batch[self.label_key]
        prediction = self.forward(x)
        prediction = torch.squeeze(prediction,1)
        y = torch.squeeze(y,1)
        batch_size = int(prediction.shape[0])
        if batch_size == 1:
            y = torch.unsqueeze(y,0)
        y = torch.squeeze(y,1)

        loss = self.calculate_loss(prediction,y,None)

        self.loss_accumulator += loss
        self.loss_accumulator_d += 1.
        try: y = torch.round(y).int()
        except: pass
        for k in self.val_metrics:
            self.val_metrics[k].update(prediction,y)        
        return loss

    def test_step(self,batch,batch_idx):
        x, y = batch[self.image_key],batch[self.label_key]
        prediction = self.forward(x)
        prediction = torch.squeeze(prediction,1)
        y = torch.squeeze(y,1)
        batch_size = int(prediction.shape[0])
        if batch_size == 1:
            y = torch.unsqueeze(y,0)
        y = torch.squeeze(y,1)
                
        loss = self.calculate_loss(prediction,y,None)

        self.loss_accumulator += loss
        self.loss_accumulator_d += 1.
        try: y = torch.round(y).int()
        except: pass
        for k in self.test_metrics:
            self.test_metrics[k].update(prediction,y)        
        return loss

    def train_dataloader(self) -> torch.utils.data.DataLoader:
        return self.training_dataloader_call()

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(
            self.parameters(),lr=self.learning_rate,
            weight_decay=self.weight_decay)
        lr_schedulers = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer,patience=5,min_lr=1e-6,factor=0.2)

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
        if self.n_classes == 2:
            C_1 = 2
            C_2 = None
            A = None
            M = "micro"
        else:
            C_1 = self.n_classes
            C_2 = self.n_classes
            A = "samplewise"
            M = "macro"
        self.train_metrics = torch.nn.ModuleDict({
            "IoU":torchmetrics.JaccardIndex(C_1,mdmc_average=A),
            "Re":torchmetrics.Recall(C_2,mdmc_average=A,average=M),
            "Pr":torchmetrics.Precision(C_2,mdmc_average=A,average=M),
            "F1":torchmetrics.FBetaScore(C_2,mdmc_average=A,average=M)})
        self.val_metrics = torch.nn.ModuleDict({
            "VIoU":torchmetrics.JaccardIndex(C_1,mdmc_average=A),
            "VRe":torchmetrics.Recall(C_2,mdmc_average=A,average=M),
            "VPr":torchmetrics.Precision(C_2,mdmc_average=A,average=M),
            "VF1":torchmetrics.FBetaScore(C_2,mdmc_average=A,average=M)})
        self.test_metrics = torch.nn.ModuleDict({
            "Test IoU":torchmetrics.JaccardIndex(C_1,mdmc_average=A),
            "Test Re":torchmetrics.Recall(C_2,mdmc_average=A,average=M),
            "Test Pr":torchmetrics.Precision(C_2,mdmc_average=A,average=M),
            "Test F1":torchmetrics.FBetaScore(C_2,mdmc_average=A,average=M)})
