import numpy as np
import torch
import torch.nn.functional as F
import torchmetrics
import pytorch_lightning as pl
from typing import Callable

from .classification import CatNet,OrdNet,ordinal_prediction_to_class

class ClassNetPL(pl.LightningModule):
    def __init__(
        self,
        net_type: str="cat",
        image_key: str="image",
        label_key: str="label",
        learning_rate: float=0.001,
        batch_size: int=4,
        weight_decay: float=0.005,
        training_dataloader_call: Callable=None,
        loss_fn: Callable=F.binary_cross_entropy,
        loss_params: dict={},
        *args,**kwargs) -> torch.nn.Module:
        """YOLO-like network implementation for Pytorch Lightning.

        Args:
            image_key (str): key corresponding to the key from the train
                dataloader.
            label_key (str): key corresponding to the label key from the train
                dataloader.
            learning_rate (float, optional): learning rate. Defaults to 0.001.
                batch_size (int, optional): batch size. Defaults to 4.
            weight_decay (float, optional): weight decay for optimizer. Defaults 
                to 0.005.
            training_dataloader_call (Callable, optional): call for the 
                training dataloader. Defaults to None.
            loss_fn (Callable, optional): loss function. Defaults to 
                F.binary_cross_entropy
            loss_params (dict, optional): classification loss parameters. 
                Defaults to {}.
            args: arguments for classification network class.
            kwargs: keyword arguments for classification network class.

        Returns:
            pl.LightningModule: a classification network module.
        """
        
        super().__init__()

        self.net_type = net_type[:3]
        self.image_key = image_key
        self.label_key = label_key
        self.learning_rate = learning_rate
        self.batch_size = batch_size
        self.weight_decay = weight_decay
        self.training_dataloader_call = training_dataloader_call
        self.loss_fn = loss_fn
        self.loss_params = loss_params
        self.args = args
        self.kwargs = kwargs
        
        self.setup_network()
        self.setup_metrics()
        
        self.loss_accumulator = 0.
        self.loss_accumulator_d = 0.
   
    def setup_network(self):
        if self.net_type == "cat":
            self.network = CatNet(*self.args,**self.kwargs)
        elif self.net_type == "ord":
            self.network = OrdNet(*self.args,**self.kwargs)
        else:
            raise Exception("net_type '{}' not valid, has to be one of \
                ['ord','cat']".format(self.net_type))
        self.forward = self.network.forward
        self.n_classes = self.network.n_classes

    def calculate_loss(self,prediction,y,with_params=True):
        if self.n_classes > 2:
            if len(y.shape) > 1:
                y = y.squeeze(1)
            y = y.to(torch.int64)
        else:
            y = y.float()
        if with_params == True:
            loss = self.loss_fn(prediction,y,**self.loss_params)
        else:
            loss = self.loss_fn(prediction,y)
        return loss.mean()

    def update_metrics(self,prediction,y,metrics,log=True):
        if self.net_type == "ord":
            prediction = ordinal_prediction_to_class(prediction)
        elif self.n_classes > 2:
            prediction = torch.argmax(prediction,1).to(torch.int64)
        if len(y.shape) > 1:
            y.squeeze(1)
        for k in metrics:
            metrics[k].update(prediction,y)
            if log == True:
                self.log(
                    k,metrics[k],on_epoch=True,
                    on_step=False,prog_bar=True)

    def training_step(self,batch,batch_idx):
        x, y = batch[self.image_key],batch[self.label_key]
        prediction = self.forward(x)
        prediction = torch.squeeze(prediction,1)

        loss = self.calculate_loss(prediction,y)
        
        self.log("train_loss", loss)
        self.update_metrics(prediction,y,self.train_metrics)
        return loss
    
    def validation_step(self,batch,batch_idx):
        x, y = batch[self.image_key],batch[self.label_key]
        prediction = self.forward(x)
        prediction = torch.squeeze(prediction,1)

        loss = self.calculate_loss(prediction,y)

        self.loss_accumulator += loss.detach().cpu().numpy()
        self.loss_accumulator_d += 1
        
        self.update_metrics(prediction,y,self.val_metrics,log=False)
        return loss

    def test_step(self,batch,batch_idx):
        x, y = batch[self.image_key],batch[self.label_key]
        prediction = self.forward(x)
        prediction = torch.squeeze(prediction,1)

        loss = self.calculate_loss(prediction,y)
            
        self.update_metrics(prediction,y,self.test_metrics,log=False)
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
        if self.n_classes == 2:
            C,A,M = None,None,"micro"
        else:
            C,A,M = self.n_classes,None,"macro"
        metric_dict = {
            "Rec":torchmetrics.Recall,
            "Prec":torchmetrics.Precision,
            "F1":torchmetrics.F1Score}
        self.train_metrics = torch.nn.ModuleDict({})
        self.val_metrics = torch.nn.ModuleDict({})
        self.test_metrics = torch.nn.ModuleDict({})
        for k in metric_dict:
            self.train_metrics[k] = metric_dict[k](
                C,mdmc_average=A,average=M)
            self.val_metrics["val"+k] = metric_dict[k](
                C,mdmc_average=A,average=M)
            if self.n_classes > 2:
                self.test_metrics["test"+k] = metric_dict[k](
                    C,mdmc_average=A,average="none")
            else:
                self.test_metrics["test"+k] = metric_dict[k](
                    C,mdmc_average=A,average=M)