import numpy as np
import torch
import torch.nn.functional as F
import torchmetrics
import pytorch_lightning as pl
from typing import Callable

from .classification import (
    CatNet,OrdNet,ordinal_prediction_to_class,SegCatNet)

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
            optimizer,patience=5,min_lr=1e-6,factor=0.5)

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

class SegCatNetPL(SegCatNet,pl.LightningModule):
    def __init__(self,
                 image_key: str="image",
                 label_key: str="label",
                 skip_conditioning_key: str=None,
                 feature_conditioning_key: str=None,
                 learning_rate: float=0.001,
                 batch_size: int=4,
                 weight_decay: float=0.005,
                 training_dataloader_call: Callable=None,
                 loss_fn: Callable=F.binary_cross_entropy_with_logits,
                 loss_params: dict={},
                 *args,**kwargs):

        super().__init__(*args,**kwargs)
        
        self.image_key = image_key
        self.label_key = label_key
        self.skip_conditioning_key = skip_conditioning_key
        self.feature_conditioning_key = feature_conditioning_key
        self.learning_rate = learning_rate
        self.batch_size = batch_size
        self.weight_decay = weight_decay
        self.trainig_dataloader_call = training_dataloader_call
        self.loss_fn = loss_fn
        self.loss_params = loss_params

        self.setup_metrics()
        
        self.loss_accumulator = 0.
        self.loss_accumulator_d = 0.
   
    def calculate_loss(self,prediction,y):
        y = y.type(torch.float32)
        if len(y.shape) > 1:
            y = y.squeeze(1)
        prediction = prediction.type(torch.float32)
        if len(prediction.shape) > 1:
            prediction = prediction.squeeze(1)
        if 'weight' in self.loss_params:
            weights = torch.ones_like(y)
            if len(self.loss_params['weight']) == 1:
                weights[y == 1] = self.loss_params['weight']
            else:
                weights = self.loss_params['weight'][y]
            loss_params = {'weight':weights}
        else:
            loss_params = {}
        loss = self.loss_fn(prediction,y,**loss_params)
        return loss.mean()

    def update_metrics(self,metrics,pred,y,**kwargs):
        y = y.long()
        if self.n_classes == 2:
            pred = F.sigmoid(pred)
        else:
            pred = F.softmax(pred,-1)
        for k in metrics:
            metrics[k].update(pred,y)
            self.log(k,metrics[k],**kwargs)

    def loss_wrapper(self,x,y,x_cond,x_fc):
        try: y = torch.round(y)
        except: y = torch.round(y.float())
        prediction = self.forward(
            x,X_skip_layer=x_cond,X_feature_conditioning=x_fc)
        prediction = torch.squeeze(prediction,1)
        if len(y.shape) > 1:
            y = torch.squeeze(y,1)
        batch_size = int(prediction.shape[0])
        if batch_size == 1:
            y = torch.unsqueeze(y,0)

        loss = self.calculate_loss(prediction,y)
        return prediction,loss

    def training_step(self,batch,batch_idx):
        x, y = batch[self.image_key],batch[self.label_key]
        if self.skip_conditioning_key is not None:
            x_cond = batch[self.skip_conditioning_key]
        else:
            x_cond = None
        if self.feature_conditioning_key is not None:
            x_fc = batch[self.feature_conditioning_key]
        else:
            x_fc = None

        pred_final,loss = self.loss_wrapper(x,y,x_cond,x_fc)

        self.log("train_loss", loss)
        try: y = torch.round(y).int()
        except: pass
        self.update_metrics(
            self.train_metrics,pred_final,y,
            on_epoch=True,on_step=False,prog_bar=True)
        return loss
    
    def validation_step(self,batch,batch_idx):
        x, y = batch[self.image_key],batch[self.label_key]
        if self.skip_conditioning_key is not None:
            x_cond = batch[self.skip_conditioning_key]
        else:
            x_cond = None
        if self.feature_conditioning_key is not None:
            x_fc = batch[self.feature_conditioning_key]
        else:
            x_fc = None

        pred_final,loss = self.loss_wrapper(x,y,x_cond,x_fc)

        self.loss_accumulator += loss
        self.loss_accumulator_d += 1.
        try: y = torch.round(y).int()
        except: pass
        self.update_metrics(
            self.val_metrics,pred_final,y,
            on_epoch=True,prog_bar=True)
        return loss

    def test_step(self,batch,batch_idx):
        x, y = batch[self.image_key],batch[self.label_key]
        if self.skip_conditioning_key is not None:
            x_cond = batch[self.skip_conditioning_key]
        else:
            x_cond = None
        if self.feature_conditioning_key is not None:
            x_fc = batch[self.feature_conditioning_key]
        else:
            x_fc = None

        pred_final,loss = self.loss_wrapper(x,y,x_cond,x_fc)

        try: y = torch.round(y).int()
        except: pass
        self.update_metrics(
            self.test_metrics,pred_final,y,
            on_epoch=True,on_step=False,prog_bar=True)
        return loss
        
    def train_dataloader(self) -> torch.utils.data.DataLoader:
        return self.training_dataloader_call()

    def configure_optimizers(self):
        optimizer = torch.optim.AdamW(
            self.parameters(),lr=self.learning_rate,
            weight_decay=self.weight_decay)
        lr_schedulers = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer,patience=10,min_lr=1e-6,factor=0.5)

        return {"optimizer":optimizer,
                "lr_scheduler":lr_schedulers,
                "monitor":"val_loss"}
    
    def on_validation_epoch_end(self):
        D = self.loss_accumulator_d
        if D > 0:
            val_loss = self.loss_accumulator/D
        else:
            val_loss = np.nan
        self.log("val_loss",val_loss,prog_bar=True)
        sch = self.lr_schedulers().state_dict()
        lr = self.learning_rate
        last_lr = sch['_last_lr'][0] if '_last_lr' in sch else lr
        self.log("lr",last_lr)
        self.loss_accumulator = 0.
        self.loss_accumulator_d = 0.

    def setup_metrics(self):
        if self.n_classes == 2:
            C_1,C_2,A,M,I = 2,None,None,"micro",None
        else:
            c = self.n_classes
            C_1,C_2,A,M,I = [c,c,"samplewise","macro",None]
        self.train_metrics = torch.nn.ModuleDict({})
        self.val_metrics = torch.nn.ModuleDict({})
        self.test_metrics = torch.nn.ModuleDict({})
        md = {"Pr":torchmetrics.Precision,
              "F1":torchmetrics.FBetaScore,
              "Re":torchmetrics.Recall,
              "AUC":torchmetrics.AUROC}
        for k in md:
            if k == "IoU":
                m,C = "macro",C_1
            else:
                m,C = M,C_2

            if k in ["F1"]:
                self.train_metrics[k] = md[k](
                    num_classes=C,mdmc_average=A,average=m,ignore_index=I).to(
                        self.device)
                self.val_metrics["V_"+k] = md[k](
                    num_classes=C,mdmc_average=A,average=m,ignore_index=I).to(
                        self.device)
            self.test_metrics["T_"+k] = md[k](
                num_classes=C,mdmc_average=A,average=m,ignore_index=I).to(
                    self.device)
