import numpy as np
import torch
import torch.nn.functional as F
import torchmetrics
import pytorch_lightning as pl
from typing import Callable
from copy import deepcopy

from .segmentation import UNet
from .segmentation_plus import UNetPlusPlus

class UNetPL(UNet,pl.LightningModule):
    def __init__(
        self,
        image_key: str="image",
        label_key: str="label",
        skip_conditioning_key: str=None,
        feature_conditioning_key: str=None,
        learning_rate: float=0.001,
        batch_size: int=4,
        weight_decay: float=0.005,
        training_dataloader_call: Callable=None,
        loss_fn: Callable=torch.nn.functional.binary_cross_entropy,
        loss_params: dict={},*args,**kwargs) -> torch.nn.Module:
        """Standard U-Net [1] implementation for Pytorch Lightning.

        Args:
            image_key (str): key corresponding to the input from the train
                dataloader.
            label_key (str): key corresponding to the label map from the train
                dataloader.
            skip_conditioning_key (str, optional): key corresponding to
                image which will be concatenated to the skip connections. 
            feature_conditioning_key (str, optional): key corresponding to
                the tabular features which will be used in the feature
                conditioning.
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
        
        super().__init__(*args,**kwargs)
        
        self.image_key = image_key
        self.label_key = label_key
        self.skip_conditioning_key = skip_conditioning_key
        self.feature_conditioning_key = feature_conditioning_key
        self.learning_rate = learning_rate
        self.batch_size = batch_size
        self.weight_decay = weight_decay
        self.training_dataloader_call = training_dataloader_call
        self.loss_fn = loss_fn
        self.loss_params = loss_params
        
        self.loss_fn_class = torch.nn.BCEWithLogitsLoss()
        self.setup_metrics()
        
        self.loss_accumulator = 0.
        self.loss_accumulator_class = 0.
        self.loss_accumulator_d = 0.
        self.bn_mult = 0.01
   
    def calculate_loss(self,prediction,y):
        loss = self.loss_fn(prediction,y,**self.loss_params)
        return loss.mean()

    def calculate_loss_class(self,prediction,y):
        y = y.type(torch.float32)
        prediction = prediction.squeeze(1).type(torch.float32)
        loss = self.loss_fn_class(prediction,y)
        return loss.mean()

    def update_metrics(self,metrics,pred,y,pred_class,y_class,**kwargs):
        y = y.long()
        if y_class is not None:
            y_class = y_class.long()
        for k in metrics:
            if 'cl:' in k:
                metrics[k].update(pred_class,y_class)
            else:    
                metrics[k].update(pred,y)
            self.log(k,metrics[k],**kwargs)

    def loss_wrapper(self,x,y,y_class,x_cond,x_fc):
        y = torch.round(y)
        prediction,pred_class = self.forward(
            x,X_skip_layer=x_cond,X_feature_conditioning=x_fc)
        prediction = torch.squeeze(prediction,1)
        y = torch.squeeze(y,1)
        batch_size = int(prediction.shape[0])
        if batch_size == 1:
            y = torch.unsqueeze(y,0)
        y = torch.squeeze(y,1)

        loss = self.calculate_loss(prediction,y)
        if self.bottleneck_classification == True:
            class_loss = self.calculate_loss_class(pred_class,y_class)
            output_loss = loss + class_loss * self.bn_mult
        else:
            class_loss = None
            output_loss = loss
        return prediction,pred_class,loss,class_loss,output_loss

    def training_step(self,batch,batch_idx):
        x, y = batch[self.image_key],batch[self.label_key]
        if self.skip_conditioning_key is not None:
            x_cond = batch[self.skip_conditioning_key]
        else:
            x_cond = None
        if self.bottleneck_classification == True:
            y_class = y.flatten(start_dim=1).max(1).values
        else:
            y_class = None
        if self.feature_conditioning_key is not None:
            x_fc = batch[self.feature_conditioning_key]
        else:
            x_fc = None

        pred_final,pred_class,loss,class_loss,output_loss = self.loss_wrapper(
            x,y,y_class,x_cond,x_fc)

        self.log("train_loss", loss)
        if class_loss is not None:
            self.log("train_cl_loss",class_loss)
        try: y = torch.round(y).int()
        except: pass
        self.update_metrics(
            self.train_metrics,pred_final,y,pred_class,y_class,
            on_epoch=True,on_step=False,prog_bar=True)
        return output_loss
    
    def validation_step(self,batch,batch_idx):
        x, y = batch[self.image_key],batch[self.label_key]
        if self.skip_conditioning_key is not None:
            x_cond = batch[self.skip_conditioning_key]
        else:
            x_cond = None
        if self.bottleneck_classification == True:
            y_class = y.flatten(start_dim=1).max(1).values
        else:
            y_class = None
        if self.feature_conditioning_key is not None:
            x_fc = batch[self.feature_conditioning_key]
        else:
            x_fc = None

        pred_final,pred_class,loss,class_loss,output_loss = self.loss_wrapper(
            x,y,y_class,x_cond,x_fc)

        self.loss_accumulator += loss
        if self.bottleneck_classification == True:
            self.loss_accumulator_class += class_loss
        self.loss_accumulator_d += 1.
        try: y = torch.round(y).int()
        except: pass
        self.update_metrics(
            self.val_metrics,pred_final,y,pred_class,y_class,
            on_epoch=True,prog_bar=True)
        return output_loss

    def test_step(self,batch,batch_idx):
        x, y = batch[self.image_key],batch[self.label_key]
        if self.skip_conditioning_key is not None:
            x_cond = batch[self.skip_conditioning_key]
        else:
            x_cond = None
        if self.bottleneck_classification == True:
            y_class = y.flatten(start_dim=1).max(1).values
        else:
            y_class = None
        if self.feature_conditioning_key is not None:
            x_fc = batch[self.feature_conditioning_key]
        else:
            x_fc = None

        pred_final,pred_class,loss,class_loss,output_loss = self.loss_wrapper(
            x,y,y_class,x_cond,x_fc)

        self.loss_accumulator += loss
        if self.bottleneck_classification == True:
            self.loss_accumulator_class += class_loss
        self.loss_accumulator_d += 1.
        try: y = torch.round(y).int()
        except: pass
        self.update_metrics(
            self.test_metrics,pred_final,y,pred_class,y_class,
            on_epoch=True,on_step=False,prog_bar=True)
        return output_loss
        
    def train_dataloader(self) -> torch.utils.data.DataLoader:
        return self.training_dataloader_call()

    def configure_optimizers(self):
        optimizer = torch.optim.AdamW(
            self.parameters(),lr=self.learning_rate,
            weight_decay=self.weight_decay)
        lr_schedulers = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer,patience=3,min_lr=1e-6,factor=0.5)

        return {"optimizer":optimizer,
                "lr_scheduler":lr_schedulers,
                "monitor":"val_loss"}
    
    def on_validation_epoch_end(self):
        for k in self.val_metrics:
            self.val_metrics[k].reset()
        D = self.loss_accumulator_d
        if D > 0:
            val_loss = self.loss_accumulator/D
            if self.bottleneck_classification == True:
                val_loss_class = self.loss_accumulator_class/D
                self.log("val_cl_loss",val_loss_class,prog_bar=True)
        else:
            val_loss = np.nan
        self.log("val_loss",val_loss,prog_bar=True)
        sch = self.lr_schedulers().state_dict()
        lr = self.learning_rate
        last_lr = sch['_last_lr'][0] if '_last_lr' in sch else lr
        self.log("lr",last_lr)
        self.loss_accumulator = 0.
        self.loss_accumulator_class = 0.
        self.loss_accumulator_d = 0.

    def setup_metrics(self):
        if self.n_classes == 2:
            C_1,C_2,A,M,I = 2,None,None,"micro",0
        else:
            C_1,C_2,A,M,I = [
                self.n_classes,self.n_classes,"samplewise","macro",None]
        self.train_metrics = torch.nn.ModuleDict({})
        self.val_metrics = torch.nn.ModuleDict({})
        self.test_metrics = torch.nn.ModuleDict({})
        md = {"IoU":torchmetrics.JaccardIndex,
              "Pr":torchmetrics.Precision,
              "F1":torchmetrics.FBetaScore}
        if self.bottleneck_classification == True:
            md["cl:F1"] = torchmetrics.FBetaScore
            md["cl:Re"] = torchmetrics.Recall
        for k in md:
            if k == "IoU":
                m,C = "macro",C_1
            else:
                m,C = M,C_2
            if k in ["cl:Re","cl:F1"]:
                I_ = None
            else:
                I_ = I
            if k in ["IoU","cl:F1"]:
                self.train_metrics[k] = md[k](
                    C,mdmc_average=A,average=m,ignore_index=I_).to(self.device)
                self.val_metrics["V_"+k] = md[k](
                    C,mdmc_average=A,average=m,ignore_index=I_).to(self.device)
            self.test_metrics["T_"+k] = md[k](
                C,mdmc_average=A,average=m,ignore_index=I_).to(self.device)

class UNetPlusPlusPL(UNetPlusPlus,pl.LightningModule):
    def __init__(
        self,
        image_key: str="image",
        label_key: str="label",
        skip_conditioning_key: str=None,
        feature_conditioning_key: str=None,
        learning_rate: float=0.001,
        batch_size: int=4,
        weight_decay: float=0.005,
        training_dataloader_call: Callable=None,
        loss_fn: Callable=torch.nn.functional.binary_cross_entropy,
        loss_params: dict={},*args,**kwargs) -> torch.nn.Module:
        """Standard U-Net++ [1] implementation for Pytorch Lightning.

        Args:
            image_key (str): key corresponding to the input from the train
                dataloader.
            label_key (str): key corresponding to the label map from the train
                dataloader.
            skip_conditioning_key (str, optional): key corresponding to
                image which will be concatenated to the skip connections.
            feature_conditioning_key (str, optional): key corresponding to
                the tabular features which will be used in the feature
                conditioning.
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
        
        super().__init__(*args,**kwargs)
        
        self.image_key = image_key
        self.label_key = label_key
        self.skip_conditioning_key = skip_conditioning_key
        self.feature_conditioning_key = feature_conditioning_key
        self.learning_rate = learning_rate
        self.batch_size = batch_size
        self.weight_decay = weight_decay
        self.training_dataloader_call = training_dataloader_call
        self.loss_fn = loss_fn
        self.loss_params = loss_params
        
        self.loss_fn_class = torch.nn.BCEWithLogitsLoss()
        self.setup_metrics()

        self.loss_accumulator = 0.
        self.loss_accumulator_class = 0.
        self.loss_accumulator_d = 0.
        self.bn_mult = 0.01
   
    def calculate_loss(self,prediction,prediction_aux,y):
        loss = self.loss_fn(prediction,y,**self.loss_params)
        n = len(prediction_aux)
        for i,p in enumerate(prediction_aux):
            l = self.loss_fn(p,y,**self.loss_params)
            # scales based on how deep the features are
            loss = loss + l / (2**(n-i+1))
        return loss.mean()
    
    def loss_wrapper(self,x,y,y_class,x_cond,x_fc):
        y = torch.round(y)
        prediction,pred_class = self.forward(
            x,X_skip_layer=x_cond,X_feature_conditioning=x_fc,
            return_aux=True)
        prediction = torch.squeeze(prediction,1)
        prediction_aux = [torch.squeeze(x,1) for x in prediction_aux]
        y = torch.squeeze(y,1)
        batch_size = int(prediction.shape[0])
        if batch_size == 1:
            y = torch.unsqueeze(y,0)
        y = torch.squeeze(y,1)

        loss = self.calculate_loss(prediction,prediction_aux,y)
        if self.bottleneck_classification == True:
            class_loss = self.calculate_loss_class(pred_class,y_class)
            output_loss = loss + class_loss * self.bn_mult
        else:
            class_loss = None
            output_loss = loss

        n = len(prediction_aux)
        D = [1/(2**(n-i+1)) for i in range(n)]
        pred_final = torch.add(
            prediction,
            sum([x*d for x,d in zip(prediction_aux,D)]))
        pred_final = pred_final / (1+sum(D))

        return pred_final,pred_class,output_loss,loss,class_loss

    def calculate_loss_class(self,prediction,y):
        loss = self.loss_fn_class(
            prediction.squeeze(1),y.type(torch.int32))
        return loss.mean()

    def update_metrics(self,metrics,pred,y,pred_class,y_class,**kwargs):
        y = y.long()
        if y_class is not None:
            y_class = y_class.long()
        for k in metrics:
            if 'cl:' in k:
                metrics[k].update(pred_class,y_class)
            else:    
                metrics[k].update(pred,y)
            self.log(k,metrics[k],**kwargs)

    def training_step(self,batch,batch_idx):
        x, y = batch[self.image_key],batch[self.label_key]
        if self.skip_conditioning_key is not None:
            x_cond = batch[self.skip_conditioning_key]
        else:
            x_cond = None
        if self.bottleneck_classification == True:
            y_class = y.flatten(start_dim=1).max(1).values
        else:
            y_class = None
        if self.feature_conditioning_key is not None:
            x_fc = batch[self.feature_conditioning_key]
        else:
            x_fc = None

        pred_final,pred_class,output_loss,loss,class_loss = self.loss_wrapper(
            x,y,y_class,x_cond,x_fc)

        if class_loss is not None:
            self.log("train_cl_loss",class_loss)
        self.log("train_loss", loss)
        try: y = torch.round(y).int()
        except: pass
        self.update_metrics(
            self.train_metrics,pred_final,y,pred_class,y_class,
            on_epoch=True,on_step=False,prog_bar=True)
        return output_loss
    
    def validation_step(self,batch,batch_idx):
        x, y = batch[self.image_key],batch[self.label_key]
        if self.skip_conditioning_key is not None:
            x_cond = batch[self.skip_conditioning_key]
        else:
            x_cond = None
        if self.bottleneck_classification == True:
            y_class = y.flatten(start_dim=1).max(1).values
        else:
            y_class = None
        if self.feature_conditioning_key is not None:
            x_fc = batch[self.feature_conditioning_key]
        else:
            x_fc = None
        
        pred_final,pred_class,output_loss,loss,class_loss = self.loss_wrapper(
            x,y,y_class,x_cond,x_fc)

        self.loss_accumulator += loss
        if self.bottleneck_classification == True:
            self.loss_accumulator_class += class_loss
        self.loss_accumulator_d += 1.
        try: y = torch.round(y).int()
        except: pass
        self.update_metrics(
            self.val_metrics,pred_final,y,pred_class,y_class,
            on_epoch=True,prog_bar=True)
        return output_loss

    def test_step(self,batch,batch_idx):
        x, y = batch[self.image_key],batch[self.label_key]
        if self.skip_conditioning_key is not None:
            x_cond = batch[self.skip_conditioning_key]
        else:
            x_cond = None
        if self.bottleneck_classification == True:
            y_class = y.flatten(start_dim=1).max(1).values
        else:
            y_class = None
        if self.feature_conditioning_key is not None:
            x_fc = batch[self.feature_conditioning_key]
        else:
            x_fc = None
            
        pred_final,pred_class,output_loss,loss,class_loss = self.loss_wrapper(
            x,y,y_class,x_cond,x_fc)

        self.loss_accumulator += loss
        if self.bottleneck_classification == True:
            self.loss_accumulator_class += class_loss
        self.loss_accumulator_d += 1.
        try: y = torch.round(y).int()
        except: pass
        self.update_metrics(
            self.test_metrics,pred_final,y,pred_class,y_class)
        return output_loss
        
    def train_dataloader(self) -> torch.utils.data.DataLoader:
        return self.training_dataloader_call()

    def configure_optimizers(self):
        optimizer = torch.optim.AdamW(
            self.parameters(),lr=self.learning_rate,
            weight_decay=self.weight_decay)
        lr_schedulers = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer,patience=3,min_lr=1e-6,factor=0.5)

        return {"optimizer":optimizer,
                "lr_scheduler":lr_schedulers,
                "monitor":"val_loss"}
    
    def on_validation_epoch_end(self):
        if self.loss_accumulator_d > 0:
            val_loss = self.loss_accumulator/self.loss_accumulator_d
        else:
            val_loss = np.nan
        sch = self.lr_schedulers().state_dict()
        lr = self.learning_rate
        last_lr = sch['_last_lr'][0] if '_last_lr' in sch else lr
        self.log("lr",last_lr)
        self.log("val_loss",val_loss,prog_bar=True)
        self.loss_accumulator = 0.
        self.loss_accumulator_d = 0.

    def setup_metrics(self):
        if self.n_classes == 2:
            C_1,C_2,A,M,I = 2,None,None,"micro",0
        else:
            C_1,C_2,A,M,I = [
                self.n_classes,self.n_classes,"samplewise","macro",None]
        self.train_metrics = torch.nn.ModuleDict({})
        self.val_metrics = torch.nn.ModuleDict({})
        self.test_metrics = torch.nn.ModuleDict({})
        md = {"IoU":torchmetrics.JaccardIndex,
              "Pr":torchmetrics.Precision,
              "F1":torchmetrics.FBetaScore}
        if self.bottleneck_classification == True:
            md["cl:F1"] = torchmetrics.FBetaScore
            md["cl:Re"] = torchmetrics.Recall
        for k in md:
            if k == "IoU":
                m,C = "macro",C_1
            else:
                m,C = M,C_2
            self.train_metrics[k] = md[k](
                C,mdmc_average=A,average=m,ignore_index=I).to(self.device)
            self.val_metrics["V_"+k] = md[k](
                C,mdmc_average=A,average=m,ignore_index=I).to(self.device)
            self.test_metrics["T_"+k] = md[k](
                C,mdmc_average=A,average=m,ignore_index=I).to(self.device)
