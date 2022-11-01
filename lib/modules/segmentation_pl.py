from pyexpat.errors import XML_ERROR_INCOMPLETE_PE
import numpy as np
import torch
import torch.nn.functional as F
import torchmetrics
import pytorch_lightning as pl
from typing import Callable
from copy import deepcopy
from picai_eval import evaluate

from .segmentation import UNet,BrUNet
from .segmentation_plus import UNetPlusPlus
from .learning_rate import polynomial_lr_decay
from .extract_lesion_candidates import extract_lesion_candidates

def split(x,n_splits,dim):
    size = int(x.shape[dim]//n_splits)
    return torch.split(x,size,dim)

def get_lesions(x):
    return extract_lesion_candidates(x)[0]

def update_metrics(cls,metrics,pred,y,pred_class,y_class,**kwargs):
    try: y = torch.round(y).int()
    except: pass
    y = y.long()
    p = pred.detach()
    if y_class is not None:
        y_class = y_class.long()
        pc = pred_class.detach()
    for k in metrics:
        if 'cl:' in k:
            if cls.n_classes == 2:
                pred_class = F.sigmoid(pc)
            else:
                pred_class = F.softmax(pc,1)
            metrics[k].update(pc,y_class)
        else:    
            metrics[k].update(p,y)
        cls.log(k,metrics[k],**kwargs,batch_size=y.shape[0],
                sync_dist=True)

class UNetPL(UNet,pl.LightningModule):
    def __init__(
        self,
        image_key: str="image",
        label_key: str="label",
        skip_conditioning_key: str=None,
        feature_conditioning_key: str=None,
        learning_rate: float=0.001,
        batch_size: int=4,
        n_epochs: int=100,
        weight_decay: float=0.005,
        training_dataloader_call: Callable=None,
        loss_fn: Callable=torch.nn.functional.binary_cross_entropy,
        loss_params: dict={},
        tta: bool=False,
        lr_encoder: float=None,*args,**kwargs) -> torch.nn.Module:
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
            n_epochs (int, optional): number of epochs. Defaults to 100.
            weight_decay (float, optional): weight decay for optimizer. Defaults 
                to 0.005.
            training_dataloader_call (Callable, optional): call for the 
            training dataloader. Defaults to None.
            loss_fn (Callable, optional): function to calculate the loss. 
                Defaults to torch.nn.functional.binary_cross_entropy.
            loss_params (dict, optional): additional parameters for the loss
                function. Defaults to {}.
            tta (bool, optional): test-time augmentation. Defaults to False.
            lr_encoder (float, optional): encoder learning rate.
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
        self.n_epochs = n_epochs
        self.weight_decay = weight_decay
        self.training_dataloader_call = training_dataloader_call
        self.loss_fn = loss_fn
        self.loss_params = loss_params
        self.tta = tta
        self.lr_encoder = lr_encoder
        
        self.loss_fn_class = torch.nn.BCEWithLogitsLoss()
        self.setup_metrics()

        # for metrics (AUC, AP) which require a list of predictions + gt
        self.all_pred = []
        self.all_true = []
        
        self.bn_mult = 0.1
   
    def calculate_loss(self,prediction,y):
        loss = self.loss_fn(prediction,y,**self.loss_params)
        return loss.mean()

    def calculate_loss_class(self,prediction,y):
        y = y.type(torch.float32)
        prediction = prediction.squeeze(1).type(torch.float32)
        loss = self.loss_fn_class(prediction,y)
        return loss.mean()

    def loss_wrapper(self,x,y,y_class,x_cond,x_fc):
        y = torch.round(y)
        output = self.forward(
            x,X_skip_layer=x_cond,X_feature_conditioning=x_fc)
        if self.deep_supervision == False:
            prediction,pred_class = output
        else:
            prediction,pred_class,deep_outputs = output
            deep_outputs = [torch.squeeze(o,1) for o in deep_outputs]
        prediction = torch.squeeze(prediction,1)
        y = torch.squeeze(y,1)
        batch_size = int(prediction.shape[0])
        if batch_size == 1:
            y = torch.unsqueeze(y,0)
        y = torch.squeeze(y,1)

        loss = self.calculate_loss(prediction,y)
        if self.deep_supervision == True:
            t = len(deep_outputs)
            additional_losses = torch.zeros_like(loss)
            for i,o in enumerate(deep_outputs):
                S = o.shape[-self.spatial_dimensions:]
                y_small = F.interpolate(torch.unsqueeze(y,1),S,mode="nearest")
                y_small = torch.squeeze(y_small,1)
                l = self.calculate_loss(o,y_small).mean()/(2**(t-i))/t
                additional_losses = additional_losses + l
            loss = loss + additional_losses
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
        
        self.log("train_loss", loss,batch_size=y.shape[0],
                 sync_dist=True)
        if class_loss is not None:
            self.log("train_cl_loss",class_loss,batch_size=y.shape[0],
                     sync_dist=True)

        update_metrics(
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
        
        for s_p,s_y in zip(pred_final.detach().cpu().numpy(),
                           y.squeeze(1).detach().cpu().numpy()):
            self.all_pred.append(s_p)
            self.all_true.append(s_y)

        self.log("val_loss",loss,prog_bar=True,
                 on_epoch=True,batch_size=y.shape[0],
                 sync_dist=True)

        update_metrics(
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

        for s_p,s_y in zip(pred_final.detach().cpu().numpy(),
                           y.squeeze(1).detach().cpu().numpy()):
            self.all_pred.append(s_p)
            self.all_true.append(s_y)

        update_metrics(
            self.test_metrics,pred_final,y,pred_class,y_class,
            on_epoch=True,on_step=False,prog_bar=True)
        return output_loss
        
    def train_dataloader(self) -> torch.utils.data.DataLoader:
        return self.training_dataloader_call()

    def configure_optimizers(self):
        encoder_params = []
        rest_of_params = []
        for k,p in self.named_parameters():
            if "encoding" in k or "encoder" in k:
                encoder_params.append(p)
            else:
                rest_of_params.append(p)
        if self.lr_encoder is None:
            lr_encoder = self.learning_rate
        else:
            lr_encoder = self.lr_encoder
        parameters = [
            {'params': encoder_params,'lr':lr_encoder},
            {'params': rest_of_params}]
        optimizer = torch.optim.AdamW(
            parameters,lr=self.learning_rate,
            weight_decay=self.weight_decay)

        return {"optimizer":optimizer,
                "monitor":"val_loss"}
    
    def on_train_epoch_end(self):
        # updating the lr here rather than as a PL lr_scheduler... 
        # basically the lr_scheduler (as I was using it at least)
        # is not terribly compatible with starting and stopping training
        opt = self.optimizers()
        polynomial_lr_decay(opt,self.current_epoch,initial_lr=self.learning_rate,
                            max_decay_steps=self.n_epochs,end_lr=1e-6,power=0.9)
        try:
            last_lr = [x["lr"] for x in opt.param_groups][-1]
        except:
            last_lr = self.learning_rate
        self.log("lr",last_lr,prog_bar=True,sync_dist=True)

    def on_validation_epoch_end(self):
        picai_eval_metrics = evaluate(
            y_det=self.all_pred,y_true=self.all_true,
            y_det_postprocess_func=get_lesions,
            num_parallel_calls=8)
        self.log("V_AP",picai_eval_metrics.AP,prog_bar=True,
                 sync_dist=True)
        self.log("V_R",picai_eval_metrics.score,prog_bar=True,
                 sync_dist=True)
        self.log("V_AUC",picai_eval_metrics.auroc,prog_bar=True,
                 sync_dist=True)
        self.all_pred = []
        self.all_true = []

    def on_test_epoch_end(self):
        picai_eval_metrics = evaluate(
            y_det=self.all_pred,y_true=self.all_true,
            y_det_postprocess_func=get_lesions,
            num_parallel_calls=8)
        self.log("T_AP",picai_eval_metrics.AP,prog_bar=True,
                 sync_dist=True)
        self.log("T_R",picai_eval_metrics.score,prog_bar=True,
                 sync_dist=True)
        self.log("T_AUC",picai_eval_metrics.auroc,prog_bar=True,
                 sync_dist=True)
        self.all_pred = []
        self.all_true = []

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
              "F1":torchmetrics.FBetaScore,
              "Dice":torchmetrics.Dice}
        if self.bottleneck_classification == True:
            md["AUC_bn"] = torchmetrics.AUROC
        for k in md:
            if k == "IoU":
                m,C = "macro",C_1
            elif k == "Dice":
                m,C = "micro",C_2
            else:
                m,C = M,C_2
            if "bn" in k:
                I_ = None
            else:
                I_ = I
            if k in []:
                self.train_metrics[k] = md[k](
                    num_classes=C,mdmc_average=A,average=m,
                    ignore_index=I_).to(self.device)
            if k in ["IoU","AUC_bn"]:
                self.val_metrics["V_"+k] = md[k](
                    num_classes=C,mdmc_average=A,average=m,
                    ignore_index=I_).to(self.device)
            self.test_metrics["T_"+k] = md[k](
                num_classes=C,mdmc_average=A,average=m,
                ignore_index=I_).to(self.device)

class UNetPlusPlusPL(UNetPlusPlus,pl.LightningModule):
    def __init__(
        self,
        image_key: str="image",
        label_key: str="label",
        skip_conditioning_key: str=None,
        feature_conditioning_key: str=None,
        learning_rate: float=0.001,
        batch_size: int=4,
        n_epochs: int=100,
        weight_decay: float=0.005,
        training_dataloader_call: Callable=None,
        loss_fn: Callable=torch.nn.functional.binary_cross_entropy,
        loss_params: dict={},
        tta: bool=False,
        lr_encoder: float=None,*args,**kwargs) -> torch.nn.Module:
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
            n_epochs (int, optional): number of epochs. Defaults to 100.
            weight_decay (float, optional): weight decay for optimizer. Defaults 
                to 0.005.
            training_dataloader_call (Callable, optional): call for the 
            training dataloader. Defaults to None.
            loss_fn (Callable, optional): function to calculate the loss. 
                Defaults to torch.nn.functional.binary_cross_entropy.
            loss_params (dict, optional): additional parameters for the loss
                function. Defaults to {}.
            tta (bool, optional): test-time augmentation. Defaults to False.
            lr_encoder (float, optional): encoder learning rate.
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
        self.n_epochs = n_epochs
        self.weight_decay = weight_decay
        self.training_dataloader_call = training_dataloader_call
        self.loss_fn = loss_fn
        self.loss_params = loss_params
        self.tta = tta
        self.lr_encoder = lr_encoder
        
        self.loss_fn_class = torch.nn.BCEWithLogitsLoss()
        self.setup_metrics()

        # for metrics (AUC, AP) which require a list of predictions + gt
        self.all_pred = []
        self.all_true = []

        self.bn_mult = 0.1
   
    def calculate_loss(self,prediction,prediction_aux,y):
        loss = self.loss_fn(prediction,y,**self.loss_params)
        n = len(prediction_aux)
        for i,p in enumerate(prediction_aux):
            l = self.loss_fn(p,y,**self.loss_params)
            # scales based on the resolution of the output
            loss = loss + l / (2**(n-i+1))
        return loss.mean()
    
    def loss_wrapper(self,x,y,y_class,x_cond,x_fc):
        y = torch.round(y)
        if self.tta == True:
            prediction,prediction_aux,pred_class = self.forward(
                torch.cat([x,x[:,:,::-1]]),
                X_skip_layer=torch.cat([x_cond,x_cond[:,:,::-1]]),
                X_feature_conditioning=torch.cat([x_fc,x_fc]),
                return_aux=True)
            prediction = sum(split(prediction,2,0)/2)
            prediction_aux = [sum(split(x,2,0)/2) for x in prediction_aux]
        else:
            prediction,prediction_aux,pred_class = self.forward(
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

        print(loss,class_loss)

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
            self.log("train_cl_loss",class_loss,batch_size=y.shape[0],
                     sync_dist=True)
        self.log("train_loss", loss,batch_size=y.shape[0],
                 sync_dist=True)

        update_metrics(
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

        for s_p,s_y in zip(pred_final.detach().cpu().numpy(),
                           y.squeeze(1).detach().cpu().numpy()):
            self.all_pred.append(s_p)
            self.all_true.append(s_y)

        self.log("val_loss",loss,prog_bar=True,on_epoch=True,
                  batch_size=y.shape[0],sync_dist=True)
        if class_loss is not None:
            self.log("val_loss_cl",class_loss,prog_bar=True,on_epoch=True,
                     batch_size=y.shape[0],sync_dist=True)

        update_metrics(
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
        
        for s_p,s_y in zip(pred_final.detach().cpu().numpy(),
                           y.squeeze(1).detach().cpu().numpy()):
            self.all_pred.append(s_p)
            self.all_true.append(s_y)

        update_metrics(
            self,self.test_metrics,pred_final,y,pred_class,y_class)
        return output_loss
        
    def configure_optimizers(self):
        encoder_params = []
        rest_of_params = []
        for k,p in self.named_parameters():
            if "encoding" in k or "encoder" in k:
                encoder_params.append(p)
            else:
                rest_of_params.append(p)
        if self.lr_encoder is None:
            lr_encoder = self.learning_rate
        else:
            lr_encoder = self.lr_encoder
        parameters = [
            {'params': encoder_params,'lr':lr_encoder},
            {'params': rest_of_params}]
        optimizer = torch.optim.AdamW(
            parameters,lr=self.learning_rate,
            weight_decay=self.weight_decay)

        return {"optimizer":optimizer,
                "monitor":"val_loss"}
    
    def on_train_epoch_end(self):
        # updating the lr here rather than as a PL lr_scheduler... 
        # basically the lr_scheduler (as I was using it at least)
        # is not terribly compatible with starting and stopping training
        opt = self.optimizers()
        polynomial_lr_decay(opt,self.current_epoch,initial_lr=self.learning_rate,
                            max_decay_steps=self.n_epochs,end_lr=1e-6,power=0.9)
        try:
            last_lr = [x["lr"] for x in opt.param_groups][-1]
        except:
            last_lr = self.learning_rate
        self.log("lr",last_lr,prog_bar=True,sync_dist=True)

class BrUNetPL(BrUNet,pl.LightningModule):
    def __init__(
        self,
        image_keys: str=["image"],
        label_key: str="label",
        skip_conditioning_key: str=None,
        feature_conditioning_key: str=None,
        learning_rate: float=0.001,
        batch_size: int=4,
        n_epochs: int=100,
        weight_decay: float=0.005,
        training_dataloader_call: Callable=None,
        loss_fn: Callable=torch.nn.functional.binary_cross_entropy,
        loss_params: dict={},
        tta: bool=False,
        lr_encoder: float=None,*args,**kwargs) -> torch.nn.Module:
        """Standard U-Net [1] implementation for Pytorch Lightning.

        Args:
            image_keys (str): list of keys corresponding to the inputs from 
                the train dataloader.
            label_key (str): key corresponding to the label map from the train
                dataloader.
            skip_conditioning_key (str, optional): key corresponding to
                image which will be concatenated to the skip connections. 
            feature_conditioning_key (str, optional): key corresponding to
                the tabular features which will be used in the feature
                conditioning.
            learning_rate (float, optional): learning rate. Defaults to 0.001.
            batch_size (int, optional): batch size. Defaults to 4.
            n_epochs (int, optional): number of epochs. Defaults to 100.
            weight_decay (float, optional): weight decay for optimizer. Defaults 
                to 0.005.
            training_dataloader_call (Callable, optional): call for the 
            training dataloader. Defaults to None.
            loss_fn (Callable, optional): function to calculate the loss. 
                Defaults to torch.nn.functional.binary_cross_entropy.
            loss_params (dict, optional): additional parameters for the loss
                function. Defaults to {}.
            tta (bool, optional): test-time augmentation. Defaults to False.
            lr_encoder (float, optional): encoder learning rate.
            args: arguments for UNet class.
            kwargs: keyword arguments for UNet class.

        [1] https://www.nature.com/articles/s41592-018-0261-2

        Returns:
            pl.LightningModule: a U-Net module.
        """
        
        super().__init__(*args,**kwargs)
        
        self.image_keys = image_keys
        self.label_key = label_key
        self.skip_conditioning_key = skip_conditioning_key
        self.feature_conditioning_key = feature_conditioning_key
        self.learning_rate = learning_rate
        self.batch_size = batch_size
        self.n_epochs = n_epochs
        self.weight_decay = weight_decay
        self.training_dataloader_call = training_dataloader_call
        self.loss_fn = loss_fn
        self.loss_params = loss_params
        self.tta = tta
        self.lr_encoder = lr_encoder
        
        self.loss_fn_class = torch.nn.BCEWithLogitsLoss()
        self.setup_metrics()

        # for metrics (AUC, AP) which require a list of predictions + gt
        self.all_pred = []
        self.all_true = []
        
        self.bn_mult = 0.1
           
    def calculate_loss(self,prediction,y):
        loss = self.loss_fn(prediction,y,**self.loss_params)
        return loss.mean()

    def calculate_loss_class(self,prediction,y):
        y = y.type(torch.float32)
        prediction = prediction.squeeze(1).type(torch.float32)
        loss = self.loss_fn_class(prediction,y)
        return loss.mean()

    def loss_wrapper(self,x,x_weights,y,y_class,x_cond,x_fc):
        y = torch.round(y)
        output = self.forward(
            x,x_weights,X_skip_layer=x_cond,X_feature_conditioning=x_fc)
        if self.deep_supervision == False:
            prediction,pred_class = output
        else:
            prediction,pred_class,deep_outputs = output
            deep_outputs = [torch.squeeze(o,1) for o in deep_outputs]
        prediction = torch.squeeze(prediction,1)
        y = torch.squeeze(y,1)
        batch_size = int(prediction.shape[0])
        if batch_size == 1:
            y = torch.unsqueeze(y,0)
        y = torch.squeeze(y,1)

        loss = self.calculate_loss(prediction,y)
        if self.deep_supervision == True:
            t = len(deep_outputs)
            additional_losses = torch.zeros_like(loss)
            for i,o in enumerate(deep_outputs):
                S = o.shape[-self.spatial_dimensions:]
                y_small = F.interpolate(torch.unsqueeze(y,1),S,mode="nearest")
                y_small = torch.squeeze(y_small,1)
                l = self.calculate_loss(o,y_small).mean()/(2**(t-i))/t
                additional_losses = additional_losses + l
            loss = loss + additional_losses
        if self.bottleneck_classification == True:
            class_loss = self.calculate_loss_class(pred_class,y_class)
            output_loss = loss + class_loss * self.bn_mult
        else:
            class_loss = None
            output_loss = loss

        return prediction,pred_class,loss,class_loss,output_loss

    def unpack_batch(self,batch):
        x, y = [batch[k] for k in self.image_keys],batch[self.label_key]
        x_weights = [batch[k+"_weight"] for k in self.image_keys]
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
        return x,x_weights,y,x_cond,x_fc,y_class

    def training_step(self,batch,batch_idx):
        x,x_weights,y,x_cond,x_fc,y_class = self.unpack_batch(batch)

        pred_final,pred_class,loss,class_loss,output_loss = self.loss_wrapper(
            x,x_weights,y,y_class,x_cond,x_fc)
        
        self.log("train_loss",loss,batch_size=y.shape[0],
                 sync_dist=True)
        if class_loss is not None:
            self.log("train_cl_loss",class_loss,batch_size=y.shape[0],
                     sync_dist=True)

        update_metrics(
            self,self.train_metrics,pred_final,y,pred_class,y_class,
            on_epoch=True,on_step=False,prog_bar=True)
        return output_loss
    
    def validation_step(self,batch,batch_idx):
        x,x_weights,y,x_cond,x_fc,y_class = self.unpack_batch(batch)
        
        pred_final,pred_class,loss,class_loss,output_loss = self.loss_wrapper(
            x,x_weights,y,y_class,x_cond,x_fc)
        
        for s_p,s_y in zip(pred_final.detach().cpu().numpy(),
                           y.squeeze(1).detach().cpu().numpy()):
            self.all_pred.append(s_p)
            self.all_true.append(s_y)

        self.log("val_loss",loss,prog_bar=True,
                 on_epoch=True,batch_size=y.shape[0],sync_dist=True)
        if class_loss is not None:
            self.log("val_cl_loss",class_loss,batch_size=y.shape[0],
                     sync_dist=True)
        
        update_metrics(
            self,self.val_metrics,pred_final,y,pred_class,y_class,
            on_epoch=True,prog_bar=True)
        return output_loss

    def test_step(self,batch,batch_idx):
        x,x_weights,y,x_cond,x_fc,y_class = self.unpack_batch(batch)

        pred_final,pred_class,loss,class_loss,output_loss = self.loss_wrapper(
            x,x_weights,y,y_class,x_cond,x_fc)

        for s_p,s_y in zip(pred_final.detach().cpu().numpy(),
                           y.squeeze(1).detach().cpu().numpy()):
            self.all_pred.append(s_p)
            self.all_true.append(s_y)

        update_metrics(
            self,self.test_metrics,pred_final,y,pred_class,y_class,
            on_epoch=True,on_step=False,prog_bar=True)
        return output_loss
        
    def train_dataloader(self) -> torch.utils.data.DataLoader:
        return self.training_dataloader_call()

    def configure_optimizers(self):
        encoder_params = []
        rest_of_params = []
        for k,p in self.named_parameters():
            if "encoding" in k or "encoder" in k:
                encoder_params.append(p)
            else:
                rest_of_params.append(p)
        if self.lr_encoder is None:
            lr_encoder = self.learning_rate
        else:
            lr_encoder = self.lr_encoder
        parameters = [
            {'params': encoder_params,'lr':lr_encoder},
            {'params': rest_of_params}]
        optimizer = torch.optim.AdamW(
            parameters,lr=self.learning_rate,
            weight_decay=self.weight_decay)

        self.initial_learning_rates = [parameters[0]["lr"],
                                       self.learning_rate]
        return {"optimizer":optimizer,
                "monitor":"val_loss"}
    
    def on_train_epoch_end(self):
        # updating the lr here rather than as a PL lr_scheduler... 
        # basically the lr_scheduler (as I was using it at least)
        # is not terribly compatible with starting and stopping training
        opt = self.optimizers()
        polynomial_lr_decay(opt,self.current_epoch,
                            initial_lr=self.initial_learning_rates,
                            max_decay_steps=self.n_epochs,end_lr=1e-6,power=0.9)
        try:
            last_lr = [x["lr"] for x in opt.param_groups][-1]
        except:
            last_lr = self.learning_rate
        self.log("lr",last_lr,prog_bar=True,sync_dist=True)

    def on_validation_epoch_end(self):
        picai_eval_metrics = evaluate(
            y_det=self.all_pred,y_true=self.all_true,
            y_det_postprocess_func=get_lesions,
            num_parallel_calls=8)
        self.log("V_AP",picai_eval_metrics.AP,prog_bar=True,
                 sync_dist=True)
        self.log("V_R",picai_eval_metrics.score,prog_bar=True,
                 sync_dist=True)
        self.log("V_AUC",picai_eval_metrics.auroc,prog_bar=True,
                 sync_dist=True)
        self.all_pred = []
        self.all_true = []

    def on_test_epoch_end(self):
        picai_eval_metrics = evaluate(
            y_det=self.all_pred,y_true=self.all_true,
            y_det_postprocess_func=get_lesions,
            num_parallel_calls=8)
        self.log("T_AP",picai_eval_metrics.AP,prog_bar=True,
                 sync_dist=True)
        self.log("T_R",picai_eval_metrics.score,prog_bar=True,
                 sync_dist=True)
        self.log("T_AUC",picai_eval_metrics.auroc,prog_bar=True,
                 sync_dist=True)
        self.all_pred = []
        self.all_true = []

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
              "F1":torchmetrics.FBetaScore,
              "Dice":torchmetrics.Dice}
        if self.bottleneck_classification == True:
            md["AUC_bn"] = torchmetrics.AUROC
        for k in md:
            if k == "IoU":
                m,C = "macro",C_1
            elif k == "Dice":
                m,C = "micro",C_2
            else:
                m,C = M,C_2
            if "bn" in k:
                I_ = None
            else:
                I_ = I
            if k in []:
                self.train_metrics[k] = md[k](
                    num_classes=C,mdmc_average=A,average=m,
                    ignore_index=I_).to(self.device)
            if k in ["IoU","AUC_bn"]:
                self.val_metrics["V_"+k] = md[k](
                    num_classes=C,mdmc_average=A,average=m,
                    ignore_index=I_).to(self.device)
            self.test_metrics["T_"+k] = md[k](
                num_classes=C,mdmc_average=A,average=m,
                ignore_index=I_).to(self.device)
