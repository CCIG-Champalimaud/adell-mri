import numpy as np
import torch
import torch.nn.functional as F
import torchmetrics
import lightning.pytorch as pl
import torchmetrics.classification as tmc
from typing import Callable,Dict,List
from abc import ABC

from .picai_eval import evaluate
from .unet import UNet,BrUNet
from .unetpp import UNetPlusPlus
from .unetr import UNETR
from .unetr import SWINUNet
from ..extract_lesion_candidates import extract_lesion_candidates
from ..learning_rate import CosineAnnealingWithWarmupLR

def binary_iou_manual(pred,truth):
    binary_pred = pred > 0.5
    intersection = torch.logical_and(
        binary_pred,truth == 1)
    intersection = intersection.sum()
    union = binary_pred.sum() + truth.sum() - intersection
    return intersection,union
 
def split(x,n_splits,dim):
    size = int(x.shape[dim]//n_splits)
    return torch.split(x,size,dim)

def get_lesions(x:torch.Tensor)->np.ndarray:
    """Wrapper for getting lesions using extract_lesion_candidates.

    Args:
        x (torch.Tensor): input tensor with segmentation probabilities.

    Returns:
        (np.ndarray): map containing indexed lesion candidates.
    """
    return extract_lesion_candidates(x)[0]

def update_metrics(cls:pl.LightningModule,
                   metrics:Dict[str,torchmetrics.Metric],
                   pred:torch.Tensor,
                   y:torch.Tensor,
                   pred_class:torch.Tensor,
                   y_class:torch.Tensor,**kwargs)->None:
    """Wraper function to update metrics.

    Args:
        cls (pl.LightningModule): a wraper function.
        metrics (Dict[str,torchmetrics.Metric]): a dictionary containing 
            strings as keys and torchmetrics metrics (with an update method
            and compatible with pl.LightningModule.log) as values.
        pred (torch.Tensor): tensor containing probability segmentation maps.
        y (torch.Tensor): segmentation ground truth.
        pred_class (torch.Tensor): class probability.
        y_class (torch.Tensor): class ground truths.
    """
    try: 
        y = torch.round(y).int()
    except: 
        pass
    y = y.long()
    pred = pred.squeeze(1)
    y = y.squeeze(1)
    p = pred.detach()
    if y_class is not None:
        y_class = y_class.long()
        pc = pred_class.detach()
        if cls.n_classes == 2:
            pc = F.sigmoid(pc)
        else:
            pc = F.softmax(pc,1)
    for k in metrics:
        if 'cl:' in k:
            metrics[k].update(pc,y_class)
        else:
            if "Dice" in k:
                metrics[k].update(p.round().long(),y)
            else:
                metrics[k].update(p,y)
        cls.log(k,metrics[k],**kwargs,batch_size=y.shape[0],
                sync_dist=True)

def get_metric_dict(nc:int,
                    bottleneck_classification:bool,
                    metric_keys:List[str]=None,
                    prefix:str="",
                    dev:str=None)->Dict[str,torchmetrics.Metric]:
    metric_dict = torch.nn.ModuleDict({})
    if nc == 2:
        md = {
            "IoU":lambda: tmc.BinaryJaccardIndex(),
            "Pr":lambda: tmc.BinaryPrecision(),
            "F1":lambda: tmc.BinaryFBetaScore(1.0),
            "Dice":lambda: torchmetrics.Dice(
                num_classes=1,multiclass=False)}
    else:
        md = {"IoU":lambda: torchmetrics.JaccardIndex(nc,average="macro"),
              "Pr":lambda: torchmetrics.Precision(nc,average="macro"),
              "F1":lambda: torchmetrics.FBetaScore(nc,average="macro"),
              "Dice":lambda: torchmetrics.Dice(nc,average="macro")}
    if bottleneck_classification is True:
        md["AUC_bn"] = torchmetrics.AUROC
    if metric_keys is None:
        metric_keys = list(md.keys())
    for k in metric_keys:
        if k in md:
            metric_dict[prefix+k] = md[k]()
    if dev is not None:
        metric_dict = {k:metric_dict[k].to(dev) for k in metric_dict}
    return metric_dict

class UNetBasePL(pl.LightningModule,ABC):
    """
    UNet base class. Has convenient methods that can be inherited by other
    UNet PyTorch-Lightning modules.
    """
    def __init__(self):
        super().__init__()
        
        self.train_batch_size = None
        self.raise_nan_loss = False

    def calculate_loss(self,prediction,y):
        loss = self.loss_fn(prediction,y,**self.loss_params)
        return loss.mean()

    def calculate_loss_class(self,prediction,y):
        y = y.type_as(prediction)
        loss = self.loss_fn_class(prediction,y)
        return loss.mean()
    
    def check_loss(self,x,y,pred,loss):
        if self.raise_nan_loss is True and torch.isnan(loss) is True:
            print("Nan loss detected! ({})".format(loss.detach()))
            for i,sx in enumerate(x):
                print("\t0",[sx.detach().max(),sx.detach().min()])
            print("\tOutput:",[pred.detach().max(),pred.detach().min()])
            print("\tTruth:",[y.min(),y.max()])
            print("\tModel parameters:")
            for n,p in self.named_parameters():
                pn = p.norm()
                if (torch.isnan(pn) is True) or (torch.isinf(pn) is True) or True:
                    print("\t\tparameter norm({})={}".format(n,pn))
            for n,p in self.named_parameters():
                if p.grad is not None:
                    pg = p.grad.mean()
                    if (torch.isnan(pg) is True) or (torch.isinf(pg) is True) or True:
                        print("\t\taverage grad({})={}".format(n,pg))
            raise RuntimeError(
                "nan found in loss (see above for details)")

    def step(self,x,y,y_class,x_cond,x_fc):
        y = torch.round(y)
        output = self.forward(
            X=x,X_skip_layer=x_cond,X_feature_conditioning=x_fc)
        if self.deep_supervision is False:
            prediction,pred_class = output
        else:
            prediction,pred_class,deep_outputs = output
        prediction = prediction

        loss = self.calculate_loss(prediction,y)
        if self.deep_supervision is True:
            t = len(deep_outputs)
            additional_losses = torch.zeros_like(loss)
            for i,o in enumerate(deep_outputs):
                S = o.shape[-self.spatial_dimensions:]
                y_small = F.interpolate(y,S,mode="nearest")
                l = self.calculate_loss(o,y_small).mean()/(2**(t-i))/(t+1)
                additional_losses = additional_losses + l
            loss = loss + additional_losses
        if self.bottleneck_classification is True:
            class_loss = self.calculate_loss_class(pred_class,y_class)
            output_loss = loss + class_loss * self.bn_mult
        else:
            class_loss = None
            output_loss = loss

        return prediction,pred_class,loss,class_loss,output_loss

    def unpack_batch(self,batch):
        x, y = batch[self.image_key],batch[self.label_key]
        if self.skip_conditioning_key is not None:
            x_cond = batch[self.skip_conditioning_key]
        else:
            x_cond = None
        if self.bottleneck_classification is True:
            y_class = y.flatten(start_dim=1).max(1).values
        else:
            y_class = None
        if self.feature_conditioning_key is not None:
            x_fc = batch[self.feature_conditioning_key]
        else:
            x_fc = None
        return x,x_cond,x_fc,y,y_class
    
    def unpack_batch_prediction(self,batch):
        x = batch[self.image_key]
        if self.skip_conditioning_key is not None:
            x_cond = batch[self.skip_conditioning_key]
        else:
            x_cond = None
        if self.feature_conditioning_key is not None:
            x_fc = batch[self.feature_conditioning_key]
        else:
            x_fc = None
        return x,x_cond,x_fc

    def predict_step(self,batch,batch_idx=0,
                     return_only_segmentation=False,*args,**kwargs):
        x,x_cond,x_fc = self.unpack_batch_prediction(batch)
        not_batched = False
        if len(x.shape) == self.spatial_dimensions + 1:
            x = x.unsqueeze(0)
            x_cond = x_cond.unsqueeze(0) if x_cond is not None else None
            x_fc = x_fc.unsqueeze(0) if x_fc is not None else None
            not_batched = True
        output = self.forward(
            X=x,X_skip_layer=x_cond,X_feature_conditioning=x_fc,
            *args,**kwargs)
        if return_only_segmentation == True:
            output = output[0]
        if not_batched == True:
            output = output[0]
        return output

    def training_step(self,batch,batch_idx):
        x,x_cond,x_fc,y,y_class = self.unpack_batch(batch)

        pred_final,pred_class,loss,class_loss,output_loss = self.step(
            x,y,y_class,x_cond,x_fc)
        
        self.log("train_loss",loss,batch_size=y.shape[0],
                 sync_dist=True)
        if class_loss is not None:
            self.log("train_cl_loss",class_loss,batch_size=y.shape[0],
                     sync_dist=True)

        self.check_loss(x,y,pred_final,output_loss)

        update_metrics(
            self,self.train_metrics,pred_final,y,pred_class,y_class,
            on_epoch=True,on_step=False,prog_bar=True)
        
        self.train_batch_size = x.shape[0]
        
        return output_loss
    
    def validation_step(self,batch,batch_idx):
        x,x_cond,x_fc,y,y_class = self.unpack_batch(batch)

        bs = x.shape[0]
        if self.train_batch_size is None:
            mbs = self.batch_size
        else:
            mbs = self.train_batch_size
        for i in range(0,bs,mbs):
            m,M = i,i+mbs
            out = self.step(x[m:M],
                            y[m:M],
                            y_class[m:M] if y_class is not None else None,
                            x_cond[m:M] if x_cond is not None else None,
                            x_fc[m:M] if x_cond is not None else None)
            pred_final,pred_class,loss,class_loss,output_loss = out
            
            if self.picai_eval is True:
                for s_p,s_y in zip(pred_final.squeeze(1).detach().cpu().numpy(),
                                   y[m:M].squeeze(1).detach().cpu().numpy()):
                    self.all_pred.append(s_p)
                    self.all_true.append(s_y)

            self.log("val_loss",loss.detach(),prog_bar=True,
                    on_epoch=True,batch_size=M-m,
                    sync_dist=True)

            update_metrics(
                self,self.val_metrics,pred_final,
                y[m:M],pred_class,y_class[m:M] if y_class is not None else None,
                on_epoch=True,prog_bar=True)

    def test_step(self,batch,batch_idx):
        x,x_cond,x_fc,y,y_class = self.unpack_batch(batch)

        bs = x.shape[0]
        if self.train_batch_size is None:
            mbs = self.batch_size
        else:
            mbs = self.train_batch_size
        for i in range(0,bs,mbs):
            m,M = i,i+mbs
            out = self.step(x[m:M],
                            y[m:M],
                            y_class[m:M] if y_class is not None else None,
                            x_cond[m:M] if x_cond is not None else None,
                            x_fc[m:M] if x_cond is not None else None)
            pred_final,pred_class,loss,class_loss,output_loss = out
        
            if self.picai_eval is True:
                for s_p,s_y in zip(pred_final.squeeze(1).detach().cpu().numpy(),
                                   y.squeeze(1).detach().cpu().numpy()):
                    self.all_pred.append(s_p)
                    self.all_true.append(s_y)

            update_metrics(
                self,self.test_metrics,pred_final,
                y[m:M],pred_class,y_class[m:M] if y_class is not None else None,
                on_epoch=True,prog_bar=True)
        
    def train_dataloader(self) -> torch.utils.data.DataLoader:
        return self.training_dataloader_call(self.batch_size)

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
            parameters = encoder_params + rest_of_params
        else:
            lr_encoder = self.lr_encoder
            parameters = [
                {'params': encoder_params,'lr':lr_encoder},
                {'params': rest_of_params}]
        optimizer = torch.optim.SGD(
            parameters,lr=self.learning_rate,momentum=0.99,
            weight_decay=self.weight_decay)
        
        if self.cosine_decay == True:
            lr_schedulers = CosineAnnealingWithWarmupLR(
                optimizer,T_max=self.n_epochs,start_decay=0,n_warmup_steps=0)
            
            return {"optimizer":optimizer,
                    "lr_scheduler":lr_schedulers,
                    "monitor":"val_loss"}
        else:
            return {"optimizer":optimizer,
                    "monitor":"val_loss"}
    
    def on_train_epoch_end(self):
        # updating the lr here rather than as a PL lr_scheduler... 
        # basically the lr_scheduler (as I was using it at least)
        # is not terribly compatible with starting and stopping training
        opt = self.optimizers()
        try:
            last_lr = [x["lr"] for x in opt.param_groups][-1]
        except:
            last_lr = self.learning_rate
        self.log("lr",last_lr,prog_bar=True,sync_dist=True)

    def on_validation_epoch_end(self):
        if self.picai_eval:
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
        if self.picai_eval:
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

    def setup_metrics(self):
        self.train_metrics = get_metric_dict(
            self.n_classes,self.bottleneck_classification,
            ["IoU","Dice"],prefix="")
        self.val_metrics = get_metric_dict(
            self.n_classes,self.bottleneck_classification,
            ["IoU","Dice","AUC_bn"],prefix="V_")
        self.test_metrics = get_metric_dict(
            self.n_classes,self.bottleneck_classification,
            None,prefix="T_")

class UNetPL(UNet,UNetBasePL):
    """Standard U-Net [1] implementation for Pytorch Lightning.

    [1] https://www.nature.com/articles/s41592-018-0261-2
    """
    def __init__(
        self,
        image_key: str="image",
        label_key: str="label",
        skip_conditioning_key: str=None,
        feature_conditioning_key: str=None,
        learning_rate: float=0.001,
        lr_encoder: float=None,
        cosine_decay: bool=True,
        batch_size: int=4,
        n_epochs: int=100,
        weight_decay: float=0.005,
        training_dataloader_call: Callable=None,
        loss_fn: Callable=torch.nn.functional.binary_cross_entropy,
        loss_params: dict={},
        picai_eval: bool=False,*args,**kwargs) -> torch.nn.Module:
        """
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
            lr_encoder (float, optional): encoder learning rate. Defaults to None
                (same as learning_rate).
            cosine_decay (bool, optional): triggers cosine learning rate
                decay. Defaults to True.
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
            picai_eval (bool, optional): evaluates network using PI-CAI 
                metrics as well (can be a bit long).
            args: arguments for UNet class.
            kwargs: keyword arguments for UNet class.
        """
        
        super().__init__(*args,**kwargs)
        
        self.image_key = image_key
        self.label_key = label_key
        self.skip_conditioning_key = skip_conditioning_key
        self.feature_conditioning_key = feature_conditioning_key
        self.learning_rate = learning_rate
        self.lr_encoder = lr_encoder
        self.cosine_decay = cosine_decay
        self.batch_size = batch_size
        self.n_epochs = n_epochs
        self.weight_decay = weight_decay
        self.training_dataloader_call = training_dataloader_call
        self.loss_fn = loss_fn
        self.loss_params = loss_params
        self.picai_eval = picai_eval
        
        self.loss_fn_class = torch.nn.BCEWithLogitsLoss()
        self.setup_metrics()

        # for metrics (AUC, AP) which require a list of predictions + gt
        self.all_pred = []
        self.all_true = []
        
        self.bn_mult = 0.1

class UNETRPL(UNETR,UNetBasePL):
    """Standard UNETR implementation for Pytorch Lightning.
    """
    def __init__(
        self,
        image_key: str="image",
        label_key: str="label",
        skip_conditioning_key: str=None,
        feature_conditioning_key: str=None,
        learning_rate: float=0.001,
        lr_encoder: float=None,
        cosine_decay: bool=True,
        batch_size: int=4,
        n_epochs: int=100,
        weight_decay: float=0.005,
        training_dataloader_call: Callable=None,
        loss_fn: Callable=torch.nn.functional.binary_cross_entropy,
        loss_params: dict={},
        picai_eval: bool=False,*args,**kwargs) -> torch.nn.Module:
        """
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
            lr_encoder (float, optional): encoder learning rate. Defaults to None
                (same as learning_rate).
            cosine_decay (bool, optional): triggers cosine learning rate
                decay. Defaults to True.
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
            picai_eval (bool, optional): evaluates network using PI-CAI 
                metrics as well (can be a bit long).
            args: arguments for UNet class.
            kwargs: keyword arguments for UNet class.
        """
        super().__init__(*args,**kwargs)
        self.image_key = image_key
        self.label_key = label_key
        self.skip_conditioning_key = skip_conditioning_key
        self.feature_conditioning_key = feature_conditioning_key
        self.learning_rate = learning_rate
        self.lr_encoder = lr_encoder
        self.cosine_decay = cosine_decay
        self.batch_size = batch_size
        self.n_epochs = n_epochs
        self.weight_decay = weight_decay
        self.training_dataloader_call = training_dataloader_call
        self.loss_fn = loss_fn
        self.loss_params = loss_params
        self.picai_eval = picai_eval
        
        self.loss_fn_class = torch.nn.BCEWithLogitsLoss()
        self.setup_metrics()

        # for metrics (AUC, AP) which require a list of predictions + gt
        self.all_pred = []
        self.all_true = []
        
        self.bn_mult = 0.1

class SWINUNetPL(SWINUNet,UNetBasePL):
    """Standard SWIN-UNet implementation for Pytorch Lightning.
    """
    def __init__(
        self,
        image_key: str="image",
        label_key: str="label",
        skip_conditioning_key: str=None,
        feature_conditioning_key: str=None,
        learning_rate: float=0.001,
        lr_encoder: float=None,
        cosine_decay: bool=True,
        batch_size: int=4,
        n_epochs: int=100,
        weight_decay: float=0.005,
        training_dataloader_call: Callable=None,
        loss_fn: Callable=torch.nn.functional.binary_cross_entropy,
        loss_params: dict={},
        picai_eval: bool=False,*args,**kwargs) -> torch.nn.Module:
        """
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
            lr_encoder (float, optional): encoder learning rate. Defaults to None
                (same as learning_rate).
            cosine_decay (bool, optional): triggers cosine learning rate
                decay. Defaults to True.
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
            picai_eval (bool, optional): evaluates network using PI-CAI 
                metrics as well (can be a bit long).
            args: arguments for UNet class.
            kwargs: keyword arguments for UNet class.
        """
        super().__init__(*args,**kwargs)
        
        self.image_key = image_key
        self.label_key = label_key
        self.skip_conditioning_key = skip_conditioning_key
        self.feature_conditioning_key = feature_conditioning_key
        self.learning_rate = learning_rate
        self.lr_encoder = lr_encoder
        self.cosine_decay = cosine_decay
        self.batch_size = batch_size
        self.n_epochs = n_epochs
        self.weight_decay = weight_decay
        self.training_dataloader_call = training_dataloader_call
        self.loss_fn = loss_fn
        self.loss_params = loss_params
        self.picai_eval = picai_eval
        
        self.loss_fn_class = torch.nn.BCEWithLogitsLoss()
        self.setup_metrics()

        # for metrics (AUC, AP) which require a list of predictions + gt
        self.all_pred = []
        self.all_true = []
        
        self.bn_mult = 0.1

class UNetPlusPlusPL(UNetPlusPlus,UNetBasePL):
    def __init__(
        self,
        image_key: str="image",
        label_key: str="label",
        skip_conditioning_key: str=None,
        feature_conditioning_key: str=None,
        learning_rate: float=0.001,
        lr_encoder: float=None,
        cosine_decay: bool=True,
        batch_size: int=4,
        n_epochs: int=100,
        weight_decay: float=0.005,
        training_dataloader_call: Callable=None,
        loss_fn: Callable=torch.nn.functional.binary_cross_entropy,
        loss_params: dict={},
        picai_eval: bool=False,*args,**kwargs) -> torch.nn.Module:
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
            lr_encoder (float, optional): encoder learning rate.
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
            picai_eval (bool, optional): evaluates network using PI-CAI 
                metrics as well (can be a bit long).
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
        self.lr_encoder = lr_encoder
        self.cosine_decay = cosine_decay
        self.batch_size = batch_size
        self.n_epochs = n_epochs
        self.weight_decay = weight_decay
        self.training_dataloader_call = training_dataloader_call
        self.loss_fn = loss_fn
        self.loss_params = loss_params
        self.picai_eval = picai_eval
        
        self.deep_supervision = True
        self.loss_fn_class = torch.nn.BCEWithLogitsLoss()
        self.setup_metrics()

        # for metrics (AUC, AP) which require a list of predictions + gt
        self.all_pred = []
        self.all_true = []

        self.bn_mult = 0.1
   
class BrUNetPL(BrUNet,UNetBasePL):
    def __init__(
        self,
        image_keys: str=["image"],
        label_key: str="label",
        skip_conditioning_key: str=None,
        feature_conditioning_key: str=None,
        learning_rate: float=0.001,
        lr_encoder: float=None,
        cosine_decay: bool=True,
        batch_size: int=4,
        n_epochs: int=100,
        weight_decay: float=0.005,
        training_dataloader_call: Callable=None,
        loss_fn: Callable=torch.nn.functional.binary_cross_entropy,
        loss_params: dict={},
        picai_eval: bool=False,*args,**kwargs) -> torch.nn.Module:
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
            lr_encoder (float, optional): encoder learning rate.
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
            picai_eval (bool, optional): evaluates network using PI-CAI 
                metrics as well (can be a bit long).
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
        self.lr_encoder = lr_encoder
        self.cosine_decay = cosine_decay
        self.batch_size = batch_size
        self.n_epochs = n_epochs
        self.weight_decay = weight_decay
        self.training_dataloader_call = training_dataloader_call
        self.loss_fn = loss_fn
        self.loss_params = loss_params
        self.picai_eval = picai_eval
        
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
        y = y.type_as(prediction)
        loss = self.loss_fn_class(prediction,y)
        return loss.mean()

    def step(self,x,x_weights,y,y_class,x_cond,x_fc):
        y = torch.round(y)
        output = self.forward(
            x,x_weights,X_skip_layer=x_cond,X_feature_conditioning=x_fc)
        if self.deep_supervision is False:
            prediction,pred_class = output
        else:
            prediction,pred_class,deep_outputs = output

        loss = self.calculate_loss(prediction,y)
        if self.deep_supervision is True:
            t = len(deep_outputs)
            additional_losses = torch.zeros_like(loss)
            for i,o in enumerate(deep_outputs):
                S = o.shape[-self.spatial_dimensions:]
                y_small = F.interpolate(y,S,mode="nearest")
                l = self.calculate_loss(o,y_small).mean()/(2**(t-i))/(t+1)
                additional_losses = additional_losses + l
            loss = loss + additional_losses
        if self.bottleneck_classification is True:
            class_loss = self.calculate_loss_class(pred_class,y_class)
            output_loss = loss + class_loss * self.bn_mult
        else:
            class_loss = None
            output_loss = loss
        
        self.check_loss(x,y,prediction,output_loss)
        
        return prediction,pred_class,loss,class_loss,output_loss

    def unpack_batch(self,batch):
        x, y = [batch[k] for k in self.image_keys],batch[self.label_key]
        x_weights = [batch[k+"_weight"] for k in self.image_keys]
        if self.skip_conditioning_key is not None:
            x_cond = batch[self.skip_conditioning_key]
        else:
            x_cond = None
        if self.bottleneck_classification is True:
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

        pred_final,pred_class,loss,class_loss,output_loss = self.step(
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
        
        pred_final,pred_class,loss,class_loss,output_loss = self.step(
            x,x_weights,y,y_class,x_cond,x_fc)
        
        if self.picai_eval is True:
            for s_p,s_y in zip(pred_final.squeeze(1).detach().cpu().numpy(),
                               y.squeeze(1).detach().cpu().numpy()):
                self.all_pred.append(s_p)
                self.all_true.append(s_y)

        self.log("val_loss",loss.detach(),prog_bar=True,
                 on_epoch=True,batch_size=y.shape[0],sync_dist=True)
        if class_loss is not None:
            self.log("val_cl_loss",class_loss.detach(),batch_size=y.shape[0],
                     sync_dist=True)
        
        update_metrics(
            self,self.val_metrics,pred_final,y,pred_class,y_class,
            on_epoch=True,prog_bar=True)

    def test_step(self,batch,batch_idx):
        x,x_weights,y,x_cond,x_fc,y_class = self.unpack_batch(batch)

        pred_final,pred_class,loss,class_loss,output_loss = self.step(
            x,x_weights,y,y_class,x_cond,x_fc)

        if self.picai_eval is True:
            for s_p,s_y in zip(pred_final.squeeze(1).detach().cpu().numpy(),
                               y.squeeze(1).detach().cpu().numpy()):
                self.all_pred.append(s_p)
                self.all_true.append(s_y)

        update_metrics(
            self,self.test_metrics,pred_final,y,pred_class,y_class,
            on_epoch=True,on_step=False,prog_bar=True)
