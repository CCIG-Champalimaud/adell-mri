import torch
import torch.nn.functional as F
import torchmetrics
import pytorch_lightning as pl
import torchmetrics.classification as tmc
from typing import Callable,List,Dict
from abc import ABC

from .classification import (
    CatNet,OrdNet,ordinal_prediction_to_class,SegCatNet,
    UNetEncoder,GenericEnsemble,ViTClassifier,FactorizedViTClassifier,
    TransformableTransformer,HybridClassifier)
from ..learning_rate import CosineAnnealingWithWarmupLR

def f1(prediction:torch.Tensor,y:torch.Tensor)->torch.Tensor:
    """
    Implementation of the sbinary F1 score for torch tensors.

    Args:
        prediction (torch.Tensor): prediction tensor.
        y (torch.Tensor): ground truth tensor.

    Returns:
        torch.Tensor: F1-score.
    """
    prediction = prediction.detach() > 0.5
    tp = torch.logical_and(prediction == y,y == 1).sum().float()
    tn = torch.logical_and(prediction == y,y == 0).sum().float()
    fp = torch.logical_and(prediction != y,y == 0).sum().float()
    fn = torch.logical_and(prediction != y,y == 1).sum().float()
    tp,tn,fp,fn = [float(x) for x in [tp,tn,fp,fn]]
    n = tp
    d = (tp+0.5*(fp + fn))
    if d > 0:
        return n/d
    else:
        return 0

def get_metric_dict(nc:int,
                    metric_keys:List[str]=None,
                    prefix:str="")->Dict[str,torchmetrics.Metric]:
    """
    Constructs a metric dictionary.

    Args:
        nc (int): number of classes.
        metric_keys (List[str], optional): keys corresponding to metrics. 
            Should be one of ["Rec","Spe","Pr","F1","AUC"]. Defaults to 
            None (all keys).
        prefix (str, optional): which prefix should be added to the metric
            key on the output dict. Defaults to "".

    Returns:
        Dict[str,torchmetrics.Metric]: dictionary containing the metrics 
            specified in metric_keys.
    """
    metric_dict = torch.nn.ModuleDict({})
    if nc == 2:
        md = {
            "Rec":lambda: tmc.BinaryRecall(),
            "Spe":lambda: tmc.BinarySpecificity(),
            "Pr":lambda: tmc.BinaryPrecision(),
            "F1":lambda: tmc.BinaryFBetaScore(1.0),
            "AUC":lambda: torchmetrics.AUROC("binary")}
    else:
        md = {"Rec":lambda: torchmetrics.Recall(nc,average="macro"),
              "Spe":lambda: torchmetrics.Specificity(),
              "Pr":lambda: torchmetrics.Precision(nc,average="macro"),
              "F1":lambda: torchmetrics.FBetaScore(nc,average="macro"),
              "AUC":lambda: torchmetrics.AUROC("multilabel")}
    if metric_keys is None:
        metric_keys = list(md.keys())
    for k in metric_keys:
        if k in md:
            metric_dict[prefix+k] = md[k]()
    return metric_dict

class ClassPLABC(pl.LightningModule,ABC):
    """
    Abstract classification class for LightningModules.
    """
    def __init__(self):
        super().__init__()

        self.raise_nan_loss = False

    def calculate_loss(self,prediction,y,with_params=False):
        y = y.to(prediction.device)
        if self.n_classes > 2:
            if len(y.shape) > 1:
                y = y.squeeze(1)
            y = y.to(torch.int64)
        else:
            y = y.float()
        if with_params is True:
            d = y.device
            params = {k:self.loss_params[k].to(d) for k in self.loss_params}
            loss = self.loss_fn(prediction,y,**params)
        else:
            loss = self.loss_fn(prediction,y)
        return loss.mean()

    def training_step(self,batch,batch_idx):
        x, y = batch[self.image_key],batch[self.label_key]
        if hasattr(self, 'training_batch_preproc'):
            if self.training_batch_preproc is not None:
                x,y = self.training_batch_preproc(x,y)
        prediction = self.forward(x)
        prediction = torch.squeeze(prediction,1)

        loss = self.calculate_loss(prediction,y,with_params=True)

        self.log("train_loss",loss,sync_dist=True)
        return loss

    def validation_step(self,batch,batch_idx):
        x, y = batch[self.image_key],batch[self.label_key]
        prediction = self.forward(x)
        prediction = torch.squeeze(prediction,1)

        loss = self.calculate_loss(prediction,y,with_params=True)
        self.log("val_loss",loss,on_epoch=True,
                 on_step=False,prog_bar=True,
                 batch_size=x.shape[0],sync_dist=True)        
        self.update_metrics(prediction,y,self.val_metrics)
        return loss

    def test_step(self,batch,batch_idx):
        x, y = batch[self.image_key],batch[self.label_key]
        prediction = self.forward(x)
        prediction = torch.squeeze(prediction,1)

        loss = self.calculate_loss(prediction,y)
        self.log("test_loss",loss,on_epoch=True,
                 on_step=False,prog_bar=True,
                 batch_size=x.shape[0],sync_dist=True)        
        self.update_metrics(prediction,y,self.test_metrics)
        return loss

    def train_dataloader(self) -> torch.utils.data.DataLoader:
        return self.training_dataloader_call()

    def configure_optimizers(self):
        optimizer = torch.optim.AdamW(
            self.parameters(),lr=self.learning_rate,
            weight_decay=self.weight_decay)
        lr_schedulers = CosineAnnealingWithWarmupLR(
            optimizer,T_max=self.n_epochs,start_decay=self.start_decay,
            n_warmup_steps=self.warmup_steps)

        return {"optimizer":optimizer,
                "lr_scheduler":lr_schedulers,
                "monitor":"val_loss"}

    def on_train_epoch_end(self):
        sch = self.lr_schedulers().state_dict()
        lr = self.learning_rate
        last_lr = sch['_last_lr'][0] if '_last_lr' in sch else lr
        self.log("lr",last_lr,sync_dist=True,prog_bar=True)

    def setup_metrics(self):
        self.train_metrics = get_metric_dict(
            self.n_classes,[],prefix="")
        self.val_metrics = get_metric_dict(
            self.n_classes,None,prefix="V_")
        self.test_metrics = get_metric_dict(
            self.n_classes,None,prefix="T_")

    def update_metrics(self,prediction,y,metrics):
        if self.n_classes > 2:
            prediction = torch.softmax(prediction,1).to(torch.int64)
        else:
            prediction = torch.sigmoid(prediction)
        if len(y.shape) > 1:
            y.squeeze(1)
        for k in metrics:
            metrics[k](prediction,y)
            self.log(
                k,metrics[k],on_epoch=True,
                on_step=False,prog_bar=True,sync_dist=True)

class ClassNetPL(ClassPLABC):
    """
    Classification network implementation for Pytorch Lightning. Can be 
    parametrised as a categorical or ordinal network, depending on the 
    specification in net_type.
    """
    def __init__(
        self,
        net_type: str="cat",
        image_key: str="image",
        label_key: str="label",
        learning_rate: float=0.001,
        batch_size: int=4,
        weight_decay: float=0.0,
        training_dataloader_call: Callable=None,
        loss_fn: Callable=F.binary_cross_entropy,
        loss_params: dict={},
        n_epochs: int=100,
        warmup_steps: int=0,
        start_decay: int=None,
        training_batch_preproc: Callable=None,
        *args,**kwargs) -> torch.nn.Module:
        """
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
            n_epochs (int, optional): number of epochs. Defaults to 100.
            warmup_steps (int, optional): number of warmup steps. Defaults 
                to 0.
            start_decay (int, optional): number of steps after which decay
                begins. Defaults to None (decay starts after warmup).
            training_batch_preproc (Callable): function to be applied to the
                entire batch before feeding it to the model during training.
                Can contain transformations such as mixup, which require access
                to the entire training batch.
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
        self.n_epochs = n_epochs
        self.warmup_steps = warmup_steps
        self.start_decay = start_decay
        self.training_batch_preproc = training_batch_preproc
        self.args = args
        self.kwargs = kwargs
        
        self.setup_network()
        self.setup_metrics()
           
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

    def update_metrics(self,prediction,y,metrics):
        if self.net_type == "ord":
            prediction = ordinal_prediction_to_class(prediction)
        elif self.n_classes > 2:
            prediction = torch.softmax(prediction,1).to(torch.int64)
        else:
            prediction = torch.sigmoid(prediction)
        if len(y.shape) > 1:
            y.squeeze(1)
        for k in metrics:
            metrics[k](prediction,y)
            self.log(
                k,metrics[k],on_epoch=True,
                on_step=False,prog_bar=True)

class SegCatNetPL(SegCatNet,pl.LightningModule):
    """
    PL module for SegCatNet.
    """
    def __init__(self,
                 image_key: str="image",
                 label_key: str="label",
                 skip_conditioning_key: str=None,
                 feature_conditioning_key: str=None,
                 learning_rate: float=0.001,
                 batch_size: int=4,
                 weight_decay: float=0.0,
                 training_dataloader_call: Callable=None,
                 loss_fn: Callable=F.binary_cross_entropy_with_logits,
                 loss_params: dict={},
                 n_epochs: int=100,
                 *args,**kwargs):
        """
        Args:
            image_key (str): key corresponding to the key from the train
                dataloader.
            label_key (str): key corresponding to the label key from the train
                dataloader.
            skip_conditioning_key (str, optional): key for the skip 
                conditioning element of the batch.
            feature_conditioning_key (str, optional): key for the feature 
                conditioning elements in the batch.
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
            n_epochs (int, optional): number of epochs. Defaults to 100.
            args: arguments for classification network class.
            kwargs: keyword arguments for classification network class.

        Returns:
            pl.LightningModule: a classification network module.
        """

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
        self.n_epochs = n_epochs

        self.setup_metrics()
        
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
            pred = torch.sigmoid(pred)
        else:
            pred = F.softmax(pred,-1)
        for k in metrics:
            metrics[k](pred,y)
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

class UNetEncoderPL(UNetEncoder,ClassPLABC):
    """
    U-Net encoder-based classification network implementation for Pytorch
    Lightning.
    """
    def __init__(
        self,
        image_key: str="image",
        label_key: str="label",
        learning_rate: float=0.001,
        batch_size: int=4,
        weight_decay: float=0.0,
        training_dataloader_call: Callable=None,
        loss_fn: Callable=F.binary_cross_entropy,
        loss_params: dict={},
        n_epochs: int=100,
        warmup_steps: int=0,
        start_decay: int=None,
        training_batch_preproc: Callable=None,
        *args,**kwargs) -> torch.nn.Module:
        """
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
            n_epochs (int, optional): number of epochs. Defaults to 100.
            warmup_steps (int, optional): number of warmup steps. Defaults 
                to 0.
            start_decay (int, optional): number of steps after which decay
                begins. Defaults to None (decay starts after warmup).
            training_batch_preproc (Callable): function to be applied to the
                entire batch before feeding it to the model during training.
                Can contain transformations such as mixup, which require access
                to the entire training batch.
            args: arguments for classification network class.
            kwargs: keyword arguments for classification network class.

        Returns:
            pl.LightningModule: a classification network module.
        """
        
        super().__init__(*args,**kwargs)

        self.image_key = image_key
        self.label_key = label_key
        self.learning_rate = learning_rate
        self.batch_size = batch_size
        self.weight_decay = weight_decay
        self.training_dataloader_call = training_dataloader_call
        self.loss_fn = loss_fn
        self.loss_params = loss_params
        self.n_epochs = n_epochs
        self.warmup_steps = warmup_steps
        self.start_decay = start_decay
        self.training_batch_preproc = training_batch_preproc
        self.args = args
        self.kwargs = kwargs
        
        self.setup_metrics()

class GenericEnsemblePL(GenericEnsemble,pl.LightningModule):
    """
    Ensemble classification network for PL.
    """
    def __init__(
        self,
        image_keys: List[str]="image",
        label_key: str="label",
        learning_rate: float=0.001,
        batch_size: int=4,
        weight_decay: float=0.0,
        training_dataloader_call: Callable=None,
        loss_fn: Callable=F.binary_cross_entropy,
        loss_params: dict={},
        n_epochs: int=100,
        *args,**kwargs) -> torch.nn.Module:
        """
        Args:
            image_keys (str): key corresponding to the key from the train
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
            n_epochs (int, optional): number of epochs. Defaults to 100.
            args: arguments for classification network class.
            kwargs: keyword arguments for classification network class.

        Returns:
            pl.LightningModule: a classification network module.
        """
        
        super().__init__(*args,**kwargs)

        self.image_keys = image_keys
        self.label_key = label_key
        self.learning_rate = learning_rate
        self.batch_size = batch_size
        self.weight_decay = weight_decay
        self.training_dataloader_call = training_dataloader_call
        self.loss_fn = loss_fn
        self.loss_params = loss_params
        self.n_epochs = n_epochs
        self.args = args
        self.kwargs = kwargs
        
        self.setup_metrics()

    def training_step(self,batch,batch_idx):
        x, y = [batch[k] for k in self.image_keys],batch[self.label_key]
        prediction = self.forward(x)
        prediction = torch.squeeze(prediction,1)

        loss = self.calculate_loss(prediction,y)
        
        self.log("train_loss", loss)
        return loss
    
    def validation_step(self,batch,batch_idx):
        x, y = [batch[k] for k in self.image_keys],batch[self.label_key]
        prediction = self.forward(x)
        prediction = torch.squeeze(prediction,1)

        loss = self.calculate_loss(prediction,y)
        self.log("val_loss",loss,on_epoch=True,
                 on_step=False,prog_bar=True,
                 batch_size=x.shape[0])        
        self.update_metrics(prediction,y,self.val_metrics)
        return loss

    def test_step(self,batch,batch_idx):
        x, y = [batch[k] for k in self.image_keys],batch[self.label_key]
        prediction = self.forward(x)
        prediction = torch.squeeze(prediction,1)

        loss = self.calculate_loss(prediction,y)
            
        self.update_metrics(prediction,y,self.test_metrics)
        return loss

    def train_dataloader(self) -> torch.utils.data.DataLoader:
        return self.training_dataloader_call()

    def configure_optimizers(self):
        optimizer = torch.optim.AdamW(
            self.parameters(),lr=self.learning_rate,
            weight_decay=self.weight_decay)
        lr_schedulers = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer,self.n_epochs)

        return {"optimizer":optimizer,
                "lr_scheduler":lr_schedulers,
                "monitor":"val_loss"}
    
    def on_train_epoch_end(self):
        sch = self.lr_schedulers().state_dict()
        lr = self.learning_rate
        last_lr = sch['_last_lr'][0] if '_last_lr' in sch else lr
        self.log("lr",last_lr)

    def setup_metrics(self):
        self.train_metrics = get_metric_dict(
            self.n_classes,[],prefix="")
        self.val_metrics = get_metric_dict(
            self.n_classes,None,prefix="V_")
        self.test_metrics = get_metric_dict(
            self.n_classes,None,prefix="T_")

class ViTClassifierPL(ViTClassifier,ClassPLABC):
    """
    ViT classification network implementation for Pytorch
    Lightning.
    """
    def __init__(
        self,
        image_key: str="image",
        label_key: str="label",
        learning_rate: float=0.001,
        batch_size: int=4,
        weight_decay: float=0.0,
        training_dataloader_call: Callable=None,
        loss_fn: Callable=F.binary_cross_entropy,
        loss_params: dict={},
        n_epochs: int=100,
        warmup_steps: int=0,
        start_decay: int=None,
        training_batch_preproc: Callable=None,
        *args,**kwargs) -> torch.nn.Module:
        """
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
            n_epochs (int, optional): number of epochs. Defaults to 100.
            warmup_steps (int, optional): number of warmup steps. Defaults 
                to 0.
            start_decay (int, optional): number of steps after which decay
                begins. Defaults to None (decay starts after warmup).
            training_batch_preproc (Callable): function to be applied to the
                entire batch before feeding it to the model during training.
                Can contain transformations such as mixup, which require access
                to the entire training batch.
            args: arguments for classification network class.
            kwargs: keyword arguments for classification network class.

        Returns:
            pl.LightningModule: a classification network module.
        """
        
        super().__init__(*args,**kwargs)

        self.image_key = image_key
        self.label_key = label_key
        self.learning_rate = learning_rate
        self.batch_size = batch_size
        self.weight_decay = weight_decay
        self.training_dataloader_call = training_dataloader_call
        self.loss_fn = loss_fn
        self.loss_params = loss_params
        self.n_epochs = n_epochs
        self.warmup_steps = warmup_steps
        self.start_decay = start_decay
        self.training_batch_preproc = training_batch_preproc
        self.args = args
        self.kwargs = kwargs
        
        self.setup_metrics()

class FactorizedViTClassifierPL(FactorizedViTClassifier,ClassPLABC):
    """
    ViT classification network implementation for Pytorch
    Lightning.
    """
    def __init__(
        self,
        image_key: str="image",
        label_key: str="label",
        learning_rate: float=0.001,
        batch_size: int=4,
        weight_decay: float=0.0,
        training_dataloader_call: Callable=None,
        loss_fn: Callable=F.binary_cross_entropy,
        loss_params: dict={},
        n_epochs: int=100,
        warmup_steps: int=0,
        start_decay: int=None,
        training_batch_preproc: Callable=None,
        *args,**kwargs) -> torch.nn.Module:
        """
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
            n_epochs (int, optional): number of epochs. Defaults to 100.
            warmup_steps (int, optional): number of warmup steps. Defaults 
                to 0.
            start_decay (int, optional): number of steps after which decay
                begins. Defaults to None (decay starts after warmup).
            training_batch_preproc (Callable): function to be applied to the
                entire batch before feeding it to the model during training.
                Can contain transformations such as mixup, which require access
                to the entire training batch.
            args: arguments for classification network class.
            kwargs: keyword arguments for classification network class.

        Returns:
            pl.LightningModule: a classification network module.
        """
            
        super().__init__(*args,**kwargs)

        self.image_key = image_key
        self.label_key = label_key
        self.learning_rate = learning_rate
        self.batch_size = batch_size
        self.weight_decay = weight_decay
        self.training_dataloader_call = training_dataloader_call
        self.loss_fn = loss_fn
        self.loss_params = loss_params
        self.n_epochs = n_epochs
        self.warmup_steps = warmup_steps
        self.start_decay = start_decay
        self.training_batch_preproc = training_batch_preproc
        self.args = args
        self.kwargs = kwargs
        
        self.setup_metrics()

class TransformableTransformerPL(TransformableTransformer,ClassPLABC):
    """
    PL module for the TransformableTransformer.
    """
    def __init__(
        self,
        image_key: str="image",
        label_key: str="label",
        learning_rate: float=0.001,
        batch_size: int=4,
        weight_decay: float=0.0,
        training_dataloader_call: Callable=None,
        loss_fn: Callable=F.binary_cross_entropy,
        loss_params: dict={},
        n_epochs: int=100,
        warmup_steps: int=0,
        start_decay: int=None,
        training_batch_preproc: Callable=None,
        *args,**kwargs) -> torch.nn.Module:
        """
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
            n_epochs (int, optional): number of epochs. Defaults to 100.
            warmup_steps (int, optional): number of warmup steps. Defaults 
                to 0.
            start_decay (int, optional): number of steps after which decay
                begins. Defaults to None (decay starts after warmup).
            training_batch_preproc (Callable): function to be applied to the
                entire batch before feeding it to the model during training.
                Can contain transformations such as mixup, which require access
                to the entire training batch.
            args: arguments for classification network class.
            kwargs: keyword arguments for classification network class.

        Returns:
            pl.LightningModule: a classification network module.
        """
            
        super().__init__(*args,**kwargs)

        self.image_key = image_key
        self.label_key = label_key
        self.learning_rate = learning_rate
        self.batch_size = batch_size
        self.weight_decay = weight_decay
        self.training_dataloader_call = training_dataloader_call
        self.loss_fn = loss_fn
        self.loss_params = loss_params
        self.n_epochs = n_epochs
        self.warmup_steps = warmup_steps
        self.start_decay = start_decay
        self.training_batch_preproc = training_batch_preproc
        self.args = args
        self.kwargs = kwargs
        
        self.setup_metrics()

class HybridClassifierPL(HybridClassifier,ClassPLABC):
    """
    PL module for the HybridClassifier.
    """
    def __init__(
        self,
        image_key: str="image",
        label_key: str="label",
        tab_key: str="tabular",
        learning_rate: float=0.001,
        batch_size: int=4,
        weight_decay: float=0.0,
        training_dataloader_call: Callable=None,
        loss_fn: Callable=F.binary_cross_entropy,
        loss_params: dict={},
        n_epochs: int=100,
        warmup_steps: int=0,
        start_decay: int=None,
        training_batch_preproc: Callable=None,
        *args,**kwargs) -> torch.nn.Module:
        """
        Args:
            image_key (str): key corresponding to the key from the train
                dataloader.
            label_key (str): key corresponding to the label key from the train
                dataloader.
            tab_key (str): key corresponding to the tabular data key in the 
                train dataloader.
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
            n_epochs (int, optional): number of epochs. Defaults to 100.
            warmup_steps (int, optional): number of warmup steps. Defaults 
                to 0.
            start_decay (int, optional): number of steps after which decay
                begins. Defaults to None (decay starts after warmup).
            training_batch_preproc (Callable): function to be applied to the
                entire batch before feeding it to the model during training.
                Can contain transformations such as mixup, which require access
                to the entire training batch.
            args: arguments for classification network class.
            kwargs: keyword arguments for classification network class.

        Returns:
            pl.LightningModule: a classification network module.
        """
            
        super().__init__(*args,**kwargs)

        self.image_key = image_key
        self.label_key = label_key
        self.tab_key = tab_key
        self.learning_rate = learning_rate
        self.batch_size = batch_size
        self.weight_decay = weight_decay
        self.training_dataloader_call = training_dataloader_call
        self.loss_fn = loss_fn
        self.loss_params = loss_params
        self.n_epochs = n_epochs
        self.warmup_steps = warmup_steps
        self.start_decay = start_decay
        self.training_batch_preproc = training_batch_preproc
        self.args = args
        self.kwargs = kwargs

        self.setup_metrics()

    def training_step(self,batch,batch_idx):
        x_conv, y = batch[self.image_key],batch[self.label_key]
        x_tab = batch[self.tab_key]
        prediction = self.forward(x_conv,x_tab)
        prediction = torch.squeeze(prediction,1)

        loss = self.calculate_loss(prediction,y,with_params=True)

        self.log("train_loss",loss,sync_dist=True)
        return loss

    def validation_step(self,batch,batch_idx):
        x_conv, y = batch[self.image_key],batch[self.label_key]
        x_tab = batch[self.tab_key]
        prediction = self.forward(x_conv,x_tab)
        prediction = torch.squeeze(prediction,1)

        loss = self.calculate_loss(prediction,y,with_params=True)
        self.log("val_loss",loss,on_epoch=True,
                 on_step=False,prog_bar=True,
                 batch_size=x_conv.shape[0],sync_dist=True)        
        self.update_metrics(prediction,y,self.val_metrics)
        return loss

    def test_step(self,batch,batch_idx):
        x_conv, y = batch[self.image_key],batch[self.label_key]
        x_tab = batch[self.tab_key]
        prediction = self.forward(x_conv,x_tab)
        prediction = torch.squeeze(prediction,1)

        loss = self.calculate_loss(prediction,y)
        self.log("test_loss",loss,on_epoch=True,
                 on_step=False,prog_bar=True,
                 batch_size=x_conv.shape[0],sync_dist=True)        
        self.update_metrics(prediction,y,self.test_metrics)
        return loss
