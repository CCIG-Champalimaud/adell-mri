import numpy as np
import torch
import torch.nn.functional as F
import torchmetrics
import pytorch_lightning as pl
from typing import Callable

from .object_detection import YOLONet3d,CoarseDetector3d,mAP

class YOLONet3dPL(YOLONet3d,pl.LightningModule):
    def __init__(
        self,
        image_key: str="image",
        label_key: str="label",
        boxes_key: str="boxes",
        box_label_key: str="labels",
        learning_rate: float=0.001,
        batch_size: int=4,
        weight_decay: float=0.005,
        training_dataloader_call: Callable=None,
        reg_loss_fn: Callable=F.mse_loss,
        classification_loss_fn: Callable=F.binary_cross_entropy,
        object_loss_fn: Callable=F.binary_cross_entropy,
        positive_weight: float=1.,
        classification_loss_params: dict={},
        object_loss_params: dict={},
        iou_threshold: float=0.5,
        *args,**kwargs) -> torch.nn.Module:
        """YOLO-like network implementation for Pytorch Lightning.

        Args:
            image_key (str): key corresponding to the key from the train
            dataloader.
            label_key (str): key corresponding to the label key from the train
            dataloader.
            boxes_key (str): key corresponding to the original bounding boxes
            from the train dataloader.
            learning_rate (float, optional): learning rate. Defaults to 0.001.
            batch_size (int, optional): batch size. Defaults to 4.
            weight_decay (float, optional): weight decay for optimizer. Defaults 
            to 0.005.
            training_dataloader_call (Callable, optional): call for the 
            training dataloader. Defaults to None.
            reg_loss_fn (Callable, optional): function to calculate the box 
            regression loss. Defaults to F.mse_loss.
            classification_loss_fn (Callable, optional): function to calculate 
            the classification loss. Defaults to F.binary_class_entropy.
            object_loss_fn (Callable, optional): function to calculate the 
            objectness loss. Defaults to F.binary_class_entropy.
            positive_weight (float, optional): weight for positive object 
            prediction. Defaults to 1.0.
            classification_loss_params (dict, optional): classification
            loss parameters. Defaults to {}.
            object_loss_params (dict, optional): object loss parameters. 
            Defaults to {}.
            args: arguments for UNet class.
            kwargs: keyword arguments for UNet class.

        Returns:
            pl.LightningModule: a U-Net module.
        """
        
        super().__init__(*args,**kwargs)

        self.image_key = image_key
        self.label_key = label_key
        self.boxes_key = boxes_key
        self.box_label_key = box_label_key
        self.learning_rate = learning_rate
        self.batch_size = batch_size
        self.weight_decay = weight_decay
        self.training_dataloader_call = training_dataloader_call
        self.reg_loss_fn = reg_loss_fn
        self.classification_loss_fn = classification_loss_fn
        self.object_loss_fn = object_loss_fn
        self.positive_weight = positive_weight
        self.classification_loss_params = classification_loss_params
        self.object_loss_params = object_loss_params
        self.iou_threshold = iou_threshold

        self.object_idxs = np.array([0])
        self.center_idxs = np.array([1,2,3])
        self.size_idxs = np.array([4,5,6])
        self.class_idxs = np.array([7])
        
        self.setup_metrics()
        
        self.loss_accumulator = 0.
        self.loss_accumulator_d = 0.
   
    def calculate_loss(self,prediction,y,y_class,b,h,w,d,a,weights=None):
        bb_center,bb_size,bb_object,bb_cl = prediction
        pred_centers = bb_center[b,:,h,w,d,a]
        y_centers = y[b,:,h,w,d,a][:,self.center_idxs]
        pred_size = torch.exp(bb_size[b,:,h,w,d,a])/2
        y_size = torch.exp(y[b,:,h,w,d,a][:,self.size_idxs])/2

        pred_corners = torch.cat(
            [pred_centers-pred_size,pred_centers+pred_size],1)
        y_corners = torch.cat(
            [y_centers-y_size,y_centers+y_size],1)
        iou,cpd,ar = self.reg_loss_fn(
            pred_corners,y_corners)
        cla_loss = self.classification_loss_fn(
            bb_cl[b,:,h,w,d],y_class[b,h,w,d],
            **self.classification_loss_params)

        y_object = torch.zeros_like(bb_object,device=self.dev)
        y_object[b,:,h,w,d,a] = torch.unsqueeze(iou,1)

        obj_loss = self.object_loss_fn(
            bb_object,y_object,**self.object_loss_params)

        obj_weight = 1.0
        box_weight = 0.1
        output = obj_loss.mean() * obj_weight
        output = output + ((1-iou).mean() + cpd.mean() + ar.mean()) * box_weight
        if self.n_c > 2:
            output = output + cla_loss.mean()
        return output.mean()
    
    def retrieve_correct(self,prediction,target,target_class,typ,b,h,w,d,a):
        typ = typ.lower()
        if typ == "center":
            p,t = prediction[0],target[:,self.center_idxs,:,:,:,:]
        elif typ == "size":
            p,t = prediction[1],target[:,self.size_idxs,:,:,:,:]
        elif typ == "obj":
            p,t = prediction[2],target[:,self.object_idxs,:,:,:,:]
            t = (t > self.iou_threshold).int()
        elif typ == "class":
            p,t = prediction[3],target_class
            t = torch.round(t).int()
        elif typ == "map":
            p,t = self.recover_boxes_batch(*prediction,to_dict=True),None
        if typ not in ["obj","map","class"]:
            p,t = p[b,:,h,w,d,a],t[b,:,h,w,d,a]
        elif typ == "class":
            p,t = p[b,:,h,w,d],t[b,h,w,d]
        return p,t

    def training_step(self,batch,batch_idx):
        x, y = batch[self.image_key],batch[self.label_key].float()
        prediction = list(self.forward(x))
        y_class,y = y[:,0,:,:,:],y[:,1:,:,:,:]
        y = torch.stack(self.split(y,self.n_b,1),-1)
        b,h,w,d,a = torch.where(y[:,0,:,:,:,:] > self.iou_threshold)
        prediction[:-1] = [torch.stack(self.split(x,self.n_b,1),-1)
                           for x in prediction[:-1]]
        batch_size = int(prediction[0].shape[0])
        if batch_size == 1:
            y = torch.unsqueeze(y,0)

        loss = self.calculate_loss(prediction,y,y_class,b,h,w,d,a)

        self.log("train_loss", loss)
        for k_typ in self.train_metrics:
            k,typ = k_typ.split('_')
            cur_pred,cur_target = self.retrieve_correct(
                prediction,y,y_class,typ,b,h,w,d,a)
            self.train_metrics[k_typ](cur_pred,cur_target)
            self.log(
                k,self.train_metrics[k_typ],on_epoch=True,
                on_step=False,prog_bar=True)
        return loss
    
    def validation_step(self,batch,batch_idx):
        x, y = batch[self.image_key],batch[self.label_key].float()
        prediction = list(self.forward(x))
        y_class,y = y[:,0,:,:,:],y[:,1:,:,:,:]
        y = torch.stack(self.split(y,self.n_b,1),-1)
        b,h,w,d,a = torch.where(y[:,0,:,:,:,:] > self.iou_threshold)
        prediction[:-1] = [torch.stack(self.split(x,self.n_b,1),-1)
                           for x in prediction[:-1]]
        batch_size = int(prediction[0].shape[0])
        if batch_size == 1:
            y = torch.unsqueeze(y,0)

        loss = self.calculate_loss(prediction,y,y_class,b,h,w,d,a)

        self.loss_accumulator += loss
        self.loss_accumulator_d += 1.
        for k_typ in self.val_metrics:
            k,typ = k_typ.split('_')
            cur_pred,cur_target = self.retrieve_correct(
                prediction,y,y_class,typ,b,h,w,d,a)
            if typ.lower() != "map":
                self.val_metrics[k_typ].update(cur_pred,cur_target)
            else:
                cur_pred,cur_target = self.retrieve_correct(
                    prediction,y,y_class,typ,b,h,w,d,a)
                cur_target = [
                    {'boxes':batch[self.boxes_key][i],
                     'labels':batch[self.box_label_key][i]}
                     for i in range(batch[self.boxes_key].shape[0])]
                for t in cur_target:
                    t['boxes'] = torch.concat(
                        [t['boxes'][:,:,0],t['boxes'][:,:,1]],axis=1)
                    t['boxes'] = t['boxes'].to(self.dev)
                    t['labels'] = torch.as_tensor(t['labels']).to(self.dev)
                self.val_metrics[k_typ].update(cur_pred,cur_target)
        return loss

    def test_step(self,batch,batch_idx):
        x, y = batch[self.image_key],batch[self.label_key].float()
        prediction = list(self.forward(x))
        y_class,y = y[:,0,:,:,:],y[:,1:,:,:,:]
        y = torch.stack(self.split(y,self.n_b,1),-1)
        b,h,w,d,a = torch.where(y[:,0,:,:,:,:] > self.iou_threshold)
        prediction[:-1] = [torch.stack(self.split(x,self.n_b,1),-1)
                           for x in prediction[:-1]]

        loss = self.calculate_loss(prediction,y,y_class,b,h,w,d,a)

        for k_typ in self.test_metrics:
            k,typ = k_typ.split('_')
            if typ.lower() != "map":
                cur_pred,cur_target = self.retrieve_correct(
                    prediction,y,y_class,typ,b,h,w,d,a)
                self.test_metrics[k_typ].update(cur_pred,cur_target)
            else:
                cur_pred,cur_target = self.retrieve_correct(
                    prediction,y,y_class,typ,b,h,w,d,a)
                cur_target = [
                    {'boxes':batch[self.boxes_key][i],
                     'labels':batch[self.box_label_key][i]}
                     for i in range(batch[self.boxes_key].shape[0])]
                for t in cur_target:
                    t['boxes'] = torch.concat(
                        [t['boxes'][:,:,0],t['boxes'][:,:,1]],axis=1)
                    t['boxes'] = t['boxes'].to(self.dev)
                    t['labels'] = torch.as_tensor(t['labels']).to(self.dev)
                self.test_metrics[k_typ].update(cur_pred,cur_target)
        return loss

    def train_dataloader(self) -> torch.utils.data.DataLoader:
        return self.training_dataloader_call()

    def configure_optimizers(self):
        optimizer = torch.optim.AdamW(
            self.parameters(),lr=self.learning_rate,
            weight_decay=self.weight_decay,amsgrad=True)
        lr_schedulers = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer,patience=5,min_lr=5e-5,factor=0.1,verbose=True,
            cooldown=5)

        return {"optimizer":optimizer,
                "lr_scheduler":lr_schedulers,
                "monitor":"val_loss"}
    
    def on_validation_epoch_end(self):
        for k_typ in self.val_metrics: 
            k,typ = k_typ.split('_')
            val = self.val_metrics[k_typ].compute()
            self.log(
                k,val,prog_bar=True)
            self.val_metrics[k_typ].reset()
        val_loss = self.loss_accumulator/self.loss_accumulator_d
        sch = self.lr_schedulers().state_dict()
        lr = self.learning_rate
        last_lr = sch['_last_lr'][0] if '_last_lr' in sch else lr
        self.log("lr",last_lr)
        self.log("val_loss",val_loss,prog_bar=True)
        self.loss_accumulator = 0.
        self.loss_accumulator_d = 0.

    def setup_metrics(self):
        if self.n_c == 2:
            C,A,M = None,None,"micro"
        else:
            C,A,M = self.n_c,"samplewise","macro"
        self.train_metrics = torch.nn.ModuleDict({
            "cMSE_center":torchmetrics.MeanSquaredError(),
            "sMSE_size":torchmetrics.MeanSquaredError(),
            "objF1_obj":torchmetrics.FBetaScore(
                None,threshold=self.iou_threshold),
            "objRec_obj":torchmetrics.Recall(
                None,threshold=self.iou_threshold)})
        self.val_metrics = torch.nn.ModuleDict({
            "v:cMSE_center":torchmetrics.MeanSquaredError(),
            "v:sMSE_size":torchmetrics.MeanSquaredError(),
            "v:objF1_obj":torchmetrics.FBetaScore(
                None,threshold=self.iou_threshold)})
        self.test_metrics = torch.nn.ModuleDict({
            "testcMSE_center":torchmetrics.MeanSquaredError(),
            "testsMSE_size":torchmetrics.MeanSquaredError(),
            "testobjRec_obj":torchmetrics.Recall(
                None,threshold=self.iou_threshold),
            "testobjPre_obj":torchmetrics.Precision(
                None,threshold=self.iou_threshold),   
            "testobjF1_obj":torchmetrics.FBetaScore(
                None,threshold=self.iou_threshold),
            "testmAP_mAP":mAP(iou_threshold=self.iou_threshold)})

        if self.n_c > 2:
            # no point in including this in the two class scenario
            self.train_metrics["clF1_class"] = torchmetrics.FBetaScore(
                C,mdmc_average=A,average=M)
            self.val_metrics["v:clF1_class"] = torchmetrics.FBetaScore(
                C,mdmc_average=A,average=M)
            self.test_metrics["t:clF1_class"] = torchmetrics.FBetaScore(
                C,mdmc_average=A,average=M)

class CoarseDetector3dPL(CoarseDetector3d,pl.LightningModule):
    def __init__(
        self,
        image_key: str="image",
        label_key: str="label",
        boxes_key: str="bb",
        learning_rate: float=0.001,
        batch_size: int=4,
        weight_decay: float=0.005,
        training_dataloader_call: Callable=None,
        object_loss_fn: Callable=F.binary_cross_entropy,
        positive_weight: float=1.,
        object_loss_params: dict={},
        iou_threshold: float=0.5,
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
            object_loss_fn (Callable, optional): function to calculate the 
            objectness loss. Defaults to F.binary_class_entropy.
            positive_weight (float, optional): weight for positive object 
            prediction. Defaults to 1.0.
            object_loss_params (dict, optional): object loss parameters. 
            Defaults to {}.
            args: arguments for CoarseDetector3d class.
            kwargs: keyword arguments for CoarseDetector3d class.

        Returns:
            pl.LightningModule: a CoarseDetector3d module.
        """
        
        super().__init__(*args,**kwargs)

        self.image_key = image_key
        self.label_key = label_key
        self.boxes_key = boxes_key
        self.learning_rate = learning_rate
        self.batch_size = batch_size
        self.weight_decay = weight_decay
        self.training_dataloader_call = training_dataloader_call
        self.object_loss_fn = object_loss_fn
        self.positive_weight = positive_weight
        self.object_loss_params = object_loss_params
        self.iou_threshold = iou_threshold

        self.object_idxs = np.array([0])
        
        self.setup_metrics()
        
        self.loss_accumulator = 0.
        self.loss_accumulator_d = 0.
   
    def calculate_loss(self,prediction,y,weights=None):
        obj_loss = self.object_loss_fn(
            prediction,y,**self.object_loss_params)

        return obj_loss.mean()
    
    def split(self,x,n_splits,dim):
        size = int(x.shape[dim]//n_splits)
        return torch.split(x,size,dim)

    def training_step(self,batch,batch_idx):
        x, y = batch[self.image_key],batch[self.label_key].float()
        prediction = self.forward(x)
        # select the objectness tensor
        y = torch.stack(self.split(y[:,1:],self.n_b,1),-1)
        y = y[:,self.object_idxs].sum(-1)
        y = torch.where(
            y>0,
            torch.ones_like(y,device=y.device),
            torch.zeros_like(y,device=y.device))
        batch_size = int(prediction.shape[0])
        if batch_size == 1:
            y = torch.unsqueeze(y,0)

        loss = self.calculate_loss(prediction,y)

        self.log("train_loss", loss)
        for k_typ in self.train_metrics:
            k,typ = k_typ.split('_')
            self.train_metrics[k_typ](prediction,y.int())
            self.log(
                k,self.train_metrics[k_typ],on_epoch=True,
                on_step=False,prog_bar=True)
        return loss
    
    def validation_step(self,batch,batch_idx):
        x, y = batch[self.image_key],batch[self.label_key].float()
        prediction = self.forward(x)
        # select the objectness tensor
        y = torch.stack(self.split(y[:,1:],self.n_b,1),-1)
        y = y[:,self.object_idxs].sum(-1)
        y = torch.where(
            y>0,
            torch.ones_like(y,device=y.device),
            torch.zeros_like(y,device=y.device))
        batch_size = int(prediction.shape[0])
        if batch_size == 1:
            y = torch.unsqueeze(y,0)

        loss = self.calculate_loss(prediction,y)

        self.loss_accumulator += loss
        self.loss_accumulator_d += 1
        for k_typ in self.val_metrics:
            k,typ = k_typ.split('_')
            self.val_metrics[k_typ](prediction,y.int())
            self.log(
                k,self.val_metrics[k_typ],on_epoch=True,
                on_step=False,prog_bar=True)
        return loss

    def test_step(self,batch,batch_idx):
        x, y = batch[self.image_key],batch[self.label_key].float()
        prediction = self.forward(x)
        # select the objectness tensor
        y = torch.stack(self.split(y[:,1:],self.n_b,1),-1)
        y = y[:,self.object_idxs].sum(-1)
        y = torch.where(
            y>0,
            torch.ones_like(y,device=y.device),
            torch.zeros_like(y,device=y.device))
        batch_size = int(prediction.shape[0])
        if batch_size == 1:
            y = torch.unsqueeze(y,0)

        loss = self.calculate_loss(prediction,y)

        for k_typ in self.test_metrics:
            k,typ = k_typ.split('_')
            self.test_metrics[k_typ](prediction,y.int())
            self.log(
                k,self.test_metrics[k_typ],on_epoch=True,on_step=False)
        return loss

    def train_dataloader(self) -> torch.utils.data.DataLoader:
        return self.training_dataloader_call()

    def configure_optimizers(self):
        optimizer = torch.optim.AdamW(
            self.parameters(),lr=self.learning_rate,
            weight_decay=self.weight_decay,amsgrad=True)
        lr_schedulers = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer,patience=10,min_lr=1e-6,factor=0.3,verbose=True)

        return {"optimizer":optimizer,
                "lr_scheduler":lr_schedulers,
                "monitor":"val_loss"}
    
    def on_validation_epoch_end(self):
        for k_typ in self.val_metrics: 
            k,typ = k_typ.split('_')
            val = self.val_metrics[k_typ].compute()
            self.log(
                k,val,prog_bar=True)
            self.val_metrics[k_typ].reset()
        val_loss = self.loss_accumulator/self.loss_accumulator_d
        sch = self.lr_schedulers().state_dict()
        lr = self.learning_rate
        last_lr = sch['_last_lr'][0] if '_last_lr' in sch else lr
        self.log("lr",last_lr)
        self.log("val_loss",val_loss,prog_bar=True)
        self.loss_accumulator = 0.
        self.loss_accumulator_d = 0.

    def setup_metrics(self):
        self.train_metrics = torch.nn.ModuleDict({
            "objF1_obj":torchmetrics.FBetaScore(
                None,threshold=self.iou_threshold),
            "objRec_obj":torchmetrics.Recall(
                None,threshold=self.iou_threshold)})
        self.val_metrics = torch.nn.ModuleDict({
            "v:objF1_obj":torchmetrics.FBetaScore(
                None,threshold=self.iou_threshold)})
        self.test_metrics = torch.nn.ModuleDict({
            "testobjRec_obj":torchmetrics.Recall(
                None,threshold=self.iou_threshold),
            "testobjPre_obj":torchmetrics.Precision(
                None,threshold=self.iou_threshold),   
            "testobjF1_obj":torchmetrics.FBetaScore(
                None,threshold=self.iou_threshold)})
