"""
Implements Lightning-based training utilities.
"""

import warnings
from abc import ABC

import lightning.pytorch as pl
import numpy as np
import torch
import torchmetrics

from typing import Any
from adell_mri.custom_types import Callable
from adell_mri.modules.layers.conv_next import ConvNeXt
from adell_mri.modules.layers.res_net import ResNet
from adell_mri.modules.learning_rate import CosineAnnealingWithWarmupLR
from adell_mri.modules.segmentation.unet import UNet
from adell_mri.modules.self_supervised.dino import DINO
from adell_mri.modules.self_supervised.ibot import iBOT
from adell_mri.modules.self_supervised.jepa import IJEPA
from adell_mri.modules.self_supervised.autoencoders import ViTMaskedAutoEncoder
from adell_mri.modules.self_supervised.losses import (
    BarlowTwinsLoss,
    DinoLoss,
    NTXentLoss,
    VICRegLocalLoss,
    VICRegLoss,
    byol_loss,
    simsiam_loss,
)


class BarlowTwinsPL(ResNet, pl.LightningModule):
    def __init__(
        self,
        image_key: str = "image",
        augmented_image_key: str = "augmented_image",
        learning_rate: float = 0.001,
        batch_size: int = 4,
        weight_decay: float = 0.005,
        training_dataloader_call: Callable = None,
        loss_lam: float = 0.02,
        *args,
        **kwargs,
    ) -> torch.nn.Module:
        super().__init__(*args, **kwargs)

        self.image_key = image_key
        self.augmented_image_key = augmented_image_key
        self.learning_rate = learning_rate
        self.batch_size = batch_size
        self.weight_decay = weight_decay
        self.train_dataloader_call = training_dataloader_call
        self.loss_lam = loss_lam

        self.loss = BarlowTwinsLoss(moving=True, lam=self.loss_lam)

    def calculate_loss(self, y1, y2, update=True):
        loss = self.loss(y1, y2, update=True)
        return loss.mean()

    def update_metrics(self, y1, y2, metrics, log=True):
        for k in metrics:
            metrics[k].update(y1, y2)
            if log is True:
                self.log(
                    k,
                    metrics[k],
                    on_epoch=True,
                    on_step=False,
                    prog_bar=True,
                    sync_dist=True,
                )

    def training_step(self, batch, batch_idx):
        x1, x2 = batch[self.image_key], batch[self.augmented_image_key]
        y1, y2 = self.forward(x1), self.forward(x2)

        loss = self.calculate_loss(y1, y2)

        self.log("train_loss", loss, prob_bar=True)
        self.update_metrics(y1, y2, self.train_metrics)
        return loss

    def validation_step(self, batch, batch_idx):
        x1, x2 = batch[self.image_key], batch[self.augmented_image_key]
        y1, y2 = self.forward(x1), self.forward(x2)

        loss = self.calculate_loss(y1, y2, False)

        self.log("val_loss", loss, prob_bar=True, sync_dist=True)
        self.update_metrics(y1, y2, self.val_metrics)
        return loss

    def test_step(self, batch, batch_idx):
        x1, x2 = batch[self.image_key], batch[self.augmented_image_key]
        y1, y2 = self.forward(x1), self.forward(x2)

        loss = self.calculate_loss(y1, y2, False)

        self.log("test_loss", loss, prob_bar=True)
        self.update_metrics(y1, y2, self.test_metrics)
        return loss

    def train_dataloader(self) -> torch.utils.data.DataLoader:
        return self.training_dataloader_call(self.batch_size)

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(
            self.parameters(),
            lr=self.learning_rate,
            weight_decay=self.weight_decay,
        )
        lr_schedulers = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer,
            patience=5,
            cooldown=5,
            min_lr=1e-6,
            factor=0.25,
            verbose=True,
        )

        return {
            "optimizer": optimizer,
            "lr_scheduler": lr_schedulers,
            "monitor": "val_loss",
        }

    def on_validation_epoch_end(self):
        sch = self.lr_schedulers().state_dict()
        lr = self.learning_rate
        last_lr = sch["_last_lr"][0] if "_last_lr" in sch else lr
        self.log("lr", last_lr)

    def setup_metrics(self):
        metric_dict = {
            "MSE": torchmetrics.MeanSquaredError,
            "R": torchmetrics.PearsonCorrCoef,
            "CS": torchmetrics.CosineSimilarity,
        }
        self.train_metrics = torch.nn.ModuleDict({})
        self.val_metrics = torch.nn.ModuleDict({})
        self.test_metrics = torch.nn.ModuleDict({})
        for k in metric_dict:
            self.train_metrics[k] = metric_dict[k]()
            self.val_metrics["V" + k] = metric_dict[k]()
            self.test_metrics["T" + k] = metric_dict[k]()


class SelfSLBasePL(pl.LightningModule, ABC):
    """
    Abstract method for non-contrastive PL modules. Features some very
    basic but helpful functions which I use consistently when training
    non-contrastive self-supervised models.
    """

    def __init__(self):
        super().__init__()

    def train_dataloader(self) -> torch.utils.data.DataLoader:
        return self.training_dataloader_call(self.batch_size)

    def update_metrics(self, y1, y2, metrics, log=True):
        # torchmetrics only allows for 1d vectors for regression (?????)
        y1 = y1.flatten()
        y2 = y2.flatten()
        for k in metrics:
            metrics[k].update(y1, y2)
            if log is True:
                self.log(
                    k,
                    metrics[k],
                    on_epoch=True,
                    batch_size=y1.shape[0],
                    on_step=False,
                    prog_bar=True,
                    sync_dist=True,
                )

    def init_loss(self):
        self.loss = simsiam_loss
        if hasattr(self, "ema"):
            if self.ema is not None:
                self.loss = byol_loss
        if self.ssl_method == "vicreg":
            self.loss = VICRegLoss(**self.vic_reg_loss_params)
        if self.ssl_method == "vicregl":
            self.loss = VICRegLocalLoss(**self.vic_reg_loss_params)
        if self.ssl_method == "simclr":
            self.loss = NTXentLoss(temperature=self.temperature)

    def calculate_loss(self, y1, y2, *args):
        if self.stop_gradient is False:
            # no need to stop gradients with VICReg or VICRegL.
            loss_value = self.loss(y1, y2, *args)
        else:
            # the famous stop gradient operation just implies detaching the
            # output tensor from the computation graph to prevent gradients
            # from being automatically propagated.
            loss_value = self.loss(y1, y2.detach(), *args)
        return loss_value

    def safe_sum(self, X):
        if isinstance(X, torch.Tensor):
            return X.sum()
        else:
            return sum(X)

    def configure_optimizers(self):
        if self.n_steps is not None:
            interval = "step"
            n = self.n_steps
        else:
            interval = "epoch"
            n = self.n_epochs
        params_no_decay = []
        params_decay = []
        for k, p in self.named_parameters():
            if "normalization" in k:
                params_no_decay.append(p)
            else:
                params_decay.append(p)
        optimizer = torch.optim.AdamW(
            params_decay + params_no_decay,
            lr=self.learning_rate,
            weight_decay=self.weight_decay,
        )
        lr_schedulers = lr_schedulers = CosineAnnealingWithWarmupLR(
            optimizer,
            T_max=n,
            start_decay=self.start_decay,
            n_warmup_steps=self.warmup_steps,
            eta_min=0.0,
        )

        return {
            "optimizer": optimizer,
            "lr_scheduler": {
                "scheduler": lr_schedulers,
                "interval": interval,
                "frequency": 1,
            },
            "monitor": "val_loss",
        }

    def on_train_epoch_end(self):
        sch = self.lr_schedulers().state_dict()
        lr = self.learning_rate
        last_lr = sch["_last_lr"][0] if "_last_lr" in sch else lr
        self.log("lr", last_lr, sync_dist=True)

    def setup_metrics(self):
        metric_dict = {
            # "MSE":torchmetrics.MeanSquaredError
        }
        self.train_metrics = torch.nn.ModuleDict({})
        self.val_metrics = torch.nn.ModuleDict({})
        self.test_metrics = torch.nn.ModuleDict({})
        for k in metric_dict:
            self.train_metrics[k] = metric_dict[k]()
            self.val_metrics["V" + k] = metric_dict[k]()
            self.test_metrics["T" + k] = metric_dict[k]()

    def on_train_start(self):
        print("Training with the following hyperparameters:")
        parameter_dict = {}
        for k, v in self.hparams.items():
            print(f"\t{k}: {v}")
            if isinstance(v, (list, tuple, torch.Tensor, np.ndarray)):
                if len(v) > 1:
                    for i in range(len(v)):
                        parameter_dict[f"{k}_{i}"] = v[i]
                else:
                    parameter_dict[k] = v[0]
            elif isinstance(v, dict):
                for kk, vv in v.items():
                    if isinstance(vv, float):
                        parameter_dict[f"{k}_{kk}"] = float(vv)
            elif isinstance(v, (int, float, bool)):
                parameter_dict[k] = v
        parameter_dict = {
            k: float(parameter_dict[k])
            for k in parameter_dict
            if isinstance(k, (int, float, bool))
        }
        self.log_dict(parameter_dict, sync_dist=True)


class SelfSLResNetPL(ResNet, SelfSLBasePL):
    """
    Operates a number of non-contrastive self-supervised learning
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
        aug_image_key_1: str = "aug_image_1",
        aug_image_key_2: str = "aug_image_2",
        box_key_1: str = "box_1",
        box_key_2: str = "box_2",
        learning_rate: float = 0.001,
        batch_size: int = 4,
        weight_decay: float = 0.005,
        training_dataloader_call: Callable = None,
        n_epochs: int = 1000,
        n_steps: int = None,
        warmup_steps: int = 0,
        start_decay: int = None,
        ema: torch.nn.Module = None,
        ssl_method: str = "simclr",
        temperature: float = 1.0,
        vic_reg_loss_params: dict = {},
        stop_gradient: bool = True,
        channels_to_batch: bool = False,
        *args,
        **kwargs,
    ):
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
            n_steps (int, optional): number of steps. Defaults to None. Only
                used if n_epochs is None.
            warmup_steps (int, optional): number of warmup steps. Defaults
                to 0.
            start_decay (int, optional): number of steps after which decay
                begins. Defaults to None (decay starts after warmup).
            ema (float, torch.nn.Module): exponential moving decay module
                (EMA). Must have an update method that takes model as input
                and updates the weights based on this. Defaults to None.
            ssl_method (str, optional): sets the SSL method. Defaults to
                "simclr".
            temperature (float, optional): temperature for NTXent (when
                simclr == True). Defaults to 1.0.
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
        self.n_steps = n_steps
        self.warmup_steps = warmup_steps
        self.start_decay = start_decay
        self.ssl_method = ssl_method
        self.temperature = temperature
        self.vic_reg_loss_params = vic_reg_loss_params
        self.stop_gradient = stop_gradient
        self.channels_to_batch = channels_to_batch

        if channels_to_batch is True:
            kwargs["backbone_args"]["in_channels"] = 1

        super().__init__(*args, **kwargs)
        self.ema = ema

        if all(
            [
                self.ssl_method not in ["vicreg", "vicregl", "simclr"],
                self.stop_gradient is False,
            ]
        ):
            warnings.warn(
                "stop_gradient=False should not (in theory) be used\
            with vic_reg=False, vic_reg_local=False or simclr=False"
            )

        self.init_loss()
        if self.ema is not None:
            self.ema.update(self)
        else:
            self.ema = None

        self.loss_str_dict = {
            "standard": [None],
            "vicreg": ["inv", "var", "cov"],
            "vicregl": ["inv", "var", "cov", "local"],
        }

        self.save_hyperparameters()
        self.setup_metrics()

    def forward_ema_stop_grad(self, x, ret):
        if self.ema is not None:
            op = self.ema.shadow.forward
        else:
            op = self.forward
        if self.stop_gradient is True:
            with torch.no_grad():
                return op(x, ret)
        else:
            return op(x, ret)

    def step(self, batch, loss_str: str, metrics: dict, train=False):
        if self.ssl_method == "simclr":
            ret_string_1 = "projection"
            ret_string_2 = "projection"
            other_args = []
        elif self.ssl_method != "vicregl":
            ret_string_1 = "prediction"
            ret_string_2 = "projection"
            other_args = []
        else:
            ret_string_1 = "representation"
            ret_string_2 = "representation"
            box_1 = batch[self.box_key_1]
            box_2 = batch[self.box_key_2]
            other_args = [box_1, box_2]

        x1, x2 = batch[self.aug_image_key_1], batch[self.aug_image_key_2]
        if self.channels_to_batch is True:
            x1 = x1.reshape(-1, 1, *x1.shape[2:])
            x2 = x2.reshape(-1, 1, *x2.shape[2:])
        y1 = self.forward(x1, ret=ret_string_1)
        y2 = self.forward_ema_stop_grad(x2, ret=ret_string_2)

        losses = self.calculate_loss(y1, y2, *other_args)
        self.update_metrics(y1, y2, metrics)

        # loss is already symmetrised for VICReg, VICRegL and SimCLR
        if self.ssl_method not in ["vicreg", "vicregl", "simclr"]:
            y1_ = self.forward_ema_stop_grad(x1, ret=ret_string_1)
            y2_ = self.forward(x2, ret=ret_string_2)
            losses = losses + self.calculate_loss(y2_, y1_, *other_args)
            self.update_metrics(y2_, y1_, metrics)

        if self.ema is not None and train is True:
            self.ema.update(self)

        loss = self.safe_sum(losses)
        self.log(
            loss_str,
            loss,
            batch_size=x1.shape[0],
            on_epoch=True,
            on_step=False,
            prog_bar=True,
            sync_dist=True,
        )

        if self.ssl_method == "vicregl":
            loss_str_list = self.loss_str_dict["vicregl"]
        elif self.ssl_method == "vicreg":
            loss_str_list = self.loss_str_dict["vicreg"]
        else:
            return loss
        for loss_s, loss_val in zip(loss_str_list, losses):
            if loss_s is not None:
                sub_loss_str = "{}:{}".format(loss_str, loss_s)
            else:
                sub_loss_str = loss_str
            self.log(
                sub_loss_str,
                loss_val,
                batch_size=x1.shape[0],
                on_epoch=True,
                on_step=False,
                prog_bar=True,
                sync_dist=True,
            )
        return loss

    def training_step(self, batch, batch_idx):
        loss = self.step(batch, "loss", self.train_metrics, train=True)
        return loss

    def validation_step(self, batch, batch_idx):
        loss = self.step(batch, "val_loss", self.val_metrics)
        return loss

    def test_step(self, batch, batch_idx):
        loss = self.step(batch, "test_loss", self.test_metrics)
        return loss


class SelfSLUNetPL(UNet, SelfSLBasePL):
    """
    Operates a number of non-contrastive self-supervised learning
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
        aug_image_key_1: str = "aug_image_1",
        aug_image_key_2: str = "aug_image_2",
        box_key_1: str = "box_1",
        box_key_2: str = "box_2",
        learning_rate: float = 0.001,
        batch_size: int = 4,
        weight_decay: float = 0.005,
        training_dataloader_call: Callable = None,
        n_epochs: int = 1000,
        n_steps: int = None,
        warmup_steps: int = 0,
        start_decay: int = None,
        ema: torch.nn.Module = None,
        ssl_method: str = "simclr",
        temperature: float = 1.0,
        vic_reg_loss_params: dict = {},
        stop_gradient: bool = True,
        channels_to_batch: bool = False,
        *args,
        **kwargs,
    ):
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
            n_steps (int, optional): number of steps. Defaults to None. Only
                used if n_epochs is None.
            warmup_steps (int, optional): number of warmup steps. Defaults
                to 0.
            start_decay (int, optional): number of steps after which decay
                begins. Defaults to None (decay starts after warmup).
            ema (torch.nn.Module, optional): exponential moving decay module
                (EMA). Must have an update method that takes model as input
                and updates the weights based on this. Defaults to None.
            ssl_method (str, optional): sets the SSL method. Defaults to
                "simclr".
            temperature (float, optional): temperature for NTXent (when
                simclr == True). Defaults to 1.0.
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
        self.n_steps = n_steps
        self.warmup_steps = warmup_steps
        self.start_decay = start_decay
        self.ema = ema
        self.ssl_method = ssl_method
        self.temperature = temperature
        self.vic_reg_loss_params = vic_reg_loss_params
        self.stop_gradient = stop_gradient
        self.channels_to_batch = channels_to_batch

        if channels_to_batch is True:
            kwargs["in_channels"] = 1

        kwargs["encoder_only"] = True
        super().__init__(*args, **kwargs)

        if all(
            [
                self.ssl_method not in ["vicreg", "vicregl", "simclr"],
                self.stop_gradient is False,
            ]
        ):
            warnings.warn(
                "stop_gradient=False should not (in theory) be used\
            with vic_reg=False, vic_reg_local=False or simclr=False"
            )

        self.init_loss()
        if self.ema is not None:
            self.ema.update(self)
        else:
            self.ema = None

        self.loss_str_dict = {
            "standard": [None],
            "vicreg": ["inv", "var", "cov"],
            "vicregl": ["inv", "var", "cov", "local"],
        }

        self.save_hyperparameters()
        self.setup_metrics()

    def forward_ema_stop_grad(self, x):
        if self.ema is not None:
            op = self.ema.shadow.forward
        else:
            op = self.forward
        if self.stop_gradient is True:
            with torch.no_grad():
                return op(x)
        else:
            return op(x)

    def step(self, batch, loss_str: str, metrics: dict, train=False):
        if self.ssl_method != "vicregl":
            other_args = []
        else:
            box_1 = batch[self.box_key_1]
            box_2 = batch[self.box_key_2]
            other_args = [box_1, box_2]

        x1, x2 = batch[self.aug_image_key_1], batch[self.aug_image_key_2]
        if self.channels_to_batch is True:
            x1 = x1.reshape(-1, 1, *x1.shape[2:])
            x2 = x2.reshape(-1, 1, *x2.shape[2:])
        y1 = self.forward(x1)
        y2 = self.forward_ema_stop_grad(x2)

        losses = self.calculate_loss(y1, y2, *other_args)
        self.update_metrics(y1, y2, metrics)

        # loss is already symmetrised for VICReg and VICRegL
        if self.ssl_method not in ["vicreg", "vicregl", "simclr"]:
            y1_ = self.forward_ema_stop_grad(x1)
            y2_ = self.forward(x2)
            losses = losses + self.calculate_loss(y2_, y1_, *other_args)
            self.update_metrics(y2_, y1_, metrics)

        if self.ema is not None and train is True:
            self.ema.update(self)

        loss = self.safe_sum(losses)
        self.log(
            loss_str,
            loss,
            batch_size=x1.shape[0],
            on_epoch=True,
            on_step=False,
            prog_bar=True,
            sync_dist=True,
        )

        if self.ssl_method == "vicregl":
            loss_str_list = self.loss_str_dict["vicregl"]
        if self.ssl_method == "vicreg":
            loss_str_list = self.loss_str_dict["vicreg"]
        else:
            return loss

        for s, loss_value in zip(loss_str_list, losses):
            if s is not None:
                sub_loss_str = "{}:{}".format(loss_str, s)
            else:
                sub_loss_str = loss_str
            self.log(
                sub_loss_str,
                loss_value,
                batch_size=x1.shape[0],
                on_epoch=True,
                on_step=False,
                prog_bar=True,
                sync_dist=True,
            )

        return loss

    def training_step(self, batch, batch_idx):
        loss = self.step(batch, "loss", self.train_metrics, train=True)
        return loss

    def validation_step(self, batch, batch_idx):
        loss = self.step(batch, "val_loss", self.val_metrics)
        return loss

    def test_step(self, batch, batch_idx):
        loss = self.step(batch, "test_loss", self.test_metrics)
        return loss


class SelfSLConvNeXtPL(ConvNeXt, SelfSLBasePL):
    """
    Operates a number of non-contrastive self-supervised learning
    methods, all of which use non-contrastive approaches to self-supervised
    learning. Included here are:
    * SimSiam (the vanilla version of this class)
    * BYOL (when an ema module is specified - this should be an
    ExponentialMovingAverage module)
    * VICReg (when vic_reg==True)
    * VICRegL (when vic_reg_local==True)

    Uses a ConvNeXt backbone, to enable the transfer of this network to a
    standard ConvNeXt.
    """

    def __init__(
        self,
        aug_image_key_1: str = "aug_image_1",
        aug_image_key_2: str = "aug_image_2",
        box_key_1: str = "box_1",
        box_key_2: str = "box_2",
        learning_rate: float = 0.001,
        batch_size: int = 4,
        weight_decay: float = 0.005,
        training_dataloader_call: Callable = None,
        n_epochs: int = 1000,
        n_steps: int = None,
        warmup_steps: int = 0,
        start_decay: int = None,
        ema: torch.nn.Module = None,
        ssl_method: str = "simclr",
        temperature: float = 1.0,
        vic_reg_loss_params: dict = {},
        stop_gradient: bool = True,
        channels_to_batch: bool = False,
        *args,
        **kwargs,
    ):
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
            n_steps (int, optional): number of steps. Defaults to None. Only
                used if n_epochs is None.
            warmup_steps (int, optional): number of warmup steps. Defaults
                to 0.
            start_decay (int, optional): number of steps after which decay
                begins. Defaults to None (decay starts after warmup).
            ema (float, torch.nn.Module): exponential moving decay module
                (EMA). Must have an update method that takes model as input
                and updates the weights based on this. Defaults to None.
            ssl_method (str, optional): sets the SSL method. Defaults to
                "simclr".
            temperature (float, optional): temperature for NTXent (when
                simclr == True). Defaults to 1.0.
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
        self.n_steps = n_steps
        self.warmup_steps = warmup_steps
        self.start_decay = start_decay
        self.ssl_method = ssl_method
        self.temperature = temperature
        self.vic_reg_loss_params = vic_reg_loss_params
        self.stop_gradient = stop_gradient
        self.channels_to_batch = channels_to_batch

        if channels_to_batch is True:
            kwargs["backbone_args"]["in_channels"] = 1

        super().__init__(*args, **kwargs)

        self.ema = ema

        if all(
            [
                self.ssl_method not in ["vicreg", "vicregl", "simclr"],
                self.stop_gradient is False,
            ]
        ):
            warnings.warn(
                "stop_gradient=False should not (in theory) be used\
            with vic_reg=False, vic_reg_local=False or simclr=False"
            )

        self.init_loss()
        if self.ema is not None:
            self.ema.update(self)
        else:
            self.ema = None

        self.loss_str_dict = {
            "standard": [None],
            "vicreg": ["inv", "var", "cov"],
            "vicregl": ["inv", "var", "cov", "local"],
        }

        self.save_hyperparameters()
        self.setup_metrics()

    def forward_ema_stop_grad(self, x, ret):
        if self.ema is not None:
            op = self.ema.shadow.forward
        else:
            op = self.forward
        if self.stop_gradient is True:
            with torch.no_grad():
                return op(x, ret)
        else:
            return op(x, ret)

    def step(self, batch, loss_str: str, metrics: dict, train=False):
        if self.ssl_method == "simclr":
            ret_string_1 = "projection"
            ret_string_2 = "projection"
            other_args = []
        elif self.ssl_method != "vicregl":
            ret_string_1 = "prediction"
            ret_string_2 = "projection"
            other_args = []
        else:
            ret_string_1 = "representation"
            ret_string_2 = "representation"
            box_1 = batch[self.box_key_1]
            box_2 = batch[self.box_key_2]
            other_args = [box_1, box_2]

        x1, x2 = batch[self.aug_image_key_1], batch[self.aug_image_key_2]

        if self.channels_to_batch is True:
            x1 = x1.reshape(-1, 1, *x1.shape[2:])
            x2 = x2.reshape(-1, 1, *x2.shape[2:])
        y1 = self.forward(x1, ret=ret_string_1)
        y2 = self.forward_ema_stop_grad(x2, ret=ret_string_2)

        losses = self.calculate_loss(y1, y2, *other_args)
        self.update_metrics(y1, y2, metrics)

        # loss is already symmetrised for VICReg, VICRegL and SimCLR
        if self.ssl_method not in ["vicreg", "vicregl", "simclr"]:
            y1_ = self.forward_ema_stop_grad(x1, ret=ret_string_1)
            y2_ = self.forward(x2, ret=ret_string_2)
            losses = losses + self.calculate_loss(y2_, y1_, *other_args)
            self.update_metrics(y2_, y1_, metrics)

        if self.ema is not None and train is True:
            self.ema.update(self)

        loss = self.safe_sum(losses)
        self.log(
            loss_str,
            loss,
            batch_size=x1.shape[0],
            on_epoch=True,
            on_step=False,
            prog_bar=True,
            sync_dist=True,
        )

        if self.ssl_method == "vicregl":
            loss_str_list = self.loss_str_dict["vicregl"]
        elif self.ssl_method == "vicreg":
            loss_str_list = self.loss_str_dict["vicreg"]
        else:
            return loss
        for s, loss_value in zip(loss_str_list, losses):
            if s is not None:
                sub_loss_str = "{}:{}".format(loss_str, s)
            else:
                sub_loss_str = loss_str
            self.log(
                sub_loss_str,
                loss_value,
                batch_size=x1.shape[0],
                on_epoch=True,
                on_step=False,
                prog_bar=True,
                sync_dist=True,
            )
        return loss

    def training_step(self, batch, batch_idx):
        loss = self.step(batch, "loss", self.train_metrics, train=True)
        return loss

    def validation_step(self, batch, batch_idx):
        loss = self.step(batch, "val_loss", self.val_metrics)
        return loss

    def test_step(self, batch, batch_idx):
        loss = self.step(batch, "test_loss", self.test_metrics)
        return loss


class IJEPAPL(IJEPA, SelfSLBasePL):
    """
    LightningModule implementation of the IJEPA architecture.
    """

    def __init__(
        self,
        image_key: str = "image",
        learning_rate: float = 0.001,
        batch_size: int = 4,
        weight_decay: float = 0.005,
        training_dataloader_call: Callable = None,
        n_epochs: int = 1000,
        n_steps: int = None,
        warmup_steps: int = 0,
        start_decay: int = None,
        ema: torch.nn.Module = None,
        ssl_method: str = "simclr",
        temperature: float = 1.0,
        vic_reg_loss_params: dict = {},
        stop_gradient: bool = True,
        channels_to_batch: bool = False,
        *args,
        **kwargs,
    ):
        """
        Args:
            image_key (str, optional): key for image. Defaults to "image".
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
            n_steps (int, optional): number of steps. Defaults to None. Only
                used if n_epochs is None.
            warmup_steps (int, optional): number of warmup steps. Defaults
                to 0.
            start_decay (int, optional): number of steps after which decay
                begins. Defaults to None (decay starts after warmup).
            ema (float, torch.nn.Module): exponential moving decay module
                (EMA). Must have an update method that takes model as input
                and updates the weights based on this. Defaults to None.
            ssl_method (str, optional): sets the SSL method. Defaults to
                "simclr".
            temperature (float, optional): temperature for NTXent (when
                simclr == True). Defaults to 1.0.
            vic_reg_loss_params (dict, optional): parameters for the VICRegLoss
                module. Defaults to {} (the default parameters).
            stop_gradient (bool, optional): stops gradients when calculating
                losses. Useful for VICReg. Defaults to True.
            channels_to_batch (bool, optional): resizes the input such that
                each channel becomes an element of the batch. Defaults to
                False.
        """
        self.image_key = image_key
        self.learning_rate = learning_rate
        self.batch_size = batch_size
        self.weight_decay = weight_decay
        self.training_dataloader_call = training_dataloader_call
        self.n_epochs = n_epochs
        self.n_steps = n_steps
        self.warmup_steps = warmup_steps
        self.start_decay = start_decay
        self.temperature = temperature
        self.vic_reg_loss_params = vic_reg_loss_params
        self.stop_gradient = stop_gradient
        self.channels_to_batch = channels_to_batch

        self.ssl_method = "ijepa"

        if channels_to_batch is True:
            kwargs["backbone_args"]["in_channels"] = 1

        super().__init__(*args, **kwargs)

        self.ema = ema

        self.init_loss()
        if self.ema is not None:
            self.ema.update(self)
        else:
            self.ema = None

        self.save_hyperparameters()
        self.setup_metrics()
        self.init_loss()

    def init_loss(self):
        self.loss = torch.nn.MSELoss()

    def calculate_loss(self, y, patches, *args):
        loss = sum([self.loss(y, patch) for patch in patches]) / len(patches)
        return loss

    def step(self, batch, loss_str: str, train=False):
        x = batch[self.image_key]
        if self.channels_to_batch is True:
            x = x.reshape(-1, 1, *x.shape[2:])
        x, patches = self.forward_training(x, self.ema)
        loss = self.calculate_loss(x, patches)
        if self.ema is not None and train is True:
            self.ema.update(self)
        # loss is rescaled for logging because values are typically too small
        # to be easily tracked from progress bar.
        self.log(
            loss_str,
            loss * 1e3,
            batch_size=x.shape[0],
            on_epoch=True,
            on_step=False,
            prog_bar=True,
            sync_dist=True,
        )
        return loss

    def training_step(self, batch, batch_idx):
        loss = self.step(batch, "loss", train=True)
        return loss

    def validation_step(self, batch, batch_idx):
        loss = self.step(batch, "val_loss")
        return loss

    def test_step(self, batch, batch_idx):
        loss = self.step(batch, "test_loss")
        return loss

    def configure_optimizers(self):
        if self.n_steps is not None:
            interval = "step"
            n = self.n_steps
        else:
            interval = "epoch"
            n = self.n_epochs
        params_no_decay = []
        params_decay = []
        for k, p in self.named_parameters():
            if "normalization" in k:
                params_no_decay.append(p)
            else:
                params_decay.append(p)
        optimizer = torch.optim.AdamW(
            params_decay + params_no_decay,
            lr=self.learning_rate,
            weight_decay=self.weight_decay,
        )
        lr_schedulers = lr_schedulers = CosineAnnealingWithWarmupLR(
            optimizer,
            T_max=n,
            start_decay=self.start_decay,
            n_warmup_steps=self.warmup_steps,
            eta_min=1e-6,
        )

        return {
            "optimizer": optimizer,
            "lr_scheduler": {
                "scheduler": lr_schedulers,
                "interval": interval,
                "frequency": 1,
            },
            "monitor": "val_loss",
        }


class DINOPL(DINO, SelfSLBasePL):
    def __init__(
        self,
        aug_image_key_1: str = "aug_image_1",
        aug_image_key_2: str = "aug_image_2",
        learning_rate: float = 1e-3,
        batch_size: int = 1,
        weight_decay: float = 1e-6,
        training_dataloader_call: Callable = None,
        n_epochs: int = 100,
        n_steps: int = None,
        warmup_steps: int = 0,
        start_decay: int = 0,
        temperature: float = 1.0,
        stop_gradient: bool = True,
        channels_to_batch: bool = False,
        ema: torch.nn.Module = None,
        centers_m: float = 0.9,
        teacher_score_method: str = "center",
        *args,
        **kwargs,
    ):
        super().__init__(*args, **kwargs)
        self.aug_image_key_1 = aug_image_key_1
        self.aug_image_key_2 = aug_image_key_2
        self.learning_rate = learning_rate
        self.batch_size = batch_size
        self.weight_decay = weight_decay
        self.training_dataloader_call = training_dataloader_call
        self.n_epochs = n_epochs
        self.n_steps = n_steps
        self.warmup_steps = warmup_steps
        self.start_decay = start_decay
        self.temperature = temperature
        self.stop_gradient = stop_gradient
        self.channels_to_batch = channels_to_batch
        self.ema = ema
        self.centers_m = centers_m
        self.teacher_score_method = teacher_score_method

        self.ssl_method = "dino"

        if channels_to_batch is True:
            kwargs["in_channels"] = 1

        super().__init__(*args, **kwargs)

        self.save_hyperparameters()
        self.setup_metrics()
        self.init_loss()

        self.ema = ema
        if self.ema is not None:
            self.ema.update(self, exclude_keys=["centers"])
        else:
            self.ema = None

    def init_loss(self):
        self.loss = DinoLoss(
            temperatures=(0.1, 0.1),
            n_features=self.out_dim,
            center_m=self.centers_m,
            teacher_score_method=self.teacher_score_method,
        )

    def calculate_loss(self, y_1, y_2, *args):
        loss = self.loss(y_1, y_2)
        return loss

    def step(
        self, batch, loss_str: str, metrics: dict | None = None, train=False
    ):
        x1, x2 = batch[self.aug_image_key_1], batch[self.aug_image_key_2]
        if self.channels_to_batch is True:
            x1 = x1.reshape(-1, 1, *x1.shape[2:])
            x2 = x2.reshape(-1, 1, *x2.shape[2:])
        stacked_x = torch.cat([x1, x2])
        sh = [x1.shape[0], x2.shape[0]]
        s_1, s_2 = self.forward(stacked_x).split(sh, dim=0)
        with torch.no_grad():
            t_1, t_2 = self.ema(stacked_x).detach().split(sh, dim=0)
        loss_value = torch.add(
            self.calculate_loss(s_1, t_2) / 2.0,
            self.calculate_loss(s_2, t_1) / 2.0,
        )
        if metrics is not None:
            self.update_metrics(
                torch.cat([s_1, s_2]),
                torch.cat([t_1, t_2]),
                metrics,
                log=True,
            )
        self.log(
            loss_str, loss_value, on_step=True, on_epoch=True, prog_bar=True
        )
        if self.ema is not None and train is True:
            self.ema.update(self, exclude_keys=["centers"])
        self.update_centers(torch.cat([t_1, t_2]))
        return loss_value

    def training_step(self, batch, batch_idx):
        loss = self.step(batch, "loss", train=True)
        return loss

    def validation_step(self, batch, batch_idx):
        loss = self.step(batch, "val_loss")
        return loss

    def test_step(self, batch, batch_idx):
        loss = self.step(batch, "test_loss")
        return loss


class iBOTPL(iBOT, SelfSLBasePL):
    def __init__(
        self,
        aug_image_key_1: str = "aug_image_1",
        aug_image_key_2: str = "aug_image_2",
        learning_rate: float = 1e-3,
        batch_size: int = 1,
        weight_decay: float = 1e-6,
        training_dataloader_call: Callable = None,
        n_epochs: int = 100,
        n_steps: int = None,
        warmup_steps: int = 0,
        start_decay: int = 0,
        temperature: float = 1.0,
        stop_gradient: bool = True,
        channels_to_batch: bool = False,
        ema: torch.nn.Module = None,
        centers_m: float = 0.9,
        teacher_score_method: str = "center",
        *args,
        **kwargs,
    ):
        super().__init__(*args, **kwargs)
        self.aug_image_key_1 = aug_image_key_1
        self.aug_image_key_2 = aug_image_key_2
        self.learning_rate = learning_rate
        self.batch_size = batch_size
        self.weight_decay = weight_decay
        self.training_dataloader_call = training_dataloader_call
        self.n_epochs = n_epochs
        self.n_steps = n_steps
        self.warmup_steps = warmup_steps
        self.start_decay = start_decay
        self.temperature = temperature
        self.stop_gradient = stop_gradient
        self.channels_to_batch = channels_to_batch
        self.ema = ema
        self.centers_m = centers_m
        self.teacher_score_method = teacher_score_method

        self.ssl_method = "dino"

        if channels_to_batch is True:
            kwargs["in_channels"] = 1

        super().__init__(*args, **kwargs)

        # self.save_hyperparameters()
        self.setup_metrics()
        self.init_loss()

        self.ema = ema
        if self.ema is not None:
            self.ema.update(self, exclude_keys=["centers"])
        else:
            self.ema = None

    def init_loss(self):
        self.loss_mask = DinoLoss(
            temperatures=(0.1, 0.1),
            n_features=self.out_dim,
            center_m=self.centers_m,
            teacher_score_method=self.teacher_score_method,
        )
        self.loss_global = DinoLoss(
            temperatures=(0.1, 0.1),
            n_features=self.out_dim,
            center_m=self.centers_m,
            teacher_score_method=self.teacher_score_method,
        )

    def calculate_loss_global(
        self, y_red_1: torch.Tensor, y_red_2: torch.Tensor
    ):
        return self.loss_global(a=y_red_1, b=y_red_2.detach())

    def calculate_loss_mask(
        self, a: torch.Tensor, b: torch.Tensor, m: list[int]
    ):
        return self.loss_mask(a[:, m], b[:, m].detach())

    def step(
        self, batch, loss_str: str, metrics: dict | None = None, train=False
    ):
        x1, x2 = batch[self.aug_image_key_1], batch[self.aug_image_key_2]
        if self.channels_to_batch is True:
            x1 = x1.reshape(-1, 1, *x1.shape[2:])
            x2 = x2.reshape(-1, 1, *x2.shape[2:])
        s_red_1, s_1, mc_1 = self.forward_training(x1, mask=True)
        s_red_2, s_2, mc_2 = self.forward_training(x2, mask=True)
        with torch.no_grad():
            t_red_1, t_1 = self.ema.shadow.forward_training(x1)
            t_red_2, t_2 = self.ema.shadow.forward_training(x2)

        # update centers
        self.loss_global.update_centers(torch.cat([t_red_1, t_red_2]))
        self.loss_mask.update_centers(torch.cat([t_1, t_2]))

        loss_global = torch.add(
            self.calculate_loss_global(s_red_1, t_red_2) / 2.0,
            self.calculate_loss_global(s_red_2, t_red_1) / 2.0,
        )
        loss_mask = torch.add(
            self.calculate_loss_mask(s_1, t_1, mc_1) / 2.0,
            self.calculate_loss_mask(s_2, t_2, mc_2) / 2.0,
        )
        if metrics is not None:
            self.update_metrics(
                torch.cat([s_red_1, s_red_2]),
                torch.cat([t_red_1, t_red_2]),
                metrics,
                log=True,
            )
        self.log(
            loss_str + "_global",
            loss_global,
            on_step=True,
            on_epoch=True,
            prog_bar=True,
        )
        self.log(
            loss_str + "_mask",
            loss_mask,
            on_step=True,
            on_epoch=True,
            prog_bar=True,
        )
        if self.ema is not None and train is True:
            self.ema.update(self, exclude_keys=["centers"])

        return loss_mask + loss_global

    def training_step(self, batch, batch_idx):
        loss = self.step(batch, "loss", train=True)
        return loss

    def validation_step(self, batch, batch_idx):
        loss = self.step(batch, "val_loss")
        return loss

    def test_step(self, batch, batch_idx):
        loss = self.step(batch, "test_loss")
        return loss


class ViTMaskedAutoEncoderPL(pl.LightningModule):
    """
    LightningModule for training ViTMaskedAutoEncoder.
    """

    def __init__(
        self,
        image_key: str,
        image_size: tuple[int, int],
        patch_size: tuple[int, int],
        in_channels: int,
        input_dim_size: int,
        encoder_args: dict[str, Any],
        decoder_args: dict[str, Any],
        n_epochs: int = 100,
        n_steps: int | None = None,
        embed_method: str = "linear",
        dropout_rate: float = 0.0,
        mask_fraction: float = 0.75,
        learning_rate: float = 1e-3,
        batch_size: int = 4,
        weight_decay: float = 1e-6,
        warmup_steps: int = 0,
        start_decay: int = 0,
        training_dataloader_call: Callable | None = None,
    ):
        """
        Args:
            image_key (str): Key for the image in the batch
            image_size (tuple[int, int]): Size of input images (height, width)
            patch_size (tuple[int, int]): Size of patches (ph, pw)
            in_channels (int): Number of input channels
            input_dim_size (int): Dimension of the input embeddings
            encoder_args (dict[str, Any]): Arguments for the encoder
            decoder_args (dict[str, Any]): Arguments for the decoder
            n_epochs (int): Number of epochs. Defaults to 100.
            n_steps (int | None): Number of steps. Defaults to None.
            embed_method (str): Embedding method. Defaults to "linear".
            dropout_rate (float): Dropout rate. Defaults to 0.0.
            mask_fraction (float): Fraction of patches to mask. Defaults to 0.75.
            learning_rate (float): Learning rate. Defaults to 1e-3.
            batch_size (int): Batch size. Defaults to 4.
            weight_decay (float): Weight decay. Defaults to 1e-6.
            warmup_steps (int): Number of warmup steps. Defaults to 0.
            start_decay (int): Number of steps before decay. Defaults to 0.
            training_dataloader_call (Callable | None): Function to get training dataloader. Defaults to None.
        """
        super().__init__()
        self.save_hyperparameters()

        self.model = ViTMaskedAutoEncoder(
            image_size=image_size,
            patch_size=patch_size,
            in_channels=in_channels,
            input_dim_size=input_dim_size,
            encoder_args=encoder_args,
            decoder_args=decoder_args,
            embed_method=embed_method,
            dropout_rate=dropout_rate,
            mask_fraction=mask_fraction,
        )

        # Loss function
        self.criterion = torch.nn.MSELoss()

        # Training parameters
        self.image_key = image_key
        self.batch_size = batch_size
        self.learning_rate = learning_rate
        self.weight_decay = weight_decay
        self.warmup_steps = warmup_steps
        self.training_dataloader_call = training_dataloader_call
        self.n_steps = n_steps
        self.n_epochs = n_epochs
        self.start_decay = start_decay

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """
Forward pass through the model."""
        return self.model(x)

    def training_step(
        self, batch: torch.Tensor, batch_idx: int
    ) -> torch.Tensor:
        """
Training step."""
        x = batch[self.image_key]
        x_recon, mask = self(x)

        # Calculate reconstruction loss
        loss = self.criterion(x_recon, x)

        # Log metrics
        self.log("train_loss", loss, on_step=True, on_epoch=True, prog_bar=True)

        return loss

    def validation_step(
        self, batch: torch.Tensor, batch_idx: int
    ) -> torch.Tensor:
        """
Validation step."""
        x = batch[self.image_key]
        x_recon, _ = self(x)

        loss = self.criterion(x_recon, x)

        # Log metrics
        self.log("val_loss", loss, on_step=False, on_epoch=True, prog_bar=True)

        return loss

    def test_step(self, batch: torch.Tensor, batch_idx: int) -> torch.Tensor:
        """
Test step."""
        x = batch[self.image_key]
        x_recon, _ = self(x)

        loss = self.criterion(x_recon, x)

        # Log metrics
        self.log("test_loss", loss, on_step=False, on_epoch=True, prog_bar=True)

        return loss

    def configure_optimizers(self) -> torch.optim.Optimizer | dict:
        """
Configure optimizers and learning rate schedulers."""
        if self.n_steps:
            n = self.n_steps
            interval = "step"
        else:
            n = self.n_epochs
            interval = "epoch"
        # Create optimizer with initial learning rate of 0.0 if using warmup
        initial_lr = 0.0 if self.warmup_steps > 0 else self.learning_rate

        optimizer = torch.optim.AdamW(
            self.parameters(),
            lr=initial_lr,  # Will be updated by the scheduler
            weight_decay=self.weight_decay,
            betas=(0.9, 0.95),
        )

        if self.warmup_steps > 0:
            # Define the learning rate schedule with warmup
            scheduler = {
                "scheduler": CosineAnnealingWithWarmupLR(
                    optimizer,
                    T_max=n,
                    start_decay=self.start_decay,
                    n_warmup_steps=self.warmup_steps,
                    eta_min=1e-6,
                ),
                "interval": interval,
                "frequency": 1,
            }

            # Set the base learning rate that will be scaled by the scheduler
            scheduler["scheduler"].base_lrs = [self.learning_rate]

            return {"optimizer": optimizer, "lr_scheduler": scheduler}

        return optimizer

    def train_dataloader(self) -> torch.utils.data.DataLoader:
        """
Get training dataloader."""
        if self.training_dataloader_call is None:
            raise ValueError(
                "training_dataloader_call must be provided for training"
            )
        return self.training_dataloader_call(self.batch_size)
