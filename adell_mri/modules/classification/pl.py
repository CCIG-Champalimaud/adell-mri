"""
PyTorch Lightning modules and utilities for classification tasks.

This module contains PyTorch Lightning implementations of various classification models
and related utilities. It includes support for:

- Binary and multi-class classification
- Ordinal classification
- Multiple instance learning
- Ensemble models
- Vision Transformer (ViT) based classifiers
- Hybrid CNN-Transformer architectures
- Deconfounded classification
- Conformal prediction

The module provides implementations of metrics, loss functions, and training loops
optimized for medical image classification tasks.
"""

import gc
from abc import ABC
from typing import Callable

import lightning.pytorch as pl
import numpy as np
import torch
import torch.nn.functional as F
import torchmetrics
import torchmetrics.classification as tmc
from tqdm import tqdm

from ..conformal_prediction import AdaptivePredictionSets
from ..learning_rate import CosineAnnealingWithWarmupLR
from .classification import (
    VGG,
    AveragingEnsemble,
    CatNet,
    FactorizedViTClassifier,
    GenericEnsemble,
    HybridClassifier,
    MultipleInstanceClassifier,
    OrdNet,
    SegCatNet,
    TransformableTransformer,
    UNetEncoder,
    ViTClassifier,
    ordinal_prediction_to_class,
)
from .classification.deconfounded_classification import DeconfoundedNetGeneric

try:
    import monai

    has_monai = True
except ModuleNotFoundError:
    has_monai = False


def f1(prediction: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
    """
    Implementation of the sbinary F1 score for torch tensors.

    Args:
        prediction (torch.Tensor): prediction tensor.
        y (torch.Tensor): ground truth tensor.

    Returns:
        torch.Tensor: F1-score.
    """
    prediction = prediction.detach() > 0.5
    tp = torch.logical_and(prediction == y, y == 1).sum().float()
    tn = torch.logical_and(prediction == y, y == 0).sum().float()
    fp = torch.logical_and(prediction != y, y == 0).sum().float()
    fn = torch.logical_and(prediction != y, y == 1).sum().float()
    tp, tn, fp, fn = [float(x) for x in [tp, tn, fp, fn]]
    n = tp
    d = tp + 0.5 * (fp + fn)
    if d > 0:
        return n / d
    else:
        return 0


def get_metric_dict(
    nc: int,
    metric_keys: list[str] = None,
    prefix: str = "",
    average: str = "macro",
) -> dict[str, torchmetrics.Metric]:
    """
    Constructs a metric dictionary.

    Args:
        nc (int): number of classes.
        metric_keys (list[str], optional): keys corresponding to metrics.
            Should be one of ["Rec","Spe","Pr","F1","AUC"]. Defaults to
            None (all keys).
        prefix (str, optional): which prefix should be added to the metric
            key on the output dict. Defaults to "".
        average (str, optional): how to average values across classes. Defaults
            to "macro".

    Returns:
        dict[str,torchmetrics.Metric]: dictionary containing the metrics
            specified in metric_keys.
    """
    metric_dict = torch.nn.ModuleDict({})
    if nc == 2:
        md = {
            "Rec": lambda: tmc.BinaryRecall(),
            "Spe": lambda: tmc.BinarySpecificity(),
            "Pr": lambda: tmc.BinaryPrecision(),
            "F1": lambda: tmc.BinaryFBetaScore(1.0),
            "AUC": lambda: torchmetrics.AUROC("binary"),
            "CalErr": lambda: torchmetrics.CalibrationError(task="binary"),
        }
    else:
        md = {
            "Rec": lambda: torchmetrics.Recall(
                task="multiclass", num_classes=nc, average=average
            ),
            "Spe": lambda: torchmetrics.Specificity(
                task="multiclass", num_classes=nc, average=average
            ),
            "Pr": lambda: torchmetrics.Precision(
                task="multiclass", num_classes=nc, average=average
            ),
            "F1": lambda: torchmetrics.FBetaScore(
                task="multiclass", num_classes=nc, average=average
            ),
            "AUC": lambda: torchmetrics.AUROC(
                task="multiclass", num_classes=nc
            ),
            "CalErr": lambda: torchmetrics.CalibrationError(
                task="multiclass", num_classes=nc
            ),
        }
    if metric_keys is None:
        metric_keys = list(md.keys())
    for k in metric_keys:
        if k in md:
            metric_dict[prefix + k] = md[k]()
    return metric_dict


def meta_tensors_to_tensors(batch):
    """
    Converts any MetaTensor instances in a batch to regular PyTorch tensors.

    Args:
        batch (dict): A dictionary containing tensors, where some values may be
            MONAI MetaTensor instances.

    Returns:
        dict: The input batch with all MetaTensor instances converted to regular
            PyTorch tensors.
    """
    if has_monai is True:
        for key in batch:
            if isinstance(batch[key], monai.data.MetaTensor):
                batch[key] = batch[key].as_tensor()
    return batch


class ClassPLABC(pl.LightningModule, ABC):
    """
    Abstract classification class for LightningModules.
    """

    def __init__(self):
        super().__init__()

        self.raise_nan_loss = False
        self.calibrated = False

    def calculate_loss(self, prediction, y, with_params=False):
        """Calculates loss between prediction and ground truth.

        Args:
            prediction (torch.Tensor): Model predictions
            y (torch.Tensor): Ground truth labels
            with_params (bool, optional): Whether to use loss parameters. Defaults to False.

        Returns:
            torch.Tensor: Mean loss value
        """
        y = y.to(prediction.device)
        if self.n_classes > 2:
            if len(y.shape) > 1:
                y = y.squeeze(1)
            y = y.to(torch.int64)
        else:
            y = y.float()
        if with_params is True:
            d = y.device
            params = {k: self.loss_params[k].to(d) for k in self.loss_params}
            loss = self.loss_fn(prediction, y, **params)
        else:
            loss = self.loss_fn(prediction, y)
        return loss.mean()

    def on_before_batch_transfer(self, batch, dataloader_idx):
        """Converts any MetaTensors to regular tensors before batch transfer.

        Args:
            batch: Input batch
            dataloader_idx: Index of dataloader

        Returns:
            dict: Batch with MetaTensors converted to regular tensors
        """
        return meta_tensors_to_tensors(batch)

    def training_step(self, batch, batch_idx):
        """Performs a single training step.

        Args:
            batch: Input batch containing images and labels
            batch_idx: Index of current batch

        Returns:
            torch.Tensor: Training loss
        """
        x, y = batch[self.image_key], batch[self.label_key]
        if hasattr(self, "training_batch_preproc"):
            if self.training_batch_preproc is not None:
                x, y = self.training_batch_preproc(x, y)
        prediction = self.forward(x)
        prediction = torch.squeeze(prediction, 1)

        loss = self.calculate_loss(prediction, y, with_params=True)

        self.log("train_loss", loss, sync_dist=True)
        return loss

    def validation_step(self, batch, batch_idx):
        """Performs a single validation step.

        Args:
            batch: Input batch containing images and labels
            batch_idx: Index of current batch

        Returns:
            torch.Tensor: Validation loss
        """
        x, y = batch[self.image_key], batch[self.label_key]
        prediction = self.forward(x)
        prediction = torch.squeeze(prediction, 1)

        loss = self.calculate_loss(prediction, y, with_params=True)
        self.log(
            "val_loss",
            loss,
            on_epoch=True,
            on_step=False,
            prog_bar=True,
            batch_size=x.shape[0],
            sync_dist=True,
        )
        self.update_metrics(prediction, y, self.val_metrics)
        return loss

    def test_step(self, batch, batch_idx):
        """
        Performs a single test step.

        Args:
            batch: Input batch containing images and labels
            batch_idx: Index of current batch

        Returns:
            tuple: Test loss and predictions
        """
        x, y = batch[self.image_key], batch[self.label_key]
        prediction = self.forward(x)
        prediction = torch.squeeze(prediction, 1)

        loss = self.calculate_loss(prediction, y)
        self.log(
            "test_loss",
            loss,
            on_epoch=True,
            on_step=False,
            prog_bar=True,
            batch_size=x.shape[0],
            sync_dist=True,
        )
        self.update_metrics(prediction, y, self.test_metrics, log=False)
        return loss, prediction

    def on_train_start(self):
        """
        Called when training begins. Logs hyperparameters.
        """
        print("Training with the following hyperparameters:")
        parameter_dict = {}
        for k, v in self.hparams.items():
            print(f"\t{k}: {v}")
            if isinstance(v, (list, tuple, torch.Tensor, np.ndarray)):
                if len(v) > 1:
                    for i in range(len(v)):
                        parameter_dict[f"{k}_{i}"] = v[i]
                else:
                    if len(v) > 0:
                        parameter_dict[k] = v[0]
            elif isinstance(v, dict):
                for kk, vv in v.items():
                    parameter_dict[f"{k}_{kk}"] = float(vv)
            elif isinstance(v, (int, float, bool)):
                parameter_dict[k] = v
        parameter_dict = {
            k: float(parameter_dict[k])
            for k in parameter_dict
            if isinstance(parameter_dict[k], (int, float, bool))
        }
        self.log_dict(parameter_dict, sync_dist=True)

    def on_train_epoch_end(self):
        """
        Called at end of training epoch. Updates learning rate.
        """
        sch = self.lr_schedulers().state_dict()
        lr = self.learning_rate
        last_lr = sch["_last_lr"][0] if "_last_lr" in sch else lr
        self.log("lr", last_lr, sync_dist=True, prog_bar=True)
        gc.collect()

    def on_fit_end(self):
        """
        Called when fitting ends. Updates Gaussian process if enabled.
        """
        if hasattr(self, "gaussian_process"):
            if self.gaussian_process is True:
                with tqdm(self.training_dataloader_call()) as pbar:
                    pbar.set_description("Fitting GP covariance")
                    for batch in pbar:
                        x, y = batch[self.image_key], batch[self.label_key]
                        self.gaussian_process_head.update_inv_cov(x, y)
                    self.cov = torch.linalg.inv(self.inv_conv)

    def calibrate(self, dataloader):
        """
        Calibrates model predictions using adaptive prediction sets.

        Args:
            dataloader: DataLoader containing calibration data
        """
        self.calibration = AdaptivePredictionSets(0.2)
        with tqdm(dataloader) as pbar:
            pbar.set_description("Calibrating adaptive prediction sets")
            for batch in pbar:
                x, y = batch[self.image_key], batch[self.label_key]
                prediction = self.forward(x)
                if self.n_classes == 2:
                    prediction = F.sigmoid(prediction)
                else:
                    prediction = F.softmax(prediction, -1)
                self.calibration.update(y, prediction)
            self.calibration.calculate()
        self.calibrated = True

    def on_test_epoch_end(self):
        """
        Called at end of test epoch. Logs test metrics.
        """
        self.log_metrics_end_epoch(self.test_metrics)

    def predict_step(self, batch, batch_idx, *args, **kwargs):
        """
        Performs prediction on a batch.

        Args:
            batch: Input batch containing images
            batch_idx: Index of current batch
            *args: Additional positional arguments
            **kwargs: Additional keyword arguments

        Returns:
            torch.Tensor: Model predictions
        """
        x = batch[self.image_key]
        prediction = self.forward(x, *args, **kwargs)
        return prediction

    def predict_calibrated_step(self, batch, batch_idx, *args, **kwargs):
        """
        Performs prediction on a batch using the calibrated model. Model must
        be calibrated.

        Args:
            batch: Input batch containing images
            batch_idx: Index of current batch
            *args: Additional positional arguments
            **kwargs: Additional keyword arguments

        Returns:
            torch.Tensor: Model predictions
        """
        prediction = self.predict_step(batch)
        if self.calibrated is True:
            prediction = self.calibration(prediction)
        else:
            raise RuntimeError(
                "Model needs to be calibrated before calibrated prediction"
            )
        return prediction

    def train_dataloader(self) -> torch.utils.data.DataLoader:
        return self.training_dataloader_call()

    def configure_optimizers(self):
        """
        Configures the optimizer for the model. If weight decay is a list,
        decouples body and head weight decay.

        Returns:
            dict: A dictionary containing the optimizer, learning rate scheduler,
                and the name of the metric to monitor for early stopping.
        """
        if isinstance(self.weight_decay, (list, tuple)):
            # decouples body and head weight decay
            wd_body, wd_head = self.weight_decay
            params_head = [
                p for (n, p) in self.named_parameters() if "classification" in n
            ]
            params_body = [
                p
                for (n, p) in self.named_parameters()
                if "classification" not in n
            ]
            parameters = [
                {"params": params_head, "weight_decay": wd_body},
                {"params": params_body, "weight_decay": wd_head},
            ]
            wd = wd_body
        else:
            parameters = self.parameters()
            wd = self.weight_decay
        optimizer = torch.optim.AdamW(
            parameters, lr=self.learning_rate, weight_decay=wd
        )
        lr_schedulers = CosineAnnealingWithWarmupLR(
            optimizer,
            T_max=self.n_epochs,
            start_decay=self.start_decay,
            n_warmup_steps=self.warmup_steps,
        )

        return {
            "optimizer": optimizer,
            "lr_scheduler": lr_schedulers,
            "monitor": "val_loss",
        }

    def setup_metrics(self):
        """
        Sets up the metrics for the model.
        """
        self.train_metrics = get_metric_dict(self.n_classes, [], prefix="")
        self.val_metrics = get_metric_dict(self.n_classes, None, prefix="V_")
        self.test_metrics = get_metric_dict(
            self.n_classes, None, prefix="T_", average="none"
        )

    def update_metrics(self, prediction, y, metrics, log=True):
        """
        Updates the metrics for the model based on the prediction and ground
        truth labels.

        Args:
            prediction (torch.Tensor): The model's prediction output.
            y (torch.Tensor): The ground truth labels.
            metrics (dict): A dictionary of metric functions to be updated.
            log (bool, optional): Whether to log the metrics. Defaults to True.

        Returns:
            None
        """
        if self.n_classes > 2:
            prediction = torch.softmax(prediction, 1)
        else:
            prediction = torch.sigmoid(prediction)
        if len(y.shape) > 1:
            y.squeeze(1)
        for k in metrics:
            metrics[k](prediction, y)
            if log is True:
                self.log(
                    k,
                    metrics[k],
                    on_epoch=True,
                    on_step=False,
                    prog_bar=True,
                    sync_dist=True,
                )

    def log_metrics_end_epoch(self, metrics):
        """
        Logs the metrics at the end of each epoch.

        Args:
            metrics (dict): A dictionary of metric functions to be logged.

        Returns:
            None
        """
        for k in metrics:
            metric = metrics[k].compute()
            if len(metric.shape) == 0:
                metric = metric.reshape(1)
            if len(metric) > 1:
                for i in range(len(metric)):
                    self.log(
                        f"{k}_{i}",
                        metric[i],
                        on_epoch=True,
                        on_step=False,
                        prog_bar=True,
                        sync_dist=True,
                    )
            else:
                self.log(
                    k,
                    metric,
                    on_epoch=True,
                    on_step=False,
                    prog_bar=True,
                    sync_dist=True,
                )


class ClassNetPL(ClassPLABC):
    """
    Classification network implementation for Pytorch Lightning. Can be
    parametrised as a categorical or ordinal network, depending on the
    specification in net_type.
    """

    def __init__(
        self,
        net_type: str = "cat",
        image_key: str = "image",
        label_key: str = "label",
        learning_rate: float = 0.001,
        batch_size: int = 4,
        weight_decay: float = 0.0,
        training_dataloader_call: Callable = None,
        loss_fn: Callable = F.binary_cross_entropy,
        loss_params: dict = {},
        n_epochs: int = 100,
        warmup_steps: int = 0,
        start_decay: int = None,
        training_batch_preproc: Callable = None,
        *args,
        **kwargs,
    ) -> torch.nn.Module:
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

        self.save_hyperparameters(
            ignore=["training_dataloader_call", "training_batch_preproc"]
        )
        self.setup_network()
        self.setup_metrics()

        if hasattr(self.network, "forward_features"):
            self.forward_features = self.network.forward_features

    def setup_network(self):
        if self.net_type == "cat":
            self.network = CatNet(*self.args, **self.kwargs)
        elif self.net_type == "ord":
            self.network = OrdNet(*self.args, **self.kwargs)
        elif self.net_type == "vgg":
            self.network = VGG(*self.args, **self.kwargs)
        else:
            raise Exception(
                "net_type '{}' not valid, has to be one of \
                ['ord', 'cat', 'vgg']".format(
                    self.net_type
                )
            )
        self.forward = self.network.forward
        self.n_classes = self.network.n_classes

    def update_metrics(self, prediction, y, metrics, log=True):
        """
        Update the metrics for the given batch.

        Args:
            prediction (torch.Tensor): The predicted values.
            y (torch.Tensor): The true labels.
            metrics (dict): A dictionary of metrics to update.
            log (bool, optional): Whether to log the metrics. Defaults to True.
        """
        if self.net_type == "ord":
            prediction = ordinal_prediction_to_class(prediction)
        elif self.n_classes > 2:
            prediction = torch.softmax(prediction, 1)
        else:
            prediction = torch.sigmoid(prediction)
        if len(y.shape) > 1:
            y = y.squeeze(1)
        for k in metrics:
            metrics[k](prediction, y)
            if log is True:
                self.log(
                    k,
                    metrics[k],
                    on_epoch=True,
                    on_step=False,
                    prog_bar=True,
                    sync_dist=True,
                )


class SegCatNetPL(SegCatNet, pl.LightningModule):
    """
    PL module for SegCatNet.
    """

    def __init__(
        self,
        image_key: str = "image",
        label_key: str = "label",
        skip_conditioning_key: str = None,
        feature_conditioning_key: str = None,
        learning_rate: float = 0.001,
        batch_size: int = 4,
        weight_decay: float = 0.0,
        training_dataloader_call: Callable = None,
        loss_fn: Callable = F.binary_cross_entropy_with_logits,
        loss_params: dict = {},
        n_epochs: int = 100,
        *args,
        **kwargs,
    ):
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

        super().__init__(*args, **kwargs)

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

    def calculate_loss(self, prediction, y):
        y = y.type(torch.float32)
        if len(y.shape) > 1:
            y = y.squeeze(1)
        prediction = prediction.type(torch.float32)
        if len(prediction.shape) > 1:
            prediction = prediction.squeeze(1)
        if "weight" in self.loss_params:
            weights = torch.ones_like(y)
            if len(self.loss_params["weight"]) == 1:
                weights[y == 1] = self.loss_params["weight"]
            else:
                weights = self.loss_params["weight"][y]
            loss_params = {"weight": weights}
        else:
            loss_params = {}
        loss = self.loss_fn(prediction, y, **loss_params)
        return loss.mean()

    def update_metrics(self, metrics, pred, y, **kwargs):
        y = y.long()
        if self.n_classes == 2:
            pred = torch.sigmoid(pred)
        else:
            pred = F.softmax(pred, -1)
        for k in metrics:
            metrics[k](pred, y)
            self.log(k, metrics[k], **kwargs)

    def loss_wrapper(self, x, y, x_cond, x_fc):
        try:
            y = torch.round(y)
        except Exception:
            y = torch.round(y.float())
        prediction = self.forward(
            x, X_skip_layer=x_cond, X_feature_conditioning=x_fc
        )
        prediction = torch.squeeze(prediction, 1)
        if len(y.shape) > 1:
            y = torch.squeeze(y, 1)
        batch_size = int(prediction.shape[0])
        if batch_size == 1:
            y = torch.unsqueeze(y, 0)

        loss = self.calculate_loss(prediction, y)
        return prediction, loss

    def training_step(self, batch, batch_idx):
        x, y = batch[self.image_key], batch[self.label_key]
        if self.skip_conditioning_key is not None:
            x_cond = batch[self.skip_conditioning_key]
        else:
            x_cond = None
        if self.feature_conditioning_key is not None:
            x_fc = batch[self.feature_conditioning_key]
        else:
            x_fc = None

        pred_final, loss = self.loss_wrapper(x, y, x_cond, x_fc)

        self.log("train_loss", loss)
        return loss

    def validation_step(self, batch, batch_idx):
        x, y = batch[self.image_key], batch[self.label_key]
        if self.skip_conditioning_key is not None:
            x_cond = batch[self.skip_conditioning_key]
        else:
            x_cond = None
        if self.feature_conditioning_key is not None:
            x_fc = batch[self.feature_conditioning_key]
        else:
            x_fc = None

        pred_final, loss = self.loss_wrapper(x, y, x_cond, x_fc)

        try:
            y = torch.round(y).int()
        except Exception:
            pass
        self.update_metrics(
            self.val_metrics, pred_final, y, on_epoch=True, prog_bar=True
        )
        return loss

    def test_step(self, batch, batch_idx):
        x, y = batch[self.image_key], batch[self.label_key]
        if self.skip_conditioning_key is not None:
            x_cond = batch[self.skip_conditioning_key]
        else:
            x_cond = None
        if self.feature_conditioning_key is not None:
            x_fc = batch[self.feature_conditioning_key]
        else:
            x_fc = None

        pred_final, loss = self.loss_wrapper(x, y, x_cond, x_fc)

        try:
            y = torch.round(y).int()
        except Exception:
            pass
        self.update_metrics(
            self.test_metrics,
            pred_final,
            y,
            on_epoch=True,
            on_step=False,
            prog_bar=True,
        )
        return loss

    def setup_metrics(self):
        if self.n_classes == 2:
            C_1, C_2, A, M, ign_idx = 2, None, None, "micro", None
        else:
            c = self.n_classes
            C_1, C_2, A, M, ign_idx = [c, c, "samplewise", "macro", None]
        self.train_metrics = torch.nn.ModuleDict({})
        self.val_metrics = torch.nn.ModuleDict({})
        self.test_metrics = torch.nn.ModuleDict({})
        md = {
            "Pr": torchmetrics.Precision,
            "F1": torchmetrics.FBetaScore,
            "Re": torchmetrics.Recall,
            "AUC": torchmetrics.AUROC,
        }
        for k in md:
            if k == "IoU":
                m, C = "macro", C_1
            else:
                m, C = M, C_2

            if k in ["F1"]:
                self.train_metrics[k] = md[k](
                    num_classes=C,
                    mdmc_average=A,
                    average=m,
                    ignore_index=ign_idx,
                ).to(self.device)
                self.val_metrics["V_" + k] = md[k](
                    num_classes=C,
                    mdmc_average=A,
                    average=m,
                    ignore_index=ign_idx,
                ).to(self.device)
            self.test_metrics["T_" + k] = md[k](
                num_classes=C,
                mdmc_average=A,
                average=m,
                ignore_index=ign_idx,
            ).to(self.device)


class UNetEncoderPL(UNetEncoder, ClassPLABC):
    """
    U-Net encoder-based classification network implementation for Pytorch
    Lightning.
    """

    def __init__(
        self,
        image_key: str = "image",
        label_key: str = "label",
        learning_rate: float = 0.001,
        batch_size: int = 4,
        weight_decay: float = 0.0,
        training_dataloader_call: Callable = None,
        loss_fn: Callable = F.binary_cross_entropy,
        loss_params: dict = {},
        n_epochs: int = 100,
        warmup_steps: int = 0,
        start_decay: int = None,
        training_batch_preproc: Callable = None,
        *args,
        **kwargs,
    ) -> torch.nn.Module:
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

        super().__init__(*args, **kwargs)

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

        self.save_hyperparameters(
            ignore=["training_dataloader_call", "training_batch_preproc"]
        )
        self.setup_metrics()


class GenericEnsemblePL(GenericEnsemble, ClassPLABC):
    """
    Ensemble classification network for PL.
    """

    def __init__(
        self,
        image_keys: list[str] = ["image"],
        label_key: str = "label",
        learning_rate: float = 0.001,
        batch_size: int = 4,
        weight_decay: float = 0.0,
        training_dataloader_call: Callable = None,
        loss_fn: Callable = F.binary_cross_entropy,
        loss_params: dict = {},
        n_epochs: int = 100,
        warmup_steps: int = 0,
        start_decay: int = None,
        *args,
        **kwargs,
    ) -> torch.nn.Module:
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
            warmup_steps (int, optional): number of warmup steps. Defaults
                to 0.
            start_decay (int, optional): number of steps after which decay
                begins. Defaults to None (decay starts after warmup).
            args: arguments for classification network class.
            kwargs: keyword arguments for classification network class.

        Returns:
            pl.LightningModule: a classification network module.
        """

        super().__init__(*args, **kwargs)

        self.image_keys = image_keys
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
        self.args = args
        self.kwargs = kwargs

        self.save_hyperparameters(
            ignore=["training_dataloader_call", "training_batch_preproc"]
        )
        self.setup_metrics()

    def training_step(self, batch, batch_idx):
        x, y = [batch[k] for k in self.image_keys], batch[self.label_key]
        prediction = self.forward(x)
        prediction = torch.squeeze(prediction, 1)

        loss = self.calculate_loss(prediction, y)

        self.log("train_loss", loss)
        return loss

    def validation_step(self, batch, batch_idx):
        x, y = [batch[k] for k in self.image_keys], batch[self.label_key]
        prediction = self.forward(x)
        prediction = torch.squeeze(prediction, 1)

        loss = self.calculate_loss(prediction, y)
        self.log(
            "val_loss",
            loss,
            on_epoch=True,
            on_step=False,
            prog_bar=True,
            batch_size=x[0].shape[0],
        )
        self.update_metrics(prediction, y, self.val_metrics)
        return loss

    def test_step(self, batch, batch_idx):
        x, y = [batch[k] for k in self.image_keys], batch[self.label_key]
        prediction = self.forward(x)
        prediction = torch.squeeze(prediction, 1)

        loss = self.calculate_loss(prediction, y)

        self.update_metrics(prediction, y, self.test_metrics, log=False)
        return loss

    def train_dataloader(self) -> torch.utils.data.DataLoader:
        return self.training_dataloader_call()

    def on_train_epoch_end(self):
        sch = self.lr_schedulers().state_dict()
        lr = self.learning_rate
        last_lr = sch["_last_lr"][0] if "_last_lr" in sch else lr
        self.log("lr", last_lr)


class AveragingEnsemblePL(AveragingEnsemble, ClassPLABC):
    """
    Ensemble average classification network for PL.
    """

    def __init__(
        self,
        image_keys: list[str] = ["image"],
        label_key: str = "label",
        learning_rate: float = 0.001,
        batch_size: int = 4,
        weight_decay: float = 0.0,
        training_dataloader_call: Callable = None,
        loss_fn: Callable = F.binary_cross_entropy,
        loss_params: dict = {},
        n_epochs: int = 100,
        warmup_steps: int = 0,
        start_decay: int = None,
        *args,
        **kwargs,
    ) -> torch.nn.Module:
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
            warmup_steps (int, optional): number of warmup steps. Defaults
                to 0.
            start_decay (int, optional): number of steps after which decay
                begins. Defaults to None (decay starts after warmup).
            args: arguments for classification network class.
            kwargs: keyword arguments for classification network class.

        Returns:
            pl.LightningModule: a classification network module.
        """

        super().__init__(*args, **kwargs)

        self.image_keys = image_keys
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
        self.args = args
        self.kwargs = kwargs

        self.save_hyperparameters(
            ignore=[
                "networks",
                "training_dataloader_call",
                "training_batch_preproc",
            ]
        )
        self.setup_metrics()

    def training_step(self, batch, batch_idx):
        x, y = [batch[k] for k in self.image_keys], batch[self.label_key]
        prediction = self.forward(x)
        prediction = torch.squeeze(prediction, 1)

        loss = self.calculate_loss(prediction, y)

        self.log("train_loss", loss)
        return loss

    def validation_step(self, batch, batch_idx):
        x, y = [batch[k] for k in self.image_keys], batch[self.label_key]
        prediction = self.forward(x)
        prediction = torch.squeeze(prediction, 1)

        loss = self.calculate_loss(prediction, y)
        self.log(
            "val_loss",
            loss,
            on_epoch=True,
            on_step=False,
            prog_bar=True,
            batch_size=x[0].shape[0],
        )
        self.update_metrics(prediction, y, self.val_metrics)
        return loss

    def test_step(self, batch, batch_idx):
        x, y = [batch[k] for k in self.image_keys], batch[self.label_key]
        prediction = self.forward(x)
        prediction = torch.squeeze(prediction, 1)

        loss = self.calculate_loss(prediction, y)

        self.update_metrics(prediction, y, self.test_metrics, log=False)
        return loss

    def train_dataloader(self) -> torch.utils.data.DataLoader:
        return self.training_dataloader_call()

    def on_train_epoch_end(self):
        sch = self.lr_schedulers().state_dict()
        lr = self.learning_rate
        last_lr = sch["_last_lr"][0] if "_last_lr" in sch else lr
        self.log("lr", last_lr)


class ViTClassifierPL(ViTClassifier, ClassPLABC):
    """
    ViT classification network implementation for Pytorch
    Lightning.
    """

    def __init__(
        self,
        image_key: str = "image",
        label_key: str = "label",
        learning_rate: float = 0.001,
        batch_size: int = 4,
        weight_decay: float = 0.0,
        training_dataloader_call: Callable = None,
        loss_fn: Callable = F.binary_cross_entropy,
        loss_params: dict = {},
        n_epochs: int = 100,
        warmup_steps: int = 0,
        start_decay: int = None,
        training_batch_preproc: Callable = None,
        *args,
        **kwargs,
    ) -> torch.nn.Module:
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

        super().__init__(*args, **kwargs)

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

        self.save_hyperparameters(
            ignore=["training_dataloader_call", "training_batch_preproc"]
        )
        self.setup_metrics()


class FactorizedViTClassifierPL(FactorizedViTClassifier, ClassPLABC):
    """
    ViT classification network implementation for Pytorch
    Lightning.
    """

    def __init__(
        self,
        image_key: str = "image",
        label_key: str = "label",
        learning_rate: float = 0.001,
        batch_size: int = 4,
        weight_decay: float = 0.0,
        training_dataloader_call: Callable = None,
        loss_fn: Callable = F.binary_cross_entropy,
        loss_params: dict = {},
        n_epochs: int = 100,
        warmup_steps: int = 0,
        start_decay: int = None,
        training_batch_preproc: Callable = None,
        *args,
        **kwargs,
    ) -> torch.nn.Module:
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

        super().__init__(*args, **kwargs)

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

        self.save_hyperparameters(
            ignore=["training_dataloader_call", "training_batch_preproc"]
        )
        self.setup_metrics()


class TransformableTransformerPL(TransformableTransformer, ClassPLABC):
    """
    PL module for the TransformableTransformer.
    """

    def __init__(
        self,
        image_key: str = "image",
        label_key: str = "label",
        learning_rate: float = 0.001,
        batch_size: int = 4,
        weight_decay: float = 0.0,
        training_dataloader_call: Callable = None,
        loss_fn: Callable = F.binary_cross_entropy,
        loss_params: dict = {},
        n_epochs: int = 100,
        warmup_steps: int = 0,
        start_decay: int = None,
        training_batch_preproc: Callable = None,
        *args,
        **kwargs,
    ) -> torch.nn.Module:
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

        super().__init__(*args, **kwargs)

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

        self.save_hyperparameters(
            ignore=[
                "module",
                "loss_fn",
                "training_dataloader_call",
                "training_batch_preproc",
            ]
        )
        self.setup_metrics()


class MultipleInstanceClassifierPL(MultipleInstanceClassifier, ClassPLABC):
    """
    PL module for the MultipleInstanceClassifier.
    """

    def __init__(
        self,
        image_key: str = "image",
        label_key: str = "label",
        learning_rate: float = 0.001,
        batch_size: int = 4,
        weight_decay: float = 0.0,
        training_dataloader_call: Callable = None,
        loss_fn: Callable = F.binary_cross_entropy,
        loss_params: dict = {},
        n_epochs: int = 100,
        warmup_steps: int = 0,
        start_decay: int = None,
        training_batch_preproc: Callable = None,
        *args,
        **kwargs,
    ) -> torch.nn.Module:
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

        super().__init__(*args, **kwargs)

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

        self.save_hyperparameters(
            ignore=[
                "module",
                "loss_fn",
                "training_dataloader_call",
                "training_batch_preproc",
            ]
        )
        self.setup_metrics()


class HybridClassifierPL(HybridClassifier, ClassPLABC):
    """
    PL module for the HybridClassifier.
    """

    def __init__(
        self,
        image_key: str = "image",
        label_key: str = "label",
        tab_key: str = "tabular",
        learning_rate: float = 0.001,
        batch_size: int = 4,
        weight_decay: float = 0.0,
        training_dataloader_call: Callable = None,
        loss_fn: Callable = F.binary_cross_entropy,
        loss_params: dict = {},
        n_epochs: int = 100,
        warmup_steps: int = 0,
        start_decay: int = None,
        training_batch_preproc: Callable = None,
        *args,
        **kwargs,
    ) -> torch.nn.Module:
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

        super().__init__(*args, **kwargs)

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

        self.save_hyperparameters(
            # explicitly ignoring as lightning seems to have some issues with
            # this during testing
            ignore=[
                "args",
                "kwargs",
                "convolutional_module",
                "tabular_module",
                "training_dataloader_call",
                "training_batch_preproc",
            ]
        )
        self.setup_metrics()

    def training_step(self, batch, batch_idx):
        x_conv, y = batch[self.image_key], batch[self.label_key]
        x_tab = batch[self.tab_key]
        prediction = self.forward(x_conv, x_tab)
        prediction = torch.squeeze(prediction, 1)

        loss = self.calculate_loss(prediction, y, with_params=True)

        self.log("train_loss", loss, sync_dist=True)
        return loss

    def validation_step(self, batch, batch_idx):
        x_conv, y = batch[self.image_key], batch[self.label_key]
        x_tab = batch[self.tab_key]
        prediction = self.forward(x_conv, x_tab)
        prediction = torch.squeeze(prediction, 1)

        loss = self.calculate_loss(prediction, y, with_params=True)
        self.log(
            "val_loss",
            loss,
            on_epoch=True,
            on_step=False,
            prog_bar=True,
            batch_size=x_conv.shape[0],
            sync_dist=True,
        )
        self.update_metrics(prediction, y, self.val_metrics)
        return loss

    def test_step(self, batch, batch_idx):
        x_conv, y = batch[self.image_key], batch[self.label_key]
        x_tab = batch[self.tab_key]
        prediction = self.forward(x_conv, x_tab)
        prediction = torch.squeeze(prediction, 1)

        loss = self.calculate_loss(prediction, y)
        self.log(
            "test_loss",
            loss,
            on_epoch=True,
            on_step=False,
            prog_bar=True,
            batch_size=x_conv.shape[0],
            sync_dist=True,
        )
        self.update_metrics(prediction, y, self.test_metrics)
        return loss


class DeconfoundedNetPL(DeconfoundedNetGeneric, ClassPLABC):
    """
    Feature deconfounder net for PyTorch Lightning.
    """

    def __init__(
        self,
        image_key: str = "image",
        label_key: str = "label",
        embedder: torch.nn.Module = None,
        cat_confounder_key: str = None,
        cont_confounder_key: str = None,
        learning_rate: float = 0.001,
        batch_size: int = 4,
        weight_decay: float = 0.0,
        training_dataloader_call: Callable = None,
        loss_fn: Callable = F.binary_cross_entropy,
        loss_params: dict = {},
        n_epochs: int = 100,
        warmup_steps: int = 0,
        start_decay: int = None,
        training_batch_preproc: Callable = None,
        *args,
        **kwargs,
    ) -> torch.nn.Module:
        """
        Args:
            image_key (str): key corresponding to the key from the train
                dataloader.
            label_key (str): key corresponding to the label key from the train
                dataloader.
            embedder (torch.nn.Module, optional): embedder for categorical
                confounders. Defaults to None but necessary if
                cat_confounder_key is specified.
            cat_confounder_key (str, optional): key for categorical confounder.
                Defaults to None.
            cont_confounder_key (str, optional): key for continuous confounder.
                Defaults to None.
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

        super().__init__(*args, **kwargs)

        self.image_key = image_key
        self.label_key = label_key
        self.embedder = embedder
        self.cat_confounder_key = cat_confounder_key
        self.cont_confounder_key = cont_confounder_key
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

        self.loss_str = ["loss", "cat_loss", "cont_loss", "feat_loss"]

        self.conf_mult = 1.0

        self.save_hyperparameters(
            ignore=[
                "loss_fn",
                "training_dataloader_call",
                "training_batch_preproc",
            ]
        )
        self.setup_metrics()

    def loss_cat_confounder(self, pred: torch.Tensor, y: torch.Tensor):
        y = self.embedder(y)
        d = pred[0].device
        return (
            sum(
                [
                    F.cross_entropy(p, y_.to(d)) / len(pred)
                    for p, y_ in zip(pred, y)
                ]
            )
            * self.conf_mult
        )

    def loss_cont_confounder(self, pred: torch.Tensor, y: torch.Tensor):
        return F.mse_loss(pred, y.to(pred)) * self.conf_mult

    def loss_features(self, features: torch.Tensor):
        if self.n_features_deconfounder > 0:
            conf_f = features[:, : self.n_features_deconfounder, None]
            deconf_f = features[:, None, self.n_features_deconfounder :]
            conf_f_norm = conf_f - conf_f.mean(0, keepdim=True)
            deconf_f_norm = deconf_f - deconf_f.mean(0, keepdim=True)
            n = (conf_f_norm * deconf_f_norm).sum(0)
            d = torch.multiply(
                # applying sqrt separately prevents overflows with low precision
                conf_f_norm.square().sum(0).sqrt(),
                deconf_f_norm.square().sum(0).sqrt(),
            ).clamp(min=1e-8)
            return n.divide(d).clamp(-1.0, 1.0).square().mean()

    def step(self, batch: dict[str, torch.Tensor], with_params: bool):
        x, y = batch[self.image_key], batch[self.label_key]
        if hasattr(self, "training_batch_preproc"):
            if self.training_batch_preproc is not None:
                x, y = self.training_batch_preproc(x, y)
        prediction = self.forward(x)
        classification = prediction[0]
        confounder_classification = prediction[1]
        confounder_regression = prediction[2]
        features = prediction[3]
        cat_conf_loss = None
        cont_conf_loss = None
        feature_loss = None
        if self.cat_confounder_key is not None:
            y_cat_confounder = batch[self.cat_confounder_key]
            cat_conf_loss = self.loss_cat_confounder(
                confounder_classification, y_cat_confounder
            )
        if self.cont_confounder_key is not None:
            y_cont_confounder = batch[self.cont_confounder_key]
            cont_conf_loss = self.loss_cont_confounder(
                confounder_regression, y_cont_confounder
            )
        if self.cat_confounder_key or self.cont_confounder_key:
            feature_loss = self.loss_features(features)
        classification_loss = self.calculate_loss(
            classification.squeeze(1), y, with_params=with_params
        )
        return (
            classification_loss,
            cat_conf_loss,
            cont_conf_loss,
            feature_loss,
            classification,
            y,
        )

    def log_losses(self, losses: list[torch.Tensor], prefix: str):
        for loss_val, s in zip(losses, self.loss_str):
            if loss_val is not None:
                self.log(f"{prefix}_{s}", loss_val, prog_bar=True)

    def training_step(self, batch, batch_idx):
        output = self.step(batch, with_params=True)
        losses, _ = output[:4], output[4:]
        losses = [
            loss_val.mean() if loss_val is not None else None
            for loss_val in losses
        ]
        self.log_losses(losses, "tr")
        return sum([loss_val for loss_val in losses if loss_val is not None])

    def validation_step(self, batch, batch_idx):
        output = self.step(batch, with_params=False)
        losses, pred_y = output[:4], output[4:]
        losses = [
            loss_val.mean() if loss_val is not None else None
            for loss_val in losses
        ]
        self.log_losses(losses, "val")
        self.update_metrics(
            pred_y[0],
            pred_y[1],
            self.val_metrics,
            on_epoch=True,
            prog_bar=True,
        )
        return sum([loss_val for loss_val in losses if loss_val is not None])

    def test_step(self, batch, batch_idx):
        output = self.step(batch, with_params=False)
        losses, pred_y = output[:4], output[4:]
        losses = [
            loss_val.mean() if loss_val is not None else None
            for loss_val in losses
        ]
        self.log_losses(losses, "test")
        self.update_metrics(
            pred_y[0],
            pred_y[1],
            self.test_metrics,
            log=False,
            on_epoch=True,
            prog_bar=True,
        )
        return sum([loss_val for loss_val in losses if loss_val is not None])

    def update_metrics(
        self, prediction, y, metrics, log: bool = True, **kwargs
    ):
        y = y.long()
        if self.n_classes == 2:
            prediction = torch.sigmoid(prediction).squeeze(1)
        else:
            prediction = F.softmax(prediction, -1)
        for k in metrics:
            metrics[k](prediction, y)
            if log is True:
                self.log(k, metrics[k], **kwargs)

    def configure_optimizers(self):
        params_confounder = []
        params_classifier = []
        for n, p in self.named_parameters():
            if "confound" in n:
                params_confounder.append(p)
            else:
                params_classifier.append(p)
        parameters = [
            {
                "params": params_confounder,
                "weight_decay": 0,
            },
            {
                "params": params_classifier,
                "weight_decay": self.weight_decay,
            },
        ]
        optimizer = torch.optim.Adam(parameters, lr=self.learning_rate)
        lr_schedulers = CosineAnnealingWithWarmupLR(
            optimizer,
            T_max=self.n_epochs,
            start_decay=self.start_decay,
            n_warmup_steps=self.warmup_steps,
        )

        return {
            "optimizer": optimizer,
            "lr_scheduler": lr_schedulers,
            "monitor": "val_loss",
        }
