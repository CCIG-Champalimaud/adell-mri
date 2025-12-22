from typing import Callable, Tuple, Union

import lightning.pytorch as pl
import numpy as np
import torch
import torch.nn.functional as F
import torchmetrics

from adell_mri.modules.classification.pl import meta_tensors_to_tensors
from adell_mri.modules.learning_rate import CosineAnnealingWithWarmupLR
from adell_mri.modules.object_detection.map import mAP
from adell_mri.modules.object_detection.nets import CoarseDetector3d, YOLONet3d


def real_boxes_from_centres_sizes(
    centres: torch.Tensor,
    sizes: torch.Tensor,
    anchors: torch.Tensor,
    h: torch.Tensor,
    w: torch.Tensor,
    d: torch.Tensor,
    a: torch.Tensor,
    correction_factor: torch.Tensor,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    centres[:, 0] = centres[:, 0] + h
    centres[:, 1] = centres[:, 1] + w
    centres[:, 2] = centres[:, 2] + d
    centres = centres * correction_factor
    anchors = torch.stack(torch.split(anchors.squeeze(), 3))
    sizes = torch.exp(sizes)
    sizes = sizes * anchors[a] / 2
    tl_corner = centres - sizes
    br_corner = centres + sizes
    return tl_corner, br_corner, centres, sizes


class YOLONet3dPL(YOLONet3d, pl.LightningModule):
    def __init__(
        self,
        image_key: str = "image",
        label_key: str = "label",
        boxes_key: str = "boxes",
        box_label_key: str = "labels",
        learning_rate: float = 0.001,
        batch_size: int = 4,
        weight_decay: float = 0.005,
        training_dataloader_call: Callable = None,
        reg_loss_fn: Callable = F.mse_loss,
        classification_loss_fn: Callable = F.binary_cross_entropy,
        object_loss_fn: Callable = F.binary_cross_entropy,
        positive_weight: float = 1.0,
        classification_loss_params: dict = {},
        object_loss_params: dict = {},
        iou_threshold: float = 0.5,
        n_epochs: int = 100,
        warmup_steps: Union[int, float] = 0,
        start_decay: int = None,
        *args,
        **kwargs,
    ) -> torch.nn.Module:
        """
        YOLO-like network implementation for Pytorch Lightning.

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

        super().__init__(*args, **kwargs)

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
        self.n_epochs = n_epochs
        self.warmup_steps = warmup_steps
        self.start_decay = start_decay

        self.object_idxs = np.array([0])
        self.center_idxs = np.array([1, 2, 3])
        self.size_idxs = np.array([4, 5, 6])
        self.class_idxs = np.array([7])

        self.setup_metrics()

        self.loss_accumulator = 0.0
        self.loss_accumulator_d = 0.0

    def on_before_batch_transfer(self, batch, dataloader_idx):
        return meta_tensors_to_tensors(batch)

    def calculate_loss(
        self,
        prediction,
        y,
        y_class,
        b,
        h,
        w,
        d,
        a,
        correction_factor,
        weights=None,
    ):
        bb_center, bb_size, bb_object, bb_cl = prediction
        pred_centers = bb_center[b, :, h, w, d, a]
        y_centers = y[:, self.center_idxs][b, :, h, w, d, a]
        pred_size = bb_size[b, :, h, w, d, a]
        y_size = y[:, self.size_idxs][b, :, h, w, d, a]

        tl_pred, br_pred, centers_pred, _ = real_boxes_from_centres_sizes(
            pred_centers,
            pred_size,
            self.anchor_tensor,
            h,
            w,
            d,
            a,
            correction_factor,
        )
        tl_y, br_y, centers_y, _ = real_boxes_from_centres_sizes(
            y_centers,
            y_size,
            self.anchor_tensor,
            h,
            w,
            d,
            a,
            correction_factor,
        )

        pred_corners = torch.cat([tl_pred, br_pred], 1)
        y_corners = torch.cat([tl_y, br_y], 1)
        iou, cpd, ar = self.reg_loss_fn(
            pred_corners, y_corners, centers_pred, centers_y
        )
        y_object = torch.zeros_like(bb_object, device=iou.device)
        y_object[b, :, h, w, d, a] = torch.unsqueeze(iou, 1)

        obj_loss = self.object_loss_fn(
            bb_object, y_object, **self.object_loss_params
        )

        obj_weight = 1.0
        box_weight = 0.1
        output = obj_loss.mean() * obj_weight
        output = (
            output + ((1 - iou).mean() + cpd.mean() + ar.mean()) * box_weight
        )
        if self.n_classes > 2:
            b_c, h_c, w_c, d_c = torch.split(
                torch.unique(torch.stack([b, h, w, d], 1), dim=0), 1, dim=1
            )
            pred_class_for_loss = bb_cl[b_c, :, h_c, w_c, d_c].squeeze(1)
            y_class_for_loss = y_class[b_c, h_c, w_c, d_c].squeeze()
            y_class_for_loss = y_class_for_loss.long()
            cla_loss = self.classification_loss_fn(
                pred_class_for_loss,
                y_class_for_loss,
                **self.classification_loss_params,
            )
            output = output + cla_loss.mean()
        return output.mean()

    def retrieve_correct(
        self,
        prediction,
        target,
        target_class,
        typ,
        b,
        h,
        w,
        d,
        a,
        correction_factor=None,
    ):
        typ = typ.lower()
        if typ == "center":
            p, t = prediction[0], target[:, self.center_idxs, :, :, :, :]
        elif typ == "size":
            p, t = prediction[1], target[:, self.size_idxs, :, :, :, :]
        elif typ == "obj":
            p, t = prediction[2], target[:, self.object_idxs, :, :, :, :]
            t = (t > self.iou_threshold).int()
        elif typ == "class":
            p, t = prediction[3], target_class
            t = torch.round(t).int()
        elif typ == "map":
            p = self.recover_boxes_batch(
                *prediction, correction_factor=correction_factor, to_dict=True
            )
            t = None
        if typ not in ["obj", "map", "class"]:
            p, t = p[b, :, h, w, d, a], t[b, :, h, w, d, a]
        elif typ == "class":
            p, t = p[b, :, h, w, d], t[b, h, w, d]
        return p, t

    def calculate_correction_factor(
        self, x: torch.Tensor, y: torch.Tensor
    ) -> torch.Tensor:
        corr_fac = torch.as_tensor(np.array(x.shape[2:]))
        corr_fac = corr_fac / torch.as_tensor(np.array(y.shape[2:]))
        corr_fac = corr_fac.to(x.device)
        return corr_fac

    def step(self, batch, batch_idx, metrics):
        x, y = batch[self.image_key], batch[self.label_key].float()
        corr_fac = self.calculate_correction_factor(x, y)
        prediction = list(self.forward(x))
        y_class, y = y[:, 0, :, :, :], y[:, 1:, :, :, :]
        y = torch.stack(self.split(y, self.n_b, 1), -1)
        b, h, w, d, a = torch.where(y[:, 0, :, :, :, :] > self.iou_threshold)
        prediction[:-1] = [
            torch.stack(self.split(x, self.n_b, 1), -1) for x in prediction[:-1]
        ]

        loss = self.calculate_loss(
            prediction, y, y_class, b, h, w, d, a, correction_factor=corr_fac
        )

        for k_typ in metrics:
            k, typ = k_typ.split("_")
            typ = typ.lower()
            cur_pred, cur_target = self.retrieve_correct(
                prediction, y, y_class, typ, b, h, w, d, a
            )
            if typ.lower() != "map":
                cur_pred, cur_target = self.retrieve_correct(
                    prediction, y, y_class, typ, b, h, w, d, a
                )
                metrics[k_typ](cur_pred, cur_target)
            else:
                cur_pred, cur_target = self.retrieve_correct(
                    prediction,
                    y,
                    y_class,
                    typ,
                    b,
                    h,
                    w,
                    d,
                    a,
                    correction_factor=corr_fac,
                )
                cur_target = [
                    {
                        "boxes": batch[self.boxes_key][i],
                        "labels": batch[self.box_label_key][i],
                    }
                    for i in range(len(batch[self.boxes_key]))
                ]
                for t in cur_target:
                    if len(t["boxes"].shape) == 2:
                        t["boxes"] = torch.concat(
                            [t["boxes"][:, :3], t["boxes"][:, 3:]], axis=1
                        )
                    else:
                        t["boxes"] = torch.concat(
                            [t["boxes"][:, :, 0], t["boxes"][:, :, 1]], axis=1
                        )
                    t["boxes"] = t["boxes"].to(loss.device)
                    t["labels"] = torch.as_tensor(t["labels"]).to(loss.device)
                metrics[k_typ](cur_pred, cur_target)
            self.log(
                k,
                metrics[k_typ],
                on_epoch=True,
                on_step=False,
                prog_bar=True,
                sync_dist=True,
                batch_size=x.shape[0],
            )
        return loss

    def training_step(self, batch, batch_idx):
        batch_size = batch[self.image_key].shape[0]
        loss = self.step(batch, batch_idx, self.train_metrics)
        self.log(
            "loss", loss, prog_bar=True, on_step=True, batch_size=batch_size
        )
        return loss

    def validation_step(self, batch, batch_idx):
        batch_size = batch[self.image_key].shape[0]
        loss = self.step(batch, batch_idx, self.val_metrics)
        self.log(
            "val_loss",
            loss,
            prog_bar=True,
            on_epoch=True,
            batch_size=batch_size,
        )
        return loss

    def test_step(self, batch, batch_idx):
        batch_size = batch[self.image_key].shape[0]
        loss = self.step(batch, batch_idx, self.test_metrics)
        self.log("test_loss", loss, on_epoch=True, batch_size=batch_size)
        return loss

    def train_dataloader(self) -> torch.utils.data.DataLoader:
        return self.training_dataloader_call()

    def configure_optimizers(self):
        optimizer = torch.optim.AdamW(
            self.parameters(),
            lr=self.learning_rate,
            weight_decay=self.weight_decay,
            amsgrad=True,
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

    def on_validation_epoch_end(self):
        sch = self.lr_schedulers().state_dict()
        lr = self.learning_rate
        last_lr = sch["_last_lr"][0] if "_last_lr" in sch else lr
        self.log("lr", last_lr)

    def setup_metrics(self):
        if self.n_classes == 2:
            C, A, M = None, None, "micro"
        else:
            C, A, M = self.n_classes, "samplewise", "macro"
        self.train_metrics = torch.nn.ModuleDict(
            {
                "cMSE_center": torchmetrics.MeanSquaredError(),
                "sMSE_size": torchmetrics.MeanSquaredError(),
                "objF1_obj": torchmetrics.FBetaScore(
                    task="binary", threshold=self.iou_threshold
                ),
                "objRec_obj": torchmetrics.Recall(
                    task="binary", threshold=self.iou_threshold
                ),
            }
        )
        self.val_metrics = torch.nn.ModuleDict(
            {
                "v:cMSE_center": torchmetrics.MeanSquaredError(),
                "v:sMSE_size": torchmetrics.MeanSquaredError(),
                "v:objF1_obj": torchmetrics.FBetaScore(
                    task="binary", threshold=self.iou_threshold
                ),
                "v:mAP_mAP": mAP(
                    iou_threshold=self.iou_threshold, n_classes=self.n_classes
                ),
            }
        )
        self.test_metrics = torch.nn.ModuleDict(
            {
                "t:cMSE_center": torchmetrics.MeanSquaredError(),
                "t:sMSE_size": torchmetrics.MeanSquaredError(),
                "t:objRec_obj": torchmetrics.Recall(
                    task="binary", threshold=self.iou_threshold
                ),
                "t:objPre_obj": torchmetrics.Precision(
                    task="binary", threshold=self.iou_threshold
                ),
                "t:objF1_obj": torchmetrics.FBetaScore(
                    task="binary", threshold=self.iou_threshold
                ),
                "t:mAP_mAP": mAP(iou_threshold=self.iou_threshold),
            }
        )

        if self.n_classes > 2:
            # no point in including this in the two class scenario
            self.train_metrics["clF1_class"] = torchmetrics.FBetaScore(
                task="multiclass", num_classes=C, mdmc_average=A, average=M
            )
            self.val_metrics["v:clF1_class"] = torchmetrics.FBetaScore(
                task="multiclass", num_classes=C, mdmc_average=A, average=M
            )
            self.test_metrics["t:clF1_class"] = torchmetrics.FBetaScore(
                task="multiclass", num_classes=C, mdmc_average=A, average=M
            )


class CoarseDetector3dPL(CoarseDetector3d, pl.LightningModule):
    def __init__(
        self,
        image_key: str = "image",
        label_key: str = "label",
        boxes_key: str = "bb",
        learning_rate: float = 0.001,
        batch_size: int = 4,
        weight_decay: float = 0.005,
        training_dataloader_call: Callable = None,
        object_loss_fn: Callable = F.binary_cross_entropy,
        positive_weight: float = 1.0,
        object_loss_params: dict = {},
        iou_threshold: float = 0.5,
        *args,
        **kwargs,
    ) -> torch.nn.Module:
        """
        YOLO-like network implementation for Pytorch Lightning.

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

        super().__init__(*args, **kwargs)

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

        self.loss_accumulator = 0.0
        self.loss_accumulator_d = 0.0

    def calculate_loss(self, prediction, y, weights=None):
        obj_loss = self.object_loss_fn(prediction, y, **self.object_loss_params)

        return obj_loss.mean()

    def split(self, x, n_splits, dim):
        size = int(x.shape[dim] // n_splits)
        return torch.split(x, size, dim)

    def training_step(self, batch, batch_idx):
        x, y = batch[self.image_key], batch[self.label_key].float()
        prediction = self.forward(x)
        # select the objectness tensor
        y = torch.stack(self.split(y[:, 1:], self.n_b, 1), -1)
        y = y[:, self.object_idxs].sum(-1)
        y = torch.where(
            y > 0,
            torch.ones_like(y, device=y.device),
            torch.zeros_like(y, device=y.device),
        )
        batch_size = int(prediction.shape[0])
        if batch_size == 1:
            y = torch.unsqueeze(y, 0)

        loss = self.calculate_loss(prediction, y)

        self.log("train_loss", loss, batch_size=x.shape[0])
        for k_typ in self.train_metrics:
            k, typ = k_typ.split("_")
            self.train_metrics[k_typ](prediction, y.int())
            self.log(
                k,
                self.train_metrics[k_typ],
                on_epoch=True,
                on_step=False,
                prog_bar=True,
                batch_size=x.shape[0],
            )
        return loss

    def validation_step(self, batch, batch_idx):
        x, y = batch[self.image_key], batch[self.label_key].float()
        prediction = self.forward(x)
        # select the objectness tensor
        y = torch.stack(self.split(y[:, 1:], self.n_b, 1), -1)
        y = y[:, self.object_idxs].sum(-1)
        y = torch.where(
            y > 0,
            torch.ones_like(y, device=y.device),
            torch.zeros_like(y, device=y.device),
        )
        batch_size = int(prediction.shape[0])
        if batch_size == 1:
            y = torch.unsqueeze(y, 0)

        loss = self.calculate_loss(prediction, y)

        self.loss_accumulator += loss
        self.loss_accumulator_d += 1
        for k_typ in self.val_metrics:
            k, typ = k_typ.split("_")
            self.val_metrics[k_typ](prediction, y.int())
            self.log(
                k,
                self.val_metrics[k_typ],
                on_epoch=True,
                on_step=False,
                prog_bar=True,
                batch_size=x.shape[0],
            )
        return loss

    def test_step(self, batch, batch_idx):
        x, y = batch[self.image_key], batch[self.label_key].float()
        prediction = self.forward(x)
        # select the objectness tensor
        y = torch.stack(self.split(y[:, 1:], self.n_b, 1), -1)
        y = y[:, self.object_idxs].sum(-1)
        y = torch.where(
            y > 0,
            torch.ones_like(y, device=y.device),
            torch.zeros_like(y, device=y.device),
        )
        batch_size = int(prediction.shape[0])
        if batch_size == 1:
            y = torch.unsqueeze(y, 0)

        loss = self.calculate_loss(prediction, y)

        for k_typ in self.test_metrics:
            k, typ = k_typ.split("_")
            self.test_metrics[k_typ](prediction, y.int())
            self.log(
                k,
                self.test_metrics[k_typ],
                on_epoch=True,
                on_step=False,
                batch_size=x.shape[0],
            )
        return loss

    def train_dataloader(self) -> torch.utils.data.DataLoader:
        return self.training_dataloader_call()

    def configure_optimizers(self):
        optimizer = torch.optim.AdamW(
            self.parameters(),
            lr=self.learning_rate,
            weight_decay=self.weight_decay,
            amsgrad=True,
        )
        lr_schedulers = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, patience=10, min_lr=1e-6, factor=0.3
        )

        return {
            "optimizer": optimizer,
            "lr_scheduler": lr_schedulers,
            "monitor": "val_loss",
        }

    def on_validation_epoch_end(self):
        for k_typ in self.val_metrics:
            k, typ = k_typ.split("_")
            val = self.val_metrics[k_typ].compute()
            self.log(k, val, prog_bar=True)
            self.val_metrics[k_typ].reset()
        val_loss = self.loss_accumulator / self.loss_accumulator_d
        sch = self.lr_schedulers().state_dict()
        lr = self.learning_rate
        last_lr = sch["_last_lr"][0] if "_last_lr" in sch else lr
        self.log("lr", last_lr)
        self.log("val_loss", val_loss, prog_bar=True)
        self.loss_accumulator = 0.0
        self.loss_accumulator_d = 0.0

    def setup_metrics(self):
        self.train_metrics = torch.nn.ModuleDict(
            {
                "objF1_obj": torchmetrics.FBetaScore(
                    None, threshold=self.iou_threshold
                ),
                "objRec_obj": torchmetrics.Recall(
                    None, threshold=self.iou_threshold
                ),
            }
        )
        self.val_metrics = torch.nn.ModuleDict(
            {
                "v:objF1_obj": torchmetrics.FBetaScore(
                    None, threshold=self.iou_threshold
                )
            }
        )
        self.test_metrics = torch.nn.ModuleDict(
            {
                "testobjRec_obj": torchmetrics.Recall(
                    None, threshold=self.iou_threshold
                ),
                "testobjPre_obj": torchmetrics.Precision(
                    None, threshold=self.iou_threshold
                ),
                "testobjF1_obj": torchmetrics.FBetaScore(
                    None, threshold=self.iou_threshold
                ),
            }
        )
