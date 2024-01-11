import torch

from ..segmentation.pl import UNetBasePL, update_metrics
from .unet import UNetSemiSL

from typing import Callable, Dict


class UNetContrastiveSemiSL(UNetSemiSL, UNetBasePL):
    """Standard supervised U-Net with support for semi-/self-supervision
    based on a contrastive scheme.
    """

    def __init__(
        self,
        image_key: str = "image",
        semi_sl_image_key_1: str = "semi_sl_image_1",
        semi_sl_image_key_2: str = "semi_sl_image_2",
        label_key: str = "label",
        skip_conditioning_key: str = None,
        feature_conditioning_key: str = None,
        learning_rate: float = 0.001,
        lr_encoder: float = None,
        cosine_decay: bool = True,
        batch_size: int = 4,
        n_epochs: int = 100,
        weight_decay: float = 0.005,
        training_dataloader_call: Callable = None,
        loss_fn: Callable = torch.nn.functional.binary_cross_entropy,
        loss_params: dict = {},
        loss_fn_semi_sl: Callable = torch.nn.functional.mse_loss,
        ema: torch.nn.Module = None,
        stop_gradient: bool = True,
        picai_eval: bool = False,
        *args,
        **kwargs
    ) -> torch.nn.Module:
        """
        Args:
            image_key (str): key corresponding to the input from the train
                dataloader.
            semi_sl_image_key_1 (str, optional): key corresponding to augmented
                image 1 with no annotations. Defaults to "semi_sl_image_1".
            semi_sl_image_key_1 (str, optional): key corresponding to augmented
                image 2 with no annotations. Defaults to "semi_sl_image_1".
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
            loss_fn_semi_sl (Callable, optional): function to calculate the
                loss during self-supervision. Should take to Tensors with
                identical shape and return a single value quantifying their
                difference. Defaults to mse_loss.
            ema (torch.nn.Module, optional): exponential moving decay module
                (EMA) for teacher. Must have an update method that takes model
                as input and updates the weights based on this. Defaults to None.
            stop_gradient (bool, optional): stops gradients when calculating
                losses. Defaults to False.
            picai_eval (bool, optional): evaluates network using PI-CAI
                metrics as well (can be a bit long).
            args: arguments for UNet class.
            kwargs: keyword arguments for UNet class.
        """

        super().__init__(*args, **kwargs)

        self.image_key = image_key
        self.semi_sl_image_key_1 = semi_sl_image_key_1
        self.semi_sl_image_key_2 = semi_sl_image_key_2
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
        self.loss_fn_semi_sl = loss_fn_semi_sl
        self.ema = ema
        self.stop_gradient = stop_gradient
        self.picai_eval = picai_eval

        self.loss_fn_class = torch.nn.BCEWithLogitsLoss()
        self.setup_metrics()

        if self.ema is not None:
            self.ema.update(self)
        else:
            self.ema = None

        # for metrics (AUC, AP) which require a list of predictions + gt
        self.all_pred = []
        self.all_true = []

        self.bn_mult = 0.1
        self.ssl_weight = 0.1

        if (
            self.semi_sl_image_key_1 is not None
            and self.semi_sl_image_key_2 is not None
        ):
            self.semi_supervised = True
        else:
            self.semi_supervised = False

    def unpack_batch(self, batch):
        if self.semi_supervised is True and "supervised" in batch:
            batch = batch["supervised"]
        return super().unpack_batch(batch)

    def unpack_batch_semi_sl(self, batch: Dict[str, torch.Tensor]):
        if self.semi_supervised is True:
            batch = batch["self_supervised"]
        x_1 = batch[self.semi_sl_image_key_1]
        x_2 = batch[self.semi_sl_image_key_2]
        if self.skip_conditioning_key is not None:
            x_cond = batch[self.skip_conditioning_key]
        else:
            x_cond = None
        if self.feature_conditioning_key is not None:
            x_fc = batch[self.feature_conditioning_key]
        else:
            x_fc = None
        return x_1, x_2, x_cond, x_fc

    def forward_features_ema_stop_grad(self, **kwargs):
        if self.ema is not None:
            op = self.ema.shadow.forward_features
        else:
            op = self.forward_features
        if self.stop_gradient is True:
            with torch.no_grad():
                return op(**kwargs)
        else:
            return op(**kwargs)

    def calculate_loss_semi_sl(
        self, output_1: torch.Tensor, output_2: torch.Tensor
    ):
        loss = self.loss_fn_semi_sl(output_1, output_2)
        return loss.mean() * self.ssl_weight

    def loss_wrapper_semi_sl(
        self,
        x_1: torch.Tensor,
        x_2: torch.Tensor,
        x_cond: torch.Tensor,
        x_fc: torch.Tensor,
    ):
        output_1 = self.forward_features_ema_stop_grad(
            X=x_1, X_skip_layer=x_cond, X_feature_conditioning=x_fc
        )
        output_2 = self.forward_features(
            X=x_2, X_skip_layer=x_cond, X_feature_conditioning=x_fc
        )
        return self.calculate_loss_semi_sl(output_1, output_2)

    def step_semi_sl(
        self,
        x_1: torch.Tensor,
        x_2: torch.Tensor,
        x_cond: torch.Tensor,
        x_fc: torch.Tensor,
    ):
        loss_a = self.loss_wrapper_semi_sl(x_1, x_2, x_cond, x_fc)
        # loss_b = self.loss_wrapper_semi_sl(x_2, x_1, x_cond, x_fc)
        loss = loss_a  # + loss_b
        return loss

    def training_step(self, batch, batch_idx):
        # supervised bit
        if self.label_key is not None:
            x, x_cond, x_fc, y, y_class = self.unpack_batch(batch)
            pred_final, pred_class, loss, class_loss = self.step(
                x, y, y_class, x_cond, x_fc
            )
            output_loss = (
                loss.mean() if class_loss is None else loss.mean() + class_loss
            )
            self.log_loss("train_loss", loss, batch_size=y.shape[0])

        # self-supervised bit
        if (
            self.semi_sl_image_key_1 is not None
            and self.semi_sl_image_key_2 is not None
        ):
            x_1, x_2, x_cond, x_fc = self.unpack_batch_semi_sl(batch)
            self_sl_loss = self.step_semi_sl(x_1, x_2, x_cond, x_fc)
            self.log(
                "train_self_sl_loss",
                self_sl_loss,
                batch_size=y.shape[0],
                sync_dist=True,
            )
            output_loss = output_loss + self_sl_loss
            if self.ema is not None:
                self.ema.update(self)

        if class_loss is not None:
            self.log(
                "train_cl_loss",
                class_loss,
                batch_size=y.shape[0],
                sync_dist=True,
            )

        self.check_loss(x, y, pred_final, output_loss)

        update_metrics(
            self,
            self.train_metrics,
            pred_final,
            y,
            pred_class,
            y_class,
            on_epoch=True,
            on_step=False,
            prog_bar=True,
        )

        self.train_batch_size = x.shape[0]

        return output_loss

    def validation_step(self, batch, batch_idx):
        output_loss = torch.as_tensor(0.0).to(self.device)
        if self.label_key is not None:
            x, x_cond, x_fc, y, y_class = self.unpack_batch(batch)

            bs = x.shape[0]
            if self.train_batch_size is None:
                mbs = self.batch_size
            else:
                mbs = self.train_batch_size
            for i in range(0, bs, mbs):
                m, M = i, i + mbs
                out = self.step(
                    x[m:M],
                    y[m:M],
                    y_class[m:M] if y_class is not None else None,
                    x_cond[m:M] if x_cond is not None else None,
                    x_fc[m:M] if x_cond is not None else None,
                )
                pred_final, pred_class, loss, class_loss = out
                output_loss += (
                    loss.mean()
                    if class_loss is None
                    else loss.mean() + class_loss
                ) / (bs // mbs)

                if self.picai_eval is True:
                    for s_p, s_y in zip(
                        pred_final.squeeze(1).detach().cpu().numpy(),
                        y[m:M].squeeze(1).detach().cpu().numpy(),
                    ):
                        self.all_pred.append(s_p)
                        self.all_true.append(s_y)

                self.log_loss(
                    "val_loss",
                    loss,
                    on_epoch=True,
                    batch_size=y.shape[0],
                )

                update_metrics(
                    self,
                    self.val_metrics,
                    pred_final,
                    y[m:M],
                    pred_class,
                    y_class[m:M] if y_class is not None else None,
                    on_epoch=True,
                    prog_bar=True,
                )

        if (
            self.semi_sl_image_key_1 is not None
            and self.semi_sl_image_key_2 is not None
        ):
            x_1, x_2, x_cond, x_fc = self.unpack_batch_semi_sl(batch)
            self_sl_loss = self.step_semi_sl(x_1, x_2, x_cond, x_fc)
            self.log(
                "val_self_sl_loss",
                self_sl_loss,
                prog_bar=True,
                batch_size=y.shape[0],
                sync_dist=True,
            )
            output_loss = output_loss + self_sl_loss.mean()
        return output_loss

    def test_step(self, batch, batch_idx):
        output_loss = torch.as_tensor(0.0).to(self.device)
        x, x_cond, x_fc, y, y_class = self.unpack_batch(batch)

        bs = x.shape[0]
        if self.train_batch_size is None:
            mbs = self.batch_size
        else:
            mbs = self.train_batch_size
        for i in range(0, bs, mbs):
            m, M = i, i + mbs
            out = self.step(
                x[m:M],
                y[m:M],
                y_class[m:M] if y_class is not None else None,
                x_cond[m:M] if x_cond is not None else None,
                x_fc[m:M] if x_cond is not None else None,
            )
            pred_final, pred_class, loss, class_loss = out
            output_loss += (
                loss.mean() if class_loss is None else loss.mean() + class_loss
            ) / (bs // mbs)

            if self.picai_eval is True:
                for s_p, s_y in zip(
                    pred_final.squeeze(1).detach().cpu().numpy(),
                    y.squeeze(1).detach().cpu().numpy(),
                ):
                    self.all_pred.append(s_p)
                    self.all_true.append(s_y)

            y, out = self.crop_if_necessary(y, pred_final)

            update_metrics(
                self,
                self.test_metrics,
                pred_final,
                y[m:M],
                pred_class,
                y_class[m:M] if y_class is not None else None,
                on_epoch=True,
                prog_bar=True,
            )

        return output_loss
