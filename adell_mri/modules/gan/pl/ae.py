import torch
import torch.nn.functional as F
import lightning.pytorch as pl
from ..losses import GaussianKLLoss
from ..ae import AutoEncoder
from ..vae import VariationalAutoEncoder


class AutoEncoderPL(AutoEncoder, pl.LightningModule):
    def __init__(
        self,
        input_image_key: str = "input_image",
        learning_rate: float = 0.0002,
        momentum_beta1: float = 0.5,
        momentum_beta2: float = 0.99,
        *args,
        **kwargs,
    ):
        super().__init__(*args, **kwargs)
        self.input_image_key = input_image_key
        self.learning_rate = learning_rate
        self.momentum_beta1 = momentum_beta1
        self.momentum_beta2 = momentum_beta2

        self.loss_fn = torch.nn.MSELoss()
        self.init_routine()

    def init_routine(self):
        self.save_hyperparameters(ignore=["loss_fn", "loss_params"])

    def step(self, batch):
        x = batch[self.input_image_key]
        output = self.forward(x)
        loss = self.loss_fn(output, x)
        return {"loss": loss}

    def training_step(self, batch, batch_idx):
        loss_dict = self.step(batch)
        for k in loss_dict:
            self.log(
                k, loss_dict[k], on_epoch=True, prog_bar=True, on_step=False
            )
        return sum([loss_dict[k] for k in loss_dict])

    def validation_step(self, batch, batch_idx):
        loss_dict = self.step(batch)
        for k in loss_dict:
            self.log(
                f"val_{k}",
                loss_dict[k],
                on_epoch=True,
                prog_bar=True,
                on_step=False,
            )
        return sum([loss_dict[k] for k in loss_dict])

    def test_step(self, batch, batch_idx):
        loss_dict = self.step(batch)
        for k in loss_dict:
            self.log(
                f"test_{k}",
                loss_dict[k],
                on_epoch=True,
                prog_bar=True,
                on_step=False,
            )
        return sum([loss_dict[k] for k in loss_dict])

    def configure_optimizers(self):
        return torch.optim.Adam(
            self.parameters(),
            lr=self.learning_rate,
            betas=(self.momentum_beta1, self.momentum_beta2),
        )


class VariationalAutoEncoderPL(VariationalAutoEncoder, pl.LightningModule):
    def __init__(
        self,
        input_image_key: str = "input_image",
        learning_rate: float = 0.0002,
        momentum_beta1: float = 0.5,
        momentum_beta2: float = 0.99,
        var_loss_mult: float = 1.0,
        *args,
        **kwargs,
    ):
        super().__init__(*args, **kwargs)
        self.input_image_key = input_image_key
        self.learning_rate = learning_rate
        self.momentum_beta1 = momentum_beta1
        self.momentum_beta2 = momentum_beta2
        self.var_loss_mult = var_loss_mult

        self.loss_fn = torch.nn.MSELoss()
        self.variational_loss_fn = GaussianKLLoss()
        self.init_routine()

    def init_routine(self):
        self.save_hyperparameters(ignore=["loss_fn", "loss_params"])

    def step(self, batch):
        x = batch[self.input_image_key]
        output, mu, logvar = self.forward(x)
        var_loss = self.variational_loss_fn(mu, logvar)
        loss = self.loss_fn(output, x)
        return {"rec_loss": loss, "var_loss": self.var_loss_mult * var_loss}

    def training_step(self, batch, batch_idx):
        loss_dict = self.step(batch)
        for k in loss_dict:
            self.log(
                k, loss_dict[k], on_epoch=True, prog_bar=True, on_step=False
            )
        return sum([loss_dict[k] for k in loss_dict])

    def validation_step(self, batch, batch_idx):
        loss_dict = self.step(batch)
        for k in loss_dict:
            self.log(
                f"val_{k}",
                loss_dict[k],
                on_epoch=True,
                prog_bar=True,
                on_step=False,
            )
        return sum([loss_dict[k] for k in loss_dict])

    def test_step(self, batch, batch_idx):
        loss_dict = self.step(batch)
        for k in loss_dict:
            self.log(
                f"test_{k}",
                loss_dict[k],
                on_epoch=True,
                prog_bar=True,
                on_step=False,
            )
        return sum([loss_dict[k] for k in loss_dict])

    def configure_optimizers(self):
        return torch.optim.Adam(
            self.parameters(),
            lr=self.learning_rate,
            betas=(self.momentum_beta1, self.momentum_beta2),
        )
