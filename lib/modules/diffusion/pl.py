import torch
import lightning.pytorch as pl
from typing import Callable,Dict,List

from .unet import DiffusionUNet
from ..learning_rate import CosineAnnealingWithWarmupLR

class DiffusionUNetPL(DiffusionUNet,pl.LightningModule):
    def __init__(self,
                 diffusion_process,
                 image_key: str="image",
                 label_key: str="label",
                 n_epochs: int=100,
                 warmup_steps: int=0,
                 start_decay: int=0,
                 training_dataloader_call: Callable=None,
                 batch_size: int=16,
                 learning_rate: float=0.001,
                 weight_decay: float=0.005,
                 *args,
                 **kwargs):
        super().__init__(*args,**kwargs)

        self.diffusion_process = diffusion_process
        self.image_key = image_key
        self.label_key = label_key
        self.n_epochs = n_epochs
        self.warmup_steps = warmup_steps
        self.start_decay = start_decay
        self.training_dataloader_call = training_dataloader_call
        self.batch_size = batch_size
        self.learning_rate = learning_rate
        self.weight_decay = weight_decay

        self.noise_steps = self.diffusion_process.noise_steps
        self.loss_fn = torch.nn.MSELoss()

    def calculate_loss(self,prediction, epsilon):
        loss = self.loss_fn(prediction,epsilon)
        return loss.mean()

    def step(self,x:torch.Tensor,t:torch.Tensor,cls:torch.Tensor=None):
        if cls is not None:
            cls = torch.round(cls)
        noisy_image,epsilon = self.diffusion_process.noise_images(x,t)
        output = self.forward(X=noisy_image,t=t / self.noise_steps,cls=cls)
        loss = self.calculate_loss(output,epsilon)
        return loss,output,noisy_image

    def unpack_batch(self,batch):
        x = batch[self.image_key]
        if self.label_key is not None:
            cls = batch[self.label_key]
        else:
            cls = None
        return x,cls

    def training_step(self,batch:dict,batch_idx:int):
        x,cls = self.unpack_batch(batch)
        t = self.diffusion_process.sample_timesteps(x.shape[0]).to(x.device)
        loss,output,noisy_image = self.step(x,t,cls)
        self.log("loss",loss,on_step=True,prog_bar=True)
        return loss

    def validation_step(self,batch:dict,batch_idx:int):
        x,cls = self.unpack_batch(batch)
        t = self.diffusion_process.sample_timesteps(x.shape[0]).to(x.device)
        loss,output,noisy_image = self.step(x,t,cls)
        self.log("val_loss",loss,on_epoch=True,prog_bar=True)
        return loss, output, noisy_image

    def test_step(self,batch:dict,batch_idx:int):
        x,cls = self.unpack_batch(batch)
        t = self.diffusion_process.sample_timesteps(x.shape[0]).to(x.device)
        loss,output,noisy_image = self.step(x,t,cls)
        self.log("test_loss",loss)
        return loss
    
    def train_dataloader(self) -> torch.utils.data.DataLoader:
        return self.training_dataloader_call(self.batch_size)
    
    def configure_optimizers(self):
        optimizer = torch.optim.AdamW(
            self.parameters(),
            lr=self.learning_rate,weight_decay=self.weight_decay)
        lr_schedulers = CosineAnnealingWithWarmupLR(
            optimizer,T_max=self.n_epochs,start_decay=self.start_decay,
            n_warmup_steps=self.warmup_steps,eta_min=0.0)

        return {"optimizer":optimizer,
                "lr_scheduler":lr_schedulers,
                "monitor":"val_loss"}
