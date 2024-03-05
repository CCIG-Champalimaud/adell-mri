import torch
import lightning.pytorch as pl
from typing import Callable, List

from generative.inferers import DiffusionInferer
from generative.networks.nets import DiffusionModelUNet
from generative.networks.schedulers import DDPMScheduler
from ..learning_rate import CosineAnnealingWithWarmupLR
from ..classification.pl import meta_tensors_to_tensors


class DiffusionUNetPL(DiffusionModelUNet, pl.LightningModule):
    def __init__(
        self,
        inferer: Callable = DiffusionInferer,
        scheduler: Callable = DDPMScheduler,
        embedder: torch.nn.Module = None,
        image_key: str = "image",
        cat_condition_key: str = "cat_condition",
        num_condition_key: str = "num_condition",
        n_epochs: int = 100,
        warmup_steps: int = 0,
        start_decay: int = 0,
        training_dataloader_call: Callable = None,
        batch_size: int = 16,
        learning_rate: float = 0.001,
        weight_decay: float = 0.0,
        seed: int = 42,
        *args,
        **kwargs,
    ):
        super().__init__(*args, **kwargs)

        self.inferer = inferer
        self.scheduler = scheduler
        self.embedder = embedder
        self.image_key = image_key
        self.cat_condition_key = cat_condition_key
        self.num_condition_key = num_condition_key
        self.n_epochs = n_epochs
        self.warmup_steps = warmup_steps
        self.start_decay = start_decay
        self.training_dataloader_call = training_dataloader_call
        self.batch_size = batch_size
        self.learning_rate = learning_rate
        self.weight_decay = weight_decay
        self.seed = seed

        self.g = torch.Generator()
        self.g.manual_seed(self.seed)
        self.noise_steps = self.scheduler.num_train_timesteps
        self.loss_fn = torch.nn.MSELoss()

    def calculate_loss(self, prediction, epsilon):
        loss = self.loss_fn(prediction, epsilon)
        return loss.mean()

    def randn_like(self, x: torch.Tensor):
        return (
            torch.randn(
                size=x.shape, generator=self.g, dtype=x.dtype, layout=x.layout
            )
            .contiguous()
            .to(x.device)
        )

    def timesteps_like(self, x: torch.Tensor):
        return (
            torch.randint(0, self.noise_steps, (x.shape[0],), generator=self.g)
            .to(x.device)
            .long()
        )

    def step(
        self,
        x: torch.Tensor,
        timesteps: torch.Tensor = None,
        context: torch.Tensor = None,
    ):
        epsilon = self.randn_like(x)
        if timesteps is None:
            timesteps = self.timesteps_like(x)
        else:
            timesteps = timesteps.long()
        epsilon_pred = self.inferer(
            inputs=x,
            diffusion_model=self,
            noise=epsilon,
            timesteps=timesteps,
            condition=context,
        )
        loss = self.calculate_loss(epsilon_pred, epsilon)
        return loss

    def unpack_batch(self, batch):
        x = batch[self.image_key]
        if self.with_conditioning is True:
            if self.cat_condition_key is not None:
                cat_condition = batch[self.cat_condition_key]
            else:
                cat_condition = None
            if self.num_condition_key is not None:
                num_condition = batch[self.num_condition_key]
            else:
                num_condition = None
            # expects three dimensions (batch, seq, embedding size)
            condition = self.embedder(cat_condition, num_condition)
            condition = condition.unsqueeze(1)
        else:
            condition = None
        return x, condition

    def on_before_batch_transfer(self, batch, dataloader_idx):
        return meta_tensors_to_tensors(batch)

    def training_step(self, batch: dict, batch_idx: int):
        x, condition = self.unpack_batch(batch)
        loss = self.step(x, context=condition)
        self.log("loss", loss, on_step=True, prog_bar=True)
        return loss

    def validation_step(self, batch: dict, batch_idx: int):
        x, condition = self.unpack_batch(batch)
        loss = self.step(x, context=condition)
        self.log("val_loss", loss, on_epoch=True, prog_bar=True)
        return loss

    def test_step(self, batch: dict, batch_idx: int):
        x, condition = self.unpack_batch(batch)
        loss = self.step(x, context=condition)
        self.log("test_loss", loss)
        return loss

    @property
    def device(self):
        return next(self.parameters()).device

    @torch.inference_mode()
    def generate_image(
        self,
        size: List[int],
        n: int,
        input_image: torch.Tensor = None,
        skip_steps: int = 0,
        cat_condition: torch.Tensor = None,
        num_condition: torch.Tensor = None,
    ):
        noise = torch.randn([n, self.in_channels, *size], device=self.device)
        if input_image is None:
            input_image = noise
        else:
            input_image = self.inferer.scheduler.add_noise(
                original_samples=input_image,
                noise=noise[0].to(input_image),
                timesteps=torch.as_tensor(
                    self.scheduler.num_train_timesteps - skip_steps
                ),
            )
        if self.embedder is not None:
            conditioning = self.embedder(
                X_cat=cat_condition,
                X_num=num_condition,
                batch_size=n,
                update_queues=False,
            ).unsqueeze(1)
        else:
            conditioning = None
        sample = self.inferer.sample(
            input_noise=input_image,
            diffusion_model=self,
            scheduler=self.scheduler,
            conditioning=conditioning,
            skip_steps=skip_steps,
        )

        return sample

    def train_dataloader(self) -> torch.utils.data.DataLoader:
        return self.training_dataloader_call(self.batch_size)

    def configure_optimizers(self):
        optimizer = torch.optim.AdamW(
            self.parameters(),
            lr=self.learning_rate,
            weight_decay=self.weight_decay,
        )
        lr_schedulers = CosineAnnealingWithWarmupLR(
            optimizer,
            T_max=self.n_epochs,
            start_decay=self.start_decay,
            n_warmup_steps=self.warmup_steps,
            eta_min=0.0,
        )

        return {
            "optimizer": optimizer,
            "lr_scheduler": lr_schedulers,
            "monitor": "val_loss",
        }
