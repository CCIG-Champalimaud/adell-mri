from datetime import datetime
from pathlib import Path

import monai.data
import monai.transforms
import torch
from lightning.pytorch import Trainer
from lightning.pytorch.callbacks import RichProgressBar

from adell_mri.modules.gan.gan.pl import ProGANPL
from adell_mri.modules.gan.gan.style import (
    ProgressiveDiscriminator,
    ProgressiveGenerator,
)
from adell_mri.utils.pl_utils import get_logger

N_WORKERS = 16
BATCH_SIZE = 16
SUBSET = 25000
IMAGE_SIZE = (128, 128)
max_epochs = 200
TRANSITION_EPOCHS = 10
EPOCHES_PER_LEVEL = 10
DEPTHS = [64, 128, 256, 512, 512, 512]


if __name__ == "__main__":

    path = "/mnt/big_disk/data/celeba/img_align_celeba/img_align_celeba/"
    all_images = [{"image": x} for x in Path(path).rglob("*jpg")]

    transform = monai.transforms.Compose(
        [
            monai.transforms.LoadImaged(keys="image"),
            monai.transforms.EnsureChannelFirstd(keys="image"),
            monai.transforms.Resized(keys="image", spatial_size=IMAGE_SIZE),
            monai.transforms.ScaleIntensityd(keys="image", minv=-1, maxv=1),
        ]
    )

    dataset = monai.data.CacheDataset(
        all_images[:SUBSET], transform=transform, num_workers=N_WORKERS
    )

    data_loader = torch.utils.data.DataLoader(
        dataset,
        batch_size=BATCH_SIZE,
        num_workers=N_WORKERS,
        shuffle=True,
        persistent_workers=True,
    )

    input_channels = 3
    output_channels = DEPTHS[-1]
    n_levels = len(DEPTHS)

    generator = ProgressiveGenerator(
        n_dim=2,
        input_channels=output_channels,
        output_channels=input_channels,
        depths=DEPTHS[::-1],
        equalized_learning_rate=True,
        noise_injection=True,
    )

    discriminator = ProgressiveDiscriminator(
        n_dim=2,
        input_channels=input_channels,
        output_channels=1,
        depths=DEPTHS,
        minibatch_std=True,
        equalized_learning_rate=True,
    )

    steps_per_epoch = len(data_loader)

    pl_progan = ProGANPL(
        generator=generator,
        discriminator=discriminator,
        gradient_penalty_lambda=10.0,
        steps_per_epoch=steps_per_epoch,
        epochs=max_epochs,
        epochs_per_level=EPOCHES_PER_LEVEL,
        transition_epochs=TRANSITION_EPOCHS,
        discriminator_step_every=1,
    )

    logger = get_logger(
        f"progan-{datetime.now().strftime('%Y-%m-%d_%H-%M-%S')}",
        summary_dir="logs",
        project_name="ProGAN-dev",
        resume="none",
        logger_type="wandb",
    )

    if SUBSET > 1000:
        logger_dict = {"logger": logger}
    else:
        logger_dict = {}

    trainer = Trainer(
        max_epochs=max_epochs,
        accelerator="gpu",
        devices=[0],
        log_every_n_steps=10,
        **logger_dict,
        precision="32",
        callbacks=RichProgressBar(),
    )

    trainer.fit(pl_progan, data_loader)
