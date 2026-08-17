import os
import sys

sys.path.append(os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))

import nibabel as nib
import numpy as np
import pytest
import torch
from monai.data import Dataset
from monai.transforms import Compose

from adell_mri.modules.config_parsing import parse_config_gan
from adell_mri.modules.diffusion.inferer import DiffusionInfererSkipSteps
from adell_mri.modules.diffusion.pl import DiffusionUNetPL
from adell_mri.modules.gan.discriminator import Discriminator
from adell_mri.modules.gan.generator import Generator
from adell_mri.modules.gan.pl import GANPL
from adell_mri.transform_factory import GenerationTransforms
from adell_mri.utils.network_factories import get_generative_network


def write_nifti(path, arr):
    nib.save(nib.Nifti1Image(arr.astype(np.float32), np.eye(4)), path)


@pytest.fixture
def nifti_files(tmp_path):
    rng = np.random.default_rng(42)
    files = {}
    files["t1"] = str(tmp_path / "t1.nii.gz")
    files["t2"] = str(tmp_path / "t2.nii.gz")
    files["mask"] = str(tmp_path / "mask.nii.gz")
    write_nifti(files["t1"], rng.random([24, 24, 20]))
    write_nifti(files["t2"], rng.random([24, 24, 20]))
    write_nifti(files["mask"], rng.integers(0, 3, [24, 24, 20]))
    return files


@pytest.mark.parametrize("n_dim", [3])
def test_generation_transforms_conditioning(nifti_files, n_dim):
    data = [
        {
            "t1": nifti_files["t1"],
            "t2": nifti_files["t2"],
            "mask": nifti_files["mask"],
        }
    ]
    factory = GenerationTransforms(
        keys=["t1", "t2"],
        image_keys=["t1"],
        input_image_keys=["t2"],
        input_mask_keys=["mask"],
        mask_classes=[3],
        target_spacing=[1.0, 1.0, 1.0],
        crop_size=[16, 16, 16],
        pad_size=[20, 20, 20],
        n_dim=n_dim,
    )
    transforms = Compose(factory.transforms())
    dataset = Dataset(data, transforms)
    out = dataset[0]

    spatial = [16, 16, 16] if n_dim == 3 else [16, 16]
    assert list(out["image"].shape) == [1, *spatial]
    # 1 input image channel + 3 one-hot mask channels
    assert list(out["cat_conditioning"].shape) == [4, *spatial]
    # one-hot mask channels sum to 1 per voxel
    mask_oh = out["cat_conditioning"][1:]
    assert torch.allclose(mask_oh.sum(0), torch.ones_like(mask_oh[0]))


def test_generation_transforms_no_conditioning(nifti_files):
    factory = GenerationTransforms(
        keys=["t1"],
        image_keys=["t1"],
        crop_size=[16, 16, 16],
        pad_size=[20, 20, 20],
    )
    transforms = Compose(factory.transforms())
    out = transforms({"t1": nifti_files["t1"]})
    assert list(out["image"].shape) == [1, 16, 16, 16]
    assert "cat_conditioning" not in out


def test_parse_config_gan_with_masks(tmp_path):
    config = tmp_path / "gan.yaml"
    config.write_text(
        """
batch_size: 2
learning_rate: 0.0002
generator:
  num_channels: [32]
discriminator:
  structure: [[16, 16, 3, 3]]
"""
    )
    network_config, gen_config, disc_config = parse_config_gan(
        str(config),
        target_keys=["t1"],
        input_keys=["t2"],
        input_mask_keys=["mask"],
        mask_classes=[3],
        spatial_dims=3,
    )
    assert gen_config["in_channels"] == 1 + 3
    assert gen_config["out_channels"] == 1
    assert disc_config["in_channels"] == 1 + 1 + 3

    with pytest.raises(ValueError):
        parse_config_gan(
            str(config),
            target_keys=["t1"],
            input_mask_keys=["mask"],
            mask_classes=[3, 3],
            spatial_dims=3,
        )


def _diffusion_model(in_channels, out_channels, concat_condition_key):
    scheduler = __import__(
        "generative.networks.schedulers", fromlist=["DDPMScheduler"]
    ).DDPMScheduler(num_train_timesteps=10)
    inferer = DiffusionInfererSkipSteps(scheduler)
    return DiffusionUNetPL(
        inferer=inferer,
        scheduler=scheduler,
        embedder=None,
        spatial_dims=2,
        in_channels=in_channels,
        out_channels=out_channels,
        num_res_blocks=1,
        num_channels=[32, 32],
        norm_num_groups=8,
        attention_levels=(False, False),
        with_conditioning=False,
        cross_attention_dim=None,
        concat_condition_key=concat_condition_key,
        n_epochs=10,
    )


def _mixed_diffusion_model(embedder=None):
    scheduler = __import__(
        "generative.networks.schedulers", fromlist=["DDPMScheduler"]
    ).DDPMScheduler(num_train_timesteps=10)
    inferer = DiffusionInfererSkipSteps(scheduler)
    return DiffusionUNetPL(
        inferer=inferer,
        scheduler=scheduler,
        embedder=embedder,
        spatial_dims=2,
        in_channels=3,
        out_channels=1,
        num_res_blocks=1,
        num_channels=[32, 32],
        norm_num_groups=8,
        attention_levels=(True, True),
        with_conditioning=True,
        cross_attention_dim=16,
        concat_condition_key="cat_conditioning",
        n_epochs=10,
    )


def test_diffusionunetpl_concat_training_step():
    model = _diffusion_model(3, 1, "cat_conditioning")
    batch = {
        "image": torch.rand([4, 1, 16, 16]),
        "cat_conditioning": torch.rand([4, 2, 16, 16]),
    }
    x, condition, concat_condition = model.unpack_batch(batch)
    assert concat_condition is not None
    loss = model.step(x, context=condition, concat_condition=concat_condition)
    assert loss.ndim == 0
    assert torch.isfinite(model.training_step(batch, 0))


def test_diffusionunetpl_concat_generate_image():
    model = _diffusion_model(3, 1, "cat_conditioning").eval()
    cond = torch.zeros([2, 2, 16, 16])
    out = model.generate_image(size=[16, 16], n=2, concat_condition=cond)
    assert list(out.shape) == [2, 1, 16, 16]
    # out_channels used for the pure-noise part
    out2 = model.generate_image(size=[16, 16], n=2, concat_condition=cond)
    assert list(out2.shape) == [2, 1, 16, 16]


def test_diffusionunetpl_mixed_conditioning_training_step():
    model = _mixed_diffusion_model()
    x = torch.randn([2, 1, 16, 16])
    concat = torch.rand([2, 2, 16, 16])
    context = torch.rand([2, 1, 16])
    # both concat and cross-attention conditioning at the same time
    loss = model.step(x, context=context, concat_condition=concat)
    assert loss.ndim == 0
    assert torch.isfinite(loss)


def test_diffusionunetpl_crossattn_training_step():
    scheduler = __import__(
        "generative.networks.schedulers", fromlist=["DDPMScheduler"]
    ).DDPMScheduler(num_train_timesteps=10)
    inferer = DiffusionInfererSkipSteps(scheduler)
    model = DiffusionUNetPL(
        inferer=inferer,
        scheduler=scheduler,
        embedder=None,
        spatial_dims=2,
        in_channels=1,
        out_channels=1,
        num_res_blocks=1,
        num_channels=[32, 32],
        norm_num_groups=8,
        attention_levels=(True, True),
        with_conditioning=True,
        cross_attention_dim=16,
        n_epochs=10,
    )
    x = torch.randn([2, 1, 16, 16])
    context = torch.rand([2, 1, 16])
    # cross-attention conditioning on its own
    assert torch.isfinite(model.step(x, context=context))


def test_diffusionunetpl_mixed_conditioning_generate_image():
    from adell_mri.modules.diffusion.embedder import Embedder

    embedder = Embedder([2], embedding_size=16)
    model = _mixed_diffusion_model(embedder=embedder).eval()
    concat = torch.zeros([2, 2, 16, 16])
    cat_condition = torch.zeros(2, 1, dtype=torch.long)
    out = model.generate_image(
        size=[16, 16],
        n=2,
        concat_condition=concat,
        cat_condition=cat_condition,
    )
    assert list(out.shape) == [2, 1, 16, 16]
    assert torch.isfinite(out).all()


def test_diffusioninfererskipsteps_sample_iter():
    model = _mixed_diffusion_model().eval()
    scheduler = model.scheduler
    inferer = model.inferer
    concat = torch.zeros([2, 2, 16, 16])
    context = torch.rand([2, 1, 16])
    steps = list(
        inferer.sample_iter(
            input_noise=torch.randn([2, 1, 16, 16]),
            diffusion_model=model,
            scheduler=scheduler,
            conditioning=context,
            concat_condition=concat,
            verbose=False,
        )
    )
    assert len(steps) == scheduler.num_train_timesteps
    assert all(s.shape == torch.Size([2, 1, 16, 16]) for s in steps)


def test_diffusioninfererskipsteps_mixed_with_guidance():
    model = _mixed_diffusion_model().eval()
    inferer = model.inferer
    concat = torch.zeros([2, 2, 16, 16])
    context = torch.rand([2, 1, 16])
    unconditioning = torch.rand([2, 1, 16])
    out = inferer.sample(
        input_noise=torch.randn([2, 1, 16, 16]),
        diffusion_model=model,
        scheduler=model.scheduler,
        conditioning=context,
        concat_condition=concat,
        unconditioning=unconditioning,
        guidance_strength=1.0,
    )
    assert list(out.shape) == [2, 1, 16, 16]
    assert torch.isfinite(out).all()


def test_generative_network_factory_concat():
    network_config = {
        "spatial_dims": 2,
        "in_channels": 3,
        "out_channels": 1,
        "num_res_blocks": 1,
        "num_channels": [32, 32],
        "norm_num_groups": 8,
        "attention_levels": (False, False),
        "with_conditioning": False,
        "cross_attention_dim": None,
    }
    net = get_generative_network(
        network_config=network_config,
        scheduler_config={
            "schedule": "scaled_linear_beta",
            "beta_start": 0.0005,
            "beta_end": 0.0195,
        },
        categorical_specification=None,
        numerical_specification=None,
        uncondition_proba=0.0,
        train_loader_call=None,
        max_epochs=10,
        warmup_steps=0,
        start_decay=0,
        diffusion_steps=10,
        concat_condition_key="cat_conditioning",
    )
    batch = {
        "image": torch.rand([4, 1, 16, 16]),
        "cat_conditioning": torch.rand([4, 2, 16, 16]),
    }
    assert torch.isfinite(net.training_step(batch, 0))


def test_ganpl_mask_conditional_training_loop():
    cond_channels = 3
    size = [1, 16, 16]
    generator = Generator(
        spatial_dims=2,
        in_channels=cond_channels,
        out_channels=1,
        num_channels=[32, 32],
        num_res_blocks=1,
        attention_levels=(False, False),
    )
    disc = Discriminator(
        "convnext",
        in_channels=1 + cond_channels,
        spatial_dim=2,
        structure=[[16, 16, 3, 3], [32, 32, 3, 3]],
    )

    def dl(batch_size):
        return torch.utils.data.DataLoader(
            [
                {
                    "image": torch.rand(*size),
                    "cat_conditioning": torch.rand(cond_channels, 16, 16),
                }
                for _ in range(4)
            ],
            batch_size=batch_size,
        )

    gan = GANPL(
        generator=generator,
        discriminator=disc,
        real_image_key="image",
        input_image_key="cat_conditioning",
        training_dataloader_call=dl,
        batch_size=2,
    )
    x = torch.rand(2, 1, 16, 16)
    cond = torch.rand(2, cond_channels, 16, 16)
    real, input_tensor = gan.prepare_image_data(
        {"image": x, "cat_conditioning": cond}
    )
    assert torch.equal(real, x)
    assert torch.equal(input_tensor, cond)

    import lightning.pytorch as pl

    trainer = pl.Trainer(
        max_epochs=1,
        limit_train_batches=2,
        accelerator="cpu",
        enable_checkpointing=False,
        logger=False,
        enable_progress_bar=False,
        enable_model_summary=False,
    )
    trainer.fit(gan)
