import os
import sys
from types import SimpleNamespace

sys.path.append(os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
import pytest
import torch

from adell_mri.modules.gan.ae import AutoEncoder
from adell_mri.modules.gan.pl.ae import (
    DEFAULT_LOSS_WEIGHTS,
    AutoEncoderPL,
    VariationalAutoEncoderPL,
)
from adell_mri.modules.gan.vae import VariationalAutoEncoder

MODEL_CLASSES = [AutoEncoder, VariationalAutoEncoder]


@pytest.fixture
def model_config():
    return {
        "spatial_dims": 2,
        "in_channels": 1,
        "out_channels": 1,
        "num_channels": [8, 16, 16],
        "num_res_blocks": 1,
        "attention_levels": (False, False, False),
        "norm_num_groups": 8,
        "no_skip_connection": True,
    }


@pytest.mark.parametrize("model_class", MODEL_CLASSES)
@pytest.mark.parametrize("latent_channels", [None, 3])
def test_encode_decode_roundtrip(model_class, latent_channels, model_config):
    torch.manual_seed(0)
    model = model_class(**model_config, latent_channels=latent_channels)
    x = torch.rand(2, 1, 32, 32)
    output = model(x)
    if model_class is VariationalAutoEncoder:
        output = output[0]
    assert list(output.shape) == list(x.shape)

    latent = model.encode(x)
    expected_channels = (
        latent_channels
        if latent_channels is not None
        else model.block_out_channels[-1]
    )
    assert latent.shape[0] == x.shape[0]
    assert latent.shape[1] == expected_channels
    factor = model.latent_downsample_factor
    factors = (
        (factor,) * len(x.shape[2:]) if isinstance(factor, int) else factor
    )
    assert latent.shape[2:] == tuple(
        size // axis_factor for size, axis_factor in zip(x.shape[2:], factors)
    )

    reconstruction = model.decode(latent)
    assert list(reconstruction.shape) == list(x.shape)


@pytest.mark.parametrize("model_class", MODEL_CLASSES)
def test_encode_is_deterministic(model_class, model_config):
    model = model_class(**model_config, latent_channels=3)
    x = torch.rand(2, 1, 32, 32)
    with torch.inference_mode():
        latent_1 = model.encode(x)
        latent_2 = model.encode(x)
    torch.testing.assert_close(latent_1, latent_2)


@pytest.mark.parametrize("model_class", MODEL_CLASSES)
def test_latent_channels_requires_no_skip(model_class, model_config):
    with pytest.raises(ValueError):
        model_class(
            **{**model_config, "no_skip_connection": False}, latent_channels=3
        )


def test_autoencoder_with_skip_connections(model_config):
    model = AutoEncoder(**{**model_config, "no_skip_connection": False})
    x = torch.rand(2, 1, 32, 32)
    assert list(model(x).shape) == list(x.shape)


@pytest.mark.parametrize("model_class", MODEL_CLASSES)
def test_generate_from_latent(model_class, model_config):
    model = model_class(**model_config, latent_channels=3)
    x = torch.rand(2, 1, 32, 32)
    latent = model.encode(x)
    if model_class is AutoEncoder:
        generated = model.generate(latent)
    else:
        generated = model.generate(tuple(latent.shape))
    assert list(generated.shape) == list(x.shape)


@pytest.mark.parametrize("model_class", MODEL_CLASSES)
def test_loss_terms_are_selected_by_weight(model_class, model_config):
    module_class = (
        AutoEncoderPL
        if model_class is AutoEncoder
        else VariationalAutoEncoderPL
    )
    model = module_class(**model_config, loss_weights={"rec": 1.0})
    x = torch.rand(2, 1, 32, 32)
    losses, _ = model.compute_losses(x, include_adversarial=False)
    assert "rec_loss" in losses
    if model_class is AutoEncoder:
        assert "kl_loss" not in losses
    else:
        assert "kl_loss" in losses
        assert model.loss_weights["kl"] == 0.0
    assert isinstance(losses["loss"], torch.Tensor)
    optimizer_config = model.configure_optimizers()
    if isinstance(optimizer_config, dict):
        optimizer_config = optimizer_config["optimizer"]
    assert len(optimizer_config.param_groups) == 1


@pytest.mark.parametrize("model_class", MODEL_CLASSES)
def test_adversarial_weight_adds_optimizer(model_class, model_config):
    module_class = (
        AutoEncoderPL
        if model_class is AutoEncoder
        else VariationalAutoEncoderPL
    )
    model = module_class(
        **model_config,
        loss_weights={"rec": 1.0, "adversarial": 0.025},
    )
    assert model.adversarial_on is True
    assert model.automatic_optimization is False
    optimizers = model.configure_optimizers()
    assert isinstance(optimizers, tuple) and len(optimizers) == 2

    x = torch.rand(2, 1, 32, 32)
    losses, output = model.compute_losses(x)
    assert "adv_loss" in losses
    recon = model.unpack_output(output)[0]
    disc_losses = model.compute_discriminator_losses(x, recon.detach())
    assert "disc_loss" in disc_losses


@pytest.mark.parametrize("model_class", MODEL_CLASSES)
def test_adversarial_schedule(model_class, model_config):
    module_class = (
        AutoEncoderPL
        if model_class is AutoEncoder
        else VariationalAutoEncoderPL
    )
    model = module_class(
        **model_config,
        loss_weights={"rec": 1.0, "adversarial": 0.025},
        adversarial_start_epoch=5,
        adversarial_warmup_epochs=10,
    )

    expected_scales = {
        4: 0.0,
        5: 0.0,
        10: 0.5,
        15: 1.0,
        20: 1.0,
    }
    for epoch, expected in expected_scales.items():
        model._trainer = SimpleNamespace(current_epoch=epoch)
        assert model.adversarial_scale == pytest.approx(expected)


@pytest.mark.parametrize("model_class", MODEL_CLASSES)
def test_discriminator_spectral_norm(model_class, model_config):
    module_class = (
        AutoEncoderPL
        if model_class is AutoEncoder
        else VariationalAutoEncoderPL
    )
    model = module_class(
        **model_config,
        loss_weights={"rec": 1.0, "adversarial": 0.025},
        spectrally_normalize_discriminator=True,
    )
    x = torch.rand(2, 1, 32, 32)
    losses, output = model.compute_losses(x)
    recon = model.unpack_output(output)[0]
    disc_losses = model.compute_discriminator_losses(x, recon.detach())


@pytest.mark.parametrize("model_class", MODEL_CLASSES)
def test_discriminator_spectral_norm_only_applies_to_conv3d(
    model_class, model_config
):
    module_class = (
        AutoEncoderPL
        if model_class is AutoEncoder
        else VariationalAutoEncoderPL
    )
    model = module_class(
        **{**model_config, "spatial_dims": 3},
        loss_weights={"rec": 1.0, "adversarial": 0.025},
        spectrally_normalize_discriminator=True,
    )
    convolutions = [
        module
        for module in model.discriminator.modules()
        if isinstance(module, (torch.nn.Conv2d, torch.nn.Conv3d))
    ]
    assert len(convolutions) > 0
    assert all(isinstance(module, torch.nn.Conv3d) for module in convolutions)
    assert all("weight" in module.parametrizations for module in convolutions)


def test_adversarial_criterion_defaults_to_bce(model_config):
    model = AutoEncoderPL(
        **model_config,
        loss_weights={"rec": 1.0, "adversarial": 0.025},
    )
    assert model.adversarial_loss.criterion == "bce"


@pytest.mark.parametrize("criterion", ["bce", "hinge"])
def test_adversarial_criterion_is_configurable(model_config, criterion):
    model = AutoEncoderPL(
        **model_config,
        loss_weights={"rec": 1.0, "adversarial": 0.025},
        adversarial_criterion=criterion,
    )
    assert model.adversarial_loss.criterion == criterion


def test_invalid_adversarial_criterion_raises(model_config):
    with pytest.raises(ValueError):
        AutoEncoderPL(
            **model_config,
            loss_weights={"rec": 1.0, "adversarial": 0.025},
            adversarial_criterion="wasserstein",
        )


@pytest.mark.parametrize("model_class", MODEL_CLASSES)
def test_discriminator_update_frequency(model_class, model_config):
    module_class = (
        AutoEncoderPL
        if model_class is AutoEncoder
        else VariationalAutoEncoderPL
    )
    model = module_class(
        **model_config,
        loss_weights={"rec": 1.0, "adversarial": 0.025},
        discriminator_update_every_n_steps=3,
    )

    expected_due = {
        0: True,
        1: False,
        2: False,
        3: True,
        4: False,
    }
    for step, expected in expected_due.items():
        model._trainer = SimpleNamespace(current_epoch=0, global_step=step)
        assert model.discriminator_update_due is expected


def test_negative_adversarial_schedule_raises(model_config):
    with pytest.raises(ValueError):
        AutoEncoderPL(
            **model_config,
            adversarial_start_epoch=-1,
        )
    with pytest.raises(ValueError):
        AutoEncoderPL(
            **model_config,
            adversarial_warmup_epochs=-1,
        )
    with pytest.raises(ValueError):
        AutoEncoderPL(
            **model_config,
            discriminator_update_every_n_steps=0,
        )


@pytest.mark.skipif(
    not torch.cuda.is_available(), reason="CUDA autocast is required"
)
def test_bce_adversarial_loss_is_safe_under_autocast(model_config):
    model = AutoEncoderPL(
        **model_config,
        loss_weights={"rec": 1.0, "adversarial": 0.01},
        adversarial_criterion="bce",
    ).cuda()
    x = torch.rand(2, 1, 32, 32, device="cuda")

    with torch.autocast(device_type="cuda", dtype=torch.float16):
        losses, _ = model.compute_losses(x)
        disc_losses = model.compute_discriminator_losses(
            x, model.forward(x)[0].detach()
        )

    assert torch.isfinite(losses["adv_loss"])
    assert torch.isfinite(disc_losses["disc_loss"])


def test_configure_optimizers_includes_lr_scheduler(model_config):
    model = AutoEncoderPL(
        **model_config,
        loss_weights={"rec": 1.0},
        n_epochs=50,
        warmup_steps=0.1,
        start_decay=10,
    )
    config = model.configure_optimizers()
    assert "optimizer" in config
    assert "lr_scheduler" in config
    assert config["lr_scheduler"].T_max == 50


def test_adversarial_configure_optimizers_returns_both_schedulers(model_config):
    model = AutoEncoderPL(
        **model_config,
        loss_weights={"rec": 1.0, "adversarial": 0.01},
        n_epochs=50,
    )
    optimizers, schedulers = model.configure_optimizers()
    assert len(optimizers) == 2
    assert len(schedulers) == 2


def test_vae_kl_weight_defaults_to_var_loss_mult(model_config):
    model = VariationalAutoEncoderPL(**model_config, var_loss_mult=0.5)
    assert model.loss_weights["kl"] == 0.5
    assert model.loss_weights["rec"] == DEFAULT_LOSS_WEIGHTS["rec"]


def test_unknown_loss_weight_raises(model_config):
    with pytest.raises(ValueError):
        AutoEncoderPL(**model_config, loss_weights={"reconstruction": 1.0})


def test_invalid_reconstruction_loss_raises(model_config):
    with pytest.raises(ValueError):
        AutoEncoderPL(**model_config, rec_loss="l2")
