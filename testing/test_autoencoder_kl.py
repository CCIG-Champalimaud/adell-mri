import os
import sys

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
    assert latent.shape[2:] == tuple(
        size // (2**model.latent_downsample_factor) for size in x.shape[2:]
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
    assert len(model.configure_optimizers().param_groups) == 1


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
