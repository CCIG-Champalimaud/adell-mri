import os
import sys

sys.path.append(os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
import pytest
import torch
from generative.networks.schedulers.ddpm import DDPMPredictionType

from adell_mri.entrypoints.generative.train_3d import split_records
from adell_mri.modules.diffusion.recipes import (
    DDPM_RECIPE,
    LDM_RECIPE,
    get_recipe,
)
from adell_mri.utils.network_factories import get_generative_network
from adell_mri.utils.torch_utils import load_checkpoint_to_model


def test_diffusion_recipes_encode_model_specific_settings():
    assert DDPM_RECIPE.scheduler_config["prediction_type"] == "epsilon"
    assert DDPM_RECIPE.scheduler_config["clip_sample"] is True
    assert DDPM_RECIPE.min_snr_gamma is None
    assert DDPM_RECIPE.scheduler_config["schedule"] == "linear_beta"

    assert LDM_RECIPE.scheduler_config["prediction_type"] == "v_prediction"
    assert LDM_RECIPE.scheduler_config["clip_sample"] is False
    assert LDM_RECIPE.min_snr_gamma == 5.0
    assert LDM_RECIPE.latent_std_decay == 0.99

    assert get_recipe("ddpm") is DDPM_RECIPE
    assert get_recipe("ldm") is LDM_RECIPE


def test_split_records_by_subject():
    records = [
        {"identifier": f"scan_{i}", "subject": f"subject_{i // 2}"}
        for i in range(8)
    ]
    train, held_out = split_records(records, 0.5, 0)
    assert len(train) == 4
    assert len(held_out) == 4
    train_subjects = {record["subject"] for record in train}
    held_out_subjects = {record["subject"] for record in held_out}
    assert train_subjects.isdisjoint(held_out_subjects)


def test_split_records_without_subjects():
    records = [{"identifier": f"scan_{i}"} for i in range(8)]
    train, held_out = split_records(records, 0.25, 0)
    assert len(train) == 6
    assert len(held_out) == 2


def test_split_records_without_fraction():
    records = [{"identifier": "scan"}]
    train, held_out = split_records(records, None, 0)
    assert train == records
    assert held_out == []


def test_prefer_ema_loads_ema_weights():
    model = torch.nn.Linear(2, 2)
    live = {key: value.clone() for key, value in model.state_dict().items()}
    ema = {key: value + 1.0 for key, value in live.items()}
    checkpoint = {"state_dict": live, "state_dict_ema": ema}

    load_checkpoint_to_model(model, checkpoint, [], prefer_ema=True)
    for key, value in model.state_dict().items():
        torch.testing.assert_close(value, ema[key])


def test_prefer_ema_falls_back_to_live_weights():
    model = torch.nn.Linear(2, 2)
    live = {key: value.clone() for key, value in model.state_dict().items()}
    load_checkpoint_to_model(model, {"state_dict": live}, [], prefer_ema=True)
    for key, value in model.state_dict().items():
        torch.testing.assert_close(value, live[key])


@pytest.fixture
def diffusion_network():
    network_config = {
        "spatial_dims": 2,
        "in_channels": 1,
        "out_channels": 1,
        "num_res_blocks": 1,
        "num_channels": [8, 8],
        "norm_num_groups": 8,
        "attention_levels": (False, False),
        "with_conditioning": False,
        "cross_attention_dim": None,
        "reconstruction_weight": 0.5,
        "reconstruction_loss": "l1",
    }
    return get_generative_network(
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
        max_epochs=1,
        warmup_steps=0,
        start_decay=0,
        diffusion_steps=10,
    )


def test_reconstruction_loss_is_computed(diffusion_network):
    x = torch.rand(2, 1, 16, 16)
    loss, reconstruction_loss = diffusion_network.step(x, target_images=x)
    assert torch.isfinite(loss)
    assert reconstruction_loss is not None
    assert torch.isfinite(reconstruction_loss)


def test_prediction_target_and_x0_roundtrip(diffusion_network):
    x_0 = torch.rand(2, 1, 16, 16)
    timesteps = torch.tensor([1, 2])
    noise = torch.randn_like(x_0)
    x_t = diffusion_network.scheduler.add_noise(
        original_samples=x_0, noise=noise, timesteps=timesteps
    )
    diffusion_network.scheduler.clip_sample = False

    for prediction_type in [
        DDPMPredictionType.EPSILON,
        DDPMPredictionType.V_PREDICTION,
        DDPMPredictionType.SAMPLE,
    ]:
        diffusion_network.scheduler.prediction_type = prediction_type
        target = diffusion_network.prediction_target(x_0, timesteps, noise)
        predicted_x0 = diffusion_network.predict_x0(x_t, timesteps, target)
        torch.testing.assert_close(predicted_x0, x_0, rtol=1e-5, atol=1e-5)


def test_min_snr_weights_are_finite(diffusion_network):
    x = torch.rand(2, 1, 16, 16)
    timesteps = torch.tensor([1, 8])
    for prediction_type in [
        DDPMPredictionType.EPSILON,
        DDPMPredictionType.V_PREDICTION,
        DDPMPredictionType.SAMPLE,
    ]:
        diffusion_network.scheduler.prediction_type = prediction_type
        diffusion_network.min_snr_gamma = 5.0
        weights = diffusion_network.min_snr_weight(x, timesteps)
        assert list(weights.shape) == [2]
        assert torch.isfinite(weights).all()
        assert (weights > 0).all()


def test_clip_sample_is_respected(diffusion_network):
    x = torch.rand(2, 1, 16, 16)
    timesteps = torch.tensor([1, 2])
    model_output = torch.full_like(x, 1000.0)
    diffusion_network.scheduler.prediction_type = DDPMPredictionType.EPSILON

    diffusion_network.scheduler.clip_sample = True
    clipped = diffusion_network.predict_x0(x, timesteps, model_output)
    assert float(clipped.min()) >= -1.0
    assert float(clipped.max()) <= 1.0

    diffusion_network.scheduler.clip_sample = False
    unclipped = diffusion_network.predict_x0(x, timesteps, model_output)
    assert (unclipped < -1.0).any()


def test_latent_std_is_updated_in_training_mode(diffusion_network):
    x = torch.rand(2, 1, 16, 16)
    diffusion_network.train()
    diffusion_network.input_encoder = lambda images: images
    diffusion_network.encode_latents(x)
    assert diffusion_network.latent_std_initialized.item() == 1.0
    first_std = diffusion_network.latent_std.clone()
    assert torch.isfinite(first_std) and first_std > 0.0

    diffusion_network.encode_latents(torch.rand(2, 1, 16, 16) * 3.0)
    assert not torch.equal(first_std, diffusion_network.latent_std)

    diffusion_network.eval()
    eval_std = diffusion_network.latent_std.clone()
    diffusion_network.encode_latents(x)
    torch.testing.assert_close(eval_std, diffusion_network.latent_std)
