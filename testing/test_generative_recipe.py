import os
import sys

sys.path.append(os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
import pytest
import torch

from adell_mri.entrypoints.generative.train_3d import split_records
from adell_mri.utils.network_factories import get_generative_network
from adell_mri.utils.torch_utils import load_checkpoint_to_model


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
