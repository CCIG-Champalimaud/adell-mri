import os
import sys

sys.path.append(os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
import pytest
import yaml

from adell_mri.modules.config_parsing import parse_config_ae

CONFIG = {
    "model": {
        "spatial_dims": 3,
        "in_channels": 1,
        "out_channels": 1,
        "num_channels": [8, 16, 16],
        "num_res_blocks": 1,
        "attention_levels": [False, False, False],
        "norm_num_groups": 8,
        "no_skip_connection": True,
        "latent_channels": 3,
        "is_variational": True,
    },
    "losses": {
        "weights": {
            "rec": 1.0,
            "kl": 1.0e-7,
            "perceptual": 0.001,
            "adversarial": 0.025,
        },
        "rec_loss": "l1",
        "perceptual_kwargs": {"pretrained": False},
    },
    "discriminator": {"num_channels": 32, "num_layers_d": 2},
    "optimizer": {
        "learning_rate": 5.0e-5,
        "discriminator_learning_rate": 1.0e-4,
    },
    "batch_size": 2,
}


@pytest.fixture
def config_file(tmp_path):
    path = tmp_path / "autoencoder.yaml"
    path.write_text(yaml.safe_dump(CONFIG))
    return str(path)


def test_parse_config_ae(config_file):
    parsed = parse_config_ae(config_file, ["t1"])
    assert parsed["is_variational"] is True
    assert parsed["model"]["in_channels"] == 1
    assert parsed["model"]["out_channels"] == 1
    assert parsed["model"]["latent_channels"] == 3
    assert "is_variational" not in parsed["model"]
    assert parsed["loss_weights"] == CONFIG["losses"]["weights"]
    assert parsed["rec_loss"] == "l1"
    assert parsed["perceptual_kwargs"] == {"pretrained": False}
    assert parsed["discriminator"]["num_layers_d"] == 2
    assert parsed["optimizer"]["discriminator_learning_rate"] == 1.0e-4
    assert parsed["batch_size"] == 2
    assert parsed["spatial_dims"] == 3


def test_parse_config_ae_multichannel_targets(config_file):
    parsed = parse_config_ae(config_file, ["t1", "t2"])
    assert parsed["model"]["in_channels"] == 2
    assert parsed["model"]["out_channels"] == 2


def test_parse_config_ae_spatial_dims_override(config_file):
    parsed = parse_config_ae(config_file, ["t1"], spatial_dims=2)
    assert parsed["model"]["spatial_dims"] == 2
    assert parsed["spatial_dims"] == 2


def test_parse_config_ae_missing_model_section(tmp_path):
    path = tmp_path / "autoencoder.yaml"
    path.write_text(yaml.safe_dump({"batch_size": 1}))
    with pytest.raises(ValueError):
        parse_config_ae(str(path), ["t1"])


def test_parse_config_ae_from_nested_section(tmp_path):
    path = tmp_path / "ldm.yaml"
    path.write_text(
        yaml.safe_dump({"ae": CONFIG, "spatial_dims": 3, "batch_size": 4})
    )
    parsed = parse_config_ae(str(path), ["t1"])
    assert parsed["model"]["latent_channels"] == 3
    assert parsed["batch_size"] == 2

    parsed_from_dict = parse_config_ae({"ae": CONFIG}, ["t1"])
    assert parsed_from_dict["model"]["latent_channels"] == 3


def test_parse_config_ae_optional_sections(tmp_path):
    path = tmp_path / "autoencoder.yaml"
    path.write_text(
        yaml.safe_dump(
            {
                "model": {
                    "num_channels": [8, 16],
                    "num_res_blocks": 1,
                    "attention_levels": [False, False],
                    "norm_num_groups": 8,
                    "no_skip_connection": True,
                }
            }
        )
    )
    parsed = parse_config_ae(str(path), ["t1"])
    assert parsed["is_variational"] is False
    assert parsed["loss_weights"] == {}
    assert parsed["rec_loss"] == "mse"
    assert parsed["discriminator"] is None
    assert parsed["optimizer"] == {}
    assert parsed["batch_size"] == 1
