import numpy as np
import torch

from adell_mri.entrypoints.generative.autoencoder import build_autoencoder
from adell_mri.modules.config_parsing import parse_config_ae
from adell_mri.utils.pl_callbacks import coerce_to_uint8, log_image

AE_MODEL_CONFIG = {
    "spatial_dims": 2,
    "in_channels": 1,
    "out_channels": 1,
    "num_channels": [8, 8],
    "num_res_blocks": 1,
    "attention_levels": [False, False],
    "norm_num_groups": 8,
}


class _FakeLogger:
    def __init__(self):
        self.calls = []

    def log_image(self, **kwargs):
        self.calls.append(kwargs)


class _FakeTrainer:
    def __init__(self):
        self.logger = _FakeLogger()
        self.global_step = 3


def test_coerce_to_uint8_handles_constant_and_non_finite():
    constant = coerce_to_uint8(np.ones((2, 2), dtype=np.float32))
    assert constant.dtype == np.uint8
    assert np.all(constant == 0)

    values = coerce_to_uint8(
        np.array([[-1.0, np.nan], [np.inf, 1.0]], dtype=np.float32)
    )
    assert values.dtype == np.uint8
    assert values.min() == 0
    assert values.max() == 255


def test_log_image_scales_each_channel_independently():
    trainer = _FakeTrainer()
    images = torch.zeros(1, 2, 4, 5)
    images[0, 0] = torch.linspace(10.0, 20.0, 20).reshape(4, 5)
    images[0, 1] = torch.linspace(100.0, 200.0, 20).reshape(4, 5)

    log_image(trainer, "images", images)

    call = trainer.logger.calls[0]
    assert len(call["images"]) == 2
    for image in call["images"]:
        array = np.asarray(image)
        assert image.mode == "L"
        assert array.dtype == np.uint8
        assert array.min() == 0
        assert array.max() == 255


def test_log_image_keeps_rgb_together_and_expands_captions():
    trainer = _FakeTrainer()
    images = torch.rand(2, 3, 4, 5)

    log_image(
        trainer,
        "rgb",
        images,
        caption=["first", "second"],
        rgb=True,
    )

    call = trainer.logger.calls[0]
    assert len(call["images"]) == 2
    assert all(image.mode == "RGB" for image in call["images"])
    assert call["caption"] == ["first", "second"]


def test_log_image_captions_include_channel_index():
    trainer = _FakeTrainer()
    images = torch.rand(1, 2, 4, 5)

    log_image(trainer, "channels", images, caption=["case"])

    call = trainer.logger.calls[0]
    assert call["caption"] == ["case (channel 0)", "case (channel 1)"]


def test_parse_config_ae_defaults_reconstruction_logging():
    parsed = parse_config_ae({"model": AE_MODEL_CONFIG}, ["t1"], spatial_dims=2)

    assert parsed["reconstruction_every_n_epochs"] == 5
    assert parsed["reconstruction_max_batches"] == 1


def test_parse_config_ae_reads_reconstruction_settings():
    config = {
        "model": AE_MODEL_CONFIG,
        "reconstruction_every_n_epochs": 5,
        "reconstruction_max_batches": 2,
    }

    parsed = parse_config_ae(config, ["t1"], spatial_dims=2)

    assert parsed["reconstruction_every_n_epochs"] == 5
    assert parsed["reconstruction_max_batches"] == 2


def test_parse_config_ae_inherits_reconstruction_settings_from_ldm_config():
    config = {
        "ae": {"model": AE_MODEL_CONFIG},
        "reconstruction_every_n_epochs": 5,
        "reconstruction_max_batches": 2,
    }

    parsed = parse_config_ae(config, ["t1"], spatial_dims=2)
    autoencoder = build_autoencoder(parsed)

    assert parsed["reconstruction_every_n_epochs"] == 5
    assert parsed["reconstruction_max_batches"] == 2
    assert autoencoder.reconstruction_every_n_epochs == 5
    assert autoencoder.reconstruction_max_batches == 2
