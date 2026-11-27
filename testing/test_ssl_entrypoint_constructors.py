"""Integration tests for SSL initialisation in the downstream entrypoints."""

import json
import os
from types import SimpleNamespace

import numpy as np
import pytest
import SimpleITK as sitk
import torch

from adell_mri.entrypoints.classification import train as classification_train
from adell_mri.entrypoints.segmentation import predict as segmentation_predict
from adell_mri.entrypoints.segmentation import train as segmentation_train
from adell_mri.modules.segmentation.pl import EncoderUNetPL
from testing.fixtures import resnet3d_checkpoint

CLASSIFICATION_CONFIG = """
spatial_dimensions: 3
resnet_structure: [[8, 8, 3, 1]]
maxpool_structure: [[2, 2, 2]]
classification_structure: [16, 8]
learning_rate: 0.001
batch_size: 1
weight_decay: 0.0
res_type: resnet
"""

SEGMENTATION_CONFIG = """
spatial_dimensions: 3
conv_type: regular
link_type: identity
upscale_type: transpose
norm_type: batch
interpolation: bilinear
padding: 1
dropout_param: 0.0
activation_fn: relu
in_channels: 1
depth: [8, 16, 32]
kernel_sizes: [3, 3, 3]
strides: [2, 2, 2]
learning_rate: 0.001
batch_size: 1
weight_decay: 0.0
loss_fn:
  dice: {}
"""


class DummyTrainer:
    """Minimal Trainer replacement that captures the constructed model."""

    models = []

    def __init__(self, *args, **kwargs):
        pass

    def fit(self, model, *args, **kwargs):
        type(self).models.append(model)

    def test(self, model, *args, **kwargs):
        return [{}]

    @property
    def test_loop(self):
        return SimpleNamespace(_results={})


@pytest.fixture
def entrypoint_setup(tmp_path):
    """
    Creates a tiny dataset, configurations and a 3D SSL checkpoint.

    Args:
        tmp_path: pytest temporary directory.

    Returns:
        dict: paths and identifiers used by the entrypoint tests.
    """
    rng = np.random.RandomState(0)
    dataset = {}
    for index in range(4):
        image = rng.rand(16, 16, 8).astype(np.float32)
        mask = np.zeros((16, 16, 8), dtype=np.uint8)
        mask[2:6, 2:6, 2:6] = 1
        image_path = tmp_path / f"image_{index}.nii.gz"
        mask_path = tmp_path / f"mask_{index}.nii.gz"
        sitk.WriteImage(sitk.GetImageFromArray(image), str(image_path))
        sitk.WriteImage(sitk.GetImageFromArray(mask), str(mask_path))
        dataset[f"case{index}"] = {
            "image": str(image_path),
            "mask": str(mask_path),
            "label": index % 2,
        }

    dataset_path = tmp_path / "dataset.json"
    with open(dataset_path, "w") as handle:
        json.dump(dataset, handle)

    classification_config = tmp_path / "classification.yaml"
    classification_config.write_text(CLASSIFICATION_CONFIG)
    segmentation_config = tmp_path / "segmentation.yaml"
    segmentation_config.write_text(SEGMENTATION_CONFIG)

    ssl_checkpoint = tmp_path / "ssl.ckpt"
    torch.save(resnet3d_checkpoint(), ssl_checkpoint)

    return {
        "dataset": str(dataset_path),
        "classification_config": str(classification_config),
        "segmentation_config": str(segmentation_config),
        "ssl_checkpoint": str(ssl_checkpoint),
        "metric_path": str(tmp_path / "metrics.csv"),
        "output_path": str(tmp_path / "prediction.nii.gz"),
        "predict_output_path": str(tmp_path / "prediction.json"),
    }


def _common_data_args(setup):
    return [
        "--dataset_json",
        setup["dataset"],
        "--image_keys",
        "image",
        "--possible_labels",
        "0",
        "1",
        "--positive_labels",
        "1",
        "--target_spacing",
        "1",
        "1",
        "1",
        "--crop_size",
        "8",
        "8",
        "8",
        "--dev",
        "cpu",
        "--n_workers",
        "1",
        "--n_folds",
        "1",
        "--max_epochs",
        "1",
        "--steps_per_epoch",
        "1",
        "--batch_size",
        "1",
    ]


def _predict_data_args(setup):
    return [
        "--dataset_json",
        setup["dataset"],
        "--image_keys",
        "image",
        "--possible_labels",
        "0",
        "1",
        "--positive_labels",
        "1",
        "--target_spacing",
        "1",
        "1",
        "1",
        "--crop_size",
        "8",
        "8",
        "8",
        "--dev",
        "cpu",
        "--n_workers",
        "0",
    ]


def test_classification_train_constructs_from_ssl(
    entrypoint_setup, monkeypatch
):
    DummyTrainer.models = []
    monkeypatch.setattr(classification_train, "Trainer", DummyTrainer)
    monkeypatch.setattr(
        classification_train, "get_logger", lambda **kwargs: False
    )

    arguments = [
        "--net_type",
        "cat",
        "--config_file",
        entrypoint_setup["classification_config"],
        "--label_keys",
        "label",
        "--encoder_checkpoint",
        entrypoint_setup["ssl_checkpoint"],
        "--metric_path",
        entrypoint_setup["metric_path"],
    ] + _common_data_args(entrypoint_setup)
    classification_train.main(arguments)

    assert len(DummyTrainer.models) == 1
    network = DummyTrainer.models[0]
    assert hasattr(network, "lr_encoder")
    ssl_state = torch.load(
        entrypoint_setup["ssl_checkpoint"], weights_only=False
    )["state_dict"]
    assert torch.allclose(
        network.network.feature_extraction.input_layer[0].weight,
        ssl_state["backbone.input_layer.0.weight"],
    )


def test_segmentation_train_constructs_from_ssl(entrypoint_setup, monkeypatch):
    DummyTrainer.models = []
    monkeypatch.setattr(segmentation_train, "Trainer", DummyTrainer)
    monkeypatch.setattr(
        segmentation_train, "get_logger", lambda **kwargs: False
    )

    arguments = [
        "--net_type",
        "encoder_unet",
        "--config_file",
        entrypoint_setup["segmentation_config"],
        "--mask_keys",
        "mask",
        "--encoder_checkpoint",
        entrypoint_setup["ssl_checkpoint"],
        "--metric_path",
        entrypoint_setup["metric_path"],
        "--optimizer",
        "adamw",
    ] + _common_data_args(entrypoint_setup)
    segmentation_train.main(arguments)

    assert len(DummyTrainer.models) == 1
    network = DummyTrainer.models[0]
    assert isinstance(network, EncoderUNetPL)
    ssl_state = torch.load(
        entrypoint_setup["ssl_checkpoint"], weights_only=False
    )["state_dict"]
    assert torch.allclose(
        network.encoder.input_layer[0].weight,
        ssl_state["backbone.input_layer.0.weight"],
    )


def test_segmentation_predict_rebuilds_encoder_from_checkpoint(
    entrypoint_setup, monkeypatch
):
    DummyTrainer.models = []
    monkeypatch.setattr(segmentation_train, "Trainer", DummyTrainer)
    monkeypatch.setattr(
        segmentation_train, "get_logger", lambda **kwargs: False
    )

    train_arguments = [
        "--net_type",
        "encoder_unet",
        "--config_file",
        entrypoint_setup["segmentation_config"],
        "--mask_keys",
        "mask",
        "--encoder_checkpoint",
        entrypoint_setup["ssl_checkpoint"],
        "--metric_path",
        entrypoint_setup["metric_path"],
        "--optimizer",
        "adamw",
    ] + _common_data_args(entrypoint_setup)
    segmentation_train.main(train_arguments)
    network = DummyTrainer.models[0]

    trained_checkpoint = entrypoint_setup["output_path"] + ".ckpt"
    torch.save(
        {
            "state_dict": network.state_dict(),
            "encoder_config": network.encoder_config,
        },
        trained_checkpoint,
    )

    predict_arguments = [
        "--net_type",
        "encoder_unet",
        "--config_file",
        entrypoint_setup["segmentation_config"],
        "--checkpoint",
        trained_checkpoint,
        "--prediction_ids",
        "missing_case",
        "--prediction_mode",
        "deep_features",
        "--output_path",
        entrypoint_setup["predict_output_path"],
    ] + _predict_data_args(entrypoint_setup)
    segmentation_predict.main(predict_arguments)

    assert os.path.exists(entrypoint_setup["predict_output_path"])
