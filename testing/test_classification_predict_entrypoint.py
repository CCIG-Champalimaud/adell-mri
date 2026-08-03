import json
import logging
import os
import shutil
import tempfile

import numpy as np
import pytest
import SimpleITK as sitk
import torch

from adell_mri.entrypoints.classification import predict
from adell_mri.utils.network_factories import get_classification_network


@pytest.fixture(scope="module")
def predict_setup():
    workdir = tempfile.mkdtemp(prefix="adell_predict_test_")
    image_path = os.path.join(workdir, "image.nii.gz")
    array = np.random.RandomState(0).rand(16, 16, 8).astype(np.float32)
    sitk.WriteImage(sitk.GetImageFromArray(array), image_path)
    dataset_json_path = os.path.join(workdir, "dataset.json")
    with open(dataset_json_path, "w") as handle:
        json.dump({"case0": {"image": image_path}}, handle)
    config_path = os.path.join(workdir, "config.yaml")
    with open(config_path, "w") as handle:
        handle.write(
            "spatial_dimensions: 3\n"
            "resnet_structure: [[1,4,[3,3,3],1]]\n"
            "maxpool_structure: []\n"
            "learning_rate: 0.001\n"
            "batch_size: 1\n"
            "weight_decay: 0.0\n"
            "res_type: convnext\n"
        )
    setup = {
        "workdir": workdir,
        "dataset_json": dataset_json_path,
        "config_file": config_path,
        "image_key": "image",
        "case_id": "case0",
    }
    yield setup
    shutil.rmtree(workdir, ignore_errors=True)


def _base_args(setup, output_path):
    return [
        "--net_type",
        "cat",
        "--dataset_json",
        setup["dataset_json"],
        "--image_keys",
        setup["image_key"],
        "--n_classes",
        "2",
        "--config_file",
        setup["config_file"],
        "--dev",
        "cpu",
        "--crop_size",
        "8",
        "8",
        "8",
        "--output_path",
        output_path,
    ]


def _build_checkpoint(setup):
    network_config = {
        "spatial_dimensions": 3,
        "resnet_structure": [[1, 4, [3, 3, 3], 1]],
        "maxpool_structure": [],
        "learning_rate": 0.001,
        "batch_size": 1,
        "weight_decay": 0.0,
        "res_type": "convnext",
        "loss_fn": torch.nn.BCEWithLogitsLoss(),
    }
    network = get_classification_network(
        net_type="cat",
        network_config=network_config,
        dropout_param=0,
        seed=None,
        n_classes=2,
        keys=["image"],
        clinical_feature_keys=[],
        train_loader_call=None,
        max_epochs=None,
        warmup_steps=None,
        start_decay=None,
        crop_size=[8, 8, 8],
    )
    checkpoint_path = os.path.join(setup["workdir"], "checkpoint.pt")
    torch.save({"state_dict": network.state_dict()}, checkpoint_path)
    return checkpoint_path


def test_predict_probability_no_checkpoint_warns(predict_setup, caplog):
    output_path = os.path.join(predict_setup["workdir"], "out_probability.json")
    with caplog.at_level(
        logging.WARNING, logger="adell_mri.entrypoints.classification.predict"
    ):
        predict.main(_base_args(predict_setup, output_path))
    assert os.path.exists(output_path)
    with open(output_path) as handle:
        output = json.load(handle)
    assert len(output) == 1
    entry = output[0]
    assert entry["iteration"] == 0
    assert entry["prediction_ids"] == [predict_setup["case_id"]]
    assert entry["checkpoint"] is None
    assert predict_setup["case_id"] in entry["prediction"]
    assert isinstance(entry["prediction"][predict_setup["case_id"]], float)
    assert any(
        "test mode" in record.message and record.levelno == logging.WARNING
        for record in caplog.records
        if record.name == "adell_mri.entrypoints.classification.predict"
    )


def test_predict_with_explicit_checkpoint(predict_setup):
    checkpoint_path = _build_checkpoint(predict_setup)
    output_path = os.path.join(predict_setup["workdir"], "out_checkpoint.json")
    predict.main(
        _base_args(predict_setup, output_path)
        + ["--checkpoints", checkpoint_path]
    )
    assert os.path.exists(output_path)
    with open(output_path) as handle:
        output = json.load(handle)
    entry = output[0]
    assert entry["checkpoint"] == checkpoint_path
    assert isinstance(entry["prediction"][predict_setup["case_id"]], float)


def test_predict_logit_type(predict_setup):
    output_path = os.path.join(predict_setup["workdir"], "out_logit.json")
    predict.main(_base_args(predict_setup, output_path) + ["--type", "logit"])
    assert os.path.exists(output_path)
    with open(output_path) as handle:
        output = json.load(handle)
    entry = output[0]
    assert isinstance(entry["prediction"][predict_setup["case_id"]], float)


def test_predict_with_prediction_ids(predict_setup):
    output_path = os.path.join(predict_setup["workdir"], "out_ids.json")
    predict.main(
        _base_args(predict_setup, output_path)
        + ["--prediction_ids", predict_setup["case_id"]]
    )
    assert os.path.exists(output_path)
    with open(output_path) as handle:
        output = json.load(handle)
    entry = output[0]
    assert entry["prediction_ids"] == [predict_setup["case_id"]]


def test_predict_pre_bias_cat_warns(predict_setup, caplog):
    output_path = os.path.join(predict_setup["workdir"], "out_pre_bias.json")
    with caplog.at_level(
        logging.WARNING, logger="adell_mri.entrypoints.classification.predict"
    ):
        predict.main(
            _base_args(predict_setup, output_path) + ["--type", "pre_bias"]
        )
    assert os.path.exists(output_path)
    assert any(
        "pre_bias" in record.message and record.levelno == logging.WARNING
        for record in caplog.records
        if record.name == "adell_mri.entrypoints.classification.predict"
    )
