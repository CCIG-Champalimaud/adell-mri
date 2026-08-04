import json
import logging
import os
import shutil
import tempfile

import numpy as np
import pytest
import SimpleITK as sitk
import torch

from adell_mri.entrypoints.classification import (
    predict as classification_predict,
)
from adell_mri.entrypoints.classification_ensemble import (
    predict as ensemble_predict,
)
from adell_mri.entrypoints.classification_mil import predict as mil_predict
from adell_mri.modules.classification.pl import GenericEnsemblePL
from adell_mri.utils.network_factories import (
    get_classification_network,
    get_deconfounded_classification_network,
)

CAT_CONFIG = (
    "spatial_dimensions: 3\n"
    "resnet_structure: [[1,4,[3,3,3],1]]\n"
    "maxpool_structure: []\n"
    "learning_rate: 0.001\n"
    "batch_size: 1\n"
    "weight_decay: 0.0\n"
    "res_type: convnext\n"
)

ENSEMBLE_CONFIG = (
    "spatial_dimensions: 3\n"
    "n_features: 1\n"
    "head_structure: [1]\n"
    "head_adn_fn:\n"
    "  norm_fn: identity\n"
    "  act_fn: gelu\n"
    "  dropout_param: 0.0\n"
    "learning_rate: 0.001\n"
    "batch_size: 1\n"
    "weight_decay: 0.0\n"
)

MIL_CONFIG = (
    "learning_rate: 0.001\n"
    "batch_size: 1\n"
    "weight_decay: 0.0\n"
    "feat_extraction_structure: [8]\n"
    "classification_structure: [4]\n"
    "classification_mode: mean\n"
    "module_out_dim: 4\n"
)

PREDICT_LOGGERS = [
    "adell_mri.entrypoints.classification.predict",
    "adell_mri.entrypoints.classification_ensemble.predict",
    "adell_mri.entrypoints.classification_mil.predict",
]


@pytest.fixture(scope="module")
def predict_setup():
    workdir = tempfile.mkdtemp(prefix="adell_predict_test_")
    image_path = os.path.join(workdir, "image.nii.gz")
    array = np.random.RandomState(0).rand(16, 16, 8).astype(np.float32)
    sitk.WriteImage(sitk.GetImageFromArray(array), image_path)
    dataset_json_path = os.path.join(workdir, "dataset.json")
    with open(dataset_json_path, "w") as handle:
        json.dump({"case0": {"image": image_path, "conf_var": "A"}}, handle)
    cat_config_path = os.path.join(workdir, "config.yaml")
    with open(cat_config_path, "w") as handle:
        handle.write(CAT_CONFIG)
    ensemble_config_path = os.path.join(workdir, "ensemble_config.yaml")
    with open(ensemble_config_path, "w") as handle:
        handle.write(ENSEMBLE_CONFIG)
    mil_config_path = os.path.join(workdir, "mil_config.yaml")
    with open(mil_config_path, "w") as handle:
        handle.write(MIL_CONFIG)
    module_path = os.path.join(workdir, "module.pt")
    module = torch.nn.Sequential(
        torch.nn.Conv2d(1, 8, 3, padding=1),
        torch.nn.Conv2d(8, 4, 3, padding=1),
    ).eval()
    scripted = torch.jit.freeze(torch.jit.trace(module, torch.rand(1, 1, 8, 8)))
    scripted.save(module_path)
    setup = {
        "workdir": workdir,
        "dataset_json": dataset_json_path,
        "config_file": cat_config_path,
        "ensemble_config_file": ensemble_config_path,
        "mil_config_file": mil_config_path,
        "module_path": module_path,
        "image_key": "image",
        "case_id": "case0",
        "confounder_key": "conf_var",
    }
    yield setup
    shutil.rmtree(workdir, ignore_errors=True)


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


def _build_deconfounder_checkpoint(setup):
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
    network = get_deconfounded_classification_network(
        network_config=network_config,
        dropout_param=0,
        seed=None,
        cat_confounder_key=None,
        cont_confounder_key=None,
        cat_vars=None,
        cont_vars=None,
        n_classes=2,
        keys=["image"],
        train_loader_call=None,
        max_epochs=None,
        warmup_steps=None,
        start_decay=None,
        n_features_deconfounder=64,
        exclude_surrogate_variables=False,
    )
    checkpoint_path = os.path.join(setup["workdir"], "deconf_checkpoint.pt")
    torch.save({"state_dict": network.state_dict()}, checkpoint_path)
    return checkpoint_path


def _build_ensemble_checkpoint(setup):
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
    inner = get_classification_network(
        net_type="cat",
        network_config=network_config,
        dropout_param=0,
        seed=42,
        n_classes=2,
        keys=["image"],
        clinical_feature_keys=[],
        train_loader_call=None,
        max_epochs=None,
        warmup_steps=None,
        start_decay=None,
        crop_size=[8, 8, 8],
    )
    ensemble = GenericEnsemblePL(
        image_keys=["image"],
        label_key="label",
        networks=[inner],
        n_classes=2,
        training_dataloader_call=None,
        n_epochs=None,
        warmup_steps=None,
        start_decay=None,
        spatial_dimensions=3,
        n_features=1,
        head_structure=[1],
        head_adn_fn=torch.nn.Identity,
        learning_rate=0.001,
        batch_size=1,
        weight_decay=0.0,
        loss_fn=torch.nn.BCEWithLogitsLoss(),
    )
    checkpoint_path = os.path.join(setup["workdir"], "ens_checkpoint.pt")
    torch.save({"state_dict": ensemble.state_dict()}, checkpoint_path)
    return checkpoint_path


def _assert_test_mode_warning(caplog, logger_name):
    assert any(
        "test mode" in record.message and record.levelno == logging.WARNING
        for record in caplog.records
        if record.name == logger_name
    )


def _load_output(path):
    with open(path) as handle:
        return json.load(handle)


def _cat_base_args(setup, output_path):
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


def _deconfounder_base_args(setup, output_path):
    return [
        "--net_type",
        "cat",
        "--dataset_json",
        setup["dataset_json"],
        "--image_keys",
        setup["image_key"],
        "--cat_confounder_keys",
        setup["confounder_key"],
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


def _ensemble_base_args(setup, output_path):
    return [
        "--dataset_json",
        setup["dataset_json"],
        "--image_keys",
        setup["image_key"],
        "--n_classes",
        "2",
        "--config_files",
        setup["config_file"],
        "--ensemble_config_file",
        setup["ensemble_config_file"],
        "--net_types",
        "cat",
        "--dev",
        "cpu",
        "--crop_size",
        "8",
        "8",
        "8",
        "--output_path",
        output_path,
    ]


def _mil_base_args(setup, output_path):
    return [
        "--dataset_json",
        setup["dataset_json"],
        "--image_keys",
        setup["image_key"],
        "--n_classes",
        "2",
        "--config_file",
        setup["mil_config_file"],
        "--mil_method",
        "standard",
        "--module_path",
        setup["module_path"],
        "--dev",
        "cpu",
        "--crop_size",
        "8",
        "8",
        "8",
        "--resize_size",
        "8",
        "8",
        "8",
        "--output_path",
        output_path,
    ]


class TestClassificationPredict:
    LOGGER = "adell_mri.entrypoints.classification.predict"

    def test_no_checkpoint_warns(self, predict_setup, caplog):
        output_path = os.path.join(
            predict_setup["workdir"], "cls_out_probability.json"
        )
        with caplog.at_level(logging.WARNING, logger=self.LOGGER):
            classification_predict.main(
                _cat_base_args(predict_setup, output_path)
            )
        assert os.path.exists(output_path)
        output = _load_output(output_path)
        assert len(output) == 1
        entry = output[0]
        assert entry["iteration"] == 0
        assert entry["prediction_ids"] == [predict_setup["case_id"]]
        assert entry["checkpoint"] is None
        assert predict_setup["case_id"] in entry["prediction"]
        assert isinstance(entry["prediction"][predict_setup["case_id"]], float)
        _assert_test_mode_warning(caplog, self.LOGGER)

    def test_explicit_checkpoint(self, predict_setup):
        checkpoint_path = _build_checkpoint(predict_setup)
        output_path = os.path.join(
            predict_setup["workdir"], "cls_out_checkpoint.json"
        )
        classification_predict.main(
            _cat_base_args(predict_setup, output_path)
            + ["--checkpoints", checkpoint_path]
        )
        assert os.path.exists(output_path)
        entry = _load_output(output_path)[0]
        assert entry["checkpoint"] == checkpoint_path
        assert isinstance(entry["prediction"][predict_setup["case_id"]], float)

    def test_logit_type(self, predict_setup):
        output_path = os.path.join(
            predict_setup["workdir"], "cls_out_logit.json"
        )
        classification_predict.main(
            _cat_base_args(predict_setup, output_path) + ["--type", "logit"]
        )
        assert os.path.exists(output_path)
        entry = _load_output(output_path)[0]
        assert isinstance(entry["prediction"][predict_setup["case_id"]], float)

    def test_prediction_ids(self, predict_setup):
        output_path = os.path.join(predict_setup["workdir"], "cls_out_ids.json")
        classification_predict.main(
            _cat_base_args(predict_setup, output_path)
            + ["--prediction_ids", predict_setup["case_id"]]
        )
        assert os.path.exists(output_path)
        entry = _load_output(output_path)[0]
        assert entry["prediction_ids"] == [predict_setup["case_id"]]

    def test_pre_bias_cat_warns(self, predict_setup, caplog):
        output_path = os.path.join(
            predict_setup["workdir"], "cls_out_pre_bias.json"
        )
        with caplog.at_level(logging.WARNING, logger=self.LOGGER):
            classification_predict.main(
                _cat_base_args(predict_setup, output_path)
                + ["--type", "pre_bias"]
            )
        assert os.path.exists(output_path)
        assert any(
            "pre_bias" in record.message and record.levelno == logging.WARNING
            for record in caplog.records
            if record.name == self.LOGGER
        )


class TestDeconfounderPredict:
    LOGGER = "adell_mri.entrypoints.classification.predict"

    def test_no_checkpoint_warns(self, predict_setup, caplog):
        output_path = os.path.join(predict_setup["workdir"], "deconf_out.json")
        with caplog.at_level(logging.WARNING, logger=self.LOGGER):
            classification_predict.main(
                _deconfounder_base_args(predict_setup, output_path)
            )
        assert os.path.exists(output_path)
        output = _load_output(output_path)
        assert len(output) == 1
        entry = output[0]
        assert entry["checkpoint"] is None
        assert predict_setup["case_id"] in entry["prediction"]
        assert isinstance(entry["prediction"][predict_setup["case_id"]], float)
        _assert_test_mode_warning(caplog, self.LOGGER)

    def test_explicit_checkpoint(self, predict_setup):
        checkpoint_path = _build_deconfounder_checkpoint(predict_setup)
        output_path = os.path.join(
            predict_setup["workdir"], "deconf_out_ckpt.json"
        )
        classification_predict.main(
            _deconfounder_base_args(predict_setup, output_path)
            + ["--checkpoints", checkpoint_path]
        )
        assert os.path.exists(output_path)
        entry = _load_output(output_path)[0]
        assert entry["checkpoint"] == checkpoint_path
        assert isinstance(entry["prediction"][predict_setup["case_id"]], float)


class TestEnsemblePredict:
    LOGGER = "adell_mri.entrypoints.classification_ensemble.predict"

    def test_no_checkpoint_warns(self, predict_setup, caplog):
        output_path = os.path.join(predict_setup["workdir"], "ens_out.json")
        with caplog.at_level(logging.WARNING, logger=self.LOGGER):
            ensemble_predict.main(
                _ensemble_base_args(predict_setup, output_path)
            )
        assert os.path.exists(output_path)
        output = _load_output(output_path)
        assert len(output) == 1
        entry = output[0]
        assert entry["checkpoint"] is None
        assert predict_setup["case_id"] in entry["prediction"]
        assert isinstance(entry["prediction"][predict_setup["case_id"]], float)
        _assert_test_mode_warning(caplog, self.LOGGER)

    def test_explicit_checkpoint(self, predict_setup):
        checkpoint_path = _build_ensemble_checkpoint(predict_setup)
        output_path = os.path.join(
            predict_setup["workdir"], "ens_out_ckpt.json"
        )
        ensemble_predict.main(
            _ensemble_base_args(predict_setup, output_path)
            + ["--checkpoints", checkpoint_path]
        )
        assert os.path.exists(output_path)
        entry = _load_output(output_path)[0]
        assert entry["checkpoint"] == checkpoint_path
        assert isinstance(entry["prediction"][predict_setup["case_id"]], float)


class TestMilPredict:
    LOGGER = "adell_mri.entrypoints.classification_mil.predict"

    def test_no_checkpoint_warns(self, predict_setup, caplog):
        output_path = os.path.join(predict_setup["workdir"], "mil_out.json")
        with caplog.at_level(logging.WARNING, logger=self.LOGGER):
            mil_predict.main(_mil_base_args(predict_setup, output_path))
        assert os.path.exists(output_path)
        output = _load_output(output_path)
        assert len(output) == 1
        entry = output[0]
        assert entry["checkpoint"] is None
        assert predict_setup["case_id"] in entry["prediction"]
        assert isinstance(entry["prediction"][predict_setup["case_id"]], float)
        _assert_test_mode_warning(caplog, self.LOGGER)

    def test_explicit_checkpoint(self, predict_setup):
        output_path = os.path.join(
            predict_setup["workdir"], "mil_out_ckpt.json"
        )
        mil_predict.main(_mil_base_args(predict_setup, output_path))
        assert os.path.exists(output_path)
        entry = _load_output(output_path)[0]
        assert entry["checkpoint"] is None
        assert isinstance(entry["prediction"][predict_setup["case_id"]], float)
