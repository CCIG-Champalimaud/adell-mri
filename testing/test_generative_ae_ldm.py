import json
import os
import sys

sys.path.append(os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
import nibabel as nib
import numpy as np
import pytest
import torch
import yaml

from adell_mri.entrypoints.generative.generate import main as generate_main
from adell_mri.entrypoints.generative.train_3d import main as train_main

AE_CONFIG = {
    "model": {
        "spatial_dims": 3,
        "in_channels": 1,
        "out_channels": 1,
        "num_channels": [8, 16],
        "num_res_blocks": 1,
        "attention_levels": [False, False],
        "norm_num_groups": 8,
        "no_skip_connection": True,
        "latent_channels": 3,
        "is_variational": True,
    },
    "losses": {
        "weights": {
            "rec": 1.0,
            "kl": 1.0e-7,
            "perceptual": 0.0,
            "adversarial": 0.0,
        },
        "rec_loss": "l1",
    },
    "optimizer": {"learning_rate": 1.0e-4},
    "batch_size": 2,
}

LDM_CONFIG = {
    "spatial_dims": 3,
    "in_channels": 3,
    "out_channels": 3,
    "num_channels": [8, 16],
    "num_res_blocks": 1,
    "attention_levels": [False, False],
    "norm_num_groups": 8,
    "with_conditioning": False,
    "cross_attention_dim": None,
    "latent_std_decay": 0.9,
    "min_snr_gamma": 3.0,
    "scheduler": {
        "prediction_type": "epsilon",
        "clip_sample": True,
    },
    "diffusion_steps": 4,
    "reconstruction_loss": "l1",
    "reconstruction_every_n_epochs": 1,
    "reconstruction_max_batches": 1,
    "reconstruction_timestep": 2,
    "batch_size": 2,
    "learning_rate": 1.0e-3,
    "ae": AE_CONFIG,
}


@pytest.fixture
def dataset(tmp_path):
    images = tmp_path / "images"
    images.mkdir()
    records = {}
    rng = np.random.default_rng(0)
    for i in range(4):
        path = images / f"case_{i}.nii.gz"
        array = rng.random((16, 16, 16)).astype(np.float32)
        nib.save(nib.Nifti1Image(array, np.eye(4)), str(path))
        records[f"case_{i}"] = {"t1": str(path), "age": float(i)}
    dataset_path = tmp_path / "dataset.json"
    dataset_path.write_text(json.dumps(records))
    ae_path = tmp_path / "ae.yaml"
    ae_path.write_text(yaml.safe_dump(AE_CONFIG))
    ldm_path = tmp_path / "ldm.yaml"
    ldm_path.write_text(yaml.safe_dump(LDM_CONFIG))
    return {
        "dataset": str(dataset_path),
        "images": str(images),
        "ae_config": str(ae_path),
        "ldm_config": str(ldm_path),
        "root": str(tmp_path),
    }


def base_arguments(config, dataset):
    return [
        "--dataset_json",
        dataset["dataset"],
        "--image_keys",
        "t1",
        "--spatial_dims",
        "3",
        "--dev",
        "cpu",
        "--n_workers",
        "0",
        "--precision",
        "32",
        "--max_epochs",
        "1",
        "--check_val_every_n_epoch",
        "1",
        "--seed",
        "0",
        "--config_file",
        config,
    ]


def test_ae_ldm_training_and_generation(dataset, tmp_path):
    checkpoint_dir = str(tmp_path / "checkpoints")
    os.makedirs(checkpoint_dir)

    train_main(
        [
            *base_arguments(dataset["ae_config"], dataset),
            "--model_type",
            "ae",
            "--checkpoint_dir",
            checkpoint_dir,
            "--checkpoint_name",
            "smoke",
            "--ae_checkpoint_name",
            "smoke_ae_custom",
            "--metric_path",
            str(tmp_path / "ae_metrics.csv"),
        ]
    )
    ae_checkpoint = os.path.join(checkpoint_dir, "smoke_ae_custom_last.ckpt")
    assert os.path.exists(ae_checkpoint)
    assert not os.path.exists(
        os.path.join(checkpoint_dir, "smoke_ae_last.ckpt")
    )

    train_main(
        [
            *base_arguments(dataset["ldm_config"], dataset),
            "--model_type",
            "ldm",
            "--checkpoint_dir",
            checkpoint_dir,
            "--checkpoint_name",
            "smoke",
            "--ae_checkpoint_name",
            "smoke_ae_custom",
            "--ae_max_epochs",
            "1",
            "--diffusion_steps",
            "4",
            "--batch_size",
            "2",
            "--metric_path",
            str(tmp_path / "ldm_metrics.csv"),
        ]
    )
    ldm_checkpoint = os.path.join(checkpoint_dir, "smoke_last.ckpt")
    assert os.path.exists(ldm_checkpoint)
    assert not os.path.exists(os.path.join(checkpoint_dir, "smoke_latents"))

    output_dir = str(tmp_path / "generated")
    generate_main(
        [
            "--dataset_json",
            dataset["dataset"],
            "--image_keys",
            "t1",
            "--config_file",
            dataset["ldm_config"],
            "--checkpoint",
            ldm_checkpoint,
            "--output_path",
            output_dir,
            "--dev",
            "cpu",
            "--n_workers",
            "0",
            "--precision",
            "32",
            "--diffusion_steps",
            "4",
            "--batch_size",
            "1",
            "--skip_steps",
            "2",
        ]
    )
    generated = sorted(
        name for name in os.listdir(output_dir) if name.endswith("_gen.mha")
    )
    assert len(generated) == 4


def test_ldm_generates_samples_without_dataset(dataset, tmp_path):
    checkpoint_dir = str(tmp_path / "checkpoints")
    os.makedirs(checkpoint_dir)
    train_main(
        [
            *base_arguments(dataset["ae_config"], dataset),
            "--model_type",
            "ae",
            "--checkpoint_dir",
            checkpoint_dir,
            "--checkpoint_name",
            "smoke",
            "--metric_path",
            str(tmp_path / "ae_metrics.csv"),
        ]
    )
    train_main(
        [
            *base_arguments(dataset["ldm_config"], dataset),
            "--model_type",
            "ldm",
            "--checkpoint_dir",
            checkpoint_dir,
            "--checkpoint_name",
            "smoke",
            "--ae_max_epochs",
            "1",
            "--diffusion_steps",
            "4",
            "--batch_size",
            "2",
            "--metric_path",
            str(tmp_path / "ldm_metrics.csv"),
        ]
    )
    output_dir = str(tmp_path / "samples")
    generate_main(
        [
            "--checkpoint",
            os.path.join(checkpoint_dir, "smoke_last.ckpt"),
            "--image_keys",
            "t1",
            "--output_path",
            output_dir,
            "--dev",
            "cpu",
            "--n_workers",
            "0",
            "--precision",
            "32",
            "--diffusion_steps",
            "4",
            "--n_samples_gen",
            "1",
        ]
    )
    assert len(os.listdir(output_dir)) == 1
