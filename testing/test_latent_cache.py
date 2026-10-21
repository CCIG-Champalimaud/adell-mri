import os
import sys

sys.path.append(os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
import nibabel as nib
import numpy as np
import pytest
import torch
from monai.transforms import Compose

from adell_mri.modules.gan.ae import AutoEncoder
from adell_mri.transform_factory import GenerationTransforms
from adell_mri.utils.latents import (
    cache_latents,
    compute_latent_statistics,
    infer_latent_shape,
)

MODEL_CONFIG = {
    "spatial_dims": 2,
    "in_channels": 1,
    "out_channels": 1,
    "num_channels": [8, 16, 16],
    "num_res_blocks": 1,
    "attention_levels": (False, False, False),
    "norm_num_groups": 8,
    "no_skip_connection": True,
    "latent_channels": 3,
}


@pytest.fixture
def autoencoder():
    return AutoEncoder(**MODEL_CONFIG)


@pytest.fixture
def latent_file(tmp_path):
    path = tmp_path / "latent.npy"
    np.save(path, np.random.rand(3, 8, 8).astype(np.float32))
    return str(path)


@pytest.fixture
def image_and_latent_files(tmp_path, latent_file):
    image = np.zeros([16, 16, 1], dtype=np.float32)
    image[4:12, 4:12, 0] = 10.0
    image_path = tmp_path / "image.nii.gz"
    nib.save(nib.Nifti1Image(image, np.eye(4)), str(image_path))
    return str(image_path), latent_file


def test_latent_only_transforms(latent_file):
    factory = GenerationTransforms(
        keys=("latent",),
        latent_keys=("latent",),
        latent_only=True,
        n_dim=2,
    )
    output = Compose(factory.transforms())({"latent": latent_file})
    assert "latent" in output
    assert list(output["image"].shape) == [3, 8, 8]


def test_latents_are_not_intensity_scaled(image_and_latent_files):
    image_path, latent_path = image_and_latent_files
    factory = GenerationTransforms(
        keys=("t1",),
        latent_keys=("latent",),
        n_dim=3,
    )
    output = Compose(factory.transforms())(
        {"t1": image_path, "latent": latent_path}
    )
    assert float(output["latent"].min()) >= 0.0
    assert float(output["latent"].max()) <= 1.0
    assert float(output["image"].max()) == pytest.approx(1.0)
    assert float(output["image"].min()) == pytest.approx(-1.0)


def test_latent_scale_is_applied(latent_file):
    source = np.load(latent_file)
    factory = GenerationTransforms(
        keys=("latent",),
        latent_keys=("latent",),
        latent_only=True,
        latent_scale=2.0,
        n_dim=2,
    )
    output = Compose(factory.transforms())({"latent": latent_file})
    np.testing.assert_allclose(output["image"].numpy(), source * 2.0)


def test_latent_only_requires_latent_keys():
    with pytest.raises(ValueError):
        GenerationTransforms(keys=("image",), latent_only=True)


def test_infer_latent_shape(autoencoder):
    assert infer_latent_shape(autoencoder, [32, 32]) == (3, 8, 8)


def test_cache_latents(autoencoder, tmp_path):
    autoencoder.eval()
    records = [
        {
            "image": torch.rand(1, 32, 32),
            "identifier": f"case_{i}",
        }
        for i in range(2)
    ]
    output = cache_latents(
        autoencoder,
        records,
        transforms=None,
        output_dir=str(tmp_path / "latents"),
        batch_size=2,
    )
    assert len(output) == 2
    for record, source in zip(output, records):
        assert os.path.exists(record["latent"])
        latent = np.load(record["latent"])
        with torch.inference_mode():
            expected = autoencoder.encode(source["image"][None])[0].numpy()
        np.testing.assert_allclose(latent, expected, rtol=1e-4, atol=1e-4)


def test_cache_latents_is_deterministic(autoencoder, tmp_path):
    records = [{"image": torch.rand(1, 32, 32), "identifier": "case"}]
    output_1 = cache_latents(
        autoencoder, records, None, str(tmp_path / "a"), overwrite=True
    )
    output_2 = cache_latents(
        autoencoder, records, None, str(tmp_path / "b"), overwrite=True
    )
    np.testing.assert_allclose(
        np.load(output_1[0]["latent"]),
        np.load(output_2[0]["latent"]),
        rtol=1e-6,
        atol=1e-6,
    )


def test_compute_latent_statistics(tmp_path):
    latents = [np.random.rand(3, 4, 4).astype(np.float32) for _ in range(2)]
    paths = []
    for i, latent in enumerate(latents):
        path = tmp_path / f"latent_{i}.npy"
        np.save(path, latent)
        paths.append(str(path))
    stats = compute_latent_statistics(paths, n_channels=3)
    stacked = np.stack(latents)
    assert stats["mean"] == pytest.approx(
        float(stacked.mean(axis=(0, 2, 3)).mean()), rel=1e-5
    )
    assert stats["std"] == pytest.approx(
        float(stacked.std(axis=(0, 2, 3)).mean()), rel=1e-5
    )
    assert stats["scale"] == pytest.approx(1.0 / stats["std"], rel=1e-5)


def test_compute_latent_statistics_rejects_empty():
    with pytest.raises(ValueError):
        compute_latent_statistics([])
