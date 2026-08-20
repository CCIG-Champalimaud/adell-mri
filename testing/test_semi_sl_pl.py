import os
import sys
from unittest.mock import patch

sys.path.append(os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))

import torch
import torch.nn.functional as F

from adell_mri.modules.semi_supervised_segmentation.pl import (
    UNetContrastiveSemiSL,
)


def make_model(label_key="mask", **kwargs):
    return UNetContrastiveSemiSL(
        spatial_dimensions=2,
        in_channels=1,
        n_classes=2,
        depth=[4, 8, 16],
        kernel_sizes=[3, 3, 3],
        strides=[2, 2, 2],
        image_key="image",
        label_key=label_key,
        batch_size=2,
        n_epochs=1,
        **kwargs,
    )


def loss_fn(pred, y):
    return [F.binary_cross_entropy(pred, y)]


def test_forward():
    model = make_model()
    x = torch.rand(2, 1, 32, 32)
    pred, bn_out = model(x)
    assert pred.shape == torch.Size([2, 1, 32, 32])
    assert bn_out is None


def test_forward_features():
    model = make_model()
    x = torch.rand(2, 1, 32, 32)
    feats = model.forward_features(X=x)
    assert feats.shape == torch.Size([2, 4, 32, 32])


def test_forward_features_with_linear_transformation():
    model = make_model()
    x = torch.rand(2, 1, 32, 32)
    feats = model.forward_features(X=x, apply_linear_transformation=True)
    assert feats.shape == torch.Size([2, 4, 32, 32])


def test_unpack_batch_semi_sl():
    model = make_model()
    batch = {
        "self_supervised": {
            "semi_sl_image_1": torch.rand(2, 1, 32, 32),
            "semi_sl_image_2": torch.rand(2, 1, 32, 32),
        }
    }
    x_1, x_2, x_cond, x_fc = model.unpack_batch_semi_sl(batch)
    assert x_1.shape == x_2.shape == torch.Size([2, 1, 32, 32])
    assert x_cond is None
    assert x_fc is None


@patch("lightning.pytorch.core.module.LightningModule.log")
def test_training_step_label_free(mock_log):
    model = make_model(label_key=None)
    batch = {
        "self_supervised": {
            "semi_sl_image_1": torch.rand(2, 1, 32, 32),
            "semi_sl_image_2": torch.rand(2, 1, 32, 32),
        }
    }
    loss = model.training_step(batch, 0)
    assert loss.ndim == 0
    assert torch.isfinite(loss)


@patch("lightning.pytorch.core.module.LightningModule.log")
def test_training_step_supervised(mock_log):
    model = make_model(loss_fn=loss_fn)
    batch = {
        "supervised": {
            "image": torch.rand(2, 1, 32, 32),
            "mask": (torch.rand(2, 1, 32, 32) > 0.5).float(),
        },
        "self_supervised": {
            "semi_sl_image_1": torch.rand(2, 1, 32, 32),
            "semi_sl_image_2": torch.rand(2, 1, 32, 32),
        },
    }
    loss = model.training_step(batch, 0)
    assert loss.ndim == 0
    assert torch.isfinite(loss)


@patch("lightning.pytorch.core.module.LightningModule.log")
def test_validation_step_label_free(mock_log):
    model = make_model(label_key=None)
    batch = {
        "self_supervised": {
            "semi_sl_image_1": torch.rand(2, 1, 32, 32),
            "semi_sl_image_2": torch.rand(2, 1, 32, 32),
        }
    }
    loss = model.validation_step(batch, 0)
    assert loss.ndim == 0
    assert torch.isfinite(loss)


@patch("lightning.pytorch.core.module.LightningModule.log")
def test_test_step_label_free_returns_zero(mock_log):
    model = make_model(label_key=None)
    batch = {
        "self_supervised": {
            "semi_sl_image_1": torch.rand(2, 1, 32, 32),
            "semi_sl_image_2": torch.rand(2, 1, 32, 32),
        }
    }
    loss = model.test_step(batch, 0)
    assert loss.item() == 0.0
