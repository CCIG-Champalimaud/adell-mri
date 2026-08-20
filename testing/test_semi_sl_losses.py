import os
import sys

sys.path.append(os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))

import pytest
import torch

from adell_mri.modules.semi_supervised_segmentation.losses import (
    AnatomicalContrastiveLoss,
    LocalContrastiveLoss,
    NearestNeighbourLoss,
    PseudoLabelCrossEntropy,
    derangement,
)

bs, c, h, w, nc = 2, 1, 32, 32, 5
n_features = 128
trials = 1


def test_derangement():
    for n in range(2, 10):
        permutation = derangement(n, seed=0)
        assert sorted(permutation) == list(range(n))
        assert all(
            permutation[i] != i for i in range(n)
        ), "derangement must not contain fixed points"


def test_derangement_requires_at_least_two_elements():
    with pytest.raises(ValueError):
        derangement(1)
    with pytest.raises(ValueError):
        derangement(0)


def test_anatomical_contrastive_buffers():
    acl = AnatomicalContrastiveLoss(
        n_classes=nc, n_features=n_features, batch_size=bs, top_k=100
    )
    for name in [
        "average_representations",
        "hard_examples",
        "hard_example_class",
    ]:
        assert name in dict(acl.named_buffers()), f"{name} must be a buffer"


def test_anatomical_contrastive_small_crop():
    acl = AnatomicalContrastiveLoss(
        n_classes=nc, n_features=n_features, batch_size=bs, top_k=100
    )
    pred = torch.rand([bs, nc, 2, 2])
    features = torch.randn([bs, n_features, 2, 2])
    labels = (torch.rand([bs, nc, 2, 2]) > 0.5).float()
    result = acl.forward(pred, labels, features)
    assert torch.isnan(result) == False


def test_nearest_neighbour_empty_queue():
    nnl = NearestNeighbourLoss(
        100,
        n_classes=nc,
        max_elements_per_batch=100,
        n_samples_per_class=10,
    )
    past_samples, past_sample_labels = nnl.get_past_samples("cpu")
    assert past_samples.shape[0] == 0
    assert past_sample_labels.shape[0] == 0
    features = torch.rand([bs, n_features, h, w])
    y = (torch.rand([bs, nc, h, w]) > 0.5).float()
    result = nnl.forward(features, y)
    assert torch.isnan(result) == False


def test_nearest_neighbour_partially_filled_queue():
    nnl = NearestNeighbourLoss(
        100,
        n_classes=nc,
        max_elements_per_batch=100,
        n_samples_per_class=10,
    )
    for _ in range(2):
        y = (torch.rand([bs, nc, h, w]) > 0.5).float()
        features = torch.rand([bs, n_features, h, w])
        nnl.put(features, y)
    y = (torch.rand([bs, nc, h, w]) > 0.5).float()
    features = torch.rand([bs, n_features, h, w])
    result = nnl.forward(features, y)
    assert torch.isnan(result) == False


def test_pseudo_label_cross_entropy_loss():
    plce = PseudoLabelCrossEntropy(threshold=0.5)

    pred = torch.randn([bs, nc, h, w])
    pseudo_labels = torch.rand([bs, nc, h, w])

    result = plce.forward(pred, pseudo_labels)
    assert len(result.shape) == 0
    assert torch.isnan(result) == False


def test_anatomical_contrastive_loss():
    for _ in range(trials):
        acl = AnatomicalContrastiveLoss(
            n_classes=nc, n_features=n_features, batch_size=bs, top_k=100
        )

        pred = torch.rand([bs, nc, h, w])
        features = torch.randn([bs, n_features, h, w])
        labels = (torch.rand([bs, nc, h, w]) > 0.5).float()

        result = acl.forward(pred, labels, features)

        assert len(result.shape) == 0
        assert torch.isnan(result) == False


def test_nearest_neighbour_loss():
    for _ in range(trials):
        nnl = NearestNeighbourLoss(
            100,
            n_classes=nc,
            max_elements_per_batch=100,
            n_samples_per_class=10,
        )

        def get_y_features():
            y = (torch.rand([bs, nc, h, w]) > 0.5).float()
            features = torch.rand([bs, n_features, h, w])
            return y, features

        for _ in range(4):
            y, features = get_y_features()
            nnl.put(features, y)

        y, features = get_y_features()
        result = nnl.forward(features, y)

        assert len(result.shape) == 0
        assert torch.isnan(result) == False


def test_local_contrastive_loss():
    for _ in range(trials):
        lcl = LocalContrastiveLoss()

        def get_features():
            features = torch.rand([bs, n_features, h, w])
            return features

        features_1 = get_features()
        features_2 = get_features()
        result = lcl.forward(features_1, features_2)

        assert len(result.shape) == 1
        assert torch.all(torch.isnan(result) == False)
