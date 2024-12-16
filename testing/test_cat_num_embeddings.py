import os
import sys

sys.path.append(os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
import pytest

import torch
from adell_mri.modules.diffusion.embedder import Embedder

cat_feature_specification = [["1", "2"], ["A", "B"]]
num_features = 4
n_features = 128


def test_cat_embedding():
    embedder = Embedder(
        cat_feat=cat_feature_specification, embedding_size=n_features
    )
    assert list(embedder.forward(X_cat=[["1", "B"], ["2", "A"]]).shape) == [
        2,
        1,
        n_features,
    ]


@pytest.mark.parametrize("batch_size", [1, 2])
def test_num_embedding(batch_size):
    embedder = Embedder(
        n_num_feat=4,
        numerical_moments=[torch.rand(4), torch.rand(4)],
        embedding_size=n_features,
    )
    assert list(embedder.forward(X_num=torch.rand(batch_size, 4)).shape) == [
        batch_size,
        1,
        n_features,
    ]


def test_cat_num_embedding():
    embedder = Embedder(
        cat_feat=cat_feature_specification,
        n_num_feat=4,
        numerical_moments=[torch.rand(4), torch.rand(4)],
        embedding_size=n_features,
    )
    assert list(
        embedder.forward(
            X_cat=[["1", "B"], ["2", "A"]], X_num=torch.rand(2, 4)
        ).shape
    ) == [
        2,
        1,
        n_features,
    ]


def test_cat_num_embedding():
    embedder = Embedder(
        cat_feat=cat_feature_specification,
        n_num_feat=4,
        numerical_moments=[torch.rand(4), torch.rand(4)],
        embedding_size=n_features,
    )
    assert list(
        embedder.forward(
            X_cat=[["1", "B"], ["2", "A"]], X_num=torch.rand(2, 4)
        ).shape
    ) == [
        2,
        1,
        n_features,
    ]
