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


@pytest.mark.parametrize(
    "uncondition_cat_idx,uncondition_num_idx",
    [["all", [1]], [[1], "all"], ["all", "all"]],
)
def test_cat_num_embedding_with_unconditioning(
    uncondition_cat_idx, uncondition_num_idx
):
    embedder = Embedder(
        cat_feat=cat_feature_specification,
        n_num_feat=4,
        numerical_moments=[torch.rand(4), torch.rand(4)],
        embedding_size=n_features,
    )
    assert list(
        embedder.forward(
            X_cat=[["1", "B"], ["2", "A"]],
            X_num=torch.rand(2, 4),
            uncondition_cat_idx=uncondition_cat_idx,
            uncondition_num_idx=uncondition_num_idx,
        ).shape
    ) == [
        2,
        1,
        n_features,
    ]


def _make_embedder():
    return Embedder(
        cat_feat=cat_feature_specification,
        n_num_feat=4,
        numerical_moments=[torch.rand(4), torch.rand(4)],
        embedding_size=n_features,
    )


def _feature_indices(uncondition_idx, n):
    if uncondition_idx is None:
        return []
    if uncondition_idx == "all":
        return range(n)
    if isinstance(uncondition_idx, int):
        return [uncondition_idx]
    return uncondition_idx


@pytest.mark.parametrize("uncondition_idx", [None, 0, [1], "all"])
def test_embed_categorical_mask_blend(uncondition_idx):
    embedder = _make_embedder()
    X_cat = [["1", "B"], ["2", "A"]]
    D = embedder.embedding_size
    K = len(embedder.cat_feat)
    cat_raw, _ = embedder.cat_embedder(X_cat, return_X=True)
    uncond_vec = embedder.unconditional_like(cat_raw)
    mask = torch.ones(K * D)
    for idx in _feature_indices(uncondition_idx, K):
        mask[idx * D : (idx + 1) * D] = 0.0
    expected = mask * cat_raw + (1 - mask) * uncond_vec[:, None].repeat(1, 1, K)
    out, _ = embedder.embed_categorical(X_cat, uncondition_idx=uncondition_idx)
    torch.testing.assert_close(out, expected)


@pytest.mark.parametrize("uncondition_idx", [None, 0, [2], "all"])
def test_embed_numerical_mask_blend(uncondition_idx):
    embedder = _make_embedder()
    X_num = torch.rand(4, 4)
    X_norm = embedder.normalize_numeric_features(X_num)
    num_raw = torch.stack(
        [
            embedder.num_embedder[i](X_norm[:, i].unsqueeze(1))
            for i in range(embedder.n_num_feat)
        ],
        0,
    )
    L = num_raw.shape[0]
    uncond_vec = embedder.unconditional_like(X_num)
    mask = torch.ones(L)
    for idx in _feature_indices(uncondition_idx, L):
        mask[idx] = 0.0
    expected = (
        mask[:, None, None] * num_raw
        + (1 - mask[:, None, None]) * uncond_vec.repeat(L, 1, 1)
    ).sum(0)
    out, _ = embedder.embed_numerical(X_num, uncondition_idx=uncondition_idx)
    torch.testing.assert_close(out, expected)


def test_forward_unconditioned_all_is_pure_unconditional():
    embedder = _make_embedder()
    X_cat = [["1", "B"], ["2", "A"]]
    X_num = torch.rand(2, 4)
    K = len(embedder.cat_feat)
    L = embedder.n_num_feat
    uncond_vec = embedder.unconditional_like(torch.zeros(2))
    cat_uncond = uncond_vec[:, None].repeat(1, 1, K)
    num_uncond = uncond_vec.repeat(L, 1, 1).sum(0)[:, None, :]
    expected = embedder.final_embedding(torch.cat([cat_uncond, num_uncond], -1))
    out = embedder.forward(
        X_cat=X_cat,
        X_num=X_num,
        uncondition_cat_idx="all",
        uncondition_num_idx="all",
    )
    torch.testing.assert_close(out, expected)


def test_unconditioned_forward_independent_of_conditions():
    embedder = _make_embedder()
    u1 = embedder.forward(
        X_cat=[["1", "B"], ["2", "A"]],
        X_num=torch.rand(2, 4),
        uncondition_cat_idx="all",
        uncondition_num_idx="all",
    )
    u2 = embedder.forward(
        X_cat=[["2", "A"], ["1", "B"]],
        X_num=torch.rand(2, 4),
        uncondition_cat_idx="all",
        uncondition_num_idx="all",
    )
    torch.testing.assert_close(u1, u2)
    conditioned = embedder.forward(
        X_cat=[["1", "B"], ["2", "A"]], X_num=torch.rand(2, 4)
    )
    assert not torch.allclose(conditioned, u1)


def test_unconditioned_embeddings_used_in_backward():
    embedder = _make_embedder()
    out = embedder.forward(
        X_cat=[["1", "B"], ["2", "A"]], X_num=torch.rand(2, 4)
    )
    out.sum().backward()
    assert embedder.unconditioned_embeddings.weight.grad is not None
