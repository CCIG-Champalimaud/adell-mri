import os
import sys

sys.path.append(os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))

import numpy as np
import pytest
import torch
from monai.data import MetaTensor

from adell_mri.utils.inference import (
    cat_array,
    get_example,
    get_shape,
    make_meta,
    multi_format_cat,
    multi_format_stack,
    multi_format_stack_or_cat,
    stack_array,
)


def test_get_example_tensor():
    X = torch.rand(2, 3)
    assert torch.equal(get_example(X), X)


def test_get_example_array():
    X = np.random.rand(2, 3)
    assert np.array_equal(get_example(X), X)


def test_get_example_dict():
    X = {"a": torch.zeros(1), "b": torch.ones(1)}
    assert torch.equal(get_example(X), torch.zeros(1))


def test_get_example_sequence():
    X = [torch.zeros(1), torch.ones(1)]
    assert torch.equal(get_example(X), torch.zeros(1))


def test_get_example_unsupported():
    with pytest.raises(NotImplementedError):
        get_example(42)


def test_get_shape_nested_list_of_tensors():
    X = [torch.zeros(2, 3), torch.zeros(2, 3)]
    assert get_shape(X) == (2, 3)


def test_get_shape_dict():
    X = {"a": torch.zeros(2, 3)}
    assert get_shape(X) == (2, 3)


def test_cat_array_torch_and_numpy():
    t = cat_array([torch.zeros(1, 2), torch.ones(1, 2)], dim=0)
    assert isinstance(t, torch.Tensor) and t.shape == (2, 2)
    a = cat_array([np.zeros((1, 2)), np.ones((1, 2))], axis=0)
    assert isinstance(a, np.ndarray) and a.shape == (2, 2)


def test_stack_array_torch_and_numpy():
    t = stack_array([torch.zeros(2), torch.ones(2)], dim=0)
    assert isinstance(t, torch.Tensor) and t.shape == (2, 2)
    a = stack_array([np.zeros(2), np.ones(2)])
    assert isinstance(a, np.ndarray) and a.shape == (2, 2)


def test_multi_format_cat_tensors():
    out = multi_format_cat([torch.zeros(1, 2), torch.ones(1, 2)])
    assert out.shape == (2, 2)


def test_multi_format_cat_dict():
    inputs = [
        {"image": torch.zeros(1, 2), "label": torch.zeros(1)},
        {"image": torch.ones(1, 2), "label": torch.ones(1)},
    ]
    out = multi_format_cat(inputs)
    assert set(out.keys()) == {"image", "label"}
    assert out["image"].shape == (2, 2)
    assert out["label"].shape == (2,)
    assert torch.equal(out["image"][0], torch.zeros(2))
    assert torch.equal(out["image"][1], torch.ones(2))


def test_multi_format_cat_list():
    inputs = [
        [torch.zeros(1, 2), torch.full((1,), 5.0)],
        [torch.ones(1, 2), torch.full((1,), 6.0)],
    ]
    out = multi_format_cat(inputs)
    assert len(out) == 2
    assert out[0].shape == (2, 2)
    assert out[1].shape == (2,)
    assert torch.equal(out[1], torch.tensor([5.0, 6.0]))


def test_multi_format_stack_tensors():
    out = multi_format_stack([torch.zeros(2), torch.ones(2)])
    assert out.shape == (2, 2)


def test_multi_format_stack_dict():
    inputs = [
        {"image": torch.zeros(1, 2)},
        {"image": torch.ones(1, 2)},
    ]
    out = multi_format_stack(inputs)
    assert out["image"].shape == (2, 1, 2)


def test_multi_format_stack_or_cat():
    tensors = [torch.zeros(1, 2), torch.ones(1, 2)]
    catted = multi_format_stack_or_cat(tensors, ndim=0)
    assert catted.shape == (2, 2)
    stacked = multi_format_stack_or_cat(tensors, ndim=1)
    assert stacked.shape == (2, 1, 2)


def test_make_meta_from_meta_source():
    source = MetaTensor(torch.zeros(1, 2, 2))
    source.meta["some_key"] = "some_value"
    out = make_meta(torch.ones(1, 2, 2), source)
    assert isinstance(out, MetaTensor)
    assert out.meta["some_key"] == "some_value"
    assert torch.equal(out, torch.ones(1, 2, 2))


def test_make_meta_plain_passthrough():
    X = torch.ones(1, 2, 2)
    out = make_meta(X, torch.zeros(1, 2, 2))
    assert not isinstance(out, MetaTensor)
    assert torch.equal(out, X)
