import os
import sys

sys.path.append(os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))

import pytest

from adell_mri.utils.dataset_filters import (
    fill_conditional,
    fill_missing_with_value,
    filter_dictionary,
    filter_dictionary_with_filters,
    filter_dictionary_with_possible_labels,
    filter_dictionary_with_presence,
)

DATASET = {
    "patient_1": {"age": 50, "class": "A", "path": "/tmp/a.nii"},
    "patient_2": {"age": 70, "class": "B", "path": "/tmp/b.nii"},
    "patient_3": {"age": 30, "class": "A", "path": "/tmp/c.nii"},
}


def test_filter_eq():
    out = filter_dictionary_with_filters(DATASET, ["class=A"])
    assert set(out.keys()) == {"patient_1", "patient_3"}


def test_filter_eq_on_list_field():
    D = {
        "p1": {"tags": ["x", "y"]},
        "p2": {"tags": ["z"]},
    }
    out = filter_dictionary_with_filters(D, ["tags=x"])
    assert set(out.keys()) == {"p1"}


def test_filter_neq():
    out = filter_dictionary_with_filters(DATASET, ["class!=A"])
    assert set(out.keys()) == {"patient_2"}


def test_filter_gt():
    out = filter_dictionary_with_filters(DATASET, ["age>40"])
    assert set(out.keys()) == {"patient_1", "patient_2"}


def test_filter_lt():
    out = filter_dictionary_with_filters(DATASET, ["age<40"])
    assert set(out.keys()) == {"patient_3"}


def test_filter_in():
    out = filter_dictionary_with_filters(DATASET, ["class(in)A,B"])
    assert len(out) == 3
    out = filter_dictionary_with_filters(DATASET, ["class(in)B"])
    assert set(out.keys()) == {"patient_2"}


def test_filter_match():
    out = filter_dictionary_with_filters(DATASET, ["path(match)nii"])
    assert len(out) == 3


def test_filter_not_match():
    out = filter_dictionary_with_filters(DATASET, ["path(!match)a"])
    assert set(out.keys()) == {"patient_2", "patient_3"}


def test_filter_multiple_combined():
    out = filter_dictionary_with_filters(DATASET, ["class=A", "age>40"])
    assert set(out.keys()) == {"patient_1"}


def test_filter_missing_key_excludes_entries_unless_optional():
    out = filter_dictionary_with_filters(DATASET, ["missing_key=1"])
    assert out == {}
    out = filter_dictionary_with_filters(
        DATASET, ["missing_key=1"], filter_is_optional=True
    )
    assert len(out) == 3


def test_filter_invalid_operator_raises():
    with pytest.raises(NotImplementedError):
        filter_dictionary_with_filters(DATASET, ["age~40"])


def test_presence_filter():
    D = {
        "p1": {"a": 1},
        "p2": {"b": 2},
        "p3": {"a": 1, "b": 2},
    }
    out = filter_dictionary_with_presence(D, ["a", "b"])
    assert set(out.keys()) == {"p3"}


def test_possible_labels_filter():
    out = filter_dictionary_with_possible_labels(
        DATASET, possible_labels=["A"], label_key="class"
    )
    assert set(out.keys()) == {"patient_1", "patient_3"}
    out = filter_dictionary_with_possible_labels(
        DATASET, possible_labels=["A"], label_key="missing"
    )
    assert out == {}


def test_fill_missing_with_value():
    D = {"p1": {"a": 1}, "p2": {}}
    out = fill_missing_with_value(D, ["a:0"])
    assert out["p1"]["a"] == 1
    assert out["p2"]["a"] == "0"


def test_fill_conditional():
    D = {
        "p1": {"sex": "M"},
        "p2": {"sex": "F"},
    }
    out = fill_conditional(D, ["prostate_cancer:true^sex:M"])
    assert out["p1"]["prostate_cancer"] == "true"
    assert "prostate_cancer" not in out["p2"]


def test_filter_dictionary_wrapper():
    out = filter_dictionary(
        dict(DATASET),
        filters_presence=["age"],
        filters=["class=A"],
        possible_labels=["A"],
        label_key="class",
    )
    assert set(out.keys()) == {"patient_1", "patient_3"}
    original = filter_dictionary(dict(DATASET))
    assert len(original) == 3
