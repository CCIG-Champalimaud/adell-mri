import json
from copy import deepcopy
from dataclasses import dataclass
from typing import Any

import numpy as np
import yaml

from ..custom_types import DatasetDict
from ..utils.parser import parse_ids
from .dataset_filters import (
    fill_conditional,
    fill_missing_with_value,
    filter_dictionary,
    print_verbose,
)


def subsample_dataset(
    data_dict: DatasetDict,
    subsample_size: int,
    rng: np.random.Generator,
    strata_key: str = None,
) -> DatasetDict:
    """
    Subsamples a DatasetDict by either randomly sampling a subset of keys
    or by stratifying based on a specified key and sampling from each
    stratum.

    Args:
        data_dict (DatasetDict): the data dictionary to subsample from.
        subsample_size (int): the number of samples to keep.
        rng (np.random.Generator): a random number generator.
        strata_key (str): a key to stratify the sampling on. If provided, each
            stratum will be sampled to match its distribution in the original
            dict.

    Returns:
        DatasetDict: the subsampled data dictionary.
    """
    if subsample_size is not None and len(data_dict) > subsample_size:
        if strata_key is not None:
            strata = {}
            for k in data_dict:
                label = data_dict[k][strata_key]
                if label not in strata:
                    strata[label] = []
                strata[label].append(k)
            ps = [len(strata[k]) / len(data_dict) for k in strata]
            split = [int(p * subsample_size) for p in ps]
            ss = []
            for k, s in zip(strata, split):
                ss.extend(
                    rng.choice(strata[k], size=s, replace=False, shuffle=False)
                )
            data_dict = {k: data_dict[k] for k in ss}
        else:
            s = subsample_size * len(data_dict)
            ss = rng.choice(
                list(data_dict.keys()), subsample_size, replace=False
            )
            data_dict = {k: data_dict[k] for k in ss}
    return data_dict


@dataclass
class Dataset:
    path: str | list[str]
    rng: np.random.Generator = None
    seed: int = 42
    verbose: bool = True
    dataset_name: str = "dataset"

    def __post_init__(self):
        self.dataset = {}
        self.load_dataset(self.path)
        self.dataset_original = deepcopy(self.dataset)

        if self.rng is None:
            self.rng = np.random.default_rng(self.seed)

    def load_dataset(self, path: str):
        if path is None:
            self.dataset = {}
        elif isinstance(path, list):
            for p in path:
                self.load_dataset(p)
        else:
            if path.endswith(".json"):
                with open(path, "r") as f:
                    dataset = json.load(f)
            elif path.endswith(".yml"):
                with open(path, "r") as f:
                    dataset = yaml.safe_load(f)
            for k in dataset:
                self.dataset[k] = dataset[k]

    def fill_conditional(self, filters: list[str]):
        if filters is not None:
            self.dataset = fill_conditional(
                self.dataset, filters, verbose=self.verbose
            )

    def fill_missing_with_value(self, filters: list[str]):
        if filters is not None:
            self.dataset = fill_missing_with_value(
                self.dataset, filters, verbose=self.verbose
            )

    def filter_dictionary(
        self,
        filters_presence: list[str] = None,
        filters_existence: list[str] = None,
        possible_labels: list[str] = None,
        label_key: str = None,
        filters: list[str] = None,
        filter_is_optional: bool = False,
        fill_conditional: list[str] = None,
        fill_missing_with_value: list[str] = None,
    ):
        self.fill_conditional(fill_conditional)
        self.fill_missing_with_value(fill_missing_with_value)
        self.dataset = filter_dictionary(
            self.dataset,
            filters_presence=filters_presence,
            filters_existence=filters_existence,
            possible_labels=possible_labels,
            label_key=label_key,
            filters=filters,
            filter_is_optional=filter_is_optional,
            verbose=self.verbose,
        )

    def to_datalist(self, key_list: list[str] = None):
        if key_list is None:
            key_list = self.keys()
        else:
            key_list = parse_ids(key_list, "list")
        return [{**self[k], "identifier": k} for k in self if k in key_list]

    def keys(self):
        return self.dataset.keys()

    def subsample_dataset(
        self,
        subsample_size: int = None,
        strata_key: str = None,
        key_list: list[str] | str = None,
        excluded_key_list: list[str] | str = None,
    ):
        n_start = len(self.dataset)
        if key_list is not None:
            key_list = parse_ids(key_list, "list")
            self.print_verbose(
                f"Selecting {len(key_list)} keys from {self.dataset_name}"
            )
            self.print_verbose(f"\tBefore: {n_start} samples")
            self.dataset = {
                k: self.dataset[k] for k in self.dataset if k in key_list
            }
        elif excluded_key_list is not None:
            excluded_key_list = parse_ids(excluded_key_list, "list")
            self.print_verbose(
                f"Excluding {len(excluded_key_list)} keys from {self.dataset_name}"
            )
            self.print_verbose(f"\tBefore: {n_start} samples")
            self.dataset = {
                k: self.dataset[k]
                for k in self.dataset
                if k not in excluded_key_list
            }
        elif subsample_size is not None:
            self.print_verbose(
                f"Reducing dataset to {subsample_size} samples from {self.dataset_name}"
            )
            self.print_verbose(f"\tBefore: {n_start} samples")
            self.dataset = subsample_dataset(
                self.dataset,
                subsample_size=subsample_size,
                rng=self.rng,
                strata_key=strata_key,
            )
        self.print_verbose(f"\tAfter: {len(self)} samples")
        self.print_verbose(f"\tDifference: {n_start - len(self)} samples")

    def print_verbose(self, *args, **kwargs):
        print_verbose(*args, **kwargs, verbose=self.verbose)

    def apply_filters(self, **filter_dict: dict[str, Any]):
        if "fill_conditional" in filter_dict:
            self.fill_conditional(filters=filter_dict["fill_conditional"])
        if "fill_missing_with_placeholder" in filter_dict:
            self.fill_missing_with_value(
                filters=filter_dict["fill_missing_with_placeholder"]
            )
        self.filter_dictionary(
            possible_labels=filter_dict.get("possible_labels", None),
            label_key=filter_dict.get("label_keys", None),
            filters_presence=filter_dict.get("presence_keys", None),
            filters_existence=filter_dict.get("filters_existence", None),
            filters=filter_dict.get("filter_on_keys", None),
            filter_is_optional=filter_dict.get("filter_is_optional", False),
        )
        if "excluded_ids" in filter_dict:
            self.subsample_dataset(
                excluded_key_list=filter_dict["excluded_ids"]
            )
        if "subsample_size" in filter_dict:
            self.subsample_dataset(
                subsample_size=filter_dict["subsample_size"],
                strata_key=filter_dict.get("label_keys", None),
            )

    def __getitem__(self, key: str | list[str]):
        if isinstance(key, (list, tuple)):
            return {k: self[k] for k in key}
        else:
            return self.dataset[key]

    def __setitem__(self, key: str, value: Any):
        self.dataset[key] = value

    def __len__(self):
        return len(self.dataset)

    def __iter__(self):
        for key in self.keys():
            yield key
