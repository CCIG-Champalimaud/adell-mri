"""
Functions for dataset handling and operations.
"""

import json
from copy import deepcopy
from dataclasses import dataclass
from typing import Any, Iterator

import numpy as np
import yaml

from adell_mri.custom_types import DatasetDict
from adell_mri.utils.dataset_filters import (
    fill_conditional,
    fill_missing_with_value,
    filter_dictionary,
)
from adell_mri.utils.parser import parse_ids
from adell_mri.utils.python_logging import get_logger

logger = get_logger(__name__)


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
    """
    A class to handle loading, subsampling and filtering of datasets.
    Expects the dataset to be a JSON file containing a dictionary with
    the following format:
    {"id1": {"key1": value1, "key2": value2, ...}, "id2": {...}, ...}

    Args:
        path (str | list[str]): path to the dataset file or list of paths.
        rng (np.random.Generator, optional): random number generator. Defaults to None.
        seed (int, optional): random seed. Defaults to 42.
    """

    path: str | list[str]
    rng: np.random.Generator = None
    seed: int = 42
    dataset_name: str = "dataset"

    def __post_init__(self):
        self.dataset = {}
        self.load_dataset(self.path)
        self.dataset_original = deepcopy(self.dataset)

        if self.rng is None:
            self.rng = np.random.default_rng(self.seed)

    def load_dataset(self, path: str | list[str] | None):
        """
        Loads a dataset from a JSON or YAML file.

        Args:
            path (str | list[str]): path to the dataset file or list of paths.
        """
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

    def fill_conditional(self, filters: list[str] | None):
        """
        Fills missing values in the dataset based on conditional logic.
        See :func:`fill_conditional` for more details.

        Args:
            filters (list[str] | None): list of filters to apply.
        """
        if filters is not None:
            self.dataset = fill_conditional(self.dataset, filters)

    def fill_missing_with_value(self, filters: list[str] | None):
        """
        Fills missing values in the dataset with a placeholder value.
        See :func:`fill_missing_with_value` for more details.

        Args:
            filters (list[str] | None): list of filters to apply.
        """
        if filters is not None:
            self.dataset = fill_missing_with_value(self.dataset, filters)

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
        """
        Filters the dataset based on a set of filters. In essence,
        this is a shorthand for fill_conditional, fill_missing_with_value
        and filter_dictionary.
        See :func:`fill_conditional`, :func:`fill_missing_with_value` and
        :func:`filter_dictionary` for more details.

        Args:
            filters_presence (list[str]): list of keys that should be present
            to keep an element.
            filters_existence (list[str]): list of keys that should exist to
            keep an element (i.e. the file value should exist in the disk).
            possible_labels (list[str]): list of possible labels.
            label_key (str): label key for `possible_labels`.
            filters (list[str]): list of filters to apply. See
                :func:`filter_dictionary` for more details.
            filter_is_optional (bool): whether the filter is optional. See
                :func:`filter_dictionary` for more details.
            fill_conditional (list[str]): list of filters to apply. See
                :func:`fill_conditional` for more details.
            fill_missing_with_value (list[str]): list of filters to apply. See
                :func:`fill_missing_with_value` for more details.
        """
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
        )

    def to_datalist(self, key_list: list[str] = None):
        """
        Converts the dataset to a list of dictionaries.

        Args:
            key_list (list[str]): list of keys to include in the list.

        Returns:
            list[dict]: list of dictionaries.
        """
        if key_list is None:
            key_list = self.keys()
        else:
            key_list = parse_ids(key_list, "list")
        return [{**self[k], "identifier": k} for k in self if k in key_list]

    def keys(self):
        """
        Returns the keys of the dataset.

        Returns:
            list[str]: list of keys.
        """
        return self.dataset.keys()

    def subsample_dataset(
        self,
        subsample_size: int = None,
        strata_key: str = None,
        key_list: list[str] | str = None,
        excluded_key_list: list[str] | str = None,
    ):
        """
        Subsamples the dataset.

        Args:
            subsample_size (int): number of samples to keep.
            strata_key (str): key to stratify on.
            key_list (list[str] | str): list of keys to include in the
                subsample.
            excluded_key_list (list[str] | str): list of keys to exclude from
                the subsample.
        """
        n_start = len(self.dataset)
        if key_list is not None:
            key_list = parse_ids(key_list, "list")
            logger.info(
                "Selecting %s keys from %s", len(key_list), self.dataset_name
            )
            logger.info("Before: %s samples", n_start)
            self.dataset = {
                k: self.dataset[k] for k in self.dataset if k in key_list
            }
        elif excluded_key_list is not None:
            excluded_key_list = parse_ids(excluded_key_list, "list")
            logger.info(
                "Excluding %s keys from %s",
                len(excluded_key_list),
                self.dataset_name,
            )
            logger.info("Before: %s samples", n_start)
            self.dataset = {
                k: self.dataset[k]
                for k in self.dataset
                if k not in excluded_key_list
            }
        elif subsample_size is not None:
            logger.info(
                "Reducing dataset to %s samples from %s",
                subsample_size,
                self.dataset_name,
            )
            logger.info("Before: %s samples", n_start)
            self.dataset = subsample_dataset(
                self.dataset,
                subsample_size=subsample_size,
                rng=self.rng,
                strata_key=strata_key,
            )
        logger.info("After: %s samples", len(self))
        logger.info("Difference: %s samples", n_start - len(self))

    def apply_filters(self, **filter_dict: dict[str, Any]):
        """
        Applies a set of filters to the dataset. Can also subset the dataset.
        The relevant filters are:
            - fill_conditional (for :func:`fill_conditional`)
            - fill_missing_with_placeholder (for :func:`fill_missing_with_value`)
            - possible_labels (for :func:`filter_dictionary`)
            - label_keys (for :func:`filter_dictionary`)
            - presence_keys (for :func:`filter_dictionary`)
            - filters_existence (for :func:`filter_dictionary`)
            - filter_on_keys (for :func:`filter_dictionary`)
            - filter_is_optional (for :func:`filter_dictionary`)
            - excluded_ids (for :func:`subsample_dataset`)
            - subsample_size (for :func:`subsample_dataset`)

        Args:
            **filter_dict: keyword arguments to pass to filter_dictionary.
        """
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
        """
        Returns the value of the dataset for the given key.

        Args:
            key (str | list[str]): key or list of keys to get.

        Returns:
            Any: value of the dataset for the given key.
        """
        if isinstance(key, (list, tuple)):
            return {k: self[k] for k in key}
        else:
            return self.dataset[key]

    def __setitem__(self, key: str, value: Any):
        """
        Sets the value of the dataset for the given key.

        Args:
            key (str): key to set.
            value (Any): value to set.
        """
        self.dataset[key] = value

    def __len__(self):
        """
        Returns the number of samples in the dataset.

        Returns:
            int: number of samples in the dataset.
        """
        return len(self.dataset)

    def __iter__(self) -> Iterator[str]:
        """
        Returns an iterator over the keys of the dataset.

        Yields:
            str: key of the dataset.
        """
        for key in self.keys():
            yield key
