"""
Shared utilities for classification entrypoints.

Consolidates logic duplicated across standard and deconfounded
classification sub-modes (train, test, predict, explain).
"""

from __future__ import annotations

from typing import Any

import torch

from adell_mri.modules.classification.losses import OrdinalSigmoidalLoss
from adell_mri.utils.network_factories import (
    get_classification_network,
    get_deconfounded_classification_network,
)
from adell_mri.utils.optimizer_factory import optimizer_eps_from_precision


def use_deconfounding(args) -> bool:
    """
    Check whether deconfounding is requested based on CLI arguments.

    Args:
        args: Parsed CLI arguments (any object with optional
            ``cat_confounder_keys`` and ``cont_confounder_keys``
            attributes).

    Returns:
        bool: True if either categorical or continuous confounder
        keys are specified.
    """
    return bool(
        getattr(args, "cat_confounder_keys", None)
        or getattr(args, "cont_confounder_keys", None)
    )


def extract_confounder_info(
    args, data_dict
) -> tuple[str | None, str | None, list[list[str]] | None, int | None]:
    """
    Derive confounder metadata from the dataset.

    Args:
        args: Parsed CLI arguments with ``cat_confounder_keys`` and
            ``cont_confounder_keys`` attributes.
        data_dict: A :class:`~adell_mri.utils.dataset.Dataset` instance.

    Returns:
        tuple: ``(cat_key, cont_key, cat_vars, cont_vars)`` where
        ``cat_key``/``cont_key`` are the keys expected by the
        deconfounder network and ``cat_vars``/``cont_vars`` are the
        variable cardinalities.
    """
    cat_vars = None
    cont_vars = None
    cat_key = None
    cont_key = None

    if args.cat_confounder_keys is not None:
        cat_key = "cat_confounder"
        cat_vars = []
        for k in args.cat_confounder_keys:
            curr_cat_vars = []
            for kk in data_dict:
                v = data_dict[kk][k]
                if v not in curr_cat_vars:
                    curr_cat_vars.append(v)
            cat_vars.append(curr_cat_vars)

    if args.cont_confounder_keys is not None:
        cont_key = "cont_confounder"
        cont_vars = len(args.cont_confounder_keys)

    return cat_key, cont_key, cat_vars, cont_vars


def resolve_n_classes_and_labels(args, data_dict):
    """
    Determine the number of classes and label grouping from user arguments.

    Args:
        args: Parsed CLI arguments (must have ``possible_labels``,
            ``label_groups``, ``positive_labels``, ``label_keys``).
        data_dict: A :class:`~adell_mri.utils.dataset.Dataset` instance.

    Returns:
        tuple: ``(n_classes, label_groups, positive_labels)``.
    """
    all_classes = []
    for k in data_dict:
        C = data_dict[k][args.label_keys]
        if isinstance(C, list):
            C = max(C)
        all_classes.append(str(C))

    label_groups = None
    positive_labels = args.positive_labels
    if args.label_groups is not None:
        n_classes = len(args.label_groups)
        label_groups = [
            label_group.split(",") for label_group in args.label_groups
        ]
        if len(label_groups) == 2:
            positive_labels = label_groups[1]
    elif args.positive_labels is None:
        n_classes = len(args.possible_labels)
    else:
        n_classes = 2

    return n_classes, label_groups, positive_labels, all_classes


def create_transform_arguments(
    args,
    *,
    n_classes: int | None = None,
    include_label_info: bool = False,
    label_groups: list[list[str]] | None = None,
    positive_labels=None,
    include_confounder_transforms: bool = False,
) -> dict[str, Any]:
    """
    Build the keyword arguments for
    :class:`~adell_mri.transform_factory.transforms.ClassificationTransforms`.

    The returned dict can be used with any classification sub-mode
    (train, test, predict).

    Args:
        args: Parsed CLI arguments.
        n_classes (int, optional): Number of classes. Defaults to None.
        include_label_info (bool, optional): When ``True``, include
            ``possible_labels``, ``positive_labels``, ``label_groups``,
            ``label_key`` and ``label_mode``. Defaults to ``False``.
        label_groups (list[list[str]], optional): Resolved label groups.
            Defaults to None.
        positive_labels (list[str], optional): Resolved positive
            labels. Required when *include_label_info* is ``True``.
        include_confounder_transforms (bool, optional): When ``True``
            and confounder keys are present on ``args``, include
            ``cat_confounder_keys`` and ``cont_confounder_keys`` in
            the returned dict so that ``ClassificationTransforms``
            processes them. For prediction/explain modes this should
            be ``False`` because the deconfounder forward pass does
            not require confounder data. Defaults to ``False``.

    Returns:
        dict: Keyword arguments for ``ClassificationTransforms``.
    """
    keys = args.image_keys
    adc_keys = args.adc_keys if args.adc_keys is not None else []
    adc_keys = [k for k in adc_keys if k in keys]
    mask_key = getattr(args, "mask_key", None)

    clinical_feature_keys = getattr(args, "clinical_feature_keys", None) or []

    transform_kwargs: dict[str, Any] = {
        "keys": keys,
        "mask_key": mask_key,
        "image_masking": getattr(args, "image_masking", False),
        "image_crop_from_mask": getattr(args, "image_crop_from_mask", False),
        "clinical_feature_keys": (
            clinical_feature_keys if not use_deconfounding(args) else []
        ),
        "adc_keys": adc_keys,
        "target_spacing": getattr(args, "target_spacing", None),
        "resample_to": getattr(args, "resample_to", None),
        "crop_size": getattr(args, "crop_size", None),
        "pad_size": getattr(args, "pad_size", None),
    }

    if use_deconfounding(args) and include_confounder_transforms:
        transform_kwargs["cat_confounder_keys"] = args.cat_confounder_keys
        transform_kwargs["cont_confounder_keys"] = args.cont_confounder_keys

    if include_label_info:
        label_mode = (
            "binary" if n_classes == 2 and label_groups is None else "cat"
        )

        transform_kwargs.update(
            {
                "possible_labels": args.possible_labels,
                "positive_labels": positive_labels,
                "label_groups": label_groups,
                "label_key": args.label_keys,
                "label_mode": label_mode,
            }
        )

    return transform_kwargs


def configure_loss_fn(network_config, net_type, n_classes, class_weights=None):
    """
    Assign a loss function to a network config dict in-place.

    Args:
        network_config (dict): Network configuration. Mutated in-place.
        net_type (str or None): Network type string (e.g. ``"cat"``,
            ``"ord"``, ``"unet"``). If ``None``, ordinal loss is never
            selected.
        n_classes (int): Number of output classes.
        class_weights (torch.Tensor, optional): Per-class loss weights.
    """
    if n_classes == 2:
        network_config["loss_fn"] = (
            torch.nn.BCEWithLogitsLoss(class_weights)
            if class_weights is not None
            else torch.nn.BCEWithLogitsLoss()
        )
    elif net_type == "ord":
        network_config["loss_fn"] = OrdinalSigmoidalLoss(
            n_classes, class_weights
        )
    else:
        network_config["loss_fn"] = (
            torch.nn.CrossEntropyLoss(class_weights)
            if class_weights is not None
            else torch.nn.CrossEntropyLoss()
        )


def create_classification_network(
    *,
    args,
    network_config: dict,
    input_keys: list[str],
    clinical_feature_keys: list[str],
    n_classes: int | None = None,
    cat_vars: list[list[str]] | None = None,
    cont_vars: int | None = None,
    cat_key: str | None = None,
    cont_key: str | None = None,
    train_loader_call: Any = None,
    max_epochs: int | None = None,
    warmup_steps: float | None = None,
    start_decay: float | None = None,
    clinical_feature_means: torch.Tensor | None = None,
    clinical_feature_stds: torch.Tensor | None = None,
):
    """
    Create a classification or deconfounded classification network.

    Dispatches to the correct factory based on whether confounder keys
    are specified.

    Args:
        args: Parsed CLI arguments.
        network_config (dict): Network configuration.
        input_keys (list[str]): Input image keys.
        clinical_feature_keys (list[str]): Clinical/tabular feature keys.
        n_classes (int, optional): number of classes. Defaults to None
            (retrieved from ``args``).
        cat_vars (list[list[str]], optional): Categorical variable
            cardinalities (deconfounder only).
        cont_vars (int, optional): Number of continuous variables
            (deconfounder only).
        cat_key (str, optional): Categorical confounder key
            (deconfounder only).
        cont_key (str, optional): Continuous confounder key
            (deconfounder only).
        train_loader_call: Training dataloader factory callable.
        max_epochs (int, optional): Maximum number of epochs.
        warmup_steps (float, optional): Warm-up steps.
        start_decay (float, optional): Decay start step.
        clinical_feature_means (torch.Tensor, optional): Means for
            clinical feature normalization.
        clinical_feature_stds (torch.Tensor, optional): Standard
            deviations for clinical feature normalization.

    Returns:
        A PyTorch Lightning classification module.
    """
    dropout_param = getattr(args, "dropout_param", 0)
    seed = getattr(args, "seed", 42)
    n_classes_val = getattr(args, "n_classes", n_classes if n_classes else 2)
    net_type = getattr(args, "net_type", "cat")
    label_smoothing_val = getattr(args, "label_smoothing", None)
    mixup_alpha_val = getattr(args, "mixup_alpha", None)
    partial_mixup_val = getattr(args, "partial_mixup", None)
    optimizer_eps = optimizer_eps_from_precision(
        getattr(args, "precision", None)
    )

    if cat_key is not None or cont_key is not None:
        return get_deconfounded_classification_network(
            network_config=network_config,
            dropout_param=dropout_param,
            seed=seed,
            n_classes=n_classes,
            keys=input_keys,
            cat_confounder_key=cat_key,
            cont_confounder_key=cont_key,
            cat_vars=cat_vars,
            cont_vars=cont_vars,
            train_loader_call=train_loader_call,
            max_epochs=max_epochs,
            warmup_steps=warmup_steps,
            start_decay=start_decay,
            n_features_deconfounder=getattr(
                args, "n_features_deconfounder", 64
            ),
            exclude_surrogate_variables=getattr(
                args, "exclude_surrogate_variables", False
            ),
            label_smoothing=label_smoothing_val,
            mixup_alpha=mixup_alpha_val,
            partial_mixup=partial_mixup_val,
            optimizer_eps=optimizer_eps,
        )
    else:
        return get_classification_network(
            net_type=net_type,
            network_config=network_config,
            dropout_param=dropout_param,
            seed=seed,
            n_classes=n_classes_val,
            keys=input_keys,
            clinical_feature_keys=clinical_feature_keys,
            train_loader_call=train_loader_call,
            max_epochs=max_epochs,
            warmup_steps=warmup_steps,
            start_decay=start_decay,
            crop_size=getattr(args, "crop_size", None),
            clinical_feature_means=clinical_feature_means,
            clinical_feature_stds=clinical_feature_stds,
            label_smoothing=label_smoothing_val,
            mixup_alpha=mixup_alpha_val,
            partial_mixup=partial_mixup_val,
            optimizer_eps=optimizer_eps,
        )
