"""
Classification network factories.
"""

from typing import Any, Callable

import torch
from lightning.pytorch import LightningModule

from adell_mri.modules.activations import activation_factory
from adell_mri.modules.classification.classification import TabularClassifier
from adell_mri.modules.classification.classification.deconfounded_classification import (
    CategoricalConversion,
)
from adell_mri.modules.classification.pl import (
    ClassNetPL,
    DeconfoundedNetPL,
    FactorizedViTClassifierPL,
    HybridClassifierPL,
    OrdNetPL,
    UNetEncoderPL,
    ViTClassifierPL,
)
from adell_mri.modules.layers.adn_fn import get_adn_fn
from adell_mri.utils.batch_preprocessing import BatchPreprocessing
from adell_mri.utils.encoder_loading import (
    load_encoder_into,
    read_encoder_manifest,
)
from adell_mri.utils.network_factories.common import (
    ALLOWED_NET_TYPES,
    _build_encoder_feature_extractor,
    _resolve_single_encoder_checkpoint,
    compile_if_necessary,
    logger,
)
from adell_mri.utils.optimizer_factory import OPTIMIZER_EPS_DEFAULT

_CLASSIFICATION_PL_KEYS = {
    "net_type",
    "image_key",
    "label_key",
    "learning_rate",
    "batch_size",
    "weight_decay",
    "training_dataloader_call",
    "loss_fn",
    "loss_params",
    "n_epochs",
    "warmup_steps",
    "start_decay",
    "training_batch_preproc",
    "optimizer_eps",
}

_CATNET_KEYS = {
    "spatial_dimensions",
    "in_channels",
    "classification_structure",
    "resnet_structure",
    "maxpool_structure",
    "res_type",
    "batch_ensemble",
    "skip_last_activation",
    "gaussian_process",
    "feature_extraction",
}

_DECONFOUNDED_KEYS = {
    "n_dim",
    "n_features_deconfounder",
    "n_cat_deconfounder",
    "n_cont_deconfounder",
    "exclude_surrogate_variables",
    "deconfounder_structure",
    "feature_extraction_module",
}


def _prepare_classification_encoder(
    net_type: str,
    network_config: dict[str, Any],
    encoder_checkpoint,
    keys: list[str],
) -> tuple[str, dict[str, Any], dict[str, Any] | None, Any]:
    """
    Prepares a classification configuration for a pretrained encoder.

    Args:
        net_type (str): requested classification network type.
        network_config (dict[str, Any]): classification configuration.
        encoder_checkpoint: self-supervised checkpoint.
        keys (list[str]): input image keys.

    Returns:
        tuple: ``(net_type, network_config, manifest, post_load_checkpoint)``.
            ``post_load_checkpoint`` is not None when the encoder weights must
            be loaded into the constructed network with
            :func:`load_encoder_into`.
    """
    if encoder_checkpoint is None:
        return net_type, network_config, None, None

    manifest = read_encoder_manifest(encoder_checkpoint)
    manifest_family = manifest["family"]
    post_load_checkpoint = None

    if (
        net_type == "unet"
        and manifest_family == "unet"
        and manifest["in_channels"] == len(keys)
    ):
        unet_args = dict(manifest["unet_args"])
        if "activation_fn" in unet_args and isinstance(
            unet_args["activation_fn"], str
        ):
            unet_args["activation_fn"] = activation_factory[
                unet_args["activation_fn"]
            ]
        network_config.update(unet_args)
        network_config["in_channels"] = manifest["in_channels"]
        post_load_checkpoint = encoder_checkpoint
    elif (
        net_type == "vit"
        and manifest_family in {"vit", "mae"}
        and manifest["in_channels"] == len(keys)
    ):
        network_config.update(manifest["backbone_args"])
        network_config.pop("in_channels", None)
        network_config.pop("spatial_dimensions", None)
        post_load_checkpoint = encoder_checkpoint
    else:
        if net_type not in {"cat", "ord"}:
            logger.warning(
                "The SSL encoder family '%s' is not compatible with "
                "net_type '%s'; falling back to a %s classifier with the "
                "pretrained encoder as the feature extractor.",
                manifest_family,
                net_type,
                "ordinal" if net_type == "ord" else "categorical",
            )
            net_type = "ord" if net_type == "ord" else "cat"
        network_config = {
            k: v
            for k, v in network_config.items()
            if k in _CLASSIFICATION_PL_KEYS or k in _CATNET_KEYS
        }
        network_config["feature_extraction"] = _build_encoder_feature_extractor(
            encoder_checkpoint, manifest, keys
        )
        network_config["in_channels"] = len(keys)
        network_config["spatial_dimensions"] = manifest.get("spatial_dim", 3)

    return net_type, network_config, manifest, post_load_checkpoint


def get_classification_network(
    net_type: str,
    network_config: dict[str, Any],
    dropout_param: float,
    seed: int,
    n_classes: int,
    keys: list[str],
    clinical_feature_keys: list[str],
    train_loader_call: Callable,
    max_epochs: int,
    warmup_steps: int,
    start_decay: int,
    crop_size: int,
    clinical_feature_means: torch.Tensor = None,
    clinical_feature_stds: torch.Tensor = None,
    label_smoothing: float | None = None,
    mixup_alpha: float | None = None,
    partial_mixup: float | None = None,
    optimizer_eps: float = OPTIMIZER_EPS_DEFAULT,
    encoder_checkpoint: str | dict[str, Any] | None = None,
    lr_encoder: float | None = None,
) -> LightningModule:
    """
    Returns a classification network.

    Args:
        net_type (str): the type of network to use.
        network_config (dict[str, Any]): the configuration of the network.
        dropout_param (float): the dropout parameter.
        seed (int): the seed for the random number generator.
        n_classes (int): the number of classes.
        keys (list[str]): the keys of the input data.
        clinical_feature_keys (list[str]): the keys of the clinical features.
        train_loader_call (Callable): the training data loader.
        max_epochs (int): the maximum number of epochs.
        warmup_steps (int): the number of warmup steps.
        start_decay (int): the number of steps to start decaying the learning rate.
        crop_size (int): the size of the crop.
        clinical_feature_means (torch.Tensor): the means of the clinical features.
        clinical_feature_stds (torch.Tensor): the standard deviations of the clinical features.
        label_smoothing (float): the label smoothing parameter.
        mixup_alpha (float): the mixup alpha parameter.
        partial_mixup (float): the partial mixup parameter.
        encoder_checkpoint (str | dict[str, Any], optional): self-supervised
            checkpoint used to initialise the encoder. Defaults to None.
        lr_encoder (float, optional): learning rate for the pretrained encoder.
            A value of 0 freezes the encoder. Defaults to None (same as the
            network learning rate).

    Returns:
        A :class:`LightningModule` classification network.
    """
    if net_type not in ALLOWED_NET_TYPES["classification"]:
        raise ValueError(
            f"net_type '{net_type}' not valid, has to be one of \
            {ALLOWED_NET_TYPES['classification']}"
        )
    if net_type == "unet":
        act_fn = network_config["activation_fn"]
        norm_fn = "batch"
    else:
        if "act_fn" in network_config:
            act_fn = network_config["act_fn"]
            del network_config["act_fn"]
        else:
            act_fn = "swish"
        if "norm_fn" in network_config:
            norm_fn = network_config["norm_fn"]
            del network_config["norm_fn"]
        else:
            norm_fn = "batch"
    adn_fn = get_adn_fn(3, norm_fn, act_fn=act_fn, dropout_param=dropout_param)
    batch_preprocessing = BatchPreprocessing(
        label_smoothing, mixup_alpha, partial_mixup, seed
    )
    boilerplate_args = {
        "in_channels": len(keys),
        "n_classes": n_classes,
        "training_dataloader_call": train_loader_call,
        "image_key": "image",
        "label_key": "label",
        "n_epochs": max_epochs,
        "warmup_steps": warmup_steps,
        "training_batch_preproc": batch_preprocessing,
        "start_decay": start_decay,
        "optimizer_eps": optimizer_eps,
    }
    encoder_checkpoint = _resolve_single_encoder_checkpoint(
        encoder_checkpoint, "classification"
    )
    (
        net_type,
        network_config,
        manifest,
        post_load_checkpoint,
    ) = _prepare_classification_encoder(
        net_type, network_config, encoder_checkpoint, keys
    )

    if net_type == "unet":
        depth_val = network_config["depth"]
        if isinstance(depth_val, (list, tuple)):
            head_structure = [depth_val[-1] for _ in range(3)]
        else:
            head_structure = [depth_val for _ in range(3)]
        network = UNetEncoderPL(
            head_structure=head_structure,
            head_adn_fn=get_adn_fn(
                1, norm_fn, act_fn="gelu", dropout_param=dropout_param
            ),
            **boilerplate_args,
            **network_config,
        )
    elif "vit" in net_type:
        if manifest is not None and manifest.get("image_size") is not None:
            image_size = tuple(int(x) for x in manifest["image_size"])
        else:
            image_size = tuple(int(x) for x in crop_size)
        if net_type == "factorized_vit" and len(image_size) != 3:
            raise ValueError(
                f"factorized_vit requires a 3D image_size, got {image_size}"
            )
        network_config["image_size"] = image_size
        if net_type == "vit":
            network = ViTClassifierPL(
                adn_fn=get_adn_fn(
                    1, "identity", act_fn="gelu", dropout_param=dropout_param
                ),
                **boilerplate_args,
                **network_config,
            )
        elif net_type == "factorized_vit":
            network_config.pop("embed_method", None)
            network = FactorizedViTClassifierPL(
                adn_fn=get_adn_fn(
                    1, "identity", act_fn="gelu", dropout_param=dropout_param
                ),
                **boilerplate_args,
                **network_config,
            )
    elif "ord" in net_type:
        for k in ["net_type", "in_channels", "n_classes", "adn_fn"]:
            network_config.pop(k, None)
        network = OrdNetPL(
            adn_fn=adn_fn,
            **boilerplate_args,
            **network_config,
        )
    else:
        for k in ["in_channels", "n_classes", "adn_fn"]:
            network_config.pop(k, None)
        network = ClassNetPL(
            net_type=net_type,
            adn_fn=adn_fn,
            **boilerplate_args,
            **network_config,
        )

    if post_load_checkpoint is not None:
        load_encoder_into(network, post_load_checkpoint, prefix="")

    if len(clinical_feature_keys) > 0:
        boilerplate_args_hybrid = {
            "training_dataloader_call": train_loader_call,
            "image_key": "image",
            "label_key": "label",
            "n_epochs": max_epochs,
            "warmup_steps": warmup_steps,
            "training_batch_preproc": batch_preprocessing,
            "start_decay": start_decay,
            "optimizer_eps": optimizer_eps,
        }

        for k in ["learning_rate", "batch_size", "loss_fn", "loss_params"]:
            if k in network_config:
                boilerplate_args_hybrid[k] = network_config[k]
        tab_network = TabularClassifier(
            len(clinical_feature_keys),
            mlp_structure=[],
            mlp_adn_fn=torch.nn.Identity,
            n_classes=n_classes,
            feature_means=clinical_feature_means,
            feature_stds=clinical_feature_stds,
        )
        network = HybridClassifierPL(
            convolutional_module=network,
            tabular_module=tab_network,
            **boilerplate_args_hybrid,
        )

    network.lr_encoder = lr_encoder
    return network


@compile_if_necessary
def get_deconfounded_classification_network(
    network_config: dict[str, Any],
    dropout_param: float,
    seed: int,
    n_classes: int,
    keys: list[str],
    cat_confounder_key: list[str],
    cont_confounder_key: list[str],
    cat_vars: list[list[str]],
    cont_vars: int,
    train_loader_call: Callable,
    max_epochs: int,
    warmup_steps: int,
    start_decay: int,
    n_features_deconfounder: int = 64,
    exclude_surrogate_variables: bool = False,
    label_smoothing: float | None = None,
    mixup_alpha: float | None = None,
    partial_mixup: float | None = None,
    optimizer_eps: float = OPTIMIZER_EPS_DEFAULT,
    encoder_checkpoint: str | dict[str, Any] | None = None,
    lr_encoder: float | None = None,
) -> LightningModule:
    """
    Returns a deconfounded classification network.

    Args:
        network_config (dict[str, Any]): the configuration of the network.
        dropout_param (float): the dropout parameter.
        seed (int): the seed for the random number generator.
        n_classes (int): the number of classes.
        keys (list[str]): the keys of the input data.
        cat_confounder_key (list[str]): the keys of the categorical confounders.
        cont_confounder_key (list[str]): the keys of the continuous confounders.
        cat_vars (list[list[str]]): the categorical variables.
        cont_vars (int): the number of continuous variables.
        train_loader_call (Callable): the training data loader.
        max_epochs (int): the maximum number of epochs.
        warmup_steps (int): the number of warmup steps.
        start_decay (int): the number of steps to start decaying the learning rate.
        n_features_deconfounder (int): the number of features in the deconfounder.
        exclude_surrogate_variables (bool): whether to exclude surrogate variables.
        label_smoothing (float): the label smoothing parameter.
        mixup_alpha (float): the mixup alpha parameter.
        partial_mixup (float): the partial mixup parameter.
        encoder_checkpoint (str | dict[str, Any], optional): self-supervised
            checkpoint used to initialise the encoder. Defaults to None.
        lr_encoder (float, optional): learning rate for the pretrained encoder.
            A value of 0 freezes the encoder. Defaults to None.

    Returns:
        A :class:`LightningModule` deconfounded classification network.
    """
    encoder_checkpoint = _resolve_single_encoder_checkpoint(
        encoder_checkpoint, "deconfounded classification"
    )

    if "act_fn" in network_config:
        act_fn = network_config["act_fn"]
        del network_config["act_fn"]
    else:
        act_fn = "relu"
    if "norm_fn" in network_config:
        norm_fn = network_config["norm_fn"]
        del network_config["norm_fn"]
    else:
        norm_fn = "batch"
    adn_fn = get_adn_fn(3, norm_fn, act_fn=act_fn, dropout_param=dropout_param)
    batch_preprocessing = BatchPreprocessing(
        label_smoothing, mixup_alpha, partial_mixup, seed
    )
    if cat_vars is not None:
        cat_conv = CategoricalConversion(cat_vars)
    else:
        cat_conv = None
    boilerplate_args = {
        "in_channels": len(keys),
        "n_classes": n_classes,
        "training_dataloader_call": train_loader_call,
        "image_key": "image",
        "label_key": "label",
        "n_epochs": max_epochs,
        "warmup_steps": warmup_steps,
        "training_batch_preproc": batch_preprocessing,
        "start_decay": start_decay,
        "optimizer_eps": optimizer_eps,
    }
    if encoder_checkpoint is not None:
        manifest = read_encoder_manifest(encoder_checkpoint)
        network_config = {
            k: v
            for k, v in network_config.items()
            if k in _CLASSIFICATION_PL_KEYS
            or k in _CATNET_KEYS
            or k in _DECONFOUNDED_KEYS
        }
        network_config.pop("in_channels", None)
        network_config.pop("n_classes", None)
        network_config["feature_extraction"] = _build_encoder_feature_extractor(
            encoder_checkpoint, manifest, keys
        )
        network_config["feature_extraction_module"] = "cat"
        network_config["n_dim"] = manifest.get("spatial_dim", 3)

    n_cat_deconfounder = (
        [len(x) for x in cat_vars] if cat_vars is not None else None
    )
    network = DeconfoundedNetPL(
        adn_fn=adn_fn,
        embedder=cat_conv,
        n_features_deconfounder=n_features_deconfounder,
        n_cat_deconfounder=n_cat_deconfounder,
        n_cont_deconfounder=cont_vars,
        cat_confounder_key=cat_confounder_key,
        cont_confounder_key=cont_confounder_key,
        exclude_surrogate_variables=exclude_surrogate_variables,
        **boilerplate_args,
        **network_config,
    )

    network.lr_encoder = lr_encoder
    return network
