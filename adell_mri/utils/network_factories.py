"""
PyTorch Lightning network factories.

This module contains factory functions for creating PyTorch Lightning
modules for various types of networks, including classification, segmentation,
self-supervised learning, and generative models.
"""

import os
from functools import wraps
from typing import Any, Callable

import torch
from lightning.pytorch import LightningModule

from adell_mri.constants import DEFAULT_SEED
from adell_mri.modules.activations import activation_factory

# classification
from adell_mri.modules.classification.classification import TabularClassifier
from adell_mri.modules.classification.classification.deconfounded_classification import (
    CategoricalConversion,
)

# confounder-free classification
from adell_mri.modules.classification.pl import (
    ClassNetPL,
    DeconfoundedNetPL,
    FactorizedViTClassifierPL,
    HybridClassifierPL,
    OrdNetPL,
    UNetEncoderPL,
    ViTClassifierPL,
)
from adell_mri.modules.diffusion.embedder import Embedder

# gan
from adell_mri.modules.gan.pl import GANPL
from adell_mri.modules.layers.adn_fn import get_adn_fn

# segmentation
from adell_mri.modules.segmentation.pl import (
    UNETRPL,
    BrUNetPL,
    EncoderUNetPL,
    MonaiSWINUNetPL,
    MonaiUNETRPL,
    SWINUNetPL,
    UNetPL,
    UNetPlusPlusPL,
)

# self-supervised learning
from adell_mri.modules.self_supervised.pl import (
    DINOPL,
    IJEPA,
    IJEPAPL,
    BarlowTwinsPL,
    ConvNeXt,
    ResNet,
    SelfSLConvNeXtPL,
    SelfSLResNetPL,
    SelfSLUNetPL,
    UNet,
    ViTMaskedAutoEncoderPL,
    iBOTPL,
)
from adell_mri.modules.semi_supervised_segmentation.losses import (
    LocalContrastiveLoss,
)

# semi-supervised segmentation
from adell_mri.modules.semi_supervised_segmentation.pl import (
    UNetContrastiveSemiSL,
)
from adell_mri.utils.batch_preprocessing import BatchPreprocessing
from adell_mri.utils.encoder_loading import (
    load_encoder_into,
    load_ssl_encoder,
    read_encoder_manifest,
)
from adell_mri.utils.optimizer_factory import OPTIMIZER_EPS_DEFAULT
from adell_mri.utils.python_logging import get_logger
from adell_mri.utils.torch_utils import ExponentialMovingAverage

logger = get_logger(__name__)

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


class _ViTFeatureExtractor(torch.nn.Module):
    """
    Adapts a ViT encoder so that it returns a spatial feature map.
    """

    def __init__(self, vit: torch.nn.Module):
        """
        Args:
            vit (torch.nn.Module): ViT encoder.
        """
        super().__init__()
        self.vit = vit

    def forward(self, X: torch.Tensor) -> torch.Tensor:
        tokens, _ = self.vit(X)
        return self.vit.tokens_to_feature_map(tokens)


def _feature_extractor_from_encoder(
    encoder: torch.nn.Module, family: str
) -> torch.nn.Module:
    """
    Converts an encoder into a module returning a spatial feature map.

    Args:
        encoder (torch.nn.Module): pretrained encoder.
        family (str): encoder family.

    Returns:
        torch.nn.Module: a module returning a spatial feature map.
    """
    if family in {"vit", "mae"}:
        return _ViTFeatureExtractor(encoder)
    return encoder


def _resolve_single_encoder_checkpoint(encoder_checkpoint, context: str):
    """
    Resolves a single encoder checkpoint from a CLI-style list.

    Args:
        encoder_checkpoint: checkpoint path, dict or list.
        context (str): description used in error messages.

    Returns:
        The single checkpoint value.

    Raises:
        ValueError: if more than one checkpoint is provided.
    """
    if isinstance(encoder_checkpoint, (list, tuple)):
        if len(encoder_checkpoint) != 1:
            raise ValueError(
                f"Only one encoder checkpoint can be used for {context}"
            )
        return encoder_checkpoint[0]
    return encoder_checkpoint


def _build_encoder_feature_extractor(
    encoder_checkpoint, manifest: dict[str, Any], keys: list[str]
) -> torch.nn.Module:
    """
    Builds a feature extractor returning a spatial map for a pretrained encoder.

    Args:
        encoder_checkpoint: self-supervised checkpoint.
        manifest (dict[str, Any]): encoder manifest.
        keys (list[str]): input image keys.

    Returns:
        torch.nn.Module: feature extractor.
    """
    pretrained_encoder = load_ssl_encoder(encoder_checkpoint, manifest=manifest)
    feature_extractor = _feature_extractor_from_encoder(
        pretrained_encoder, manifest["family"]
    )
    if manifest["in_channels"] != len(keys):
        spatial_dimensions = manifest.get("spatial_dim", 3)
        conv_op = (
            torch.nn.Conv2d if spatial_dimensions == 2 else torch.nn.Conv3d
        )
        adapter = conv_op(len(keys), manifest["in_channels"], 1)
        feature_extractor = torch.nn.Sequential(adapter, feature_extractor)
    if manifest.get("image_size") is not None:
        feature_extractor.input_shape = (
            len(keys),
            *[int(x) for x in manifest["image_size"]],
        )
    return feature_extractor


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


ALLOWED_NET_TYPES = {
    "classification": [
        "unet",
        "vit",
        "factorized_vit",
        "cat",
        "ord",
        "vgg",
    ],
    "segmentation": [
        "unet",
        "brunet",
        "unetpp",
        "unetr",
        "monai_unetr",
        "swin",
        "monai_swin",
        "encoder_unet",
    ],
}


def compile_if_necessary(func: Callable) -> Callable:
    """
    Decorator to compile the output of a function if TORCH_COMPILE is set to
    True.

    Args:
        func: a function outputing a :class:`torch.nn.Module`.

    Returns:
        The compiled :class:`torch.nn.Module`.
    """

    @wraps(func)
    def wrapper(*args, **kwargs):
        model = func(*args, **kwargs)
        tc = os.environ.get("TORCH_COMPILE", "False").lower()
        if tc in ["true", "1"]:
            logger.info(f"Compiling model because TORCH_COMPILE={tc}")
            if not isinstance(model, torch._dynamo.eval_frame.OptimizedModule):
                model = torch.compile(model)
        return model

    return wrapper


@compile_if_necessary
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


@compile_if_necessary
def get_segmentation_network(
    net_type: str,
    network_config: dict[str, Any],
    bottleneck_classification: bool,
    clinical_feature_keys: list[str],
    all_aux_keys: list[str],
    clinical_feature_params: dict[str, torch.Tensor],
    clinical_feature_key_net: str,
    aux_key_net: str,
    max_epochs: int,
    encoding_operations: list[torch.nn.Module],
    picai_eval: bool,
    lr_encoder: float,
    encoder_checkpoint: str,
    deep_supervision: bool,
    n_classes: int,
    keys: list[str],
    optimizer_str: str = "sgd",
    optimizer_eps: float = OPTIMIZER_EPS_DEFAULT,
    start_decay: float | int = 1.0,
    warmup_steps: float | int = 0.0,
    train_loader_call: Callable = None,
    random_crop_size: list[int] = None,
    crop_size: list[int] = None,
    pad_size: list[int] = None,
    resize_size: list[int] = None,
    semi_supervised: bool = False,
    max_steps_optim: int = None,
    seed: int = DEFAULT_SEED,
) -> LightningModule:
    """
    Returns a segmentation network.

    Args:
        net_type (str): the type of network to use.
        network_config (dict[str, Any]): the configuration of the network.
        bottleneck_classification (bool): whether to use bottleneck
            classification.
        clinical_feature_keys (list[str]): the keys of the clinical features.
        all_aux_keys (list[str]): the keys of the auxiliary features.
        clinical_feature_params (dict[str, torch.Tensor]): the parameters of
            the clinical features.
        clinical_feature_key_net (str): the key of the clinical features in the
            network.
        aux_key_net (str): the key of the auxiliary features in the network.
        max_epochs (int): the maximum number of epochs.
        encoding_operations (list[torch.nn.Module]): the encoding operations.
        picai_eval (bool): whether to use picai evaluation.
        lr_encoder (float): the learning rate for the encoder.
        encoder_checkpoint (str): self-supervised checkpoint used to initialise
            the encoder. Required when ``net_type == "encoder_unet"``.
        deep_supervision (bool): whether to use deep supervision.
        n_classes (int): the number of classes.
        keys (list[str]): the keys of the input data.
        optimizer_str (str, optional): the optimizer to use. Defaults to "sgd".
        start_decay (float | int, optional): the start decay. Defaults to 1.0.
        warmup_steps (float | int, optional): the warmup steps. Defaults to 0.0.
        train_loader_call (Callable, optional): the training data loader.
                Defaults to None.
        random_crop_size (list[int], optional): the random crop size. Defaults
            to None.
        crop_size (list[int], optional): the crop size. Defaults to None.
        pad_size (list[int], optional): the pad size. Defaults to None.
        resize_size (list[int], optional): the resize size. Defaults to None.
        semi_supervised (bool, optional): whether to use semi-supervised
            learning. Defaults to False.
        max_steps_optim (int, optional): the maximum number of steps for
            optimization. Defaults to None.
        seed (int, optional): the seed for the random number generator.
            Defaults to 42.

    Returns:
        A :class:`LightningModule` segmentation network.
    """

    if net_type not in ALLOWED_NET_TYPES["segmentation"]:
        raise ValueError(
            f"net_type '{net_type}' not valid, has to be one of "
            f"{ALLOWED_NET_TYPES['segmentation']}"
        )

    def get_size(*size_list):
        for size in size_list:
            if size is not None:
                return size

    size = get_size(random_crop_size, crop_size, pad_size, resize_size)

    boilerplate = dict(
        training_dataloader_call=train_loader_call,
        label_key="mask",
        n_classes=n_classes,
        bottleneck_classification=bottleneck_classification,
        skip_conditioning=len(all_aux_keys),
        skip_conditioning_key=aux_key_net,
        feature_conditioning=len(clinical_feature_keys),
        feature_conditioning_params=clinical_feature_params,
        feature_conditioning_key=clinical_feature_key_net,
        n_epochs=max_epochs,
        picai_eval=picai_eval,
        lr_encoder=lr_encoder,
        start_decay=start_decay,
        warmup_steps=warmup_steps,
        optimizer_str=optimizer_str,
        optimizer_eps=optimizer_eps,
    )

    if net_type == "unet" and semi_supervised is True:
        ema_params = {
            "decay": 0.99,
            "final_decay": 1.0,
            "n_steps": max_steps_optim,
        }
        ema = ExponentialMovingAverage(**ema_params)
        encoding_operations = encoding_operations[0]
        unet = UNetContrastiveSemiSL(
            encoding_operations=encoding_operations,
            image_key="image",
            semi_sl_image_key_1="semi_sl_image_1",
            semi_sl_image_key_2="semi_sl_image_2",
            deep_supervision=deep_supervision,
            ema=ema,
            loss_fn_semi_sl=LocalContrastiveLoss(seed=seed),
            **boilerplate,
            **network_config,
        )

    elif net_type == "brunet":
        nc = network_config["in_channels"]
        network_config["in_channels"] = nc // len(keys)
        unet = BrUNetPL(
            encoders=encoding_operations,
            image_keys=keys,
            n_input_branches=len(keys),
            deep_supervision=deep_supervision,
            **boilerplate,
            **network_config,
        )

    elif net_type == "encoder_unet":
        if encoder_checkpoint is None:
            raise ValueError(
                "net_type 'encoder_unet' requires --encoder_checkpoint"
            )
        checkpoint = (
            encoder_checkpoint[0]
            if isinstance(encoder_checkpoint, (list, tuple))
            else encoder_checkpoint
        )
        manifest = read_encoder_manifest(checkpoint)
        encoder = load_ssl_encoder(checkpoint, manifest=manifest)
        unet = EncoderUNetPL(
            encoder=encoder,
            encoder_in_channels=manifest.get("in_channels"),
            input_shape=size,
            encoder_config=manifest,
            image_key="image",
            deep_supervision=deep_supervision,
            **boilerplate,
            **network_config,
        )

    elif net_type == "unetpp":
        encoding_operations = encoding_operations[0]
        unet = UNetPlusPlusPL(
            encoding_operations=encoding_operations,
            image_key="image",
            **boilerplate,
            **network_config,
        )

    elif net_type == "unet":
        encoding_operations = encoding_operations[0]
        unet = UNetPL(
            encoding_operations=encoding_operations,
            image_key="image",
            deep_supervision=deep_supervision,
            **boilerplate,
            **network_config,
        )

    elif net_type == "unetr":
        sd = network_config["spatial_dimensions"]
        network_config["image_size"] = size[:sd]
        network_config["patch_size"] = network_config["patch_size"][:sd]
        unet = UNETRPL(
            image_key="image",
            deep_supervision=deep_supervision,
            **boilerplate,
            **network_config,
        )

    elif net_type == "monai_unetr":
        sd = network_config["spatial_dimensions"]
        network_config["image_size"] = size[:sd]
        network_config["patch_size"] = network_config["patch_size"][:sd]
        unet = MonaiUNETRPL(
            image_key="image",
            deep_supervision=deep_supervision,
            **boilerplate,
            **network_config,
        )

    elif net_type == "swin":
        sd = network_config["spatial_dimensions"]
        network_config["image_size"] = size[:sd]
        unet = SWINUNetPL(
            image_key="image",
            deep_supervision=deep_supervision,
            **boilerplate,
            **network_config,
        )

    elif net_type == "monai_swin":
        sd = network_config["spatial_dimensions"]
        network_config["image_size"] = size[:sd]
        network_config["patch_size"] = network_config["patch_size"][:sd]
        unet = MonaiSWINUNetPL(
            image_key="image",
            deep_supervision=deep_supervision,
            **boilerplate,
            **network_config,
        )

    return unet


def _default_resnet_backbone_args() -> dict[str, Any]:
    """
    Returns the default ResNet backbone arguments for SSL methods.

    Returns:
        dict[str, Any]: default backbone arguments.
    """
    return {
        "spatial_dim": 2,
        "in_channels": 1,
        "structure": [
            (64, 64, 3, 2),
            (128, 128, 3, 2),
            (256, 256, 3, 2),
            (512, 512, 3, 2),
        ],
        "maxpool_structure": [2, 2, 2, 2],
        "adn_fn": torch.nn.Identity,
        "res_type": "resnet",
    }


def _default_resnet_projection_head_args() -> dict[str, Any]:
    """
    Returns the default ResNet projection head arguments for SSL methods.

    Returns:
        dict[str, Any]: default projection head arguments.
    """
    return {
        "in_channels": 512,
        "structure": [512, 128],
        "adn_fn": torch.nn.Identity,
    }


def _default_resnet_prediction_head_args() -> dict[str, Any]:
    """
    Returns the default ResNet prediction head arguments for SSL methods.

    Returns:
        dict[str, Any]: default prediction head arguments.
    """
    return {
        "in_channels": 128,
        "structure": [512, 128],
        "adn_fn": torch.nn.Identity,
    }


def _vit_backbone_defaults() -> dict[str, Any]:
    """
    Returns the default ViT backbone arguments for the SSL methods that
    require a ViT backbone (IJEPA, DINO, iBOT).

    Returns:
        dict[str, Any]: default backbone arguments.
    """
    return {
        "image_size": [224, 224],
        "patch_size": [16, 16],
        "in_channels": 1,
        "number_of_blocks": 4,
        "attention_dim": 96,
        "embedding_size": 96,
        "n_heads": 3,
    }


def _compute_vit_dims(
    backbone_args: dict[str, Any],
) -> tuple[list[int], int]:
    """
    Computes the feature map dimensions and the number of encoder features
    from ViT backbone arguments.

    Args:
        backbone_args (dict[str, Any]): ViT backbone arguments containing
            ``image_size``, ``patch_size`` and either ``embedding_size`` or
            ``attention_dim``.

    Returns:
        tuple[list[int], int]: feature map dimensions (image size divided by
            patch size, element-wise) and the number of encoder features.
    """
    feature_map_dimensions = [
        s // p
        for s, p in zip(
            backbone_args["image_size"], backbone_args["patch_size"]
        )
    ]
    n_encoder_features = backbone_args.get(
        "embedding_size", backbone_args["attention_dim"]
    )
    return feature_map_dimensions, n_encoder_features


SSL_METHODS_WITH_EMA = {"byol", "ijepa", "dino", "ibot"}
SSL_METHODS_WITH_STOP_GRADIENT = {"byol", "ijepa", "dino", "ibot"}


def get_ema(ssl_method: str, n_steps: int) -> ExponentialMovingAverage | None:
    """
    Returns the exponential moving average teacher for a given SSL method.

    Args:
        ssl_method (str): the SSL method.
        n_steps (int): the number of optimization steps.

    Returns:
        ExponentialMovingAverage | None: the EMA module if the method uses
            an EMA teacher, otherwise ``None``.
    """
    if ssl_method not in SSL_METHODS_WITH_EMA:
        return None
    decay = 0.99 if ssl_method == "ijepa" else 0.996
    return ExponentialMovingAverage(
        decay=decay, final_decay=1.0, n_steps=n_steps
    )


def get_stop_gradients(ssl_method: str) -> bool:
    """
    Returns whether gradients should be stopped for a given SSL method.

    Args:
        ssl_method (str): the SSL method.

    Returns:
        bool: ``True`` if gradients should be stopped, ``False`` otherwise.
    """
    return ssl_method in SSL_METHODS_WITH_STOP_GRADIENT


@compile_if_necessary
def get_ssl_network(
    train_loader_call: Callable,
    max_epochs: int,
    max_steps_optim: int,
    warmup_steps: int,
    ssl_method: str,
    ema: torch.nn.Module,
    net_type: str,
    network_config: dict[str, Any],
    stop_gradient: bool,
    optimizer_eps: float = OPTIMIZER_EPS_DEFAULT,
) -> LightningModule:
    """
    Returns a SSL network.

    Args:
        train_loader_call (Callable): the training data loader.
        max_epochs (int): the maximum number of epochs.
        max_steps_optim (int): the maximum number of steps for optimization.
        warmup_steps (int): the number of warmup steps.
        ssl_method (str): the SSL method to use.
        ema (torch.nn.Module): the exponential moving average.
        net_type (str): the type of network to use.
        network_config (dict[str, Any]): the configuration of the network.
        stop_gradient (bool): whether to stop gradient.

    Returns:
        A :class:`LightningModule` SSL network.
    """
    # Common configuration for SSL methods
    common_params = {
        "training_dataloader_call": train_loader_call,
        "n_epochs": max_epochs,
        "n_steps": max_steps_optim,
        "warmup_steps": warmup_steps,
        "ema": ema,
        "batch_size": network_config.get("batch_size", 32),
        "optimizer_eps": optimizer_eps,
    }

    # pass the optimisation hyperparameters through when present in the
    # configuration file (keeps the module-level defaults otherwise)
    for key in ["learning_rate", "weight_decay"]:
        if key in network_config:
            common_params[key] = network_config[key]

    if ssl_method in ["simclr", "byol", "vicreg", "vicregl"]:
        # These methods use the standard ResNet architecture
        config = {
            "backbone_args": network_config.get(
                "backbone_args", _default_resnet_backbone_args()
            ),
            "projection_head_args": network_config.get(
                "projection_head_args",
                _default_resnet_projection_head_args(),
            ),
            "prediction_head_args": (
                None
                if ssl_method == "simclr"
                else network_config.get(
                    "prediction_head_args",
                    _default_resnet_prediction_head_args(),
                )
            ),
            "ssl_method": ssl_method,
            "aug_image_key_1": "augmented_image_1",
            "aug_image_key_2": "augmented_image_2",
            "stop_gradient": stop_gradient,
            "temperature": network_config.get("temperature", 0.1),
            "vic_reg_loss_params": network_config.get(
                "vic_reg_loss_params", {}
            ),
        }
        ssl = SelfSLResNetPL(**{**common_params, **config})

    elif ssl_method == "ijepa":
        # IJEPA specific configuration (ViT backbone + transformer predictor)
        if net_type != "vit":
            raise TypeError(
                "IJEPA only supports net_type='vit', got %s" % net_type
            )
        backbone_args: dict = network_config.get(
            "backbone_args", _vit_backbone_defaults()
        )
        feature_map_dimensions, n_encoder_features = _compute_vit_dims(
            backbone_args
        )
        predictor_head_args: dict = network_config.get(
            "projection_head_args",
            {
                "number_of_blocks": 2,
                "attention_dim": n_encoder_features,
                "hidden_dim": n_encoder_features,
                "n_heads": 3,
            },
        )
        config = {
            "image_key": "image",
            "backbone_args": backbone_args,
            "projection_head_args": predictor_head_args,
            "feature_map_dimensions": feature_map_dimensions,
            "n_encoder_features": n_encoder_features,
            "min_patch_size": network_config.get("min_patch_size", [4, 4]),
            "max_patch_size": network_config.get("max_patch_size", [8, 8]),
            "n_patches": network_config.get("n_patches", 1),
            "n_masked_patches": network_config.get("n_masked_patches", 4),
            "predictor_dim": network_config.get("predictor_dim", None),
            "stop_gradient": stop_gradient,
            "optimizer_eps": optimizer_eps,
        }
        ssl = IJEPAPL(**{**common_params, **config})

    elif ssl_method == "mae":
        # MAE specific configuration
        encoder_args: dict = network_config.get("encoder_args", {})
        decoder_args: dict = network_config.get("decoder_args", {})
        config = {
            "image_key": "image",
            "image_size": encoder_args.get("image_size", (224, 224)),
            "patch_size": encoder_args.get("patch_size", (16, 16)),
            "in_channels": encoder_args.get("in_channels", 1),
            "input_dim_size": encoder_args.get("embed_dim", 96),
            "encoder_args": encoder_args,
            "decoder_args": decoder_args,
            "mask_fraction": network_config.get("mask_fraction", 0.75),
            "optimizer_eps": optimizer_eps,
        }
        del common_params["ema"]
        ssl = ViTMaskedAutoEncoderPL(**{**common_params, **config})

    elif ssl_method == "dino":
        # DINO specific configuration (ViT backbone + MLP projection head)
        if net_type != "vit":
            raise TypeError(
                "DINO only supports net_type='vit', got %s" % net_type
            )
        backbone_args: dict = network_config.get(
            "backbone_args", _vit_backbone_defaults()
        )
        projection_head_args: dict = network_config.get(
            "projection_head_args",
            {"structure": [512, 256, 128]},
        )
        projection_head_args.setdefault(
            "adn_fn", get_adn_fn(1, "identity", "gelu", dropout_param=0.0)
        )
        config = {
            "aug_image_key_1": "augmented_image_1",
            "aug_image_key_2": "augmented_image_2",
            "backbone_args": backbone_args,
            "projection_head_args": projection_head_args,
            "out_dim": network_config.get("out_dim", 65536),
            "temperature": network_config.get("temperature", 0.1),
            "centers_m": network_config.get("centers_m", 0.9),
            "teacher_score_method": network_config.get(
                "teacher_score_method", "center"
            ),
            "teacher_temperature": network_config.get(
                "teacher_temperature", 0.04
            ),
            "teacher_temperature_final": network_config.get(
                "teacher_temperature_final", 0.07
            ),
            "teacher_temperature_warmup_epochs": network_config.get(
                "teacher_temperature_warmup_epochs", 30
            ),
            "freeze_last_layer_epochs": network_config.get(
                "freeze_last_layer_epochs", 1
            ),
            "stop_gradient": stop_gradient,
            "optimizer_eps": optimizer_eps,
        }
        ssl = DINOPL(**{**common_params, **config})

    elif ssl_method == "ibot":
        # iBOT specific configuration (ViT backbone + MLP projection head)
        if net_type != "vit":
            raise TypeError(
                "iBOT only supports net_type='vit', got %s" % net_type
            )
        backbone_args: dict = network_config.get(
            "backbone_args", _vit_backbone_defaults()
        )
        feature_map_dimensions, n_encoder_features = _compute_vit_dims(
            backbone_args
        )
        projection_head_args: dict = network_config.get(
            "projection_head_args",
            {"structure": [512, 256, 128]},
        )
        projection_head_args.setdefault(
            "adn_fn", get_adn_fn(1, "identity", "gelu", dropout_param=0.0)
        )
        config = {
            "aug_image_key_1": "augmented_image_1",
            "aug_image_key_2": "augmented_image_2",
            "backbone_args": backbone_args,
            "projection_head_args": projection_head_args,
            "out_dim": network_config.get("out_dim", 65536),
            "feature_map_dimensions": feature_map_dimensions,
            "n_encoder_features": n_encoder_features,
            "min_patch_size": network_config.get("min_patch_size", [4, 4]),
            "max_patch_size": network_config.get("max_patch_size", [8, 8]),
            "temperature": network_config.get("temperature", 0.1),
            "centers_m": network_config.get("centers_m", 0.9),
            "teacher_score_method": network_config.get(
                "teacher_score_method", "center"
            ),
            "stop_gradient": stop_gradient,
            "optimizer_eps": optimizer_eps,
        }
        ssl = iBOTPL(**{**common_params, **config})

    elif ssl_method == "barlow":
        # barlow twins-specific configuration (resnet backbone, projector
        # head whose output is used directly in the cross-correlation loss)
        backbone_args: dict = network_config.get(
            "backbone_args", _default_resnet_backbone_args()
        )
        projection_head_args: dict = network_config.get(
            "projection_head_args",
            {"in_channels": 512, "structure": [2048, 8192]},
        )
        config = {
            "image_key": "augmented_image_1",
            "augmented_image_key": "augmented_image_2",
            "backbone_args": backbone_args,
            "projection_head_args": projection_head_args,
            "loss_lam": network_config.get("loss_lam", 0.005),
            "optimizer_eps": optimizer_eps,
        }
        ssl = BarlowTwinsPL(
            training_dataloader_call=train_loader_call,
            learning_rate=network_config.get("learning_rate", 0.001),
            weight_decay=network_config.get("weight_decay", 0.005),
            batch_size=network_config.get("batch_size", 32),
            **config,
        )

    else:
        boilerplate = {
            "training_dataloader_call": train_loader_call,
            "aug_image_key_1": "augmented_image_1",
            "aug_image_key_2": "augmented_image_2",
            "box_key_1": "box_1",
            "box_key_2": "box_2",
            "n_epochs": max_epochs,
            "n_steps": max_steps_optim,
            "warmup_steps": warmup_steps,
            "ssl_method": ssl_method,
            "ema": ema,
            "stop_gradient": stop_gradient,
            "temperature": 0.1,
            "optimizer_eps": optimizer_eps,
        }
        if net_type == "unet_encoder":
            ssl = SelfSLUNetPL(**boilerplate, **network_config)
        elif net_type == "convnext":
            network_config["backbone_args"] = {
                k: network_config["backbone_args"][k]
                for k in network_config["backbone_args"]
                if k not in ["res_type"]
            }
            ssl = SelfSLConvNeXtPL(**boilerplate, **network_config)
        else:
            ssl = SelfSLResNetPL(**boilerplate, **network_config)

    return ssl


@compile_if_necessary
def get_ssl_network_no_pl(
    ssl_method: str, net_type: str, network_config: dict[str, Any]
) -> torch.nn.Module:
    """
    Returns a SSL network.

    Args:
        ssl_method (str): the SSL method to use.
        net_type (str): the type of network to use.
        network_config (dict[str, Any]): the configuration of the network.

    Returns:
        A :class:`torch.nn.Module` SSL network.
    """
    if ssl_method == "ijepa":
        ssl = IJEPA(**network_config)

    else:
        if net_type == "unet_encoder":
            ssl = UNet(**network_config)
        elif net_type == "vit":
            from adell_mri.modules.layers.vit import ViT

            ssl = ViT(**network_config)
        elif net_type == "convnext":
            network_config["backbone_args"] = {
                k: network_config["backbone_args"][k]
                for k in network_config["backbone_args"]
                if k not in ["res_type"]
            }
            ssl = ConvNeXt(**network_config)
        else:
            ssl = ResNet(**network_config)

    return ssl


@compile_if_necessary
def get_generative_network(
    network_config: dict[str, Any],
    scheduler_config: dict[str, Any],
    categorical_specification: list[list[str] | int],
    numerical_specification: int,
    uncondition_proba: float,
    train_loader_call: Callable,
    max_epochs: int,
    warmup_steps: int,
    start_decay: int,
    diffusion_steps: int,
    concat_condition_key: str = None,
    optimizer_eps: float = OPTIMIZER_EPS_DEFAULT,
) -> LightningModule:
    """
    Returns a generative network.

    Args:
        network_config (dict[str, Any]): the configuration of the network.
        scheduler_config (dict[str, Any]): the configuration of the scheduler.
        categorical_specification (list[list[str] | int]): the categorical specification.
        numerical_specification (int): the numerical specification.
        uncondition_proba (float): the uncondition probability.
        train_loader_call (Callable): the training data loader.
        max_epochs (int): the maximum number of epochs.
        warmup_steps (int): the number of warmup steps.
        start_decay (int): the number of steps to start decay.
        diffusion_steps (int): the number of diffusion steps.

    Returns:
        A :class:`LightningModule` generative network.
    """
    try:
        import monai.networks.schedulers  # noqa: F401

        from adell_mri.modules.diffusion.inferer import (
            DiffusionInfererSkipSteps,
        )
        from adell_mri.modules.diffusion.pl import DiffusionUNetPL
        from adell_mri.modules.diffusion.scheduler import DDPMScheduler
    except ImportError:
        raise ImportError(
            "Please install the generative package to diffusion models"
            "(go to https://github.com/Project-MONAI/GenerativeModels for "
            "instructions)"
        )

    scheduler = DDPMScheduler(
        num_train_timesteps=diffusion_steps, **scheduler_config
    )
    inferer = DiffusionInfererSkipSteps(scheduler)
    if any(
        [
            categorical_specification is not None,
            numerical_specification is not None,
        ]
    ):
        embedder = Embedder(
            categorical_specification,
            numerical_specification,
            embedding_size=network_config["cross_attention_dim"],
        )
    else:
        embedder = None

    boilerplate_args = {
        "training_dataloader_call": train_loader_call,
        "image_key": "image",
        "cat_condition_key": None,
        "num_condition_key": None,
        "concat_condition_key": concat_condition_key,
        "n_epochs": max_epochs,
        "warmup_steps": warmup_steps,
        "start_decay": start_decay,
        "uncondition_proba": uncondition_proba,
        "optimizer_eps": optimizer_eps,
    }

    if categorical_specification is not None:
        boilerplate_args["cat_condition_key"] = "cat"
    if numerical_specification is not None:
        boilerplate_args["num_condition_key"] = "num"

    network = DiffusionUNetPL(
        inferer=inferer,
        scheduler=scheduler,
        embedder=embedder,
        **boilerplate_args,
        **network_config,
    )

    return network


@compile_if_necessary
def get_gan_network(
    network_config: dict[str, Any],
    generator_config: dict[str, Any],
    discriminator_config: dict[str, Any],
    training_dataloader_call: Callable,
    input_image_key: str,
    categorical_specification: list[list[str] | int] | None,
    numerical_specification: int | None,
    numerical_moments: tuple[list[float], list[float]] | None,
    max_epochs: int,
    steps_per_epoch: int,
    pct_start: int,
    optimizer_eps: float = OPTIMIZER_EPS_DEFAULT,
) -> LightningModule:
    """
    Returns a GAN network.

    Args:
        network_config (dict[str, Any]): the configuration of the network.
        generator_config (dict[str, Any]): the configuration of the generator.
        discriminator_config (dict[str, Any]): the configuration of the discriminator.
        training_dataloader_call (Callable): the training data loader.
        input_image_key (str): the key corresponding to the input image.
        categorical_specification (list[list[str] | int] | None): the categorical specification.
        numerical_specification (int | None): the numerical specification.
        numerical_moments (tuple[list[float], list[float]] | None): the numerical moments.
        max_epochs (int): the maximum number of epochs.
        steps_per_epoch (int): the number of steps per epoch.
        pct_start (int): the percentage of steps for warm-up.

    Returns:
        A :class:`LightningModule` GAN network.
    """
    try:
        from adell_mri.modules.gan.discriminator import Discriminator
        from adell_mri.modules.gan.unet_generator import UNetGenerator
    except ImportError:
        raise ImportError(
            "Please install the generative package to use gan models"
            "(go to https://github.com/Project-MONAI/GenerativeModels for "
            "instructions)"
        )
    boilerplate_args = {
        "real_image_key": "image",
        "classification_target_key": None,
        "regression_target_key": None,
        "epochs": max_epochs,
        "steps_per_epoch": steps_per_epoch,
        "pct_start": pct_start,
        "training_dataloader_call": training_dataloader_call,
        "class_target_specification": categorical_specification,
        "reg_target_specification": numerical_specification,
        "numerical_moments": numerical_moments,
        "optimizer_eps": optimizer_eps,
    }

    for key in [
        "lambda_gp",
        "lambda_feature_matching",
        "lambda_feature_map_matching",
        "lambda_identity",
        "n_critic",
        "momentum_beta1",
        "momentum_beta2",
        "learning_rate",
        "batch_size",
        "patch_size",
    ]:
        if key in network_config:
            boilerplate_args[key] = network_config[key]

    if categorical_specification is not None:
        boilerplate_args["classification_target_key"] = "cat"
        discriminator_config["additional_classification_targets"] = [
            x if isinstance(x, int) else len(x)
            for x in categorical_specification
        ]
    if numerical_specification is not None:
        num_spec = numerical_specification
        boilerplate_args["regression_target_key"] = "num"
        discriminator_config["additional_regression_targets"] = num_spec
    if input_image_key is not None:
        boilerplate_args["input_image_key"] = input_image_key

    boilerplate_args["generator"] = UNetGenerator(**generator_config)
    boilerplate_args["discriminator"] = Discriminator(**discriminator_config)

    if network_config.get("cycle_consistency", False) is True:
        if network_config.get("cycle_symmetry", False) is True:
            boilerplate_args = {
                **boilerplate_args,
                "generator_cycle": boilerplate_args["generator"],
                "discriminator_cycle": boilerplate_args["discriminator"],
                "cycle_consistency": True,
                "cycle_symmetry": True,
            }
        else:
            cycle_gen_conf = {k: generator_config[k] for k in generator_config}
            cycle_gen_conf["in_channels"] = generator_config["out_channels"]
            cycle_gen_conf["out_channels"] = generator_config["in_channels"]
            boilerplate_args = {
                **boilerplate_args,
                "generator_cycle": UNetGenerator(**cycle_gen_conf),
                "discriminator_cycle": Discriminator(**discriminator_config),
                "cycle_consistency": True,
            }

    network = GANPL(**boilerplate_args)

    return network
