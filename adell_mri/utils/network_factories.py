"""
PyTorch Lightning network factories.

This module contains factory functions for creating PyTorch Lightning
modules for various types of networks, including classification, segmentation,
self-supervised learning, and generative models.
"""

from typing import Any, Callable

import os
import numpy as np
import torch
import torch.nn.functional as F
from functools import wraps
from lightning.pytorch import LightningModule


# confounder-free classification
from adell_mri.modules.classification.pl import (
    ClassNetPL,
    DeconfoundedNetPL,
    FactorizedViTClassifierPL,
    HybridClassifierPL,
    UNetEncoderPL,
    ViTClassifierPL,
)
from adell_mri.modules.diffusion.embedder import Embedder

# gan
from adell_mri.modules.gan.pl import GANPL
from adell_mri.modules.layers.adn_fn import get_adn_fn

# detection
from adell_mri.modules.object_detection.losses import complete_iou_loss
from adell_mri.modules.object_detection.pl import YOLONet3dPL

# segmentation
from adell_mri.modules.segmentation.pl import (
    UNETRPL,
    BrUNetPL,
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
    ConvNeXt,
    ResNet,
    SelfSLConvNeXtPL,
    SelfSLResNetPL,
    SelfSLUNetPL,
    UNet,
    iBOTPL,
    ViTMaskedAutoEncoderPL,
)
from adell_mri.modules.semi_supervised_segmentation.losses import (
    LocalContrastiveLoss,
)

# semi-supervised segmentation
from adell_mri.modules.semi_supervised_segmentation.pl import (
    UNetContrastiveSemiSL,
)

# classification
from adell_mri.modules.classification.classification import TabularClassifier
from adell_mri.modules.classification.classification.deconfounded_classification import (
    CategoricalConversion,
)
from adell_mri.utils.batch_preprocessing import BatchPreprocessing
from adell_mri.utils.utils import (
    ExponentialMovingAverage,
    get_loss_param_dict,
    loss_factory,
)

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
        if os.environ.get("TORCH_COMPILE", "False").lower() in ["true", "1"]:
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
    label_smoothing=None,
    mixup_alpha=None,
    partial_mixup=None,
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
    }
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
            for k in ["embed_method"]:
                if k in network_config:
                    del network_config[k]
            network = FactorizedViTClassifierPL(
                adn_fn=get_adn_fn(
                    1, "identity", act_fn="gelu", dropout_param=dropout_param
                ),
                **boilerplate_args,
                **network_config,
            )

    else:
        for k in ["in_channels", "n_classes", "adn_fn"]:
            if k in network_config:
                del network_config[k]
        network = ClassNetPL(
            net_type=net_type,
            adn_fn=adn_fn,
            **boilerplate_args,
            **network_config,
        )

    if len(clinical_feature_keys) > 0:
        boilerplate_args_hybrid = {
            "training_dataloader_call": train_loader_call,
            "image_key": "image",
            "label_key": "label",
            "n_epochs": max_epochs,
            "warmup_steps": warmup_steps,
            "training_batch_preproc": batch_preprocessing,
            "start_decay": start_decay,
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
    label_smoothing=None,
    mixup_alpha=None,
    partial_mixup=None,
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

    Returns:
        A :class:`LightningModule` deconfounded classification network.
    """
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
    }
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

    return network


@compile_if_necessary
def get_detection_network(
    network_config: dict[str, Any],
    dropout_param: float,
    loss_gamma: float,
    loss_comb: float,
    class_weights: torch.Tensor,
    train_loader_call: Callable,
    iou_threshold: float,
    n_classes: int,
    anchor_array: np.ndarray,
    n_epochs: int,
    warmup_steps: int,
    boxes_key: str,
    box_class_key: str,
    dev: str,
) -> LightningModule:
    if "activation_fn" in network_config:
        act_fn = network_config["activation_fn"]
    else:
        act_fn = "swish"

    if "classification_loss_fn" in network_config:
        k = "binary" if n_classes == 2 else "categorical"
        classification_loss_fn = loss_factory[k][
            network_config["classification_loss_fn"]
        ]
    else:
        if n_classes == 2:
            classification_loss_fn = F.binary_cross_entropy
        else:
            classification_loss_fn = F.cross_entropy

    if "object_loss_fn" in network_config:
        object_loss_key = network_config["object_loss_fn"]
        object_loss_fn = loss_factory["binary"][
            network_config["object_loss_fn"]
        ]
    else:
        object_loss_fn = F.binary_cross_entropy

    net_cfg = {
        k: network_config[k]
        for k in network_config
        if k
        not in ["activation_fn", "classification_loss_fn", "object_loss_fn"]
    }

    if "batch_size" not in net_cfg:
        net_cfg["batch_size"] = 1

    classification_loss_params = {}
    if (loss_gamma is None) or (loss_comb is None) or (class_weights is None):
        object_loss_params = {}
    else:
        object_loss_params = get_loss_param_dict(
            1.0, loss_gamma, loss_comb, 0.5
        )[object_loss_key]

    adn_fn = get_adn_fn(
        3, norm_fn="batch", act_fn=act_fn, dropout_param=dropout_param
    )
    network = YOLONet3dPL(
        training_dataloader_call=train_loader_call,
        image_key="image",
        label_key="bb_map",
        boxes_key=boxes_key,
        box_label_key=box_class_key,
        anchor_sizes=anchor_array,
        adn_fn=adn_fn,
        iou_threshold=iou_threshold,
        classification_loss_fn=classification_loss_fn,
        object_loss_fn=object_loss_fn,
        reg_loss_fn=complete_iou_loss,
        object_loss_params=object_loss_params,
        classification_loss_params=classification_loss_params,
        n_epochs=n_epochs,
        warmup_steps=warmup_steps,
        n_classes=n_classes,
        **net_cfg,
    )

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
    res_config_file: str | None,
    deep_supervision: bool,
    n_classes: int,
    keys: list[str],
    optimizer_str: str = "sgd",
    start_decay: float | int = 1.0,
    warmup_steps: float | int = 0.0,
    train_loader_call: Callable = None,
    random_crop_size: list[int] = None,
    crop_size: list[int] = None,
    pad_size: list[int] = None,
    resize_size: list[int] = None,
    semi_supervised: bool = False,
    max_steps_optim: int = None,
    seed: int = 42,
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
        encoder_checkpoint (str): the checkpoint of the encoder.
        res_config_file (str | None): the configuration file for the residual
            network.
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
        if encoder_checkpoint is not None and res_config_file is None:
            for encoder, ckpt in zip(unet.encoders, encoder_checkpoint):
                encoder.load_state_dict(
                    torch.load(ckpt, weights_only=False)["state_dict"]
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
    }

    if ssl_method in ["simclr", "byol", "vicreg", "vicregl"]:
        # These methods use the standard ResNet architecture
        config = {
            "backbone_args": network_config.get(
                "backbone_args",
                {
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
                },
            ),
            "projection_head_args": network_config.get(
                "projection_head_args",
                {
                    "in_channels": 512,
                    "structure": [512, 128],
                    "adn_fn": torch.nn.Identity,
                },
            ),
            "prediction_head_args": (
                None
                if ssl_method == "simclr"
                else network_config.get(
                    "prediction_head_args",
                    {
                        "in_channels": 128,
                        "structure": [512, 128],
                        "adn_fn": torch.nn.Identity,
                    },
                )
            ),
            "ssl_method": ssl_method,
            "stop_gradient": stop_gradient,
        }
        ssl = SelfSLResNetPL(**{**common_params, **config})

    elif ssl_method == "ijepa":
        # IJEPA specific configuration
        backbone_args: dict = network_config.get("backbone_args", {})
        config = {
            "image_key": "image",
            "backbone_args": {
                "in_channels": backbone_args.get("in_channels", 1),
                "patch_size": backbone_args.get("patch_size", (16, 16)),
                "img_size": backbone_args.get("img_size", (224, 224)),
                "embed_dim": backbone_args.get("embed_dim", 96),
                "depth": backbone_args.get("depth", 4),
                "num_heads": backbone_args.get("num_heads", 3),
                "mlp_ratio": backbone_args.get("mlp_ratio", 4.0),
                "qkv_bias": backbone_args.get("qkv_bias", True),
                "norm_layer": backbone_args.get(
                    "norm_layer", torch.nn.LayerNorm
                ),
            },
            "feature_map_dimensions": network_config.get(
                "feature_map_dimensions", [14, 14]
            ),
            "stop_gradient": stop_gradient,
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
        }
        del common_params["ema"]
        ssl = ViTMaskedAutoEncoderPL(**{**common_params, **config})

    elif ssl_method == "dino":
        # DINO specific configuration
        config = {
            "aug_image_key_1": "augmented_image_1",
            "aug_image_key_2": "augmented_image_2",
            "backbone_args": {
                "in_channels": 1,
                "patch_size": (16, 16),
                "img_size": (224, 224),
                "embed_dim": 96,
                "depth": 4,
                "num_heads": 3,
                "mlp_ratio": 4.0,
                "qkv_bias": True,
                "norm_layer": torch.nn.LayerNorm,
            },
            "projection_head_args": {
                "in_dim": 96
                * 14
                * 14,  # embed_dim * (img_size // patch_size) ** 2
                "hidden_dim": 512,
                "out_dim": 128,
                "num_layers": 3,
            },
            "temperature": 0.1,
            "stop_gradient": stop_gradient,
        }
        ssl = DINOPL(**{**common_params, **config})

    elif ssl_method == "ibot":
        # iBOT specific configuration
        config = {
            "aug_image_key_1": "augmented_image_1",
            "aug_image_key_2": "augmented_image_2",
            "backbone_args": {
                "in_channels": 1,
                "patch_size": (16, 16),
                "img_size": (224, 224),
                "embed_dim": 96,
                "depth": 4,
                "num_heads": 3,
                "mlp_ratio": 4.0,
                "qkv_bias": True,
                "norm_layer": torch.nn.LayerNorm,
            },
            "projection_head_args": {
                # embed_dim * (img_size // patch_size) ** 2
                "in_dim": 96 * 14 * 14,
                "hidden_dim": 512,
                "out_dim": 128,
                "num_layers": 3,
            },
            "feature_map_dimensions": [14, 14],
            "n_encoder_features": 96,
            "min_patch_size": 2,
            "max_patch_size": 8,
            "temperature": 0.1,
            "stop_gradient": stop_gradient,
        }
        ssl = iBOTPL(**{**common_params, **config})
    else:
        if ssl_method == "simclr":
            # simclr only uses a projection head, no prediction head
            del network_config["prediction_head_args"]
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
        from generative.networks.schedulers import DDPMScheduler
        from adell_mri.modules.diffusion.inferer import (
            DiffusionInfererSkipSteps,
        )
        from adell_mri.modules.diffusion.pl import DiffusionUNetPL
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
        "n_epochs": max_epochs,
        "warmup_steps": warmup_steps,
        "start_decay": start_decay,
        "uncondition_proba": uncondition_proba,
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
        from adell_mri.modules.gan.generator import Generator
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
    }

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
                "generator_cycle": Generator(**cycle_gen_conf),
                "discriminator_cycle": Discriminator(**discriminator_config),
                "cycle_consistency": True,
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

    boilerplate_args["generator"] = Generator(**generator_config)
    boilerplate_args["discriminator"] = Discriminator(**discriminator_config)

    network = GANPL(**boilerplate_args)

    return network
