"""
Segmentation network factories.
"""

from typing import Any, Callable

import torch
from lightning.pytorch import LightningModule

from adell_mri.constants import DEFAULT_SEED
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
from adell_mri.modules.semi_supervised_segmentation.losses import (
    LocalContrastiveLoss,
)
from adell_mri.modules.semi_supervised_segmentation.pl import (
    UNetContrastiveSemiSL,
)
from adell_mri.utils.encoder_loading import (
    load_ssl_encoder,
    read_encoder_manifest,
)
from adell_mri.utils.network_factories.common import (
    ALLOWED_NET_TYPES,
    compile_if_necessary,
)
from adell_mri.utils.optimizer_factory import OPTIMIZER_EPS_DEFAULT
from adell_mri.utils.torch_utils import ExponentialMovingAverage


def _get_segmentation_size(*sizes):
    """
    Returns the first non-None spatial size.

    Args:
        *sizes: candidate sizes.

    Returns:
        The first non-None size, or None.
    """
    for size in sizes:
        if size is not None:
            return size
    return None


def _set_image_and_patch_size(
    network_config: dict[str, Any], size, include_patch_size: bool
) -> None:
    """
    Sets the image and (optionally) patch sizes in a network configuration.

    Args:
        network_config (dict[str, Any]): network configuration.
        size: spatial size to use.
        include_patch_size (bool): whether to slice the patch size too.
    """
    spatial_dimensions = network_config["spatial_dimensions"]
    network_config["image_size"] = size[:spatial_dimensions]
    if include_patch_size:
        network_config["patch_size"] = network_config["patch_size"][
            :spatial_dimensions
        ]


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

    size = _get_segmentation_size(
        random_crop_size, crop_size, pad_size, resize_size
    )

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
        _set_image_and_patch_size(network_config, size, True)
        unet = UNETRPL(
            image_key="image",
            deep_supervision=deep_supervision,
            **boilerplate,
            **network_config,
        )

    elif net_type == "monai_unetr":
        _set_image_and_patch_size(network_config, size, True)
        unet = MonaiUNETRPL(
            image_key="image",
            deep_supervision=deep_supervision,
            **boilerplate,
            **network_config,
        )

    elif net_type == "swin":
        _set_image_and_patch_size(network_config, size, False)
        unet = SWINUNetPL(
            image_key="image",
            deep_supervision=deep_supervision,
            **boilerplate,
            **network_config,
        )

    elif net_type == "monai_swin":
        _set_image_and_patch_size(network_config, size, True)
        unet = MonaiSWINUNetPL(
            image_key="image",
            deep_supervision=deep_supervision,
            **boilerplate,
            **network_config,
        )

    return unet
