from typing import Any, Callable, Dict, List

import numpy as np
import torch
import torch.nn.functional as F

from generative.networks.schedulers import DDPMScheduler

from ..modules.classification.classification import TabularClassifier
from ..modules.classification.classification.deconfounded_classification import \
    CategoricalConversion
# confounder-free classification
from ..modules.classification.pl import (ClassNetPL, DeconfoundedNetPL,
                                         FactorizedViTClassifierPL,
                                         HybridClassifierPL, UNetEncoderPL,
                                         ViTClassifierPL)
from ..modules.diffusion.embedder import Embedder
from ..modules.diffusion.inferer import DiffusionInfererSkipSteps
# diffusion
from ..modules.diffusion.pl import DiffusionUNetPL
from ..modules.gan.discriminator import Discriminator
from ..modules.gan.generator import Generator
# gan
from ..modules.gan.pl import GANPL
from ..modules.layers.adn_fn import get_adn_fn
# detection
from ..modules.object_detection.losses import complete_iou_loss
from ..modules.object_detection.pl import YOLONet3dPL
# segmentation
from ..modules.segmentation.pl import (UNETRPL, BrUNetPL, MonaiSWINUNetPL,
                                       MonaiUNETRPL, SWINUNetPL, UNetPL,
                                       UNetPlusPlusPL)
# self-supervised learning
from ..modules.self_supervised.pl import (DINOPL, IJEPA, IJEPAPL, ConvNeXt,
                                          ResNet, SelfSLConvNeXtPL,
                                          SelfSLResNetPL, SelfSLUNetPL, UNet,
                                          iBOTPL)
from ..modules.semi_supervised_segmentation.losses import LocalContrastiveLoss
# semi-supervised segmentation
from ..modules.semi_supervised_segmentation.pl import UNetContrastiveSemiSL
# classification
from ..utils.batch_preprocessing import BatchPreprocessing
from ..utils.utils import (ExponentialMovingAverage, get_loss_param_dict,
                           loss_factory)


def get_classification_network(
    net_type: str,
    network_config: Dict[str, Any],
    dropout_param: float,
    seed: int,
    n_classes: int,
    keys: List[str],
    clinical_feature_keys: List[str],
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
) -> torch.nn.Module:
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
        "n_channels": len(keys),
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
        network = UNetEncoderPL(
            head_structure=[network_config["depth"][-1] for _ in range(3)],
            head_adn_fn=get_adn_fn(
                1, norm_fn, act_fn="gelu", dropout_param=dropout_param
            ),
            **boilerplate_args,
            **network_config,
        )
    elif "vit" in net_type:
        image_size = [int(x) for x in crop_size]
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


def get_deconfounded_classification_network(
    network_config: Dict[str, Any],
    dropout_param: float,
    seed: int,
    n_classes: int,
    keys: List[str],
    cat_confounder_key: List[str],
    cont_confounder_key: List[str],
    cat_vars: List[List[str]],
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
) -> torch.nn.Module:
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
        "n_channels": len(keys),
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


def get_detection_network(
    net_type: str,
    network_config: Dict[str, Any],
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
) -> torch.nn.Module:
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


def get_segmentation_network(
    net_type: str,
    network_config: Dict[str, Any],
    bottleneck_classification: bool,
    clinical_feature_keys: List[str],
    all_aux_keys: List[str],
    clinical_feature_params: Dict[str, torch.Tensor],
    clinical_feature_key_net: str,
    aux_key_net: str,
    max_epochs: int,
    encoding_operations: List[torch.nn.Module],
    picai_eval: bool,
    lr_encoder: float,
    encoder_checkpoint: str,
    res_config_file: str,
    deep_supervision: bool,
    n_classes: int,
    keys: List[str],
    optimizer_str: str = "sgd",
    start_decay: float | int = 1.0,
    warmup_steps: float | int = 0.0,
    train_loader_call: Callable = None,
    random_crop_size: List[int] = None,
    crop_size: List[int] = None,
    pad_size: List[int] = None,
    resize_size: List[int] = None,
    semi_supervised: bool = False,
    max_steps_optim: int = None,
    seed: int = 42,
) -> torch.nn.Module:
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
        nc = network_config["n_channels"]
        network_config["n_channels"] = nc // len(keys)
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


def get_ssl_network(
    train_loader_call: Callable,
    max_epochs: int,
    max_steps_optim: int,
    warmup_steps: int,
    ssl_method: str,
    ema: torch.nn.Module,
    net_type: str,
    network_config_correct: Dict[str, Any],
    stop_gradient: bool,
):
    if ssl_method == "ijepa":
        boilerplate = {
            "training_dataloader_call": train_loader_call,
            "image_key": "image",
            "n_epochs": max_epochs,
            "n_steps": max_steps_optim,
            "warmup_steps": warmup_steps,
            "ema": ema,
            "stop_gradient": stop_gradient,
            "temperature": 0.1,
        }
        ssl = IJEPAPL(**boilerplate, **network_config_correct)

    elif ssl_method == "dino":
        boilerplate = {
            "training_dataloader_call": train_loader_call,
            "aug_image_key_1": "augmented_image_1",
            "aug_image_key_2": "augmented_image_2",
            "n_epochs": max_epochs,
            "n_steps": max_steps_optim,
            "warmup_steps": warmup_steps,
            "ema": ema,
            "stop_gradient": stop_gradient,
            "temperature": 0.1,
        }
        ssl = DINOPL(**boilerplate, **network_config_correct)

    elif ssl_method == "ibot":
        boilerplate = {
            "training_dataloader_call": train_loader_call,
            "aug_image_key_1": "augmented_image_1",
            "aug_image_key_2": "augmented_image_2",
            "n_epochs": max_epochs,
            "n_steps": max_steps_optim,
            "warmup_steps": warmup_steps,
            "ema": ema,
            "stop_gradient": stop_gradient,
            "temperature": 0.1,
        }
        ssl = iBOTPL(**boilerplate, **network_config_correct)

    else:
        if ssl_method == "simclr":
            # simclr only uses a projection head, no prediction head
            del network_config_correct["prediction_head_args"]
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
            ssl = SelfSLUNetPL(**boilerplate, **network_config_correct)
        elif net_type == "convnext":
            network_config_correct["backbone_args"] = {
                k: network_config_correct["backbone_args"][k]
                for k in network_config_correct["backbone_args"]
                if k not in ["res_type"]
            }
            ssl = SelfSLConvNeXtPL(**boilerplate, **network_config_correct)
        else:
            ssl = SelfSLResNetPL(**boilerplate, **network_config_correct)

    return ssl


def get_ssl_network_no_pl(
    ssl_method: str, net_type: str, network_config_correct: Dict[str, Any]
):
    if ssl_method == "ijepa":
        ssl = IJEPA(**network_config_correct)

    else:
        if net_type == "unet_encoder":
            ssl = UNet(**network_config_correct)
        elif net_type == "convnext":
            network_config_correct["backbone_args"] = {
                k: network_config_correct["backbone_args"][k]
                for k in network_config_correct["backbone_args"]
                if k not in ["res_type"]
            }
            ssl = ConvNeXt(**network_config_correct)
        else:
            ssl = ResNet(**network_config_correct)

    return ssl


def get_generative_network(
    network_config: Dict[str, Any],
    scheduler_config: Dict[str, Any],
    categorical_specification: List[List[str] | int],
    numerical_specification: int,
    uncondition_proba: float,
    train_loader_call: Callable,
    max_epochs: int,
    warmup_steps: int,
    start_decay: int,
    diffusion_steps: int,
) -> DiffusionUNetPL:
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


def get_gan_network(
    network_config: Dict[str, Any],
    generator_config: Dict[str, Any],
    discriminator_config: Dict[str, Any],
    training_dataloader_call: Callable,
    input_image_key: str,
    categorical_specification: List[List[str] | int] | None,
    numerical_specification: int | None,
    numerical_moments: tuple[list[float], list[float]] | None,
    max_epochs: int,
    steps_per_epoch: int,
    pct_start: int,
) -> torch.nn.Module:
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
