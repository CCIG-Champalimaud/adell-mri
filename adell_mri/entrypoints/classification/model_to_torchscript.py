import argparse

import numpy as np
import torch

from adell_mri.modules.config_parsing import parse_config_cat, parse_config_unet
from adell_mri.utils.network_factories import (
    get_classification_network,
    get_deconfounded_classification_network,
)
from adell_mri.utils.python_logging import get_logger
from adell_mri.utils.torch_utils import load_checkpoint_to_model

logger = get_logger(__name__)


def main(arguments):
    parser = argparse.ArgumentParser(
        description="Converts a model to torchscript"
    )

    parser.add_argument(
        "--input_shape",
        dest="input_shape",
        type=int,
        nargs="+",
        help="Input shape",
        required=True,
    )
    parser.add_argument(
        "-in_channels",
        dest="in_channels",
        type=int,
        help="Number of input channels",
        required=True,
    )
    parser.add_argument(
        "--n_classes",
        dest="n_classes",
        type=int,
        help="Number of classes",
        required=True,
    )
    parser.add_argument(
        "--n_clinical_features",
        dest="n_clinical_features",
        type=int,
        help="Number of clinical features",
        default=0,
    )
    parser.add_argument(
        "--config_file",
        dest="config_file",
        help="Path to network configuration file (yaml)",
        required=True,
    )
    parser.add_argument(
        "--checkpoint",
        dest="checkpoint",
        help="Path to checkpoint",
        required=True,
    )
    parser.add_argument(
        "--output_path",
        dest="output_path",
        help="Path to traced model",
        required=True,
    )
    parser.add_argument(
        "--net_type",
        dest="net_type",
        help="Classification type.",
        choices=["cat", "ord", "unet", "vit", "factorized_vit"],
        default="cat",
    )
    parser.add_argument(
        "--dev",
        dest="dev",
        default="cpu",
        help="Device for PyTorch training",
        type=str,
    )
    parser.add_argument(
        "--cat_confounder_keys",
        dest="cat_confounder_keys",
        type=str,
        nargs="+",
        default=None,
        help="Keys corresponding to categorical confounder.",
    )
    parser.add_argument(
        "--cont_confounder_keys",
        dest="cont_confounder_keys",
        type=str,
        nargs="+",
        default=None,
        help="Keys corresponding to continuous confounder.",
    )
    parser.add_argument(
        "--n_features_deconfounder",
        dest="n_features_deconfounder",
        type=int,
        default=64,
        help="Number of features used for deconfounding.",
    )
    parser.add_argument(
        "--exclude_surrogate_variables",
        dest="exclude_surrogate_variables",
        action="store_true",
        default=False,
        help="Excludes variables used in deconfounding from prediction.",
    )

    args = parser.parse_args(arguments)

    if args.net_type == "unet":
        network_config, _ = parse_config_unet(
            args.config_file, args.in_channels, args.n_classes
        )
    else:
        network_config = parse_config_cat(args.config_file)

    network_config["batch_size"] = 1

    if args.n_clinical_features > 0:
        clinical_feature_means = np.zeros([args.n_clinical_features])
        clinical_feature_stds = np.ones([args.n_clinical_features])
    else:
        clinical_feature_means = None
        clinical_feature_stds = None

    input_keys = ["image_{}".format(i) for i in range(args.in_channels)]
    clinical_feature_keys = [
        "tab_{}".format(i) for i in range(args.n_clinical_features)
    ]

    use_deconfounder = bool(
        args.cat_confounder_keys or args.cont_confounder_keys
    )

    if use_deconfounder:
        cat_vars_from_keys = None
        if args.cat_confounder_keys is not None:
            cat_vars_from_keys = [
                ["placeholder"] for _ in args.cat_confounder_keys
            ]
        cont_vars_from_keys = (
            len(args.cont_confounder_keys)
            if args.cont_confounder_keys is not None
            else None
        )
        network = get_deconfounded_classification_network(
            network_config=network_config,
            dropout_param=0.0,
            seed=42,
            n_classes=args.n_classes,
            keys=input_keys,
            cat_confounder_key=(
                "cat_confounder"
                if args.cat_confounder_keys is not None
                else None
            ),
            cont_confounder_key=(
                "cont_confounder"
                if args.cont_confounder_keys is not None
                else None
            ),
            cat_vars=cat_vars_from_keys,
            cont_vars=cont_vars_from_keys,
            train_loader_call=None,
            max_epochs=1,
            warmup_steps=None,
            start_decay=None,
            n_features_deconfounder=args.n_features_deconfounder,
            exclude_surrogate_variables=(args.exclude_surrogate_variables),
            label_smoothing=False,
            mixup_alpha=False,
            partial_mixup=False,
        )
    else:
        network = get_classification_network(
            net_type=args.net_type,
            network_config=network_config,
            dropout_param=0.0,
            seed=42,
            n_classes=args.n_classes,
            keys=input_keys,
            clinical_feature_keys=clinical_feature_keys,
            train_loader_call=None,
            max_epochs=1,
            warmup_steps=None,
            start_decay=None,
            crop_size=args.input_shape,
            clinical_feature_means=clinical_feature_means,
            clinical_feature_stds=clinical_feature_stds,
            label_smoothing=False,
            mixup_alpha=False,
            partial_mixup=False,
        )

    load_checkpoint_to_model(
        network, args.checkpoint, exclude_from_state_dict=["loss_fn"]
    )
    network.eval()

    traced_network = network.to_torchscript()

    torch.jit.save(traced_network, args.output_path)
