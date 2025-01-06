import argparse
import sys

import numpy as np
import torch

from adell_mri.modules.config_parsing import (parse_config_cat,
                                              parse_config_unet)
from adell_mri.utils.network_factories import get_classification_network

sys.path.append(r"..")


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
        "--n_channels",
        dest="n_channels",
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

    args = parser.parse_args(arguments)

    if args.net_type == "unet":
        network_config, _ = parse_config_unet(
            args.config_file, args.n_channels, args.n_classes
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

    network = get_classification_network(
        net_type=args.net_type,
        network_config=network_config,
        dropout_param=0.0,
        seed=42,
        n_classes=args.n_classes,
        keys=["image_{}".format(i) for i in range(args.n_channels)],
        clinical_feature_keys=[
            "tab_{}".format(i) for i in range(args.n_clinical_features)
        ],
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

    state_dict = torch.load(args.checkpoint, map_location=args.dev)[
        "state_dict"
    ]
    state_dict = {k: state_dict[k] for k in state_dict if "loss_fn" not in k}
    inc = network.load_state_dict(state_dict)
    print(inc)
    network.eval()

    # example = torch.rand(1, args.n_channels, *args.input_shape).to(args.dev)
    traced_network = network.to_torchscript()

    torch.jit.save(traced_network, args.output_model_path)
