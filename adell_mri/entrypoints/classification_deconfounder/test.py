import sys
from copy import deepcopy

import monai
import numpy as np
import torch
from lightning.pytorch import Trainer

from ...modules.classification.losses import OrdinalSigmoidalLoss
from ...modules.classification.pl import AveragingEnsemblePL
from ...modules.config_parsing import parse_config_cat, parse_config_unet
from ...monai_transforms import get_transforms_classification as get_transforms
from ...utils import safe_collate
from ...utils.bootstrap_metrics import bootstrap_metric
from ...utils.dataset import Dataset
from ...utils.network_factories import get_deconfounded_classification_network
from ...utils.parser import get_params, merge_args, parse_ids
from ...utils.pl_utils import get_devices
from ...utils.torch_utils import (get_generator_and_rng,
                                  load_checkpoint_to_model)
from ..assemble_args import Parser


def main(arguments):
    parser = Parser()

    parser.add_argument_by_key(
        [
            ("classification_net_type", "net_type"),
            "params_from",
            "dataset_json",
            "image_keys",
            "clinical_feature_keys",
            "adc_keys",
            "label_keys",
            "mask_key",
            "cat_confounder_keys",
            "cont_confounder_keys",
            "exclude_surrogate_variables",
            "n_features_deconfounder",
            "image_masking",
            "image_crop_from_mask",
            "filter_on_keys",
            "fill_missing_with_placeholder",
            "fill_conditional",
            "possible_labels",
            "positive_labels",
            "label_groups",
            "cache_rate",
            "target_spacing",
            "pad_size",
            "crop_size",
            "subsample_size",
            "batch_size",
            "config_file",
            "dev",
            "seed",
            "n_workers",
            "metric_path",
            "test_ids",
            "one_to_one",
            "ensemble",
            ("test_checkpoints", "checkpoints"),
            "exclude_from_state_dict",
        ]
    )

    args = parser.parse_args(arguments)

    if args.params_from is not None:
        param_dict = get_params(args.params_from)
        args = merge_args(args, param_dict, sys.argv[1:])

    g, rng = get_generator_and_rng(args.seed)

    accelerator, devices, strategy = get_devices(args.dev)

    output_file = open(args.metric_path, "w")

    if args.clinical_feature_keys is None:
        clinical_feature_keys = []
    else:
        clinical_feature_keys = args.clinical_feature_keys

    data_dict = Dataset(args.dataset_json, rng=rng, verbose=True)

    presence_keys = args.image_keys + [args.label_keys] + clinical_feature_keys
    if args.mask_key is not None:
        presence_keys.append(args.mask_key)
    cat_key = None
    cont_key = None
    if args.cat_confounder_keys is not None:
        cat_key = "cat_confounder"
    if args.cont_confounder_keys is not None:
        cont_key = "cont_confounder"
    data_dict.filter_dictionary(
        filters_presence=presence_keys,
        possible_labels=args.possible_labels,
        label_key=args.label_keys,
        filters=args.filter_on_keys,
    )
    data_dict.subsample_dataset(
        subsample_size=args.subsample_size,
        strata_key=args.label_keys,
    )

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
    elif positive_labels is None:
        n_classes = len(args.possible_labels)
    else:
        n_classes = 2

    if len(data_dict) == 0:
        raise Exception(
            "No data available for testing \
                (dataset={}; keys={}; labels={})".format(
                args.dataset_json, args.image_keys, args.label_keys
            )
        )

    keys = args.image_keys
    adc_keys = args.adc_keys if args.adc_keys is not None else []
    mask_key = args.mask_key
    input_keys = deepcopy(keys)
    if mask_key is not None:
        input_keys.append(mask_key)
    adc_keys = [k for k in adc_keys if k in keys]

    cat_vars = None
    cont_vars = None
    if args.cat_confounder_keys is not None:
        cat_vars = []
        for k in args.cat_confounder_keys:
            curr_cat_vars = []
            for kk in data_dict:
                v = data_dict[kk][k]
                if v not in curr_cat_vars:
                    curr_cat_vars.append(v)
            cat_vars.append(curr_cat_vars)
    if args.cont_confounder_keys is not None:
        cont_vars = len(args.cont_confounder_keys)

    if args.net_type == "unet":
        network_config, _ = parse_config_unet(
            args.config_file, len(input_keys), n_classes
        )
    else:
        network_config = parse_config_cat(args.config_file)

    if args.batch_size is not None:
        network_config["batch_size"] = args.batch_size

    if "batch_size" not in network_config:
        network_config["batch_size"] = 1

    if network_config["batch_size"] > len(data_dict):
        network_config["batch_size"] = len(data_dict)

    all_pids = [k for k in data_dict]

    print("Setting up transforms...")
    label_mode = "binary" if n_classes == 2 and label_groups is None else "cat"
    transform_arguments = {
        "keys": keys,
        "mask_key": mask_key,
        "image_masking": args.image_masking,
        "image_crop_from_mask": args.image_crop_from_mask,
        "clinical_feature_keys": clinical_feature_keys,
        "adc_keys": adc_keys,
        "target_spacing": args.target_spacing,
        "crop_size": args.crop_size,
        "pad_size": args.pad_size,
        "possible_labels": args.possible_labels,
        "positive_labels": positive_labels,
        "label_groups": label_groups,
        "label_key": args.label_keys,
        "label_mode": label_mode,
        "cat_confounder_keys": args.cat_confounder_keys,
        "cont_confounder_keys": args.cont_confounder_keys,
    }

    transforms_val = monai.transforms.Compose(
        [
            *get_transforms("pre", **transform_arguments),
            *get_transforms("post", **transform_arguments),
        ]
    )
    transforms_val.set_random_state(args.seed)

    if args.test_ids is not None:
        all_test_ids = parse_ids(args.test_ids)
    else:
        all_test_ids = [all_pids]
    for iteration in range(len(all_test_ids)):
        test_ids = all_test_ids[iteration]
        test_list = [data_dict[pid] for pid in test_ids if pid in data_dict]

        print("Testing fold", iteration)
        for u, c in zip(
            *np.unique(
                [x[args.label_keys] for x in test_list], return_counts=True
            )
        ):
            print(f"\tCases({u}) = {c}")

        test_dataset = monai.data.CacheDataset(
            test_list,
            transforms_val,
            num_workers=args.n_workers,
            cache_rate=args.cache_rate,
        )

        # PL sometimes needs a little hint to detect GPUs.
        torch.ones([1]).to("cuda" if "cuda" in args.dev else "cpu")

        if n_classes == 2:
            network_config["loss_fn"] = torch.nn.BCEWithLogitsLoss()
        elif args.net_type == "ord":
            network_config["loss_fn"] = OrdinalSigmoidalLoss(
                n_classes=n_classes
            )
        else:
            network_config["loss_fn"] = torch.nn.CrossEntropyLoss()

        test_loader = monai.data.ThreadDataLoader(
            test_dataset,
            batch_size=network_config["batch_size"],
            shuffle=False,
            num_workers=args.n_workers,
            collate_fn=safe_collate,
        )

        print("Setting up testing...")
        if args.net_type == "unet":
            act_fn = network_config["activation_fn"]
        else:
            act_fn = "swish"  # noqa
        batch_preprocessing = None  # noqa

        if args.one_to_one is True:
            checkpoint_list = [args.checkpoints[iteration]]
        else:
            checkpoint_list = args.checkpoints
        all_networks = []
        if args.exclude_from_state_dict is None:
            exclude_from_state_dict = []
        else:
            exclude_from_state_dict = args.exclude_from_state_dict
        for checkpoint in checkpoint_list:
            network = get_deconfounded_classification_network(
                network_config=network_config,
                dropout_param=0,
                seed=None,
                cat_confounder_key=cat_key,
                cont_confounder_key=cont_key,
                cat_vars=cat_vars,
                cont_vars=cont_vars,
                n_classes=n_classes,
                keys=input_keys,
                train_loader_call=None,
                max_epochs=None,
                warmup_steps=None,
                start_decay=None,
                label_smoothing=None,
                mixup_alpha=None,
                partial_mixup=None,
                n_features_deconfounder=args.n_features_deconfounder,
                exclude_surrogate_variables=args.exclude_surrogate_variables,
            )
            load_checkpoint_to_model(
                network,
                checkpoint,
                ["loss_fn.weight"] + exclude_from_state_dict,
            )
            all_networks.append(network)

        if args.ensemble is None:
            for network_idx, network in enumerate(all_networks):
                checkpoint = checkpoint_list[network_idx]
                network = network.eval()
                trainer = Trainer(accelerator=accelerator, devices=devices)
                test_metrics = trainer.test(network, test_loader)[0]
                for k in test_metrics:
                    out = test_metrics[k]
                    try:
                        value = float(out.detach().numpy())
                    except Exception:
                        value = float(out)
                    if n_classes > 2:
                        k = k.split("_")
                        if k[-1].isdigit():
                            k, idx = "_".join(k[:-1]), k[-1]
                        else:
                            k, idx = "_".join(k), 0
                    else:
                        idx = 0
                    x = "{},{},{},{},{}".format(
                        k, checkpoint, iteration, idx, value
                    )
                    output_file.write(x + "\n")
                    print(x)
                    # bootstrap AUC estimate
                mean, (upper, lower) = bootstrap_metric(
                    network.test_metrics["T_AUC"], 100, 0.5
                )
                for idx, (m, u, low) in enumerate(
                    zip(mean, upper, lower)
                ):  # noqa
                    x = "{},{},{},{},{}".format(
                        "T_AUC_mean", checkpoint, iteration, idx, m
                    )
                    output_file.write(x + "\n")
                    print(x)
                    x = "{},{},{},{},{}".format(
                        "T_AUC_lower", checkpoint, iteration, idx, u
                    )
                    output_file.write(x + "\n")
                    print(x)
                    x = "{},{},{},{},{}".format(
                        "T_AUC_upper", checkpoint, iteration, idx, low
                    )
                    output_file.write(x + "\n")
                    print(x)

        else:
            ensemble_network = AveragingEnsemblePL(
                networks=all_networks, n_classes=n_classes, idx=0
            )
            ensemble_network = ensemble_network.eval()
            trainer = Trainer(accelerator=accelerator, devices=devices)
            test_metrics = trainer.test(ensemble_network, test_loader)[0]
            for k in test_metrics:
                out = test_metrics[k]
                try:
                    value = float(out.detach().numpy())
                except Exception:
                    value = float(out)
                if n_classes > 2:
                    k = k.split("_")
                    if k[-1].isdigit():
                        k, idx = "_".join(k[:-1]), k[-1]
                    else:
                        k, idx = "_".join(k), 0
                else:
                    idx = 0
                x = "{},{},{},{},{}".format(
                    k, "ensemble", iteration, idx, value
                )
                output_file.write(x + "\n")
                print(x)
            # bootstrap AUC estimate
            mean, (upper, lower) = bootstrap_metric(
                ensemble_network.test_metrics["T_AUC"],
                samples=10000,
                sample_size=0.5,
            )
            for idx, (m, u, l) in enumerate(zip(mean, upper, lower)):  # noqa
                x = "{},{},{},{},{}".format(
                    "T_AUC_mean", "ensemble", iteration, idx, m
                )
                output_file.write(x + "\n")
                print(x)
                x = "{},{},{},{},{}".format(
                    "T_AUC_lower", "ensemble", iteration, idx, u
                )
                output_file.write(x + "\n")
                print(x)
                x = "{},{},{},{},{}".format(
                    "T_AUC_upper", "ensemble", iteration, idx, l
                )
                output_file.write(x + "\n")
                print(x)
