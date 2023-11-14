import argparse
import random
import json
import numpy as np
import torch
import monai
from copy import deepcopy

from lightning.pytorch import Trainer

import sys
from ..assemble_args import Parser
from ...utils import safe_collate, subsample_dataset
from ...utils.pl_utils import get_devices
from ...monai_transforms import get_transforms_classification as get_transforms
from ...modules.classification.losses import OrdinalSigmoidalLoss
from ...modules.config_parsing import parse_config_unet, parse_config_cat
from ...utils.dataset_filters import filter_dictionary
from ...utils.network_factories import get_classification_network
from ...utils.parser import get_params, merge_args, parse_ids


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
            "image_masking",
            "image_crop_from_mask",
            "filter_on_keys",
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
            ("test_checkpoints", "checkpoints"),
        ]
    )

    args = parser.parse_args(arguments)

    if args.params_from is not None:
        param_dict = get_params(args.params_from)
        args = merge_args(args, param_dict, sys.argv[1:])

    torch.manual_seed(args.seed)
    random.seed(args.seed)
    np.random.seed(args.seed)
    g = torch.Generator()
    g.manual_seed(args.seed)
    rng = np.random.default_rng(args.seed)

    accelerator, devices, strategy = get_devices(args.dev)

    output_file = open(args.metric_path, "w")

    if args.clinical_feature_keys is None:
        clinical_feature_keys = []
    else:
        clinical_feature_keys = args.clinical_feature_keys

    data_dict = json.load(open(args.dataset_json, "r"))
    presence_keys = args.image_keys + [args.label_keys] + clinical_feature_keys
    if args.mask_key is not None:
        presence_keys.append(args.mask_key)
    data_dict = filter_dictionary(
        data_dict,
        filters_presence=presence_keys,
        possible_labels=args.possible_labels,
        label_key=args.label_keys,
        filters=args.filter_on_keys,
    )
    data_dict = subsample_dataset(
        data_dict=data_dict,
        subsample_size=args.subsample_size,
        rng=rng,
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
    elif args.positive_labels is None:
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
        "positive_labels": args.positive_labels,
        "label_groups": label_groups,
        "label_key": args.label_keys,
        "label_mode": label_mode,
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
            act_fn = "swish"
        batch_preprocessing = None

        if args.one_to_one is True:
            checkpoint_list = [args.checkpoints[iteration]]
        else:
            checkpoint_list = args.checkpoints
        for checkpoint in checkpoint_list:
            network = get_classification_network(
                net_type=args.net_type,
                network_config=network_config,
                dropout_param=0,
                seed=None,
                n_classes=n_classes,
                keys=input_keys,
                clinical_feature_keys=clinical_feature_keys,
                train_loader_call=None,
                max_epochs=None,
                warmup_steps=None,
                start_decay=None,
                crop_size=args.crop_size,
                clinical_feature_means=None,
                clinical_feature_stds=None,
                label_smoothing=None,
                mixup_alpha=None,
                partial_mixup=None,
            )

            state_dict = torch.load(checkpoint)["state_dict"]
            state_dict = {
                k: state_dict[k]
                for k in state_dict
                if "loss_fn.weight" not in k
            }
            network.load_state_dict(state_dict)
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
