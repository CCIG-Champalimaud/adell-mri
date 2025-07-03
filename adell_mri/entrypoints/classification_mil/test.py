import json
import sys
from pathlib import Path

import monai
import torch
from lightning.pytorch import Trainer

from adell_mri.entrypoints.assemble_args import Parser
from adell_mri.modules.classification.pl import (
    MultipleInstanceClassifierPL,
    TransformableTransformerPL,
)
from adell_mri.modules.config_parsing import parse_config_2d_classifier_3d
from adell_mri.transform_factory.transforms import ClassificationTransforms
from adell_mri.utils.dataset import Dataset
from adell_mri.utils.monai_transforms import (
    EinopsRearranged,
    ScaleIntensityAlongDimd,
)
from adell_mri.utils.parser import get_params, merge_args, parse_ids
from adell_mri.utils.pl_utils import get_devices
from adell_mri.utils.torch_utils import get_generator_and_rng
from adell_mri.utils.utils import safe_collate


def main(arguments):
    parser = Parser()

    parser.add_argument_by_key(
        [
            "params_from",
            "dataset_json",
            "image_keys",
            "clinical_feature_keys",
            "adc_keys",
            "label_keys",
            "filter_on_keys",
            "fill_missing_with_placeholder",
            "fill_conditional",
            "possible_labels",
            "positive_labels",
            "label_groups",
            "target_spacing",
            "pad_size",
            "crop_size",
            "resize_size",
            "subsample_size",
            "batch_size",
            "config_file",
            "mil_method",
            "module_path",
            "dev",
            "seed",
            "n_workers",
            "metric_path",
            "test_ids",
            "one_to_one",
            "cache_rate",
            "excluded_ids",
            ("test_checkpoints", "checkpoints"),
            "ensemble",
        ]
    )

    args = parser.parse_args(arguments)
    args = parser.parse_args(arguments)

    if args.params_from is not None:
        param_dict = get_params(args.params_from)
        args = merge_args(args, param_dict, sys.argv[1:])

    g, rng = get_generator_and_rng(args.seed)

    accelerator, devices, strategy = get_devices(args.dev)

    data_dict = Dataset(args.dataset_json, rng=rng, verbose=True)

    if args.excluded_ids is not None:
        data_dict.subsample_dataset(excluded_key_list=args.excluded_ids)

    data_dict.filter_dictionary(
        filters_presence=args.image_keys + [args.label_keys],
        possible_labels=args.possible_labels,
        label_key=args.label_keys,
        filters=args.filter_on_keys,
        fill_conditional=args.fill_conditional,
        fill_missing_with_value=args.fill_missing_with_placeholder,
    )
    data_dict.subsample_dataset(
        subsample_size=args.subsample_size,
        strata_key=args.label_keys,
    )

    if args.test_ids is not None:
        all_test_pids = parse_ids(args.test_ids)
    else:
        all_test_pids = [[key for key in data_dict]]

    all_classes = []
    for k in data_dict:
        C = data_dict[k][args.label_keys]
        if isinstance(C, list):
            C = max(C)
        all_classes.append(str(C))
    if args.positive_labels is None:
        n_classes = len(args.possible_labels)
    else:
        n_classes = 2

    if len(data_dict) == 0:
        raise Exception(
            "No data available for training (dataset={}; keys={}; labels={})".format(
                args.dataset_json, args.image_keys, args.label_keys
            )
        )

    keys = args.image_keys
    adc_keys = []

    network_config, _ = parse_config_2d_classifier_3d(args.config_file, 0.0)

    if n_classes == 2:
        network_config["loss_fn"] = torch.nn.BCEWithLogitsLoss(torch.ones([]))
    else:
        network_config["loss_fn"] = torch.nn.CrossEntropy(
            torch.ones([n_classes])
        )

    if args.batch_size is not None:
        network_config["batch_size"] = args.batch_size

    if "batch_size" not in network_config:
        network_config["batch_size"] = 1

    all_pids = [k for k in data_dict]  # noqa

    print("Setting up transforms...")
    label_mode = "binary" if n_classes == 2 else "cat"
    transform_arguments = {
        "keys": keys,
        "adc_keys": adc_keys,
        "target_spacing": args.target_spacing,
        "target_size": args.resize_size,
        "crop_size": args.crop_size,
        "pad_size": args.pad_size,
        "possible_labels": args.possible_labels,
        "positive_labels": args.positive_labels,
        "label_key": args.label_keys,
        "clinical_feature_keys": [],
        "label_mode": label_mode,
    }

    transforms = ClassificationTransforms(**transform_arguments).transforms(
        final_transforms=[
            EinopsRearranged("image", "c h w d -> 1 h w (d c)"),
            ScaleIntensityAlongDimd("image", dim=-1),
        ]
    )

    all_metrics = []
    for iteration, test_pids in enumerate(all_test_pids):
        test_list = [data_dict[pid] for pid in test_pids if pid in data_dict]
        test_dataset = monai.data.CacheDataset(
            test_list,
            transforms,
            cache_rate=args.cache_rate,
            num_workers=args.n_workers,
        )

        # PL sometimes needs a little hint to detect GPUs.
        torch.ones([1]).to("cuda" if "cuda" in args.dev else "cpu")

        test_loader = monai.data.ThreadDataLoader(
            test_dataset,
            batch_size=network_config["batch_size"],
            shuffle=False,
            num_workers=args.n_workers,
            collate_fn=safe_collate,
        )

        if args.one_to_one is True:
            checkpoint_list = [args.checkpoints[iteration]]
        else:
            checkpoint_list = args.checkpoints
        for checkpoint in checkpoint_list:
            print(f"Iteration {iteration} with checkpoint {checkpoint}")
            n_slices = int(len(keys) * args.crop_size[-1])
            boilerplate_args = {
                "n_classes": n_classes,
                "training_dataloader_call": None,
                "image_key": "image",
                "label_key": "label",
                "n_epochs": 0,
                "warmup_steps": 0,
                "training_batch_preproc": None,
                "start_decay": 0,
                "n_slices": n_slices,
            }

            network_config["module"] = torch.jit.load(args.module_path).to(
                args.dev
            )
            network_config["module"].requires_grad = False
            network_config["module"] = network_config["module"].eval()
            network_config["module"] = torch.jit.freeze(
                network_config["module"]
            )
            if "module_out_dim" not in network_config:
                print("2D module output size not specified, inferring...")
                input_example = torch.rand(
                    1, 1, *[int(x) for x in args.crop_size][:2]
                ).to(args.dev.split(":")[0])
                output = network_config["module"](input_example)
                network_config["module_out_dim"] = int(output.shape[1])
                print(
                    "2D module output size={}".format(
                        network_config["module_out_dim"]
                    )
                )
            if args.mil_method == "transformer":
                network = TransformableTransformerPL(
                    **boilerplate_args, **network_config
                )
            elif args.mil_method == "standard":
                network = MultipleInstanceClassifierPL(
                    **boilerplate_args, **network_config
                )

            train_loader_call = None  # noqa
            state_dict = torch.load(checkpoint, weights_only=False)[
                "state_dict"
            ]
            network.load_state_dict(state_dict)
            network = network.eval().to(args.dev)
            trainer = Trainer(accelerator=accelerator, devices=devices)
            test_metrics = trainer.test(network, test_loader)[0]
            test_metrics["checkpoint"] = checkpoint
            test_metrics["pids"] = test_pids
            all_metrics.append(test_metrics)

    Path(args.metric_path).parent.mkdir(exist_ok=True, parents=True)
    with open(args.metric_path, "w") as o:
        json.dump(all_metrics, o)
