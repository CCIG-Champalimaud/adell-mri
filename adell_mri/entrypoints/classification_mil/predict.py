import json
import sys
from pathlib import Path

import monai
import torch
from tqdm import tqdm

from ...entrypoints.assemble_args import Parser
from ...modules.classification.pl import (
    MultipleInstanceClassifierPL,
    TransformableTransformerPL,
)
from ...modules.config_parsing import parse_config_2d_classifier_3d
from ...monai_transforms import get_transforms_classification as get_transforms
from ...utils import EinopsRearranged, ScaleIntensityAlongDimd, safe_collate
from ...utils.dataset import Dataset
from ...utils.parser import get_params, merge_args, parse_ids
from ...utils.torch_utils import get_generator_and_rng


def main(arguments):
    parser = Parser()

    parser.add_argument_by_key(
        [
            "params_from",
            "dataset_json",
            "image_keys",
            "clinical_feature_keys",
            "adc_keys",
            "n_classes",
            "filter_on_keys",
            "target_spacing",
            "pad_size",
            "crop_size",
            "resize_size",
            "subsample_size",
            "batch_size",
            "cache_rate",
            "config_file",
            "mil_method",
            "module_path",
            "dev",
            "n_workers",
            "seed",
            "one_to_one",
            "prediction_ids",
            (
                "prediction_type",
                "type",
                {"choices": ["probability", "logit", "attention"]},
            ),
            "excluded_ids",
            ("prediction_checkpoints", "checkpoints"),
            "output_path",
        ]
    )

    args = parser.parse_args(arguments)

    if args.params_from is not None:
        param_dict = get_params(args.params_from)
        args = merge_args(args, param_dict, sys.argv[1:])

    g, rng = get_generator_and_rng(args.seed)

    data_dict = Dataset(args.dataset_json, rng=rng, verbose=True)
    all_prediction_pids = parse_ids(args.prediction_ids)
    if args.excluded_ids is not None:
        excluded_ids = parse_ids(args.excluded_ids, output_format="list")
        a = len(data_dict)
        data_dict = {
            k: data_dict[k] for k in data_dict if k not in excluded_ids
        }
        print(
            "Excluded {} cases with --excluded_ids".format(a - len(data_dict))
        )

    data_dict.filter_dictionary(
        filters_presence=args.image_keys,
        filters=args.filter_on_keys,
    )
    data_dict.subsample_dataset(subsample_size=args.subsample_size)

    if len(data_dict) == 0:
        raise Exception(
            "No data available for training \
                (dataset={}; keys={})".format(
                args.dataset_json, args.image_keys
            )
        )

    keys = args.image_keys
    adc_keys = []

    network_config, _ = parse_config_2d_classifier_3d(args.config_file, 0.0)

    if args.n_classes == 2:
        network_config["loss_fn"] = torch.nn.BCEWithLogitsLoss(torch.ones([]))
    else:
        network_config["loss_fn"] = torch.nn.CrossEntropy(
            torch.ones([args.n_classes])
        )

    if args.batch_size is not None:
        network_config["batch_size"] = args.batch_size

    if "batch_size" not in network_config:
        network_config["batch_size"] = 1

    all_pids = [k for k in data_dict]  # noqa

    print("Setting up transforms...")
    label_mode = "binary" if args.n_classes == 2 else "cat"
    transform_arguments = {
        "keys": keys,
        "adc_keys": adc_keys,
        "target_spacing": args.target_spacing,
        "target_size": args.resize_size,
        "crop_size": args.crop_size,
        "pad_size": args.pad_size,
        "possible_labels": None,
        "positive_labels": None,
        "label_key": None,
        "clinical_feature_keys": [],
        "label_mode": label_mode,
    }

    transforms = monai.transforms.Compose(
        [
            *get_transforms("pre", **transform_arguments),
            *get_transforms("post", **transform_arguments),
            EinopsRearranged("image", "c h w d -> 1 h w (d c)"),
            ScaleIntensityAlongDimd("image", dim=-1),
        ]
    )

    all_metrics = []
    for iteration, prediction_pids in enumerate(all_prediction_pids):
        prediction_pids = [pid for pid in prediction_pids if pid in data_dict]
        prediction_list = [data_dict[pid] for pid in prediction_pids]
        prediction_dataset = monai.data.CacheDataset(
            prediction_list,
            transforms,
            cache_rate=args.cache_rate,
            num_workers=args.n_workers,
        )

        # PL sometimes needs a little hint to detect GPUs.
        torch.ones([1]).to("cuda" if "cuda" in args.dev else "cpu")

        prediction_loader = monai.data.ThreadDataLoader(
            prediction_dataset,
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
            n_slices = int(len(keys) * args.crop_size[-1])
            boilerplate_args = {
                "n_classes": args.n_classes,
                "training_dataloader_call": None,
                "image_key": "image",
                "label_key": None,
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

            state_dict = torch.load(checkpoint)["state_dict"]
            network.load_state_dict(state_dict)
            network = network.eval().to(args.dev)
            kwargs = {}
            if args.type == "attention":
                kwargs["return_attention"] = True
            prediction_output = []
            attention_output = []
            with tqdm(prediction_loader, total=len(prediction_loader)) as pbar:
                for idx, batch in enumerate(pbar):
                    batch = {
                        k: batch[k].to(args.dev)
                        for k in batch
                        if isinstance(batch[k], torch.Tensor)
                    }
                    prediction = network.predict_step(batch, idx, **kwargs)
                    if args.type == "probability":
                        if args.n_classes == 2:
                            prediction = torch.nn.functional.sigmoid(prediction)
                        else:
                            prediction = torch.nn.functional.softmax(
                                prediction, axis=-1
                            )
                    elif args.type == "logit":
                        prediction = prediction
                    elif args.type == "attention":
                        prediction, attention = prediction
                        attention_output.extend(attention)
                    prediction_output.extend(prediction)

            prediction_output = {
                k: x.detach().cpu().numpy().tolist()
                for x, k in zip(prediction_output, prediction_pids)
            }
            prediction_output = {"prediction": prediction_output}
            if len(attention_output) > 0:
                prediction_output["attention"] = {
                    k: x.detach().cpu().numpy().tolist()
                    for x, k in zip(attention_output, prediction_pids)
                }
            prediction_output["checkpoint"] = checkpoint
            all_metrics.append(prediction_output)

    Path(args.output_path).parent.mkdir(exist_ok=True, parents=True)
    with open(args.output_path, "w") as o:
        json.dump(all_metrics, o)
