import random
import json
import numpy as np
import torch
import monai
from copy import deepcopy
from pathlib import Path
from tqdm import tqdm

import sys
from lib.entrypoints.assemble_args import Parser
from lib.utils.utils import subsample_dataset
from ...monai_transforms import get_transforms_classification as get_transforms
from ...modules.classification.losses import OrdinalSigmoidalLoss
from ...modules.config_parsing import parse_config_unet, parse_config_cat
from ...utils.dataset_filters import filter_dictionary
from ...utils.network_factories import get_classification_network
from ...utils.parser import parse_ids
from ...utils.parser import get_params, merge_args


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
            "mask_key",
            "image_masking",
            "image_crop_from_mask",
            "n_classes",
            "filter_on_keys",
            "target_spacing",
            "pad_size",
            "crop_size",
            "subsample_size",
            "batch_size",
            "cache_rate",
            "config_file",
            "dev",
            "n_workers",
            "seed",
            "one_to_one",
            "prediction_ids",
            ("prediction_type", "type"),
            ("prediction_checkpoints", "checkpoints"),
            "output_path",
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

    if args.clinical_feature_keys is None:
        clinical_feature_keys = []
    else:
        clinical_feature_keys = args.clinical_feature_keys

    data_dict = json.load(open(args.dataset_json, "r"))
    presence_keys = args.image_keys + clinical_feature_keys
    if args.mask_key is not None:
        presence_keys.append(args.mask_key)
    data_dict = filter_dictionary(
        data_dict,
        filters_presence=presence_keys,
        filters=args.filter_on_keys,
    )
    data_dict = subsample_dataset(
        data_dict=data_dict, subsample_size=args.subsample_size, rng=rng
    )

    if len(data_dict) == 0:
        raise Exception(
            "No data available for prediction \
                (dataset={}; keys={})".format(
                args.dataset_json, args.image_keys
            )
        )

    keys = args.image_keys
    adc_keys = args.adc_keys if args.adc_keys is not None else []
    adc_keys = [k for k in adc_keys if k in keys]
    mask_key = args.mask_key
    input_keys = deepcopy(keys)
    if mask_key is not None:
        input_keys.append(mask_key)

    if args.net_type == "unet":
        network_config, _ = parse_config_unet(
            args.config_file, len(keys), args.n_classes
        )
    else:
        network_config = parse_config_cat(args.config_file)

    if args.batch_size is not None:
        network_config["batch_size"] = args.batch_size

    if "batch_size" not in network_config:
        network_config["batch_size"] = 1

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
    }

    transforms_prediction = monai.transforms.Compose(
        [
            *get_transforms("pre", **transform_arguments),
            *get_transforms("post", **transform_arguments),
        ]
    )

    global_output = []
    if args.type in ["probability", "logit"]:
        extra_args = {}
    else:
        extra_args = {"return_features": True}

    if args.type == "probability":
        if args.n_classes > 2:
            post_proc_fn = torch.nn.Softmax(-1)
        else:
            post_proc_fn = torch.nn.Sigmoid()
    else:
        post_proc_fn = torch.nn.Identity()

    if args.prediction_ids:
        prediction_ids = parse_ids(args.prediction_ids)
    else:
        prediction_ids = [[k for k in data_dict]]
    for iteration in range(len(prediction_ids)):
        curr_prediction_ids = [
            pid for pid in prediction_ids[iteration] if pid in data_dict
        ]
        prediction_list = [data_dict[pid] for pid in curr_prediction_ids]

        prediction_dataset = monai.data.CacheDataset(
            prediction_list,
            transforms_prediction,
            num_workers=args.n_workers,
            cache_rate=args.cache_rate,
        )

        # PL sometimes needs a little hint to detect GPUs.
        torch.ones([1]).to("cuda" if "cuda" in args.dev else "cpu")

        if args.n_classes == 2:
            network_config["loss_fn"] = torch.nn.BCEWithLogitsLoss()
        elif args.net_type == "ord":
            network_config["loss_fn"] = OrdinalSigmoidalLoss(
                n_classes=args.n_classes
            )
        else:
            network_config["loss_fn"] = torch.nn.CrossEntropy()

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
            print(f"Predicting for {checkpoint}")
            network = get_classification_network(
                net_type=args.net_type,
                network_config=network_config,
                dropout_param=0,
                seed=None,
                n_classes=args.n_classes,
                keys=keys,
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
            network = network.eval().to(args.dev)

            output_dict = {
                "iteration": iteration,
                "prediction_ids": curr_prediction_ids,
                "checkpoint": checkpoint,
                "predictions": {},
            }
            with tqdm(total=len(curr_prediction_ids)) as pbar:
                for identifier, element in zip(
                    curr_prediction_ids, prediction_dataset
                ):
                    pbar.set_description("Predicting {}".format(identifier))
                    output = network.forward(
                        element["image"].unsqueeze(0).to(args.dev),
                        **extra_args,
                    ).detach()
                    if args.type == "features":
                        output = (
                            output.flatten(start_dim=2).max(-1).values.cpu()
                        )
                    else:
                        output = output.cpu()
                    output = post_proc_fn(output)
                    output = output.numpy()[0].tolist()
                    output_dict["predictions"][identifier] = output
                    pbar.update()
            global_output.append(output_dict)

    Path(args.output_path).parent.mkdir(exist_ok=True, parents=True)
    with open(args.output_path, "w") as o:
        o.write(json.dumps(global_output))
