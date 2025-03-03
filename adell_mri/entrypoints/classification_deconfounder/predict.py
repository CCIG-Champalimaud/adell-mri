import json
import sys
from copy import deepcopy
from pathlib import Path

import monai
import numpy as np
import torch
from tqdm import tqdm

from adell_mri.entrypoints.assemble_args import Parser
from adell_mri.utils.prediction_utils import get_ensemble_prediction
from adell_mri.utils.torch_utils import get_generator_and_rng

from ...modules.config_parsing import parse_config_cat, parse_config_unet
from ...transform_factory.transforms import ClassificationTransforms
from ...utils.dataset import Dataset
from ...utils.network_factories import get_deconfounded_classification_network
from ...utils.parser import get_params, merge_args, parse_ids


def main(arguments):
    parser = Parser()

    parser.add_argument_by_key(
        [
            ("classification_net_type", "net_type"),
            "params_from",
            "dataset_json",
            "image_keys",
            "adc_keys",
            "mask_key",
            "cat_confounder_keys",
            "cont_confounder_keys",
            "exclude_surrogate_variables",
            "n_features_deconfounder",
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
            "ensemble",
            "output_path",
        ]
    )

    args = parser.parse_args(arguments)

    if args.params_from is not None:
        param_dict = get_params(args.params_from)
        args = merge_args(args, param_dict, sys.argv[1:])

    g, rng = get_generator_and_rng(args.seed)

    data_dict = Dataset(args.dataset_json, rng=rng, verbose=True)
    presence_keys = args.image_keys
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
        filters=args.filter_on_keys,
    )
    data_dict.subsample_dataset(subsample_size=args.subsample_size)

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
        "clinical_feature_keys": [],
        "adc_keys": adc_keys,
        "target_spacing": args.target_spacing,
        "crop_size": args.crop_size,
        "pad_size": args.pad_size,
    }

    transforms_prediction = ClassificationTransforms(
        **transform_arguments
    ).transforms()

    global_output = []

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
        else:
            network_config["loss_fn"] = torch.nn.CrossEntropy()

        if args.net_type == "unet":
            act_fn = network_config["activation_fn"]
        else:
            act_fn = "swish"  # noqa
        batch_preprocessing = None  # noqa

        if args.one_to_one is True and args.ensemble is None:
            checkpoint_list = [args.checkpoints[iteration]]
        else:
            checkpoint_list = args.checkpoints
        for checkpoint in checkpoint_list:
            print(f"Predicting for {checkpoint}")
            network = get_deconfounded_classification_network(
                network_config=network_config,
                dropout_param=0,
                seed=None,
                cat_confounder_key=cat_key,
                cont_confounder_key=cont_key,
                cat_vars=cat_vars,
                cont_vars=cont_vars,
                n_classes=args.n_classes,
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

            state_dict = torch.load(checkpoint, weights_only=False)[
                "state_dict"
            ]
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
                        element["image"].unsqueeze(0).to(args.dev)
                    )
                    if args.type == "features":
                        output = output[-1].detach()
                        output = output.flatten(start_dim=2)
                        output = output.max(-1).values.cpu()
                    else:
                        output = output[0].detach()
                        output = output.cpu()
                    output = post_proc_fn(output)
                    output = output.numpy()[0].tolist()
                    output_dict["predictions"][identifier] = output
                    pbar.update()
            global_output.append(output_dict)

        if args.ensemble is not None:
            global_output.append(
                get_ensemble_prediction(global_output, args.ensemble)
            )
    Path(args.output_path).parent.mkdir(exist_ok=True, parents=True)
    with open(args.output_path, "w") as o:
        o.write(json.dumps(global_output))
