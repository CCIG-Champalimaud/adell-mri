import json
import sys
from copy import deepcopy
from pathlib import Path

import monai
import torch
from tqdm import tqdm

from adell_mri.entrypoints.assemble_args import Parser
from adell_mri.entrypoints.classification._utils import (
    configure_loss_fn,
    create_classification_network,
    create_transform_arguments,
    extract_confounder_info,
    use_deconfounding,
)
from adell_mri.entrypoints.classification.predict_utils import (
    ClassificationPredictionAccumulator,
    predict_sample,
    resolve_checkpoint_list,
    resolve_postprocessing,
)
from adell_mri.modules.config_parsing import parse_config_cat, parse_config_unet
from adell_mri.transform_factory.transforms import ClassificationTransforms
from adell_mri.utils.dataset import Dataset
from adell_mri.utils.parser import get_params, merge_args, parse_ids
from adell_mri.utils.prediction_utils import get_ensemble_prediction
from adell_mri.utils.python_logging import get_logger
from adell_mri.utils.torch_utils import (
    get_generator_and_rng,
    load_checkpoint_to_model,
)

logger = get_logger(__name__)


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
            "cat_confounder_keys",
            "cont_confounder_keys",
            "exclude_surrogate_variables",
            "n_features_deconfounder",
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
            (
                "prediction_type",
                "type",
                {
                    "choices": [
                        "probability",
                        "logit",
                        "pre_bias",
                    ],
                    "help": "Returns either the classification "
                    "probability, the logits or the pre-bias "
                    "ordinal values.",
                },
            ),
            ("prediction_checkpoints", "checkpoints"),
            "ensemble",
            "output_path",
        ]
    )

    args = parser.parse_args(arguments)

    if args.params_from is not None:
        param_dict = get_params(args.params_from)
        args = merge_args(args, param_dict, sys.argv[1:])

    _g, rng = get_generator_and_rng(args.seed)

    if args.clinical_feature_keys is None:
        clinical_feature_keys = []
    else:
        clinical_feature_keys = args.clinical_feature_keys

    data_dict = Dataset(args.dataset_json, rng=rng)
    presence_keys = list(args.image_keys)
    if args.mask_key is not None:
        presence_keys.append(args.mask_key)
    if args.cat_confounder_keys is not None:
        presence_keys.extend(args.cat_confounder_keys)
    if args.cont_confounder_keys is not None:
        presence_keys.extend(args.cont_confounder_keys)
    if not use_deconfounding(args):
        presence_keys.extend(clinical_feature_keys)

    data_dict.filter_dictionary(
        filters_presence=presence_keys,
        filters=args.filter_on_keys,
    )
    data_dict.subsample_dataset(subsample_size=args.subsample_size)

    if len(data_dict) == 0:
        raise Exception(
            "No data available for prediction "
            "(dataset={}; keys={})".format(args.dataset_json, args.image_keys)
        )

    keys = args.image_keys
    adc_keys = args.adc_keys if args.adc_keys is not None else []
    adc_keys = [k for k in adc_keys if k in keys]
    mask_key = args.mask_key
    input_keys = deepcopy(keys)
    if mask_key is not None:
        input_keys.append(mask_key)

    cat_key, cont_key, cat_vars, cont_vars = extract_confounder_info(
        args, data_dict
    )

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

    transform_arguments = create_transform_arguments(args)
    transforms_prediction = ClassificationTransforms(
        **transform_arguments
    ).transforms()

    post_proc_fn, extra_args = resolve_postprocessing(
        args.type,
        None if use_deconfounding(args) else args.net_type,
        args.n_classes,
        caller_logger=logger,
    )

    global_output = []
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

        torch.ones([1]).to("cuda" if "cuda" in args.dev else "cpu")

        configure_loss_fn(
            network_config,
            None if use_deconfounding(args) else args.net_type,
            args.n_classes,
        )

        checkpoint_list = resolve_checkpoint_list(
            args.checkpoints,
            args.one_to_one,
            args.ensemble,
            iteration,
            caller_logger=logger,
        )
        for checkpoint in checkpoint_list:
            if checkpoint is not None:
                logger.info("Predicting for %s", checkpoint)
            network = create_classification_network(
                args=args,
                network_config=network_config,
                input_keys=input_keys,
                n_classes=args.n_classes,
                clinical_feature_keys=clinical_feature_keys,
                cat_vars=cat_vars,
                cont_vars=cont_vars,
                cat_key=cat_key,
                cont_key=cont_key,
                train_loader_call=None,
                max_epochs=None,
                warmup_steps=None,
                start_decay=None,
                clinical_feature_means=None,
                clinical_feature_stds=None,
            )

            load_checkpoint_to_model(
                network,
                checkpoint,
                exclude_from_state_dict=["loss_fn.weight"],
            )
            network = network.eval().to(args.dev)
            if getattr(network, "gaussian_process", False):
                network.gaussian_process_head.get_cov()

            accumulator = ClassificationPredictionAccumulator(
                iteration=iteration,
                prediction_ids=curr_prediction_ids,
                checkpoint=checkpoint,
            )
            with tqdm(total=len(curr_prediction_ids)) as pbar:
                for identifier, element in zip(
                    curr_prediction_ids, prediction_dataset
                ):
                    pbar.set_description("Predicting {}".format(identifier))
                    output = predict_sample(
                        network,
                        element,
                        args.dev,
                        post_proc_fn,
                        extra_args,
                        include_tabular=not use_deconfounding(args),
                    )
                    accumulator.add(identifier, output)
                    pbar.update()
            global_output.append(accumulator.as_dict())

        if args.ensemble is not None:
            global_output.append(
                get_ensemble_prediction(global_output, args.ensemble)
            )
    Path(args.output_path).parent.mkdir(exist_ok=True, parents=True)
    with open(args.output_path, "w") as o:
        o.write(json.dumps(global_output))
