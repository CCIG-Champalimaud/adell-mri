import json
import sys
from pathlib import Path

import monai
import torch
from tqdm import tqdm

from adell_mri.entrypoints.assemble_args import Parser
from adell_mri.entrypoints.classification.predict_utils import (
    ClassificationPredictionAccumulator,
    predict_sample,
    resolve_checkpoint_list,
    resolve_postprocessing,
)
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
from adell_mri.utils.prediction_utils import get_ensemble_prediction
from adell_mri.utils.python_logging import get_logger
from adell_mri.utils.torch_utils import (
    get_generator_and_rng,
    load_checkpoint_to_model,
)


def main(arguments):
    logger = get_logger(__name__)
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
            "ensemble",
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

    data_dict = Dataset(args.dataset_json, rng=rng)
    if args.prediction_ids:
        prediction_ids = parse_ids(args.prediction_ids)
    else:
        prediction_ids = [[k for k in data_dict]]
    if args.excluded_ids is not None:
        excluded_ids = parse_ids(args.excluded_ids, output_format="list")
        a = len(data_dict)
        data_dict = {
            k: data_dict[k] for k in data_dict if k not in excluded_ids
        }
        logger.info("Excluded %s cases with --excluded_ids", a - len(data_dict))

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
        network_config["loss_fn"] = torch.nn.CrossEntropyLoss(
            torch.ones([args.n_classes])
        )

    if args.batch_size is not None:
        network_config["batch_size"] = args.batch_size

    if "batch_size" not in network_config:
        network_config["batch_size"] = 1

    logger.info("Setting up transforms...")
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

    transforms = ClassificationTransforms(**transform_arguments).transforms(
        final_transforms=[
            EinopsRearranged("image", "c h w d -> 1 h w (d c)"),
            ScaleIntensityAlongDimd("image", dim=-1),
        ]
    )

    post_proc_fn, extra_args = resolve_postprocessing(
        args.type, None, args.n_classes, caller_logger=logger
    )

    global_output = []
    for iteration in range(len(prediction_ids)):
        curr_prediction_ids = [
            pid for pid in prediction_ids[iteration] if pid in data_dict
        ]
        prediction_list = [data_dict[pid] for pid in curr_prediction_ids]
        prediction_dataset = monai.data.CacheDataset(
            prediction_list,
            transforms,
            cache_rate=args.cache_rate,
            num_workers=args.n_workers,
        )

        # PL sometimes needs a little hint to detect GPUs.
        torch.ones([1]).to("cuda" if "cuda" in args.dev else "cpu")

        checkpoint_list = resolve_checkpoint_list(
            args.checkpoints,
            args.one_to_one,
            None,
            iteration,
            caller_logger=logger,
        )
        for checkpoint in checkpoint_list:
            if checkpoint is not None:
                logger.info(f"Predicting for {checkpoint}")
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
                logger.info("2D module output size not specified, inferring...")
                input_example = torch.rand(
                    1, 1, *[int(x) for x in args.crop_size][:2]
                ).to(args.dev)
                output = network_config["module"](input_example)
                network_config["module_out_dim"] = int(output.shape[1])
                logger.info(
                    "2D module output size=%s", network_config["module_out_dim"]
                )
            if args.mil_method == "transformer":
                network = TransformableTransformerPL(
                    **boilerplate_args, **network_config
                )
            elif args.mil_method == "standard":
                network = MultipleInstanceClassifierPL(
                    **boilerplate_args, **network_config
                )

            load_checkpoint_to_model(network, checkpoint)
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
                        process_features=False,
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
