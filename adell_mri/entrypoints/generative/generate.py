import os
import sys
from pathlib import Path
from typing import Any

import monai
import numpy as np
import SimpleITK as sitk
import torch
from tqdm import tqdm

from adell_mri.entrypoints.assemble_args import Parser
from adell_mri.entrypoints.generative.train_3d import return_first_not_none
from adell_mri.transform_factory import GenerationTransforms
from adell_mri.utils.dataset import Dataset
from adell_mri.utils.generic_utils import safe_collate
from adell_mri.utils.logging import get_progress
from adell_mri.utils.network_factories import get_generative_network
from adell_mri.utils.parser import compose, get_params, merge_args, parse_ids
from adell_mri.utils.pl_utils import get_devices
from adell_mri.utils.python_logging import get_logger
from adell_mri.utils.torch_utils import (
    get_generator_and_rng,
    load_checkpoint_to_model,
)


def image_to_sitk_array(image: torch.Tensor) -> np.ndarray:
    """
    Converts a channel-first tensor (from ``GenerationTransforms``) into the
    array layout expected by ``SimpleITK.GetImageFromArray`` (spatial dims in
    x, y, [z] order with the channel dimension moved last).

    Args:
        image (torch.Tensor): channel-first image tensor with shape
            [C, H, W] (2D) or [C, H, W, D] (3D).

    Returns:
        numpy.ndarray: array with shape [H, W, C] or [D, H, W, C].
    """
    if image.ndim == 4:
        perm = (3, 1, 2, 0)
    elif image.ndim == 3:
        perm = (1, 2, 0)
    else:
        raise ValueError(
            f"Expected a channel-first 2D or 3D tensor, got shape "
            f"{list(image.shape)}"
        )
    return image.permute(perm).numpy()


def fetch_specifications(state_dict: dict[str, Any]):
    cbks = state_dict["callbacks"]
    ckpt_cbk = cbks[[k for k in cbks if "ModelCheckpointWithMetadata" in k][0]]
    metadata = ckpt_cbk["metadata"]
    cat_spec = None
    num_spec = None
    if "network_config" in metadata:
        network_config = metadata["network_config"]
    else:
        network_config = None
    if "categorical_specification" in metadata:
        if metadata["categorical_specification"] is not None:
            cat_spec = metadata["categorical_specification"]
            cat_spec = [[str(v) for v in C] for C in cat_spec]
    if "numerical_specification" in metadata:
        if metadata["numerical_specification"] is not None:
            num_spec = metadata["numerical_specification"]
    transform_args = metadata["transform_arguments"]
    if ("pre" in transform_args) and ("post" in transform_args):
        transform_args = transform_args["pre"] | transform_args["post"]
    spacing = metadata["transform_arguments"]["target_spacing"]
    return network_config, cat_spec, num_spec, spacing, transform_args


def main(arguments):
    logger = get_logger(__name__)
    parser = Parser()

    parser.add_argument_by_key(
        [
            ("dataset_json", "dataset_json", {"required": False}),
            "prediction_ids",
            "keep_original",
            "params_from",
            "image_keys",
            "cat_condition_keys",
            "num_condition_keys",
            "uncondition_cat_idx",
            "uncondition_num_idx",
            "cat_condition",
            "num_condition",
            "filter_on_keys",
            "excluded_ids",
            "cache_rate",
            "subsample_size",
            "target_spacing",
            "pad_size",
            "crop_size",
            "config_file",
            "overrides",
            "dev",
            "n_workers",
            "precision",
            "seed",
            "checkpoint",
            "batch_size",
            "learning_rate",
            "diffusion_steps",
            "skip_steps",
            "n_samples_gen",
            "guidance_strength",
            "output_path",
            "overwrite",
        ]
    )

    args = parser.parse_args(arguments)

    if args.params_from is not None:
        param_dict = get_params(args.params_from)
        args = merge_args(args, param_dict, sys.argv[1:])

    g, rng = get_generator_and_rng(args.seed)

    ckpt = torch.load(args.checkpoint[0], weights_only=False)

    accelerator, devices, strategy = get_devices(args.dev)
    n_devices = len(devices) if isinstance(devices, list) else devices
    n_devices = 1 if isinstance(devices, str) else n_devices

    specs = fetch_specifications(ckpt)
    network_config = specs[0]
    categorical_specification = specs[1]
    numerical_specification = specs[2]
    spacing = specs[3]
    transform_args = specs[4]

    presence_keys = [*args.image_keys]
    with_conditioning = False
    if args.cat_condition_keys is not None:
        presence_keys.extend(args.cat_condition_keys)
        with_conditioning = True
    if args.num_condition_keys is not None:
        presence_keys.extend(args.num_condition_keys)
        with_conditioning = True

    if network_config is None:
        network_config = compose(args.config_file, "diffusion", args.overrides)
        network_config["batch_size"] = return_first_not_none(
            args.batch_size, network_config.get("batch_size")
        )
        network_config["learning_rate"] = return_first_not_none(
            args.learning_rate, network_config.get("learning_rate")
        )
        network_config["with_conditioning"] = with_conditioning
        network_config["cross_attention_dim"] = (
            256 if with_conditioning else None
        )

    network = get_generative_network(
        network_config=network_config,
        categorical_specification=categorical_specification,
        numerical_specification=numerical_specification,
        train_loader_call=None,
        max_epochs=None,
        warmup_steps=None,
        start_decay=None,
        diffusion_steps=args.diffusion_steps,
        scheduler_config={
            "schedule": "scaled_linear_beta",
            "beta_start": 0.0005,
            "beta_end": 0.0195,
        },
        uncondition_proba=0.0,
        concat_condition_key=(
            "cat_conditioning"
            if transform_args.get("input_image_keys") is not None
            or transform_args.get("input_mask_keys") is not None
            else None
        ),
    )

    load_checkpoint_to_model(network, ckpt, [])

    # PL needs a little hint to detect GPUs.
    torch.ones([1]).to("cuda" if "cuda" in args.dev else "cpu")

    if "cuda" in args.dev:
        network = network.to(args.dev)

    cat_condition = None
    num_condition = None
    if args.cat_condition is not None:
        cat_condition = [c.split("=") for c in args.cat_condition]
        cat_condition = {k: v for k, v in cat_condition}
    if args.num_condition is not None:
        num_condition = [c.split("=") for c in args.num_condition]
        num_condition = {k: float(v) for k, v in num_condition}

    if args.precision == "32":
        inference_dtype = torch.float32
    elif args.precision == "16":
        inference_dtype = torch.float16
    else:
        logger.info("Invalid precision. Using 32-bit precision.")
        inference_dtype = torch.float32
    network = network.eval()
    network = torch.compile(network)
    network = network.to(dtype=inference_dtype)
    progress = get_progress(transient=True)
    progress.start()
    if args.dataset_json is not None:
        logger.info("Setting up transforms...")
        transforms = GenerationTransforms(**transform_args).transforms()
        transforms.set_random_state(args.seed)
        data_dict = Dataset(args.dataset_json, rng=rng)
        for k in list(data_dict.keys()):
            data_dict[k] = {**data_dict[k], "key": k}
        if args.excluded_ids is not None:
            args.excluded_ids = parse_ids(
                args.excluded_ids, output_format="list"
            )
            logger.info("Removing IDs specified in --excluded_ids")
            prev_len = len(data_dict)
            data_dict.subsample_dataset(excluded_key_list=args.excluded_ids)
            logger.info("Removed %s IDs", prev_len - len(data_dict))
        data_dict.filter_dictionary(
            filters_presence=presence_keys,
            filters=args.filter_on_keys,
        )
        if args.cat_condition_keys:
            data_dict.apply(str, args.cat_condition_keys)
        if args.subsample_size is not None:
            data_dict.subsample_dataset(args.subsample_size)

        if len(data_dict) == 0:
            raise Exception(
                "No data available for prediction \
                    (dataset={}; keys={}; labels={})".format(
                    args.dataset_json, args.image_keys, args.label_keys
                )
            )

        pred_list = data_dict.to_datalist(args.prediction_ids)

        logger.info("Prediction set size=%s", len(pred_list))

        logger.info("Number of cases: %s", len(pred_list))

        dataset = monai.data.CacheDataset(
            pred_list,
            transforms,
            cache_rate=args.cache_rate,
            num_workers=args.n_workers,
        )
        dataloader = torch.utils.data.DataLoader(
            dataset,
            batch_size=network_config["batch_size"],
            shuffle=False,
            num_workers=args.n_workers,
            pin_memory=True,
            collate_fn=safe_collate,
        )

        Path(args.output_path).mkdir(exist_ok=True, parents=True)
        for data in progress.track(dataloader, description="Generating images"):
            output_paths = [
                os.path.join(args.output_path, f"{k}_gen.mha")
                for k in data["key"]
            ]
            if (
                all(map(os.path.exists, output_paths))
                and args.overwrite is False
            ):
                continue
            images = data["image"].to(args.dev)
            curr_cat, curr_num = None, None
            if args.cat_condition_keys is not None:
                curr_cat = [[] for _ in range(len(data["key"]))]
                for k in args.cat_condition_keys:
                    for i in range(len(curr_cat)):
                        if cat_condition is not None:
                            C = cat_condition.get(k, data.get(k, None)[0])
                        else:
                            C = data.get(k, None)[0]
                        curr_cat[i].append(C)
            if args.num_condition_keys is not None:
                curr_num = [[] for _ in range(len(data["key"]))]
                for k in args.num_condition_keys:
                    for i in range(len(curr_num)):
                        if num_condition is not None:
                            N = num_condition.get(k, data.get(k, None)[0])
                        else:
                            N = data.get(k, None)[0]
                        curr_num[i].append(N)
                curr_num = torch.as_tensor(
                    curr_num, device=args.dev, dtype=inference_dtype
                )
            outputs = network.generate_image(
                input_image=images.to(inference_dtype),
                size=images.shape[2:],
                n=1,
                skip_steps=args.skip_steps,
                cat_condition=curr_cat,
                num_condition=curr_num,
                uncondition_cat_idx=args.uncondition_cat_idx,
                uncondition_num_idx=args.uncondition_num_idx,
                guidance_strength=args.guidance_strength,
                concat_condition=(
                    data["cat_conditioning"].to(args.dev).to(inference_dtype)
                    if "cat_conditioning" in data
                    else None
                ),
            )
            outputs = outputs.detach().float().cpu()
            for image, output_path, output in zip(
                images, output_paths, outputs
            ):
                output = sitk.GetImageFromArray(image_to_sitk_array(output))
                output.SetSpacing(spacing)
                output.SetMetaData("checkpoint", args.checkpoint[0])
                sitk.WriteImage(output, output_path, useCompression=True)
                if args.keep_original:
                    image = sitk.GetImageFromArray(
                        image_to_sitk_array(image.detach().cpu())
                    )
                    image.SetSpacing(spacing)
                    image_path = output_path.replace("_gen.mha", "_orig.mha")
                    sitk.WriteImage(image, image_path, useCompression=True)

    elif args.n_samples_gen is not None:
        size = return_first_not_none(args.crop_size, args.pad_size)
        size = [int(i) for i in size]
        Path(args.output_path).mkdir(exist_ok=True, parents=True)
        logger.info("Generating %s samples", args.n_samples_gen)
        cat_condition = [cat_condition[k] for k in args.cat_condition_keys]
        num_condition = [num_condition[k] for k in args.num_condition_keys]
        num_condition = torch.as_tensor(
            [num_condition], device=args.dev, dtype=inference_dtype
        )
        concat_condition = None
        n_cond_channels = network.in_channels - network.out_channels
        if n_cond_channels > 0:
            concat_condition = torch.zeros(
                [1, n_cond_channels, *size],
                device=args.dev,
                dtype=inference_dtype,
            )
        for i in progress.track(
            range(args.n_samples_gen), description="Generating images"
        ):
            output = network.generate_image(
                size=size,
                n=1,
                skip_steps=0,
                cat_condition=cat_condition,
                num_condition=num_condition,
                concat_condition=concat_condition,
            )
            output = image_to_sitk_array(output.detach().cpu()[0])
            output = sitk.GetImageFromArray(output)
            output.SetSpacing(spacing)
            output.SetMetaData("checkpoint", args.checkpoint[0])
            output_path = os.path.join(args.output_path, f"{i}.mha")
            sitk.WriteImage(output, output_path, useCompression=True)

    else:
        raise Exception(
            "one of dataset_json, n_samples_gen should be specified"
        )

    progress.stop()
