import os
import sys
from pathlib import Path
from typing import Any

import monai
import SimpleITK as sitk
import torch
from tqdm import tqdm

from ...monai_transforms import \
    get_post_transforms_generation as get_post_transforms
from ...monai_transforms import \
    get_pre_transforms_generation as get_pre_transforms
from ...utils import safe_collate
from ...utils.dataset import Dataset
from ...utils.network_factories import get_generative_network
from ...utils.parser import compose, get_params, merge_args, parse_ids
from ...utils.pl_utils import get_devices
from ...utils.torch_utils import (get_generator_and_rng,
                                  load_checkpoint_to_model)
from ..assemble_args import Parser
from .train import return_first_not_none


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
    spacing = metadata["transform_arguments"]["pre"]["target_spacing"]
    return network_config, cat_spec, num_spec, spacing, transform_args


def main(arguments):
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

    ckpt = torch.load(args.checkpoint[0])

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

    inference_dtype = torch.float16
    network = network.eval()
    network = torch.compile(network)
    network = network.to(dtype=inference_dtype)
    if args.dataset_json is not None:
        print("Setting up transforms...")
        transform_pre_arguments = transform_args["pre"]
        transform_post_arguments = transform_args["post"]

        transforms = [
            *get_pre_transforms(**transform_pre_arguments),
            *get_post_transforms(**transform_post_arguments),
        ]
        transforms = monai.transforms.Compose(transforms)
        transforms.set_random_state(args.seed)
        data_dict = Dataset(args.dataset_json, rng=rng, verbose=True)
        data_dict.dataset = {
            k: {**v, "key": k} for k, v in data_dict.dataset.items()
        }
        if args.excluded_ids is not None:
            args.excluded_ids = parse_ids(
                args.excluded_ids, output_format="list"
            )
            print("Removing IDs specified in --excluded_ids")
            prev_len = len(data_dict)
            data_dict = {
                k: data_dict[k] for k in data_dict if k not in args.excluded_ids
            }
            print("\tRemoved {} IDs".format(prev_len - len(data_dict)))
        data_dict.filter_dictionary(
            filters_presence=presence_keys,
            filters=args.filter_on_keys,
        )
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

        print("\tPrediction set size={}".format(len(pred_list)))

        print(f"Number of cases: {len(pred_list)}")

        dataset = monai.data.CacheDataset(
            pred_list,
            transforms,
            cache_rate=args.cache_rate,
            num_workers=args.n_workers,
        )
        dataloader = torch.utils.data.DataLoader(
            dataset,
            batch_size=args.batch_size,
            shuffle=False,
            num_workers=args.n_workers,
            pin_memory=True,
            collate_fn=safe_collate,
        )

        Path(args.output_path).mkdir(exist_ok=True, parents=True)
        for data in tqdm(dataloader, desc="Generating images"):
            output_paths = [
                os.path.join(args.output_path, f"{k}_gen.mha")
                for k in data["key"]
            ]
            if (
                all(map(os.path.exists, output_paths))
                and args.overwrite is False
            ):
                continue
            images = data["image"].to(args.dev).float()
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
            )
            outputs = outputs.detach().float().cpu()
            for image, output_path, output in zip(
                images, output_paths, outputs
            ):
                output = sitk.GetImageFromArray(
                    output.permute(3, 1, 2, 0).numpy()
                )
                output.SetSpacing(spacing)
                output.SetMetaData("checkpoint", args.checkpoint[0])
                sitk.WriteImage(output, output_path, useCompression=True)
                if args.keep_original:
                    image = image.detach().cpu().permute(3, 1, 2, 0).numpy()
                    image = sitk.GetImageFromArray(image)
                    image.SetSpacing(spacing)
                    image_path = output_path.replace("_gen.mha", "_orig.mha")
                    sitk.WriteImage(image, image_path, useCompression=True)

    elif args.n_samples_gen is not None:
        size = return_first_not_none(args.crop_size, args.pad_size)
        size = [int(i) for i in size]
        Path(args.output_path).mkdir(exist_ok=True, parents=True)
        print(f"Generating {args.n_samples_gen} samples")
        cat_condition = [cat_condition[k] for k in args.cat_condition_keys]
        num_condition = [num_condition[k] for k in args.num_condition_keys]
        num_condition = torch.as_tensor(
            [num_condition], device=args.dev, dtype=inference_dtype
        )
        for i in range(args.n_samples_gen):
            output = network.generate_image(
                size=size,
                n=1,
                skip_steps=0,
                cat_condition=cat_condition,
                num_condition=num_condition,
            )
            output = output.detach().cpu()[0].permute(3, 1, 2, 0).numpy()
            output = sitk.GetImageFromArray(output)
            output.SetSpacing(spacing)
            output.SetMetaData("checkpoint", args.checkpoint[0])
            output_path = os.path.join(args.output_path, f"{i}.mha")
            sitk.WriteImage(output, output_path, useCompression=True)

    else:
        raise Exception(
            "one of dataset_json, n_samples_gen should be specified"
        )
