import json
import sys

import monai
import numpy as np
import torch
from lightning.pytorch import Trainer
from lightning.pytorch.callbacks import RichProgressBar

from ...modules.config_parsing import parse_config_gan
from ...transform_factory import GenerationTransforms
from ...transform_factory import get_augmentations_class as get_augmentations
from ...utils.dicom_dataset import filter_dicom_dict_on_presence
from ...utils.dicom_loader import DICOMDataset, SliceSampler
from ...utils.network_factories import get_gan_network
from ...utils.parser import get_params, merge_args
from ...utils.pl_callbacks import LogImageFromGAN
from ...utils.pl_utils import get_ckpt_callback, get_devices, get_logger
from ...utils.torch_utils import get_generator_and_rng, load_checkpoint_to_model
from ...utils.utils import safe_collate
from ..assemble_args import Parser


def get_conditional_specification(d: dict, cond_key: str):
    possible_values = []
    for k in d:
        if cond_key in d[k]:
            v = d[k][cond_key]
            if v not in possible_values:
                possible_values.append(d[k][cond_key])
    return possible_values


def get_mean_and_std(d: dict, regression_keys: list[str]):
    if regression_keys is None:
        return None, None
    values = {k: [] for k in regression_keys}
    for study_uid in d:
        for series_uid in d[study_uid]:
            for instance in d[study_uid][series_uid]:
                for k in values:
                    values[k].append(instance[k])
    means = [np.mean(values[k]) for k in values]
    stds = [np.std(values[k]) for k in values]
    return means, stds


def return_first_not_none(*size_list):
    for size in size_list:
        if size is not None:
            return size


def main(arguments):
    parser = Parser()

    parser.add_argument_by_key(
        [
            "dataset_json",
            "params_from",
            "image_keys",
            "train_pids",
            "input_image_keys",
            "cat_condition_keys",
            "num_condition_keys",
            "augment",
            "augment_args",
            "subsample_size",
            "target_spacing",
            "pad_size",
            "crop_size",
            "config_file",
            "warmup_steps",
            "dev",
            "n_workers",
            "seed",
            "max_epochs",
            "steps_per_epoch",
            "precision",
            "check_val_every_n_epoch",
            "gradient_clip_val",
            "accumulate_grad_batches",
            "checkpoint_dir",
            "checkpoint_name",
            "checkpoint",
            "resume_from_last",
            "exclude_from_state_dict",
            "logger_type",
            "project_name",
            "log_model",
            "summary_dir",
            "summary_name",
            "tracking_uri",
            "monitor",
            "metric_path",
            "n_series_iterations",
            "resume",
            "batch_size",
            "learning_rate",
        ]
    )

    args = parser.parse_args(arguments)

    if args.params_from is not None:
        param_dict = get_params(args.params_from)
        args = merge_args(args, param_dict, sys.argv[1:])

    g, rng = get_generator_and_rng(args.seed)

    accelerator, devices, strategy = get_devices(args.dev)
    n_devices = len(devices) if isinstance(devices, list) else devices
    n_devices = 1 if isinstance(devices, str) else n_devices

    output_file = open(args.metric_path, "w")

    all_keys = [x for x in args.image_keys]
    all_image_keys = [x for x in args.image_keys]
    if args.input_image_keys is not None:
        all_keys.append(args.input_image_keys)
        all_image_keys.append(args.input_image_keys)

    data_dict = json.load(open(args.dataset_json, "r"))
    data_dict = filter_dicom_dict_on_presence(data_dict, all_keys=all_keys)

    categorical_specification = None
    numerical_specification = None
    if args.cat_condition_keys is not None:
        categorical_specification = [
            get_conditional_specification(data_dict, k)
            for k in args.cat_condition_keys
        ]
        all_keys.extend(args.cat_condition_keys)
    if args.num_condition_keys is not None:
        numerical_specification = len(args.num_condition_keys)
        all_keys.extend(args.num_condition_keys)

    means, stds = get_mean_and_std(data_dict, args.num_condition_keys)

    for k in data_dict:
        for kk in data_dict[k]:
            for i in range(len(data_dict[k][kk])):
                data_dict[k][kk][i]["pid"] = k

    if len(data_dict) == 0:
        print("No data in dataset JSON")
        exit()

    if args.subsample_size is not None:
        ss = np.random.choice(
            list(data_dict.keys()), args.subsample_size, replace=False
        )
        data_dict = {k: data_dict[k] for k in ss}

    network_config, gen_config, disc_config = parse_config_gan(
        args.config_file,
        args.image_keys,
        args.input_image_keys,
        batch_size=args.batch_size,
        learning_rate=args.learning_rate,
    )

    transform_arguments = {
        "keys": all_image_keys,
        "target_spacing": args.target_spacing,
        "crop_size": args.crop_size,
        "pad_size": args.pad_size,
        "n_dim": 2,
        "cat_keys": args.cat_condition_keys,
        "num_keys": args.num_condition_keys,
    }

    augmentation_args = {}

    transforms = GenerationTransforms(**transform_arguments)

    if args.train_pids is not None:
        train_pids = {pid: "" for pid in args.train_pids}
    else:
        # checking intersections is much much faster in dicts
        train_pids = {pid: "" for pid in data_dict}

    train_list = [
        value for pid, value in data_dict.items() if pid in train_pids
    ]
    train_pids = list(train_pids.keys())

    print(f"Training set size: {len(train_list)}")

    transforms = monai.transforms.Compose(transforms)
    transforms.set_random_state(args.seed)

    train_dataset = DICOMDataset(train_list, transforms)
    if args.steps_per_epoch is not None:
        n_samples = args.steps_per_epoch * network_config["batch_size"]
    else:
        n_samples = None
    sampler = SliceSampler(
        train_list, n_iterations=args.n_series_iterations, n_samples=n_samples
    )

    if isinstance(devices, list):
        n_workers = args.n_workers // len(devices)
    else:
        n_workers = args.n_workers // devices

    def train_loader_call(batch_size):
        return monai.data.ThreadDataLoader(
            train_dataset,
            batch_size=batch_size,
            num_workers=n_workers,
            sampler=sampler,
            collate_fn=safe_collate,
            pin_memory=True,
            persistent_workers=n_workers > 1,
            drop_last=True,
        )

    train_loader = train_loader_call(network_config["batch_size"])
    val_loader = monai.data.ThreadDataLoader(
        train_dataset,
        batch_size=network_config["batch_size"],
        num_workers=n_workers,
        collate_fn=safe_collate,
        sampler=sampler,
        drop_last=True,
    )

    n_devices = len(devices) if isinstance(devices, list) else 1
    agb = args.accumulate_grad_batches
    steps_per_epoch = len(train_loader) // agb // n_devices

    model = get_gan_network(
        network_config=network_config,
        generator_config=gen_config,
        discriminator_config=disc_config,
        training_dataloader_call=train_loader_call,
        categorical_specification=categorical_specification,
        numerical_specification=numerical_specification,
        numerical_moments=(means, stds),
        input_image_key=args.input_image_keys,
        max_epochs=args.max_epochs,
        steps_per_epoch=steps_per_epoch,
        pct_start=args.warmup_steps,
    )

    load_checkpoint_to_model(
        model, args.checkpoint, args.exclude_from_state_dict
    )

    callbacks = [RichProgressBar()]

    ckpt_callback, ckpt_path, status = get_ckpt_callback(
        checkpoint_dir=args.checkpoint_dir,
        checkpoint_name=args.checkpoint_name,
        max_epochs=args.max_epochs,
        resume_from_last=args.resume_from_last,
        val_fold=None,
        monitor="val_loss",
        metadata={
            "train_pids": train_pids,
            "network_config": network_config,
            "transform_arguments": transform_arguments,
            "categorical_specification": categorical_specification,
            "numerical_specification": numerical_specification,
        },
    )

    if ckpt_callback is not None:
        callbacks.append(ckpt_callback)
    ckpt = ckpt_callback is not None
    if status == "finished":
        print("Training has finished")
        exit()

    logger = get_logger(
        summary_name=args.summary_name,
        summary_dir=args.summary_dir,
        project_name=args.project_name,
        resume=args.resume,
        log_model=args.log_model,
        logger_type=args.logger_type,
        tracking_uri=args.tracking_uri,
        fold=None,
        tags={
            "train_pids": train_pids,
            "network_config": network_config,
            "transform_arguments": transform_arguments,
            "categorical_specification": categorical_specification,
            "numerical_specification": numerical_specification,
        },
    )

    if logger is not None:
        size = return_first_not_none(args.pad_size, args.crop_size)
        callbacks.append(
            LogImageFromGAN(
                n_images=5,
                size=[len(all_image_keys)]
                + [int(x) for x in size][: gen_config["spatial_dims"]],
            )
        )

    precision = args.precision

    trainer = Trainer(
        accelerator=accelerator,
        devices=devices,
        logger=logger,
        callbacks=callbacks,
        strategy=strategy,
        max_epochs=args.max_epochs,
        sync_batchnorm=True if strategy is not None else False,
        enable_checkpointing=ckpt,
        precision=precision,
        check_val_every_n_epoch=args.check_val_every_n_epoch,
        accumulate_grad_batches=args.accumulate_grad_batches,
        gradient_clip_val=args.gradient_clip_val,
        limit_train_batches=len(train_loader) // n_devices,
        limit_val_batches=len(val_loader) // n_devices,
    )

    torch.cuda.empty_cache()

    trainer.fit(model, val_dataloaders=val_loader, ckpt_path=ckpt_path)

    print("Validating...")
    test_metrics = trainer.test(model, val_loader)[0]
    for k in test_metrics:
        out = test_metrics[k]
        try:
            value = float(out.detach().numpy())
        except Exception:
            value = float(out)
        x = "{},{},{},{}".format(k, 0, 0, value)
        output_file.write(x + "\n")
    x = "{},{},{},{}".format("train_ids", 0, 0, ":".join(train_pids))
    output_file.write(x + "\n")

    # just in case
    torch.cuda.empty_cache()

    output_file.close()
