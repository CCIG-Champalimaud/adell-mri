import os
import argparse
import random
import json
import numpy as np
import torch
import monai
from copy import deepcopy
from lightning.pytorch import Trainer
from lightning.pytorch.callbacks import RichProgressBar

from ...entrypoints.assemble_args import Parser
from ...utils import safe_collate, ExponentialMovingAverage
from ...utils.pl_utils import get_devices, get_ckpt_callback, get_logger
from ...modules.config_parsing import parse_config_ssl, parse_config_unet
from ...utils.dicom_loader import DICOMDataset, SliceSampler
from ...utils.network_factories import get_ssl_network
from ...monai_transforms import (
    get_pre_transforms_ssl,
    get_post_transforms_ssl,
    get_augmentations_ssl,
)

torch.backends.cudnn.benchmark = True


def keep_first_not_none(*args):
    for arg in args:
        if arg is not None:
            return arg


def force_cudnn_initialization():
    """Convenience function to initialise CuDNN (and avoid the lazy loading
    from PyTorch).
    """
    s = 16
    dev = torch.device("cuda")
    torch.nn.functional.conv2d(
        torch.zeros(s, s, s, s, device=dev),
        torch.zeros(s, s, s, s, device=dev),
    )


def filter_dicom_dict_on_presence(data_dict, all_keys):
    def check_intersection(a, b):
        return len(set.intersection(set(a), set(b))) == len(set(b))

    for k in data_dict:
        for kk in data_dict[k]:
            data_dict[k][kk] = [
                element
                for element in data_dict[k][kk]
                if check_intersection(element.keys(), all_keys)
            ]
    return data_dict


def filter_dicom_dict_by_size(data_dict, max_size):
    print("Filtering by size (max={})".format(max_size, len(data_dict)))
    output_dict = {}
    removed_series = 0
    for k in data_dict:
        for kk in data_dict[k]:
            if len(data_dict[k][kk]) < max_size:
                if k not in output_dict:
                    output_dict[k] = {}
                output_dict[k][kk] = data_dict[k][kk]
            else:
                removed_series += 1
    print("Removed={}".format(removed_series))
    return output_dict


def main(arguments):
    parser = Parser()

    parser.add_argument_by_key(
        [
            "dataset_json",
            "image_keys",
            "adc_keys",
            "jpeg_dataset",
            "num_samples",
            "train_pids",
            "pad_size",
            "crop_size",
            "random_crop_size",
            "scaled_crop_size",
            "different_crop",
            "target_spacing",
            "subsample_size",
            "cache_rate",
            "precision",
            ("ssl_net_type", "net_type"),
            "batch_size",
            "max_slices",
            "config_file",
            "ssl_method",
            "ema",
            "stop_gradient",
            "checkpoint_dir",
            "checkpoint_name",
            "checkpoint",
            "resume_from_last",
            "project_name",
            "resume",
            "summary_dir",
            "summary_name",
            "metric_path",
            "n_series_iterations",
            "n_transforms",
            "dev",
            "n_workers",
            "seed",
            "max_epochs",
            "check_val_every_n_epoch",
            "accumulate_grad_batches",
            "gradient_clip_val",
            "steps_per_epoch",
            ("warmup_steps", "warmup_epochs"),
            "dropout_param",
        ]
    )

    args = parser.parse_args(arguments)

    torch.manual_seed(args.seed)
    random.seed(args.seed)
    np.random.seed(args.seed)
    g = torch.Generator()
    g.manual_seed(args.seed)

    n_iterations = args.n_series_iterations
    accelerator, devices, strategy = get_devices(args.dev)

    output_file = open(args.metric_path, "w")

    keys = args.image_keys
    copied_keys = [k + "_copy" for k in keys]
    if args.adc_keys is None:
        adc_image_keys = []
    adc_image_keys = [k for k in adc_image_keys if k in keys]
    non_adc_keys = [k for k in keys if k not in adc_image_keys]
    all_keys = [*keys]

    if args.jpeg_dataset is True:
        tmp = [x.strip() for x in open(args.dataset_json)]
        data_dict = {k.split(os.sep)[-1]: {args.image_keys[0]: k} for k in tmp}
    else:
        data_dict = json.load(open(args.dataset_json, "r"))
        data_dict = filter_dicom_dict_on_presence(data_dict, all_keys)
        if args.max_slices is not None:
            data_dict = filter_dicom_dict_by_size(data_dict, args.max_slices)
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

    if args.net_type == "unet_encoder":
        network_config, _ = parse_config_unet(args.config_file, len(keys), 2)
        network_config_correct = deepcopy(network_config)
        for k in network_config:
            if k in ["loss_fn"]:
                del network_config_correct[k]
    else:
        network_config, network_config_correct = parse_config_ssl(
            args.config_file,
            args.dropout_param,
            len(keys),
            args.ssl_method == "ijepa",
        )

    if args.batch_size is not None:
        network_config["batch_size"] = args.batch_size
        network_config_correct["batch_size"] = args.batch_size

    if args.random_crop_size is None:
        roi_size = [128, 128]
    else:
        roi_size = [int(x) for x in args.random_crop_size]

    all_pids = [k for k in data_dict]

    is_ijepa = args.ssl_method == "ijepa"
    pre_transform_args = {
        "all_keys": all_keys,
        "copied_keys": copied_keys,
        "adc_keys": adc_image_keys,
        "non_adc_keys": non_adc_keys,
        "target_spacing": args.target_spacing,
        "crop_size": args.crop_size,
        "pad_size": args.pad_size,
        "n_channels": 1,
        "n_dim": 2,
        "skip_augmentations": is_ijepa,
        "jpeg_dataset": args.jpeg_dataset,
    }

    post_transform_args = {
        "all_keys": all_keys,
        "copied_keys": copied_keys,
        "skip_augmentations": is_ijepa,
    }

    augmentation_args = {
        "all_keys": all_keys,
        "copied_keys": copied_keys if is_ijepa is False else [],
        "scaled_crop_size": args.scaled_crop_size,
        "roi_size": roi_size,
        "different_crop": args.different_crop,
        "vicregl": args.ssl_method == "vicregl",
        "n_transforms": args.n_transforms,
        "n_dim": 2,
        "skip_augmentations": is_ijepa,
    }

    if is_ijepa is True:
        image_size = keep_first_not_none(args.scaled_crop_size, args.crop_size)
        patch_size = network_config_correct["backbone_args"]["patch_size"]
        feature_map_size = [i // pi for i, pi in zip(image_size, patch_size)]
        network_config_correct["backbone_args"]["image_size"] = image_size
        network_config_correct["feature_map_dimensions"] = feature_map_size

    transforms = [
        *get_pre_transforms_ssl(**pre_transform_args),
        *get_augmentations_ssl(**augmentation_args),
        *get_post_transforms_ssl(**post_transform_args),
    ]

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

    if args.jpeg_dataset is True:
        train_dataset = monai.data.Dataset(
            train_list, monai.transforms.Compose(transforms)
        )
        sampler = torch.utils.data.RandomSampler(
            train_list, num_samples=args.num_samples, generator=g
        )
        val_sampler = torch.utils.data.RandomSampler(
            train_list, num_samples=args.num_samples, generator=g
        )
    else:
        train_dataset = DICOMDataset(
            train_list, monai.transforms.Compose(transforms)
        )
        if args.steps_per_epoch is not None:
            n_samples = args.steps_per_epoch * network_config["batch_size"]
        elif args.num_samples is not None:
            n_samples = args.num_samples
        else:
            n_samples = None
        sampler = SliceSampler(
            train_list, n_iterations=n_iterations, n_samples=n_samples
        )
        val_sampler = SliceSampler(
            train_list, n_iterations=n_iterations, n_samples=n_samples
        )

    n_devices = len(devices) if isinstance(devices, list) else 1
    agb = args.accumulate_grad_batches
    if args.steps_per_epoch is not None:
        steps_per_epoch = args.steps_per_epoch
        steps_per_epoch_optim = int(np.ceil(args.steps_per_epoch / agb))
        max_steps = args.max_epochs * steps_per_epoch
        max_epochs = -1
        max_steps_optim = max_epochs * steps_per_epoch_optim
        warmup_steps = args.warmup_epochs * steps_per_epoch_optim
        check_val_every_n_epoch = None
        val_check_interval = args.check_val_every_n_epoch * steps_per_epoch
    else:
        bs = network_config_correct["batch_size"]
        steps_per_epoch = len(sampler) // (bs * n_devices)
        steps_per_epoch = int(np.ceil(steps_per_epoch / agb))
        max_epochs = args.max_epochs
        max_steps = -1
        max_steps_optim = args.max_epochs * steps_per_epoch
        warmup_steps = args.warmup_epochs * steps_per_epoch
        check_val_every_n_epoch = args.check_val_every_n_epoch
        val_check_interval = None
    warmup_steps = int(warmup_steps)
    max_steps_optim = int(max_steps_optim)

    if args.ema is True:
        if is_ijepa is True:
            ema_params = {
                "decay": 0.99,
                "final_decay": 1.0,
                "n_steps": max_steps_optim,
            }
        else:
            ema_params = {
                "decay": 0.996,
                "final_decay": 1.0,
                "n_steps": max_steps_optim,
            }
        ema = ExponentialMovingAverage(**ema_params)
    else:
        ema = None

    if isinstance(devices, list):
        n_workers = args.n_workers // len(devices)
    else:
        n_workers = args.n_workers // devices

    def train_loader_call(batch_size, shuffle=True):
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

    train_loader = train_loader_call(
        network_config_correct["batch_size"], False
    )
    val_loader = monai.data.ThreadDataLoader(
        train_dataset,
        batch_size=network_config_correct["batch_size"],
        num_workers=n_workers,
        collate_fn=safe_collate,
        sampler=val_sampler,
        drop_last=True,
    )

    ssl = get_ssl_network(
        train_loader_call=train_loader_call,
        max_epochs=max_epochs,
        max_steps_optim=max_steps_optim,
        warmup_steps=warmup_steps,
        ssl_method=args.ssl_method,
        ema=ema,
        net_type=args.net_type,
        network_config_correct=network_config_correct,
        stop_gradient=args.stop_gradient,
    )

    if args.checkpoint is not None:
        state_dict = torch.load(args.checkpoint, map_location=args.dev)[
            "state_dict"
        ]
        inc = ssl.load_state_dict(state_dict)

    callbacks = [RichProgressBar()]

    epochs_ckpt = max_epochs
    ckpt_callback, ckpt_path, status = get_ckpt_callback(
        checkpoint_dir=args.checkpoint_dir,
        checkpoint_name=args.checkpoint_name,
        max_epochs=epochs_ckpt,
        resume_from_last=args.resume_from_last,
        val_fold=None,
        monitor="val_loss",
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
        fold=None,
    )

    precision = args.precision
    trainer = Trainer(
        accelerator=accelerator,
        devices=devices,
        logger=logger,
        callbacks=callbacks,
        strategy=strategy,
        max_epochs=max_epochs,
        max_steps=max_steps,
        sync_batchnorm=True if strategy is not None else False,
        enable_checkpointing=ckpt,
        val_check_interval=val_check_interval,
        check_val_every_n_epoch=check_val_every_n_epoch,
        precision=precision,
        accumulate_grad_batches=args.accumulate_grad_batches,
        gradient_clip_val=args.gradient_clip_val,
        limit_train_batches=len(train_loader) // n_devices,
        limit_val_batches=len(val_loader) // n_devices,
    )

    torch.cuda.empty_cache()
    force_cudnn_initialization()

    trainer.fit(ssl, val_dataloaders=val_loader, ckpt_path=ckpt_path)

    print("Validating...")
    test_metrics = trainer.test(ssl, val_loader)[0]
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
