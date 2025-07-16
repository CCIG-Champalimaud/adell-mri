from copy import deepcopy

import monai
import numpy as np
import torch
from lightning.pytorch import Trainer
from lightning.pytorch.callbacks import RichProgressBar

from adell_mri.utils.torch_utils import get_generator_and_rng

from adell_mri.entrypoints.assemble_args import Parser
from adell_mri.modules.config_parsing import parse_config_ssl, parse_config_unet
from adell_mri.transform_factory import SSLTransforms, get_augmentations_ssl
from adell_mri.utils.dataset import Dataset
from adell_mri.utils.network_factories import get_ssl_network
from adell_mri.utils.pl_utils import get_ckpt_callback, get_devices, get_logger
from adell_mri.utils.utils import ExponentialMovingAverage, safe_collate

torch.backends.cudnn.benchmark = True


def keep_first_not_none(*args):
    for arg in args:
        if arg is not None:
            return arg


def force_cudnn_initialization():
    """
    Convenience function to initialise CuDNN (and avoid the lazy loading
    from PyTorch).
    """
    s = 16
    dev = torch.device("cuda")
    torch.nn.functional.conv2d(
        torch.zeros(s, s, s, s, device=dev),
        torch.zeros(s, s, s, s, device=dev),
    )


def main(arguments):
    parser = Parser()

    parser.add_argument_by_key(
        [
            "dataset_json",
            "image_keys",
            ("adc_keys", "adc_image_keys"),
            "target_spacing",
            "pad_size",
            "crop_size",
            "random_crop_size",
            "different_crop",
            "subsample_size",
            "filter_on_keys",
            "cache_rate",
            "precision",
            ("ssl_net_type", "net_type"),
            "batch_size",
            "n_transforms",
            "stop_gradient",
            "scaled_crop_size",
            "config_file",
            "ssl_method",
            "ema",
            "checkpoint_dir",
            "checkpoint_name",
            "checkpoint",
            "resume_from_last",
            "logger_type",
            "project_name",
            "log_model",
            "summary_dir",
            "summary_name",
            "tracking_uri",
            "resume",
            "metric_path",
            "check_val_every_n_epoch",
            "dev",
            "n_workers",
            "seed",
            "max_epochs",
            "accumulate_grad_batches",
            "gradient_clip_val",
            ("warmup_steps", "warmup_epochs"),
            "dropout_param",
        ]
    )

    args = parser.parse_args(arguments)

    g, rng = get_generator_and_rng(args.seed)

    output_file = open(args.metric_path, "w")

    keys = args.image_keys
    copied_keys = [k + "_copy" for k in keys]
    if args.adc_image_keys is None:
        args.adc_image_keys = []
    args.adc_image_keys = [k for k in args.adc_image_keys if k in keys]
    intp = []
    intp_resampling_augmentations = []
    for k in keys:
        intp.append("area")
        intp_resampling_augmentations.append("bilinear")
    non_adc_keys = [k for k in keys if k not in args.adc_image_keys]
    all_keys = [*keys]

    data_dict = Dataset(args.dataset_json, rng=rng)
    data_dict.apply_filters(**vars(args), presence_keys=all_keys)

    if len(data_dict) == 0:
        print("No data in dataset JSON")
        exit()

    for k in data_dict:
        data_dict[k]["pid"] = k

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

    if (args.batch_size is not None) and (args.batch_size != "tune"):
        network_config["batch_size"] = args.batch_size
        network_config_correct["batch_size"] = args.batch_size

    if args.batch_size is not None:
        network_config["batch_size"] = args.batch_size
        network_config_correct["batch_size"] = args.batch_size

    if args.random_crop_size is None:
        roi_size = [128, 128]
    else:
        roi_size = [int(x) for x in args.random_crop_size]

    is_ijepa = args.ssl_method == "ijepa"
    transform_args = {
        "all_keys": all_keys,
        "copied_keys": copied_keys,
        "adc_keys": [],
        "non_adc_keys": non_adc_keys,
        "target_spacing": args.target_spacing,
        "crop_size": args.crop_size,
        "pad_size": args.pad_size,
        "in_channels": 1,
        "n_dim": 3,
        "skip_augmentations": is_ijepa,
        "jpeg_dataset": False,
    }

    augmentation_args = {
        "all_keys": all_keys,
        "copied_keys": copied_keys if is_ijepa is False else [],
        "scaled_crop_size": args.scaled_crop_size,
        "roi_size": roi_size,
        "different_crop": args.different_crop,
        "vicregl": args.ssl_method == "vicregl",
        "n_transforms": args.n_transforms,
        "n_dim": 3,
        "skip_augmentations": is_ijepa,
    }

    if is_ijepa is True:
        image_size = keep_first_not_none(args.scaled_crop_size, args.crop_size)
        patch_size = network_config_correct["backbone_args"]["patch_size"]
        feature_map_size = [i // pi for i, pi in zip(image_size, patch_size)]
        network_config_correct["backbone_args"]["image_size"] = image_size
        network_config_correct["feature_map_dimensions"] = feature_map_size

    transforms = SSLTransforms(**transform_args).transforms(
        get_augmentations_ssl(**augmentation_args)
    )

    print(f"Training set size: {len(data_dict)}")

    train_pids = list(data_dict.keys())
    train_list = [value for pid, value in data_dict.items()]

    accelerator, devices, strategy = get_devices(args.dev)

    transforms = monai.transforms.Compose(transforms)
    transforms.set_random_state(args.seed)
    train_dataset = monai.data.CacheDataset(
        train_list,
        monai.transforms.Compose(transforms),
        cache_rate=args.cache_rate,
        num_workers=args.n_workers,
    )

    n_devices = len(devices) if isinstance(devices, list) else 1
    agb = args.accumulate_grad_batches
    bs = network_config_correct["batch_size"]
    steps_per_epoch = len(data_dict) // (bs * n_devices)
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
        network_config=network_config_correct,
        stop_gradient=args.stop_gradient,
    )

    if args.checkpoint is not None:
        state_dict = torch.load(
            args.checkpoint, map_location=args.dev, weights_only=False
        )["state_dict"]
        inc = ssl.load_state_dict(state_dict)

    callbacks = [RichProgressBar()]

    if "cuda" in args.dev:
        ssl = ssl.to("cuda")

    if args.checkpoint is not None:
        state_dict = torch.load(
            args.checkpoint, map_location=args.dev, weights_only=False
        )["state_dict"]
        inc = ssl.load_state_dict(state_dict)
        print(inc)

    callbacks = [RichProgressBar()]

    epochs_ckpt = max_epochs
    ckpt_callback, ckpt_path, status = get_ckpt_callback(
        checkpoint_dir=args.checkpoint_dir,
        checkpoint_name=args.checkpoint_name,
        max_epochs=epochs_ckpt,
        resume_from_last=args.resume_from_last,
        val_fold=None,
        monitor="val_loss",
        metadata={
            "train_pids": train_pids,
            "network_config": network_config,
            "transform_arguments": transform_args,
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
            "network_config": network_config,
            "augment_arguments": augmentation_args,
            "transform_arguments": transform_args,
        },
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
