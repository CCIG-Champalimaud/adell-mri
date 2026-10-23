import ast
import sys

import monai
import numpy as np
import torch
from lightning.pytorch import Trainer
from lightning.pytorch.callbacks import RichProgressBar
from sklearn.model_selection import GroupShuffleSplit, train_test_split

from adell_mri.entrypoints.assemble_args import Parser
from adell_mri.entrypoints.generative.autoencoder import (
    attach_autoencoder,
    ensure_autoencoder,
    get_latent_shape,
)
from adell_mri.modules.config_parsing import parse_config_ae, parse_config_gan
from adell_mri.modules.diffusion.recipes import DDPM_RECIPE, LDM_RECIPE
from adell_mri.transform_factory import GenerationTransforms, get_augmentations
from adell_mri.utils.dataset import Dataset
from adell_mri.utils.generic_utils import collate_last_slice, safe_collate
from adell_mri.utils.monai_transforms import RandomSlices
from adell_mri.utils.network_factories import (
    get_gan_network,
    get_generative_network,
)
from adell_mri.utils.optimizer_factory import optimizer_eps_from_precision
from adell_mri.utils.parser import compose, get_params, merge_args
from adell_mri.utils.pl_callbacks import (
    EMACallback,
    LogImageFromDiffusionProcess,
    LogImageFromGAN,
    SpectralNorm,
)
from adell_mri.utils.pl_utils import get_ckpt_callback, get_devices, get_logger
from adell_mri.utils.python_logging import get_logger as get_python_logger
from adell_mri.utils.samplers import DistributedRandomSampler
from adell_mri.utils.torch_utils import (
    conditional_parameter_freezing,
    get_generator_and_rng,
    load_checkpoint_to_model,
)


def get_conditional_specification(d: dict, cond_key: str):
    possible_values = []
    for k in d:
        if cond_key in d[k]:
            v = d[k][cond_key]
            if v not in possible_values:
                possible_values.append(d[k][cond_key])
    return sorted(possible_values)


def split_records(
    records: list[dict], fraction: float | None, seed: int
) -> tuple[list[dict], list[dict]]:
    """
    Splits records into a training and a held-out set.

    If every record has a ``subject`` entry, the split is grouped by subject;
    otherwise the split is random. A fraction of zero or None returns all
    records as training data.

    Args:
        records (list[dict]): records to split.
        fraction (float | None): fraction of records in the held-out set.
        seed (int): random seed.

    Returns:
        tuple[list[dict], list[dict]]: training and held-out records.
    """
    if fraction is None or fraction <= 0.0 or len(records) < 2:
        return records, []
    groups = [record.get("subject") for record in records]
    if all(group is not None for group in groups):
        splitter = GroupShuffleSplit(
            n_splits=1, test_size=fraction, random_state=seed
        )
        train_idx, other_idx = next(splitter.split(records, groups=groups))
    else:
        train_idx, other_idx = train_test_split(
            np.arange(len(records)), test_size=fraction, random_state=seed
        )
    return (
        [records[i] for i in train_idx],
        [records[i] for i in other_idx],
    )


def get_mean_and_std(d: dict, regression_keys: list[str]):
    if regression_keys is None:
        return None, None
    values = {k: [] for k in regression_keys}
    for study_uid in d:
        for k in values:
            if k in d[study_uid]:
                values[k].append(d[study_uid][k])
    means = [np.mean(values[k]) for k in values]
    stds = [np.std(values[k]) for k in values]
    return means, stds


def return_first_not_none(*size_list):
    for size in size_list:
        if size is not None:
            return size


def main(arguments):
    logger = get_python_logger(__name__)
    parser = Parser()

    parser.add_argument_by_key(
        [
            "dataset_json",
            "params_from",
            "image_keys",
            "input_image_keys",
            "input_mask_keys",
            "mask_classes",
            "cat_condition_keys",
            "num_condition_keys",
            "uncondition_proba",
            "filter_on_keys",
            "excluded_ids",
            "augment",
            "augment_args",
            "cache_rate",
            "subsample_size",
            "val_from_train",
            "test_from_train",
            "target_spacing",
            "pad_size",
            "crop_size",
            "config_file",
            "overrides",
            "model_type",
            "ae_max_epochs",
            "spatial_dims",
            "warmup_steps",
            "start_decay",
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
            "freeze_regex",
            "not_freeze_regex",
            "logger_type",
            "project_name",
            "log_model",
            "summary_dir",
            "summary_name",
            "tracking_uri",
            "monitor",
            "metric_path",
            "resume",
            "dropout_param",
            "batch_size",
            "learning_rate",
            "diffusion_steps",
            "ema_decay",
            "fill_missing_with_placeholder",
            "fill_conditional",
            "spectral_norm_power_iterations",
        ]
    )

    args = parser.parse_args(arguments)

    if args.params_from is not None:
        param_dict = get_params(args.params_from)
        args = merge_args(args, param_dict, sys.argv[1:])

    if args.model_type not in ["diffusion", "gan", "ae", "ldm"]:
        raise ValueError(
            f"--model_type must be one of ['diffusion', 'gan', 'ae', 'ldm'], "
            f"got '{args.model_type}'"
        )

    g, rng = get_generator_and_rng(args.seed)

    accelerator, devices, strategy = get_devices(args.dev)
    n_devices = len(devices) if isinstance(devices, list) else 1

    output_file = open(args.metric_path, "w")

    data_dict = Dataset(args.dataset_json, rng=rng)
    data_dict.fill_missing_with_value(args.fill_missing_with_placeholder)

    presence_keys = [*args.image_keys]

    categorical_specification = None
    numerical_specification = None
    with_conditioning = False
    if args.cat_condition_keys is not None:
        categorical_specification = [
            get_conditional_specification(data_dict, k)
            for k in args.cat_condition_keys
        ]
        presence_keys.extend(args.cat_condition_keys)
        with_conditioning = True
    if args.num_condition_keys is not None:
        numerical_specification = len(args.num_condition_keys)
        presence_keys.extend(args.num_condition_keys)
        with_conditioning = True

    if args.input_image_keys is not None:
        presence_keys.extend(args.input_image_keys)

    if args.input_mask_keys is not None:
        if args.mask_classes is None or len(args.mask_classes) != len(
            args.input_mask_keys
        ):
            raise ValueError(
                "--mask_classes must have one entry per --input_mask_keys key"
            )
        presence_keys.extend(args.input_mask_keys)

    data_dict.apply_filters(**vars(args), presence_keys=presence_keys)
    if args.cat_condition_keys:
        data_dict.apply(str, to_keys=args.cat_condition_keys)

    if len(data_dict) == 0:
        raise Exception(
            "No data available for training \
                (dataset={}; keys={})".format(
                args.dataset_json, args.image_keys
            )
        )

    keys = args.image_keys
    all_image_keys = (
        [
            *args.image_keys,
            *args.input_image_keys,
        ]
        if args.input_image_keys is not None
        else [*args.image_keys]
    )

    conditioning_channels = 0
    if args.input_image_keys is not None:
        conditioning_channels += len(args.input_image_keys)
    if args.input_mask_keys is not None:
        conditioning_channels += sum(args.mask_classes)
    input_image_key = "cat_conditioning" if conditioning_channels > 0 else None

    if args.model_type == "gan":
        network_config, gen_config, disc_config = parse_config_gan(
            args.config_file,
            args.image_keys,
            args.input_image_keys,
            input_mask_keys=args.input_mask_keys,
            mask_classes=args.mask_classes,
            spatial_dims=args.spatial_dims or 3,
            batch_size=args.batch_size,
            learning_rate=args.learning_rate,
        )
    elif args.model_type == "diffusion":
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
        network_config["in_channels"] = len(keys) + conditioning_channels
        network_config["out_channels"] = len(keys)

    recipe = None
    if args.model_type == "diffusion":
        recipe = DDPM_RECIPE
    elif args.model_type == "ldm":
        recipe = LDM_RECIPE
    scheduler_config = None
    diffusion_steps = None

    all_pids = [k for k in data_dict]

    logger.info("Setting up transforms...")
    transform_arguments = {
        "keys": all_image_keys,
        "image_keys": keys,
        "input_image_keys": args.input_image_keys,
        "input_mask_keys": args.input_mask_keys,
        "mask_classes": args.mask_classes,
        "target_spacing": args.target_spacing,
        "crop_size": args.crop_size,
        "pad_size": args.pad_size,
        "cat_keys": args.cat_condition_keys,
        "num_keys": args.num_condition_keys,
    }
    augmentation_args = {
        "augment": args.augment,
        "image_keys": all_image_keys,
        "t2_keys": all_image_keys,
        "flip_axis": [0],
        "data_range": (-1.0, 1.0),
    } | (
        ast.literal_eval(args.augment_args)
        if args.augment_args is not None
        else {}
    )

    transform_factory = GenerationTransforms(**transform_arguments)
    transforms_train = transform_factory.transforms(
        get_augmentations(**augmentation_args)
    )

    train_list = data_dict.to_datalist(all_pids)
    transforms_eval = transform_factory.transforms()

    if args.model_type in ["ae", "ldm"]:
        val_fraction = args.val_from_train
        if val_fraction is None:
            val_fraction = 0.1
            logger.info(
                "No --val_from_train specified; using 10% for validation"
            )
        train_val, test_list = split_records(
            train_list, args.test_from_train, args.seed
        )
        train_list, val_list = split_records(train_val, val_fraction, args.seed)
        if len(val_list) == 0:
            val_list = train_list
        if len(test_list) == 0:
            test_list = val_list
        logger.info(
            "Split sizes: train=%s, validation=%s, test=%s",
            len(train_list),
            len(val_list),
            len(test_list),
        )
    else:
        val_list = train_list
        test_list = train_list

    logger.info("Train set size=%s", len(train_list))

    ae_config = None
    ae_checkpoint = None
    latent_channels = None
    latent_shape = None
    if args.model_type in ["ae", "ldm"]:
        if args.spatial_dims not in [None, 3]:
            raise ValueError(
                "--model_type ae and ldm currently support 3D data only"
            )
        if args.model_type == "ldm":
            network_config = compose(
                args.config_file, "diffusion", args.overrides
            )
            if "ae" not in network_config:
                raise ValueError(
                    "--model_type ldm requires an 'ae' section in "
                    "--config_file"
                )
            ae_config = parse_config_ae(
                network_config.pop("ae"),
                args.image_keys,
                spatial_dims=args.spatial_dims or 3,
            )
        else:
            ae_config = parse_config_ae(
                args.config_file,
                args.image_keys,
                spatial_dims=args.spatial_dims or 3,
            )
        ae_logger_name = (
            args.summary_name or args.checkpoint_name or "generative"
        ) + "_ae"
        ae_pl_logger = get_logger(
            summary_name=ae_logger_name,
            summary_dir=args.summary_dir,
            project_name=args.project_name,
            resume=args.resume,
            log_model=args.log_model,
            logger_type=args.logger_type,
            tracking_uri=args.tracking_uri,
            fold=None,
            tags={"model_type": "ae", "ae_config": ae_config},
        )
        autoencoder, ae_checkpoint, _ = ensure_autoencoder(
            args,
            ae_config,
            train_list,
            val_list,
            test_list,
            transforms_train,
            transforms_eval,
            args.checkpoint_dir,
            args.checkpoint_name,
            accelerator,
            devices,
            strategy,
            n_devices,
            ae_pl_logger,
            output_file,
            logger,
        )
        if args.model_type == "ae":
            output_file.close()
            return

        if (
            args.input_image_keys is not None
            or args.input_mask_keys is not None
        ):
            raise ValueError(
                "concat conditioning is not supported for --model_type ldm; "
                "use --cat_condition_keys or --num_condition_keys"
            )
        latent_channels = (
            autoencoder.latent_channels or autoencoder.block_out_channels[-1]
        )
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
        network_config["in_channels"] = latent_channels
        network_config["out_channels"] = latent_channels
        network_config["spatial_dims"] = 3
        input_image_key = None

    if recipe is not None:
        (
            scheduler_config,
            diffusion_steps,
            min_snr_gamma,
            latent_std_decay,
        ) = recipe.resolve(network_config, args.diffusion_steps)
        for key in [
            "scheduler",
            "diffusion_steps",
            "min_snr_gamma",
            "latent_std_decay",
        ]:
            network_config.pop(key, None)
        network_config["min_snr_gamma"] = min_snr_gamma
        if latent_std_decay is not None:
            network_config["latent_std_decay"] = latent_std_decay

    checkpoint_metadata = {
        "train_pids": [record["identifier"] for record in train_list],
        "val_pids": [record["identifier"] for record in val_list],
        "test_pids": [record["identifier"] for record in test_list],
        "network_config": network_config,
        "transform_arguments": transform_arguments,
        "categorical_specification": categorical_specification,
        "numerical_specification": numerical_specification,
        "model_type": args.model_type,
    }
    if recipe is not None:
        checkpoint_metadata.update(
            recipe.to_metadata(scheduler_config, diffusion_steps)
        )
    if args.model_type == "ldm":
        checkpoint_metadata.update(
            {
                "ae_config": ae_config,
                "ae_checkpoint": ae_checkpoint,
                "latent_channels": latent_channels,
            }
        )
    ckpt_callback, ckpt_path, status = get_ckpt_callback(
        checkpoint_dir=args.checkpoint_dir,
        checkpoint_name=args.checkpoint_name,
        max_epochs=args.max_epochs,
        resume_from_last=args.resume_from_last,
        val_fold=None,
        monitor=args.monitor,
        metadata=checkpoint_metadata,
    )
    ckpt = ckpt_callback is not None
    if status == "finished":
        logger.info("Training has finished")
        sys.exit(0)

    logger.info("Number of cases: %s", len(train_list))

    # PL needs a little hint to detect GPUs.
    torch.ones([1]).to("cuda" if "cuda" in args.dev else "cpu")

    spatial_dims = (
        gen_config["spatial_dims"]
        if args.model_type == "gan"
        else network_config["spatial_dims"]
    )
    if spatial_dims == 2:
        slice_keys = ["image"]
        if input_image_key is not None:
            slice_keys.append("cat_conditioning")
        transforms_train.append(RandomSlices(slice_keys, None, n=1))
        collate_fn = collate_last_slice
    else:
        collate_fn = safe_collate
    transforms_train = monai.transforms.Compose(transforms_train)
    transforms_train.set_random_state(args.seed)

    if args.model_type in ["ae", "ldm"]:
        train_dataset = monai.data.Dataset(train_list, transforms_train)
    else:
        train_dataset = monai.data.CacheDataset(
            train_list,
            transforms_train,
            cache_rate=args.cache_rate,
            num_workers=args.n_workers,
        )

    if args.model_type == "ldm":
        sample = train_dataset[0]["image"]
        latent_shape = get_latent_shape(autoencoder, sample.shape[1:])
        checkpoint_metadata["latent_shape"] = latent_shape
        logger.info("Latent shape inferred from the data: %s", latent_shape)

    n_workers = args.n_workers // n_devices
    bs = network_config["batch_size"]
    real_bs = bs * n_devices
    if len(train_dataset) < real_bs:
        new_bs = len(train_dataset) // n_devices
        logger.info(
            "Batch size changed from %s to %s (dataset too small)",
            bs,
            new_bs,
        )
        bs = new_bs
        real_bs = bs * n_devices

    def train_loader_call(batch_size):
        steps_per_epoch = args.steps_per_epoch or max(
            1, len(train_dataset) // real_bs
        )
        num_samples = steps_per_epoch * real_bs
        if n_devices == 1:
            num_samples = min(num_samples, len(train_dataset))
        if n_devices > 1:
            sampler = DistributedRandomSampler(
                train_dataset,
                # this gets divided by num_replicas (as expected), so if we want
                # to keep the same number of steps we have to multiply it again
                # by n_devices
                num_samples=num_samples * n_devices,
                num_replicas=n_devices,
                seed=args.seed,
            )
        else:
            sampler = torch.utils.data.RandomSampler(
                train_dataset,
                replacement=False,
                num_samples=num_samples,
                generator=g,
            )
        loader_kwargs = {
            "batch_size": batch_size,
            "num_workers": n_workers,
            "collate_fn": collate_fn,
            "pin_memory": True,
            "drop_last": True,
            "sampler": sampler,
        }
        if n_workers > 0:
            loader_kwargs["persistent_workers"] = True
            loader_kwargs["prefetch_factor"] = 8
        return monai.data.ThreadDataLoader(train_dataset, **loader_kwargs)

    train_loader = train_loader_call(bs)

    if args.model_type in ["ae", "ldm"]:
        val_loader = monai.data.ThreadDataLoader(
            monai.data.Dataset(val_list, transforms_eval),
            batch_size=bs,
            num_workers=n_workers,
            collate_fn=safe_collate,
            pin_memory=True,
            shuffle=False,
        )
        test_loader = monai.data.ThreadDataLoader(
            monai.data.Dataset(test_list, transforms_eval),
            batch_size=bs,
            num_workers=n_workers,
            collate_fn=safe_collate,
            pin_memory=True,
            shuffle=False,
        )
    else:
        val_loader = train_loader
        test_loader = train_loader

    if args.model_type == "gan":
        means, stds = get_mean_and_std(data_dict, args.num_condition_keys)
        network = get_gan_network(
            network_config=network_config,
            generator_config=gen_config,
            discriminator_config=disc_config,
            training_dataloader_call=train_loader_call,
            categorical_specification=categorical_specification,
            numerical_specification=numerical_specification,
            numerical_moments=(means, stds),
            input_image_key=input_image_key,
            max_epochs=args.max_epochs,
            steps_per_epoch=args.steps_per_epoch or 1,
            pct_start=args.warmup_steps,
            optimizer_eps=optimizer_eps_from_precision(args.precision),
        )
    else:
        network = get_generative_network(
            network_config=network_config,
            scheduler_config=scheduler_config,
            categorical_specification=categorical_specification,
            numerical_specification=numerical_specification,
            train_loader_call=train_loader_call,
            max_epochs=args.max_epochs,
            warmup_steps=args.warmup_steps,
            start_decay=args.start_decay,
            diffusion_steps=diffusion_steps,
            uncondition_proba=args.uncondition_proba,
            concat_condition_key=input_image_key,
            optimizer_eps=optimizer_eps_from_precision(args.precision),
        )

    if args.checkpoint is not None:
        load_checkpoint_to_model(
            network, args.checkpoint[0], args.exclude_from_state_dict
        )

    conditional_parameter_freezing(
        network, args.freeze_regex, args.not_freeze_regex
    )

    if args.model_type == "ldm":
        attach_autoencoder(network, autoencoder)

    # instantiate callbacks and loggers
    callbacks = [RichProgressBar()]

    if ckpt_callback is not None:
        callbacks.append(ckpt_callback)

    if args.spectral_norm_power_iterations is not None:
        spectral_norm = SpectralNorm(
            n_power_iterations=args.spectral_norm_power_iterations
        )
        callbacks.append(spectral_norm)

    if args.ema_decay is not None:
        callbacks.append(
            EMACallback(decay=args.ema_decay, use_ema_weights=True)
        )

    pl_logger = get_logger(
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
            "augment_arguments": None,
            "transform_arguments": transform_arguments,
            "categorical_specification": categorical_specification,
            "numerical_specification": numerical_specification,
        },
    )

    if pl_logger is not None:
        if args.model_type == "ldm":
            size = list(latent_shape[1:])
        else:
            size = return_first_not_none(args.pad_size, args.crop_size)
            size = [int(x) for x in size][:spatial_dims]
        if args.model_type == "gan":
            callbacks.append(
                LogImageFromGAN(
                    n_images=5,
                    size=[len(args.image_keys)] + size,
                )
            )
        else:
            callbacks.append(
                LogImageFromDiffusionProcess(n_images=1, size=size)
            )

    trainer = Trainer(
        accelerator=accelerator,
        devices=devices,
        logger=pl_logger,
        callbacks=callbacks,
        max_epochs=args.max_epochs,
        enable_checkpointing=ckpt,
        gradient_clip_val=args.gradient_clip_val,
        strategy=strategy,
        accumulate_grad_batches=args.accumulate_grad_batches,
        check_val_every_n_epoch=args.check_val_every_n_epoch,
        precision=args.precision,
        deterministic="warn",
    )

    trainer.fit(network, train_loader, val_loader, ckpt_path=ckpt_path)

    # assessing performance on validation set
    logger.info("Validating...")

    if ckpt is True:
        ckpt_list = ["last", "best"]
    else:
        ckpt_list = ["last"]
    for ckpt_key in ckpt_list:
        if ckpt_callback is not None:
            path = (
                ckpt_callback.last_model_path
                if ckpt_key == "last"
                else ckpt_callback.best_model_path
            )
            if path:
                load_checkpoint_to_model(network, path, [], prefer_ema=True)
        test_metrics = trainer.test(network, test_loader)[0]
        for key, value in test_metrics.items():
            if isinstance(value, torch.Tensor):
                value = value.detach().cpu().item()
            output_file.write(f"{ckpt_key}_{key},{value}\n")
