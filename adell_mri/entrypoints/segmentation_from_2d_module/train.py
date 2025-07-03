import gc
from copy import deepcopy

import monai
import numpy as np
import torch
from lightning.pytorch import Trainer
from lightning.pytorch.callbacks import EarlyStopping, RichProgressBar
from sklearn.model_selection import KFold, train_test_split
from tqdm import tqdm

from adell_mri.utils.logging import CSVLogger
from adell_mri.utils.torch_utils import get_generator_and_rng

from adell_mri.entrypoints.assemble_args import Parser
from adell_mri.modules.config_parsing import parse_config_unet
from adell_mri.modules.layers.adn_fn import get_adn_fn
from adell_mri.modules.segmentation.pl import MIMUNetPL
from adell_mri.transform_factory import SegmentationTransforms
from adell_mri.transform_factory import (
    get_augmentations_unet as get_augmentations,
)
from adell_mri.utils.dataset import Dataset
from adell_mri.utils.monai_transforms import (
    GetAllCropsd,
    RandomSlices,
    SlicesToFirst,
)
from adell_mri.utils.parser import parse_ids
from adell_mri.utils.pl_utils import get_ckpt_callback, get_devices, get_logger
from adell_mri.utils.samplers import PartiallyRandomSampler
from adell_mri.utils.sitk_utils import (
    get_spacing_quantile,
    spacing_values_from_dataset_json,
)
from adell_mri.utils.utils import (
    collate_last_slice,
    get_loss_param_dict,
    safe_collate,
    safe_collate_crops,
)

torch.backends.cudnn.benchmark = True


def if_none_else(x, obj):
    if x is None:
        return obj
    return x


def inter_size(a, b):
    return len(set.intersection(set(a), set(b)))


def get_first(*lists):
    for ll in lists:
        if ll is not None:
            return ll


def main(arguments):
    parser = Parser()

    parser.add_argument_by_key(
        [
            "dataset_json",
            "image_keys",
            "mask_image_keys",
            "mask_keys",
            "skip_keys",
            "skip_mask_keys",
            "feature_keys",
            "subsample_size",
            "excluded_ids",
            "use_val_as_train_val",
            "cache_rate",
            "adc_keys",
            "t2_keys",
            "target_spacing",
            "resize_size",
            "resize_keys",
            "pad_size",
            "crop_size",
            "random_crop_size",
            "n_crops",
            "possible_labels",
            "positive_labels",
            "bottleneck_classification",
            "deep_supervision",
            "constant_ratio",
            "config_file",
            "module_path",
            ("segmentation_net_type", "net_type"),
            "res_config_file",
            "encoder_checkpoint",
            "lr_encoder",
            "dev",
            "n_workers",
            "seed",
            "augment",
            "augment_args",
            "loss_gamma",
            "loss_comb",
            "loss_scale",
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
            "monitor",
            "learning_rate",
            "batch_size",
            "gradient_clip_val",
            "max_epochs",
            "dataset_iterations_per_epoch",
            "precision",
            "n_folds",
            "folds",
            "check_val_every_n_epoch",
            "accumulate_grad_batches",
            "picai_eval",
            "metric_path",
            "early_stopping",
            ("class_weights", "class_weights", {"default": [1.0]}),
        ]
    )

    args = parser.parse_args(arguments)

    g, rng = get_generator_and_rng(args.seed)

    accelerator, devices, strategy = get_devices(args.dev)
    dev = args.dev.split(":")[0]

    if args.possible_labels == 2 or args.positive_labels is not None:
        n_classes = 2
    else:
        n_classes = args.possible_labels

    keys = args.image_keys
    label_keys = args.mask_keys

    mask_image_keys = if_none_else(args.mask_image_keys, [])
    adc_keys = if_none_else(args.adc_keys, [])
    t2_keys = if_none_else(args.t2_keys, [])
    resize_keys = if_none_else(args.resize_keys, [])

    adc_keys = [k for k in adc_keys if k in keys]
    intp = []
    intp_resampling_augmentations = []
    for k in keys:
        if k in mask_image_keys:
            intp.append("nearest")
            intp_resampling_augmentations.append("nearest")
        else:
            intp.append("area")
            intp_resampling_augmentations.append("bilinear")
    non_adc_keys = [k for k in keys if k not in adc_keys]
    intp.extend(["nearest"] * len(label_keys))
    intp_resampling_augmentations.extend(["nearest"] * len(label_keys))
    all_keys = [*keys, *label_keys]
    all_keys_t = [*all_keys]
    if args.resize_size is not None:
        args.resize_size = [round(x) for x in args.resize_size]
    if args.crop_size is not None:
        args.crop_size = [round(x) for x in args.crop_size]
    if args.pad_size is not None:
        args.pad_size = [round(x) for x in args.pad_size]
    if args.random_crop_size is not None:
        args.random_crop_size = [round(x) for x in args.random_crop_size]
    label_mode = "binary" if n_classes == 2 else "cat"

    data_dict = Dataset(args.dataset_json, rng=rng)
    if args.missing_to_empty is None:
        data_dict.dataset = {
            k: data_dict.dataset[k]
            for k in data_dict.dataset
            if inter_size(data_dict[k], set(all_keys_t)) == len(all_keys_t)
        }
    else:
        if "mask" in args.missing_to_empty:
            data_dict.dataset = {
                k: data_dict.dataset[k]
                for k in data_dict.dataset
                if inter_size(data_dict[k], set(mask_image_keys)) >= 0
            }

    data_dict.apply_filters(**vars(args))

    if args.target_spacing[0] == "infer":
        target_spacing_dict = spacing_values_from_dataset_json(
            data_dict, key=keys[0], n_workers=args.n_workers
        )

    all_pids = [k for k in data_dict]

    network_config, loss_key = parse_config_unet(
        args.config_file, len(keys), n_classes
    )
    if args.learning_rate is not None:
        network_config["learning_rate"] = args.learning_rate
    if args.batch_size is not None:
        network_config["batch_size"] = args.batch_size

    if args.folds is None:
        if args.n_folds > 1:
            fold_generator = KFold(
                args.n_folds, shuffle=True, random_state=args.seed
            ).split(all_pids)
        else:
            fold_generator = iter(
                [train_test_split(range(len(all_pids)), test_size=0.2)]
            )
    else:
        args.folds = parse_ids(args.folds)
        folds = []
        for fold_idx, val_ids in enumerate(args.folds):
            train_idxs = [i for i, x in enumerate(all_pids) if x not in val_ids]
            val_idxs = [i for i, x in enumerate(all_pids) if x in val_ids]
            if len(train_idxs) == 0:
                print("No train samples in fold {}".format(fold_idx))
                continue
            if len(val_idxs) == 0:
                print("No val samples in fold {}".format(fold_idx))
                continue
            folds.append([train_idxs, val_idxs])
        args.n_folds = len(folds)
        fold_generator = iter(folds)

    csv_logger = CSVLogger(args.metric_path, not args.resume_from_last)
    for val_fold in range(args.n_folds):
        print("=" * 80)
        print("Starting fold={}".format(val_fold))

        train_idxs, val_idxs = next(fold_generator)
        if args.use_val_as_train_val is False:
            train_idxs, train_val_idxs = train_test_split(
                train_idxs, test_size=0.15
            )
        else:
            train_val_idxs = val_idxs
        train_pids = [all_pids[i] for i in train_idxs]
        train_val_pids = [all_pids[i] for i in train_val_idxs]
        val_pids = [all_pids[i] for i in val_idxs]
        train_list = [data_dict[pid] for pid in train_pids]
        train_val_list = [data_dict[pid] for pid in train_val_pids]
        val_list = [data_dict[pid] for pid in val_pids]

        if args.target_spacing[0] == "infer":
            target_spacing = get_spacing_quantile(
                {k: target_spacing_dict[k] for k in train_pids}
            )
        else:
            target_spacing = [float(x) for x in args.target_spacing]

        transform_arguments = {
            "all_keys": all_keys,
            "image_keys": keys,
            "label_keys": label_keys,
            "non_adc_keys": non_adc_keys,
            "adc_keys": adc_keys,
            "target_spacing": target_spacing,
            "intp": intp,
            "intp_resampling_augmentations": intp_resampling_augmentations,
            "possible_labels": args.possible_labels,
            "positive_labels": args.positive_labels,
            "adc_factor": args.adc_factor,
            "all_aux_keys": [],
            "resize_keys": resize_keys,
            "feature_keys": [],
            "aux_key_net": [],
            "feature_key_net": [],
            "resize_size": args.resize_size,
            "pad_size": args.pad_size,
            "crop_size": args.crop_size,
            "random_crop_size": args.random_crop_size,
            "label_mode": label_mode,
            "fill_missing": args.missing_to_empty is not None,
            "brunet": False,
        }
        transform_arguments_val = {
            k: transform_arguments[k] for k in transform_arguments
        }
        transform_arguments_val["random_crop_size"] = None
        transform_arguments_val["crop_size"] = None
        augment_arguments = {
            "augment": args.augment,
            "all_keys": all_keys,
            "image_keys": keys,
            "t2_keys": t2_keys,
            "random_crop_size": args.random_crop_size,
            "n_crops": args.n_crops,
        } | (eval(args.augment_args) if args.augment_args is not None else {})
        if args.random_crop_size:
            get_all_crops_transform = [
                GetAllCropsd(args.image_keys + ["mask"], args.random_crop_size)
            ]
        else:
            get_all_crops_transform = []
        transform_factory = SegmentationTransforms(**transform_arguments)
        transforms_train = transform_factory.transforms(
            get_augmentations(**augment_arguments)
        )

        transforms_train_val = transform_factory.transforms(
            get_all_crops_transform
        )

        transforms_val = transform_factory.transforms()

        if network_config["spatial_dimensions"] == 2:
            transforms_train.append(
                RandomSlices(["image", "mask"], "mask", n=8, base=0.05)
            )
            transforms_train_val.append(SlicesToFirst(["image", "mask"]))
            transforms_val.append(SlicesToFirst(["image", "mask"]))
            collate_fn_train = collate_last_slice
            collate_fn_val = collate_last_slice
        elif args.random_crop_size is not None:
            collate_fn_train = safe_collate_crops
            collate_fn_val = safe_collate
        else:
            collate_fn_train = safe_collate
            collate_fn_val = safe_collate

        callbacks = [RichProgressBar()]
        ckpt_path = None

        ckpt_callback, ckpt_path, status = get_ckpt_callback(
            checkpoint_dir=args.checkpoint_dir,
            checkpoint_name=args.checkpoint_name,
            max_epochs=args.max_epochs,
            resume_from_last=args.resume_from_last,
            val_fold=val_fold,
            monitor=args.monitor,
            metadata={
                "train_pids": train_pids,
                "val_pids": val_pids,
                "transform_arguments": transform_arguments,
            },
        )
        ckpt = ckpt_callback is not None
        if status == "finished":
            continue
        if ckpt_callback is not None:
            callbacks.append(ckpt_callback)

        if args.checkpoint is not None:
            if len(args.checkpoint) >= (val_fold + 1):
                ckpt_path = args.checkpoint[val_fold]
                print(
                    "Resuming training from checkpoint in {}".format(ckpt_path)
                )

        transforms_train.set_random_state(args.seed)
        train_dataset = monai.data.CacheDataset(
            train_list,
            transforms_train,
            num_workers=args.n_workers,
            cache_rate=args.cache_rate,
        )
        train_dataset_val = monai.data.CacheDataset(
            train_val_list,
            transforms_train_val,
            num_workers=args.n_workers,
        )
        validation_dataset = monai.data.Dataset(val_list, transforms_val)

        n_samples = int(len(train_dataset) * args.dataset_iterations_per_epoch)
        sampler = torch.utils.data.RandomSampler(
            ["element" for _ in train_idxs],
            num_samples=n_samples,
            replacement=len(train_dataset) < n_samples,
            generator=g,
        )
        if isinstance(args.class_weights[0], str):
            ad = "adaptive" in args.class_weights[0]
        else:
            ad = False
        # include some constant label images
        if args.constant_ratio is not None or ad:
            cl = []
            pos_pixel_sum = 0
            total_pixel_sum = 0
            with tqdm(train_list) as t:
                t.set_description(
                    "Setting up partially random sampler/adaptive weights"
                )
                for x in t:
                    intersection = set.intersection(
                        set(label_keys), set(x.keys())
                    )
                    if len(intersection) > 0:
                        masks = monai.transforms.LoadImaged(
                            keys=label_keys, allow_missing_keys=True
                        )(x)
                        total = []
                        for k in intersection:
                            for u, c in zip(
                                *np.unique(masks[k], return_counts=True)
                            ):
                                if u not in total:
                                    total.append(u)
                                if u != 0:
                                    pos_pixel_sum += c
                                total_pixel_sum += c
                        if len(total) > 1:
                            cl.append(1)
                        else:
                            cl.append(0)
                    else:
                        cl.append(0)
            adaptive_weights = len(cl) / np.sum(cl)
            adaptive_pixel_weights = total_pixel_sum / pos_pixel_sum
            if args.constant_ratio is not None:
                sampler = PartiallyRandomSampler(
                    cl, non_keep_ratio=args.constant_ratio, seed=args.seed
                )
                if args.class_weights[0] == "adaptive":
                    adaptive_weights = 1 + args.constant_ratio
        # weights to tensor
        if args.class_weights[0] == "adaptive":
            weights = adaptive_weights
        elif args.class_weights[0] == "adaptive_pixel":
            weights = adaptive_pixel_weights
        else:
            weights = torch.as_tensor(
                np.float32(np.array(args.class_weights)),
                dtype=torch.float32,
                device=dev,
            )
        print("Weights set to:", weights)

        # get loss function parameters
        loss_params = get_loss_param_dict(
            weights=weights,
            gamma=args.loss_gamma,
            comb=args.loss_comb,
            scale=args.loss_scale,
        )[loss_key]
        if "eps" in loss_params and args.precision != "32":
            if loss_params["eps"] < 1e-4:
                loss_params["eps"] = 1e-4

        if isinstance(devices, list):
            nw = args.n_workers // len(devices)
        else:
            nw = args.n_workers

        def train_loader_call(batch_size):
            return monai.data.ThreadDataLoader(
                dataset=train_dataset,  # noqa: F821
                batch_size=batch_size,  # noqa: F821
                num_workers=nw,
                generator=g,
                sampler=sampler,
                collate_fn=collate_fn_train,
                pin_memory=True,
                persistent_workers=True,
                drop_last=True,
            )

        train_loader = train_loader_call(network_config["batch_size"])
        train_val_loader = monai.data.ThreadDataLoader(
            train_dataset_val,
            batch_size=1,
            shuffle=False,
            num_workers=nw,
            collate_fn=collate_fn_train,
            persistent_workers=True,
        )
        validation_loader = monai.data.ThreadDataLoader(
            validation_dataset,
            batch_size=1,
            shuffle=False,
            num_workers=args.n_workers,
            collate_fn=collate_fn_val,
            persistent_workers=True,
        )

        # make other network configurations compatible with mimunet
        network_config_corr = deepcopy(network_config)
        adn_fn_args = {
            "spatial_dim": 3,
            "norm_fn": "batch",
            "act_fn": "swish",
            "dropout_param": 0.1,
        }
        if "norm_type" in network_config_corr:
            adn_fn_args["norm_fn"] = network_config_corr["norm_type"]
        if "activation_fn" in network_config_corr:
            adn_fn_args["act_fn"] = network_config_corr["activation_fn"]
        if "dropout_param" in network_config_corr:
            adn_fn_args["dropout_param"] = network_config_corr["dropout_param"]
        network_config_corr["adn_fn"] = get_adn_fn(**adn_fn_args)
        network_config_corr = {
            k: network_config_corr[k]
            for k in network_config_corr
            if k
            not in [
                "depth",
                "spatial_dimensions",
                "conv_type",
                "norm_type",
                "interpolation",
                "dropout_param",
                "activation_fn",
                "kernel_sizes",
                "strides",
                "padding",
            ]
        }
        module_2d = torch.jit.load(args.module_path, map_location=args.dev)
        module_2d.requires_grad = False
        module_2d = module_2d.eval()
        size = get_first(args.random_crop_size, args.crop_size, args.pad_size)
        unet = MIMUNetPL(
            module=module_2d,
            n_classes=n_classes,
            n_slices=size[-1],
            image_key="image",
            label_key="mask",
            cosine_decay=args.cosine_decay,
            n_epochs=args.max_epochs,
            training_dataloader_call=train_loader_call,
            deep_supervision=args.deep_supervision,
            loss_params=loss_params,
            **network_config_corr,
        )
        unet.module = torch.jit.freeze(unet.module)

        if args.early_stopping is not None:
            early_stopping = EarlyStopping(
                "val_loss",
                patience=args.early_stopping,
                strict=True,
                mode="min",
            )
            callbacks.append(early_stopping)

        logger = get_logger(
            summary_name=args.summary_name,
            summary_dir=args.summary_dir,
            project_name=args.project_name,
            resume=args.resume,
            log_model=args.log_model,
            logger_type=args.logger_type,
            tracking_uri=args.tracking_uri,
            fold=val_fold,
            tags={
                "network_config": network_config,
                "augment_arguments": augment_arguments,
                "transform_arguments": transform_arguments,
            },
        )

        precision = {"32": 32, "16": 16, "bf16": "bf16"}[args.precision]
        trainer = Trainer(
            accelerator=dev,
            devices=devices,
            logger=logger,
            callbacks=callbacks,
            strategy=strategy,
            max_epochs=args.max_epochs,
            enable_checkpointing=ckpt,
            accumulate_grad_batches=args.accumulate_grad_batches,
            check_val_every_n_epoch=args.check_val_every_n_epoch,
            log_every_n_steps=10,
            precision=precision,
            gradient_clip_val=args.gradient_clip_val,
            detect_anomaly=False,
        )

        trainer.fit(unet, train_loader, train_val_loader, ckpt_path=ckpt_path)

        print("Validating...")
        ckpt_list = ["last", "best"] if ckpt is True else ["last"]
        for ckpt_key in ckpt_list:
            test_metrics = trainer.test(
                unet, validation_loader, ckpt_path=ckpt_key
            )[0]
            for k in test_metrics:
                out = test_metrics[k]
                if n_classes == 2:
                    try:
                        value = float(out.detach().numpy())
                    except Exception:
                        value = float(out)
                    x = {
                        "metric": k,
                        "checkpoint": ckpt_key,
                        "val_fold": val_fold,
                        "idx": 0,
                        "value": value,
                        "n_train": len(train_pids),
                        "n_val": len(val_pids),
                    }
                    csv_logger.log(x)
                    print(x)
                else:
                    for i, v in enumerate(out):
                        x = {
                            "metric": k,
                            "checkpoint": ckpt_key,
                            "val_fold": val_fold,
                            "idx": i,
                            "value": v,
                            "n_train": len(train_pids),
                            "n_val": len(val_pids),
                        }
                        csv_logger.log(x)
                        print(x)

        print("=" * 80)
        gc.collect()

        # just for safety
        del trainer
        del train_dataset
        del train_loader
        del validation_dataset
        del validation_loader
        del train_dataset_val
        del train_val_loader
