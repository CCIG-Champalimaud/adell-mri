import random
import json
import numpy as np
import torch
import monai
from copy import deepcopy
from sklearn.model_selection import train_test_split, StratifiedKFold

from lightning.pytorch import Trainer
from lightning.pytorch.callbacks import EarlyStopping, RichProgressBar

import sys
from ...entrypoints.assemble_args import Parser
from ...utils import (
    safe_collate,
    set_classification_layer_bias,
    conditional_parameter_freezing,
)
from ...utils.pl_utils import (
    get_ckpt_callback,
    get_logger,
    get_devices,
    delete_checkpoints,
)
from ...utils.torch_utils import load_checkpoint_to_model, get_class_weights
from ...utils.dataset import Dataset
from ...monai_transforms import get_transforms_classification as get_transforms
from ...monai_transforms import get_augmentations_class as get_augmentations
from ...modules.losses import OrdinalSigmoidalLoss
from ...modules.config_parsing import (
    parse_config_unet,
    parse_config_cat,
    parse_config_ensemble,
)
from ...modules.classification.pl import GenericEnsemblePL
from ...utils.network_factories import get_classification_network
from ...utils.parser import get_params, merge_args, parse_ids


def main(arguments):
    parser = Parser()

    parser.add_argument_by_key(
        [
            "params_from",
            "dataset_json",
            "image_keys",
            "clinical_feature_keys",
            "label_keys",
            "mask_key",
            "image_masking",
            "image_crop_from_mask",
            "t2_keys",
            "adc_keys",
            "filter_on_keys",
            "possible_labels",
            "positive_labels",
            "label_groups",
            "cache_rate",
            "target_spacing",
            "pad_size",
            "crop_size",
            "subsample_size",
            "subsample_training_data",
            "val_from_train",
            "config_files",
            "ensemble_config_file",
            "branched",
            ("classification_net_types", "net_types"),
            "module_paths",
            "dev",
            "n_workers",
            "seed",
            "augment",
            "label_smoothing",
            "mixup_alpha",
            "partial_mixup",
            "dropout_param",
            "max_epochs",
            "n_folds",
            "folds",
            "excluded_ids",
            "excluded_ids_from_training_data",
            "checkpoint_dir",
            "checkpoint_name",
            ("checkpoint_ensemble", "checkpoint"),
            "freeze_regex",
            "not_freeze_regex",
            "exclude_from_state_dict",
            "delete_checkpoints",
            "logger_type",
            "project_name",
            "log_model",
            "summary_dir",
            "summary_name",
            "tracking_uri",
            "resume",
            "monitor",
            "metric_path",
            "early_stopping",
            "resume_from_last",
            "warmup_steps",
            "start_decay",
            "gradient_clip_val",
            "accumulate_grad_batches",
            "weighted_sampling",
            "class_weights",
            "correct_classification_bias",
            "batch_size",
            "learning_rate",
        ]
    )

    args = parser.parse_args(arguments)

    if args.params_from is not None:
        param_dict = get_params(args.params_from)
        args = merge_args(args, param_dict, sys.argv[1:])

    torch.manual_seed(args.seed)
    random.seed(args.seed)
    np.random.seed(args.seed)
    g = torch.Generator()
    g.manual_seed(args.seed)
    rng = np.random.default_rng(args.seed)

    accelerator, devices, strategy = get_devices(args.dev)
    n_devices = len(devices) if isinstance(devices, list) else devices
    n_devices = 1 if isinstance(devices, str) else n_devices

    output_file = open(args.metric_path, "w")

    data_dict = Dataset(args.dataset_json, rng=rng)

    if args.clinical_feature_keys is None:
        clinical_feature_keys = []
    else:
        clinical_feature_keys = args.clinical_feature_keys

    if args.excluded_ids_from_training_data is not None:
        excluded_ids_from_training_data = parse_ids(
            args.excluded_ids_from_training_data, output_format="list"
        )
    else:
        excluded_ids_from_training_data = []

    presence_keys = args.image_keys + [args.label_keys] + clinical_feature_keys
    if args.mask_key is not None:
        presence_keys.append(args.mask_key)

    data_dict.apply_filters(**vars(args), presence_keys=presence_keys)
    data_dict.filter_dictionary(
        filters=[f"{k}!=nan" for k in clinical_feature_keys]
    )

    all_classes = []
    for k in data_dict:
        C = data_dict[k][args.label_keys]
        if isinstance(C, list):
            C = max(C)
        all_classes.append(str(C))
    label_groups = None
    if args.label_groups is not None:
        n_classes = len(args.label_groups)
        label_groups = [
            label_group.split(",") for label_group in args.label_groups
        ]
    elif args.positive_labels is None:
        n_classes = len(args.possible_labels)
    else:
        n_classes = 2

    if len(data_dict) == 0:
        raise Exception(
            "No data available for training \
                (dataset={}; keys={}; labels={})".format(
                args.dataset_json, args.image_keys, args.label_keys
            )
        )

    keys = args.image_keys
    adc_keys = args.adc_keys if args.adc_keys is not None else []
    t2_keys = args.t2_keys if args.t2_keys is not None else []
    adc_keys = [k for k in adc_keys if k in keys]
    t2_keys = [k for k in t2_keys if k in keys]
    mask_key = args.mask_key
    input_keys = deepcopy(keys)
    if mask_key is not None:
        input_keys.append(mask_key)

    ensemble_config = parse_config_ensemble(
        args.ensemble_config_file, n_classes
    )

    if args.module_paths is not None:
        config_files = None
        module_paths = args.module_paths
        network_configs = None
    else:
        network_configs = [
            parse_config_unet(config_file, len(keys), n_classes)
            if net_type == "unet"
            else parse_config_cat(config_file)
            for config_file, net_type in zip(config_files, args.net_types)
        ]
        if len(args.config_files) == 1:
            config_files = [args.config_files[0] for _ in args.net_types]
        else:
            config_files = args.config_files

    if args.batch_size is not None:
        ensemble_config["batch_size"] = args.batch_size
    if args.learning_rate is not None:
        ensemble_config["learning_rate"] = args.learning_rate
    if "batch_size" not in ensemble_config:
        ensemble_config["batch_size"] = 1

    all_pids = [k for k in data_dict]

    print("Setting up transforms...")
    label_mode = "binary" if n_classes == 2 and label_groups is None else "cat"
    transform_arguments = {
        "keys": keys,
        "mask_key": mask_key,
        "image_masking": args.image_masking,
        "image_crop_from_mask": args.image_crop_from_mask,
        "clinical_feature_keys": clinical_feature_keys,
        "adc_keys": adc_keys,
        "target_spacing": args.target_spacing,
        "crop_size": args.crop_size,
        "pad_size": args.pad_size,
        "possible_labels": args.possible_labels,
        "positive_labels": args.positive_labels,
        "label_groups": label_groups,
        "label_key": args.label_keys,
        "label_mode": label_mode,
        "branched": args.branched,
    }
    augment_arguments = {
        "augment": args.augment,
        "t2_keys": t2_keys,
        "all_keys": keys,
        "image_keys": keys,
        "mask_key": mask_key,
    }

    transforms_train = monai.transforms.Compose(
        [
            *get_transforms("pre", **transform_arguments),
            get_augmentations(**augment_arguments),
            *get_transforms("post", **transform_arguments),
        ]
    )
    transforms_train.set_random_state(args.seed)

    transforms_val = monai.transforms.Compose(
        [
            *get_transforms("pre", **transform_arguments),
            *get_transforms("post", **transform_arguments),
        ]
    )

    if args.folds is None:
        if args.n_folds > 1:
            fold_generator = StratifiedKFold(
                args.n_folds, shuffle=True, random_state=args.seed
            ).split(all_pids, all_classes)
        else:
            fold_generator = iter(
                [train_test_split(range(len(all_pids)), test_size=0.2)]
            )
    else:
        args.folds = parse_ids(args.folds)
        folds = []
        for fold_idx, val_ids in enumerate(args.folds):
            train_idxs = [
                i for i, x in enumerate(all_pids) if x not in val_ids
            ]
            val_idxs = [i for i, x in enumerate(all_pids) if x in val_ids]
            if len(train_idxs) == 0:
                print("No train samples in fold {}".format(fold_idx))
                continue
            if len(val_idxs) == 0:
                print("No val samples in fold {}".format(fold_idx))
                continue
            else:
                N = len(
                    [
                        i
                        for i in train_idxs
                        if all_pids[i] not in excluded_ids_from_training_data
                    ]
                )
                print(
                    "Fold {}: {} train samples; {} val samples".format(
                        fold_idx, N, len(val_idxs)
                    )
                )
            folds.append([train_idxs, val_idxs])
        args.n_folds = len(folds)
        fold_generator = iter(folds)

    for val_fold in range(args.n_folds):
        train_idxs, val_idxs = next(fold_generator)
        train_pids = [all_pids[i] for i in train_idxs]

        if args.subsample_training_data is not None:
            train_pids = rng.choice(
                train_pids,
                size=int(len(train_pids) * args.subsample_training_data),
                replace=False,
            )
        if args.excluded_ids_from_training_data is not None:
            train_idxs = [
                idx
                for idx in train_idxs
                if all_pids[idx] not in excluded_ids_from_training_data
            ]

        val_pids = [all_pids[i] for i in val_idxs]
        if args.val_from_train is not None:
            n_train_val = int(len(train_pids) * args.val_from_train)
            train_val_pids = rng.choice(train_pids, n_train_val, replace=False)
            train_pids = [
                pid for pid in train_pids if pid not in train_val_pids
            ]
        else:
            train_val_pids = val_pids
        train_list = [data_dict[pid] for pid in train_pids]
        train_val_list = [data_dict[pid] for pid in train_val_pids]
        val_list = [data_dict[pid] for pid in val_pids]

        print("Current fold={}".format(val_fold))
        print("\tTrain set size={}".format(len(train_idxs)))
        print("\tTrain validation set size={}".format(len(train_val_pids)))
        print("\tValidation set size={}".format(len(val_idxs)))

        if len(clinical_feature_keys) > 0:
            clinical_feature_values = [
                [train_sample[k] for train_sample in train_list]
                for k in clinical_feature_keys
            ]
            clinical_feature_values = np.array(
                clinical_feature_values, dtype=np.float32
            )
            clinical_feature_means = np.mean(clinical_feature_values, axis=1)
            clinical_feature_stds = np.std(clinical_feature_values, axis=1)
        else:
            clinical_feature_means = None
            clinical_feature_stds = None

        ckpt_callback, ckpt_path, status = get_ckpt_callback(
            checkpoint_dir=args.checkpoint_dir,
            checkpoint_name=args.checkpoint_name,
            max_epochs=args.max_epochs,
            resume_from_last=args.resume_from_last,
            val_fold=val_fold,
            monitor=args.monitor,
            metadata={
                "train_pids": train_pids,
                "transform_arguments": transform_arguments,
            },
        )
        ckpt = ckpt_callback is not None
        if status == "finished":
            continue

        train_dataset = monai.data.CacheDataset(
            train_list,
            transforms_train,
            cache_rate=args.cache_rate,
            num_workers=args.n_workers,
        )
        train_dataset_val = monai.data.CacheDataset(
            train_val_list,
            transforms_val,
            cache_rate=args.cache_rate,
            num_workers=args.n_workers,
        )
        validation_dataset = monai.data.Dataset(val_list, transforms_val)

        classes = []
        for p in train_list:
            P = str(p[args.label_keys])
            if isinstance(P, list) or isinstance(P, tuple):
                P = max(P)
            classes.append(P)
        U, C = np.unique(classes, return_counts=True)
        for u, c in zip(U, C):
            print("Number of {} cases: {}".format(u, c))
        if args.weighted_sampling is True:
            weights = {k: 0 for k in args.possible_labels}
            for c in classes:
                if c in weights:
                    weights[c] += 1
            weight_sum = np.sum([weights[c] for c in args.possible_labels])
            weights = {
                k: weight_sum / (1 + weights[k] * len(weights))
                for k in weights
            }
            weight_vector = np.array([weights[k] for k in classes])
            weight_vector = np.where(weight_vector < 0.25, 0.25, weight_vector)
            weight_vector = np.where(weight_vector > 4, 4, weight_vector)
            sampler = torch.utils.data.WeightedRandomSampler(
                weight_vector, len(weight_vector), generator=g
            )
        else:
            sampler = None

        # PL needs a little hint to detect GPUs.
        torch.ones([1]).to("cuda" if "cuda" in args.dev else "cpu")

        # get class weights if necessary
        class_weights = get_class_weights(
            args.class_weights,
            classes=classes,
            n_classes=n_classes,
            positive_labels=args.positive_labels,
            possible_labels=args.possible_labels,
            label_groups=label_groups,
        )
        if class_weights is not None:
            class_weights = torch.as_tensor(
                np.array(class_weights),
                device=args.dev.split(":")[0],
                dtype=torch.float32,
            )

        print("Initializing loss with class_weights: {}".format(class_weights))
        if n_classes == 2:
            ensemble_config["loss_fn"] = torch.nn.BCEWithLogitsLoss(
                class_weights
            )
        elif args.net_types[0] == "ord":
            ensemble_config["loss_fn"] = OrdinalSigmoidalLoss(
                class_weights, n_classes
            )
        else:
            ensemble_config["loss_fn"] = torch.nn.CrossEntropyLoss(
                class_weights
            )

        n_workers = args.n_workers // n_devices
        bs = ensemble_config["batch_size"]
        real_bs = bs * n_devices
        if len(train_dataset) < real_bs:
            new_bs = len(train_dataset) // n_devices
            print(
                f"Batch size changed from {bs} to {new_bs} (dataset too small)"
            )
            bs = new_bs
            real_bs = bs * n_devices

        def train_loader_call():
            return monai.data.ThreadDataLoader(
                train_dataset,
                batch_size=bs,
                shuffle=sampler is None,
                num_workers=n_workers,
                generator=g,
                collate_fn=safe_collate,
                pin_memory=True,
                sampler=sampler,
                persistent_workers=args.n_workers > 0,
                drop_last=True,
            )

        train_loader = train_loader_call()
        train_val_loader = monai.data.ThreadDataLoader(
            train_dataset_val,
            batch_size=ensemble_config["batch_size"],
            shuffle=False,
            num_workers=n_workers,
            collate_fn=safe_collate,
        )
        validation_loader = monai.data.ThreadDataLoader(
            validation_dataset,
            batch_size=ensemble_config["batch_size"],
            shuffle=False,
            num_workers=n_workers,
            collate_fn=safe_collate,
        )

        if network_configs is not None:
            networks = [
                get_classification_network(
                    net_type=net_type,
                    network_config=network_config,
                    dropout_param=args.dropout_param,
                    seed=args.seed,
                    n_classes=n_classes,
                    keys=input_keys,
                    clinical_feature_keys=clinical_feature_keys,
                    train_loader_call=train_loader_call,
                    max_epochs=args.max_epochs,
                    warmup_steps=args.warmup_steps,
                    start_decay=args.start_decay,
                    crop_size=args.crop_size,
                    clinical_feature_means=clinical_feature_means,
                    clinical_feature_stds=clinical_feature_stds,
                    label_smoothing=args.label_smoothing,
                    mixup_alpha=args.mixup_alpha,
                    partial_mixup=args.partial_mixup,
                )
                for net_type, network_config in zip(
                    args.net_types, network_configs
                )
            ]
        else:
            networks = []
            for module_path in module_paths:
                network = torch.jit.load(module_path)
                network.requires_grad = False
                network.eval()
                network = torch.jit.freeze(network)
                networks.append(network)

        if args.checkpoint is not None:
            if len(args.checkpoint) > 1:
                checkpoints = args.checkpoint[val_fold]
            else:
                checkpoints = args.checkpoint
            # here each checkpoint should actually be a list of len(networks)
            # checkpoints or size 1. If checkpoint is empty, this is skipped
            checkpoints = checkpoints.split(",")
            if len(checkpoints) == 1:
                checkpoints = [checkpoints[0] for _ in networks]
            if len(checkpoints) != len(networks):
                raise Exception(
                    "len(checkpoints) should be the same as len(networks)"
                )
            for network, checkpoint in zip(networks, checkpoints):
                load_checkpoint_to_model(
                    network, checkpoint, args.exclude_from_state_dict
                )
                conditional_parameter_freezing(
                    network, args.freeze_regex, args.not_freeze_regex
                )
                if args.correct_classification_bias is True and n_classes == 2:
                    pos = len(
                        [x for x in classes if x in args.positive_labels]
                    )
                    neg = len(classes) - pos
                    set_classification_layer_bias(pos, neg, network)

        ensemble = GenericEnsemblePL(
            image_keys=args.image_keys
            if args.branched is False
            else ["image"],
            label_key="label",
            networks=networks,
            n_classes=n_classes,
            training_dataloader_call=train_loader_call,
            n_epochs=args.max_epochs,
            warmup_steps=args.warmup_steps,
            start_decay=args.start_decay,
            **ensemble_config,
        )

        # instantiate callbacks and loggers
        callbacks = [RichProgressBar()]
        if args.early_stopping is not None:
            early_stopping = EarlyStopping(
                "val_loss",
                patience=args.early_stopping,
                strict=True,
                mode="min",
            )
            callbacks.append(early_stopping)

        if ckpt_callback is not None:
            callbacks.append(ckpt_callback)

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
                "network_config": ensemble_config,
                "augment_arguments": augment_arguments,
                "transform_arguments": transform_arguments,
            },
        )

        trainer = Trainer(
            accelerator=accelerator,
            devices=devices,
            logger=logger,
            callbacks=callbacks,
            max_epochs=args.max_epochs,
            enable_checkpointing=ckpt,
            gradient_clip_val=args.gradient_clip_val,
            strategy=strategy,
            accumulate_grad_batches=args.accumulate_grad_batches,
            check_val_every_n_epoch=1,
            deterministic="warn",
        )

        trainer.fit(
            ensemble, train_loader, train_val_loader, ckpt_path=ckpt_path
        )

        # assessing performance on validation set
        print("Validating...")

        if ckpt is True:
            ckpt_list = ["last", "best"]
        else:
            ckpt_list = ["last"]
        for ckpt_key in ckpt_list:
            test_metrics = trainer.test(
                ensemble, validation_loader, ckpt_path=ckpt_key
            )
            test_metrics = test_metrics[0]
            for k in test_metrics:
                out = test_metrics[k]
                try:
                    value = float(out.detach().numpy())
                except Exception:
                    value = float(out)
                if n_classes > 2:
                    k = k.split("_")
                    if k[-1].isdigit():
                        k, idx = "_".join(k[:-1]), k[-1]
                    else:
                        k, idx = "_".join(k), 0
                else:
                    idx = 0
                x = "{},{},{},{},{}".format(k, ckpt_key, val_fold, idx, value)
                output_file.write(x + "\n")
                print(x)

        if args.delete_checkpoints == True:
            delete_checkpoints(trainer)
