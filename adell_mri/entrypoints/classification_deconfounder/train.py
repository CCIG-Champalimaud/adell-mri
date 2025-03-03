import sys
from copy import deepcopy

import monai
import numpy as np
import torch
from lightning.pytorch import Trainer
from lightning.pytorch.callbacks import (EarlyStopping, RichProgressBar,
                                         StochasticWeightAveraging)
from sklearn.model_selection import StratifiedKFold, train_test_split

from ...modules.classification.losses import OrdinalSigmoidalLoss
from ...modules.config_parsing import parse_config_cat, parse_config_unet
from ...transform_factory import ClassificationTransforms
from ...transform_factory import get_augmentations_class as get_augmentations
from ...utils.dataset import Dataset
from ...utils.logging import CSVLogger
from ...utils.network_factories import get_deconfounded_classification_network
from ...utils.parser import get_params, merge_args, parse_ids
from ...utils.pl_utils import (delete_checkpoints, get_ckpt_callback,
                               get_devices, get_logger)
from ...utils.torch_utils import (conditional_parameter_freezing,
                                  get_class_weights, get_generator_and_rng,
                                  load_checkpoint_to_model,
                                  set_classification_layer_bias)
from ...utils.utils import safe_collate
from ..assemble_args import Parser


def main(arguments):
    parser = Parser()

    # params
    parser.add_argument_by_key(
        [
            ("classification_net_type", "net_type"),
            "params_from",
            "dataset_json",
            "image_keys",
            "label_keys",
            "cat_confounder_keys",
            "cont_confounder_keys",
            "exclude_surrogate_variables",
            "fill_missing_with_placeholder",
            "fill_conditional",
            "n_features_deconfounder",
            "mask_key",
            "image_masking",
            "image_crop_from_mask",
            "t2_keys",
            "adc_keys",
            "possible_labels",
            "positive_labels",
            "label_groups",
            "cache_rate",
            "target_spacing",
            "pad_size",
            "crop_size",
            "subsample_size",
            "subsample_training_data",
            "filter_on_keys",
            "val_from_train",
            "config_file",
            "dev",
            "n_workers",
            "seed",
            "augment",
            "augment_args",
            "label_smoothing",
            "mixup_alpha",
            "partial_mixup",
            "max_epochs",
            "n_folds",
            "folds",
            "excluded_ids",
            "excluded_ids_from_training_data",
            "checkpoint_dir",
            "checkpoint_name",
            "checkpoint",
            "delete_checkpoints",
            "freeze_regex",
            "not_freeze_regex",
            "exclude_from_state_dict",
            "resume_from_last",
            "monitor",
            "logger_type",
            "project_name",
            "log_model",
            "summary_dir",
            "summary_name",
            "tracking_uri",
            "metric_path",
            "resume",
            "warmup_steps",
            "start_decay",
            "dropout_param",
            "accumulate_grad_batches",
            "gradient_clip_val",
            "early_stopping",
            "swa",
            "learning_rate",
            "batch_size",
            "precision",
            "class_weights",
            "weighted_sampling",
            "correct_classification_bias",
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

    data_dict = Dataset(args.dataset_json, rng=rng)

    if args.excluded_ids_from_training_data is not None:
        excluded_ids_from_training_data = parse_ids(
            args.excluded_ids_from_training_data, output_format="list"
        )
    else:
        excluded_ids_from_training_data = []

    presence_keys = args.image_keys + [args.label_keys]
    cat_key = None
    cont_key = None
    if args.mask_key is not None:
        presence_keys.append(args.mask_key)
    if args.cat_confounder_keys is not None:
        presence_keys.extend(args.cat_confounder_keys)
        cat_key = "cat_confounder"
    if args.cont_confounder_keys is not None:
        presence_keys.extend(args.cont_confounder_keys)
        cont_key = "cont_confounder"
    data_dict.apply_filters(**vars(args), presence_keys=presence_keys)

    cat_vars = None
    cont_vars = None
    if args.cat_confounder_keys is not None:
        cat_vars = []
        for k in args.cat_confounder_keys:
            curr_cat_vars = []
            for kk in data_dict:
                v = data_dict[kk][k]
                if v not in curr_cat_vars:
                    curr_cat_vars.append(v)
            cat_vars.append(curr_cat_vars)
    if args.cont_confounder_keys is not None:
        cont_vars = len(args.cont_confounder_keys)

    all_classes = []
    for k in data_dict:
        C = data_dict[k][args.label_keys]
        if isinstance(C, list):
            C = max(C)
        all_classes.append(str(C))

    label_groups = None
    positive_labels = args.positive_labels
    if args.label_groups is not None:
        n_classes = len(args.label_groups)
        label_groups = [
            label_group.split(",") for label_group in args.label_groups
        ]
        if len(label_groups) == 2:
            positive_labels = label_groups[1]
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
    mask_key = args.mask_key
    input_keys = deepcopy(keys)
    if mask_key is not None:
        input_keys.append(mask_key)
    adc_keys = [k for k in adc_keys if k in keys]
    t2_keys = [k for k in t2_keys if k in keys]

    if args.net_type == "unet":
        network_config, _ = parse_config_unet(
            args.config_file, len(input_keys), n_classes
        )
    else:
        network_config = parse_config_cat(args.config_file)

    if args.batch_size is not None:
        network_config["batch_size"] = args.batch_size
    if args.learning_rate is not None:
        network_config["learning_rate"] = args.learning_rate

    if "batch_size" not in network_config:
        network_config["batch_size"] = 1

    all_pids = [k for k in data_dict]

    print("Setting up transforms...")
    label_mode = "binary" if n_classes == 2 and label_groups is None else "cat"
    transform_arguments = {
        "keys": keys,
        "mask_key": mask_key,
        "image_masking": args.image_masking,
        "image_crop_from_mask": args.image_crop_from_mask,
        "adc_keys": adc_keys,
        "target_spacing": args.target_spacing,
        "crop_size": args.crop_size,
        "pad_size": args.pad_size,
        "clinical_feature_keys": [],
        "possible_labels": args.possible_labels,
        "positive_labels": args.positive_labels,
        "label_groups": label_groups,
        "label_key": args.label_keys,
        "label_mode": label_mode,
        "cat_confounder_keys": args.cat_confounder_keys,
        "cont_confounder_keys": args.cont_confounder_keys,
    }
    augment_arguments = {
        "augment": args.augment,
        "t2_keys": t2_keys,
        "image_keys": keys,
        "mask_key": mask_key,
    } | (eval(args.augment_args) if args.augment_args is not None else {})

    transform_factory = ClassificationTransforms(**transform_arguments)
    transforms_train = transform_factory.transforms(
        get_augmentations(**augment_arguments)
    )
    transforms_train.set_random_state(args.seed)
    transforms_val = transform_factory.transforms()

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
            train_idxs = [i for i, x in enumerate(all_pids) if x not in val_ids]
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

    csv_logger = CSVLogger(args.metric_path, not args.resume_from_last)
    for val_fold in range(args.n_folds):
        train_idxs, val_idxs = next(fold_generator)
        if args.subsample_training_data is not None:
            train_idxs = rng.choice(
                train_idxs,
                size=int(len(train_idxs) * args.subsample_training_data),
                replace=False,
            )
        if args.excluded_ids_from_training_data is not None:
            train_idxs = [
                idx
                for idx in train_idxs
                if all_pids[idx] not in excluded_ids_from_training_data
            ]

        train_pids = [all_pids[i] for i in train_idxs]

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
                "cat_vars": cat_vars,
                "cont_vars": cont_vars,
                "cat_key": cat_key,
                "cont_key": cont_key,
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
                k: weight_sum / (1 + weights[k] * len(weights)) for k in weights
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
            positive_labels=positive_labels,
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
            network_config["loss_fn"] = torch.nn.BCEWithLogitsLoss(
                class_weights
            )
        elif args.net_type == "ord":
            network_config["loss_fn"] = OrdinalSigmoidalLoss(
                class_weights, n_classes
            )
        else:
            network_config["loss_fn"] = torch.nn.CrossEntropyLoss(class_weights)

        n_workers = args.n_workers // n_devices
        bs = network_config["batch_size"]
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
            batch_size=network_config["batch_size"],
            shuffle=False,
            num_workers=n_workers,
            collate_fn=safe_collate,
        )
        validation_loader = monai.data.ThreadDataLoader(
            validation_dataset,
            batch_size=network_config["batch_size"],
            shuffle=False,
            num_workers=n_workers,
            collate_fn=safe_collate,
        )

        network = get_deconfounded_classification_network(
            network_config=network_config,
            dropout_param=args.dropout_param,
            seed=args.seed,
            n_classes=n_classes,
            keys=input_keys,
            cat_confounder_key=cat_key,
            cont_confounder_key=cont_key,
            cat_vars=cat_vars,
            cont_vars=cont_vars,
            train_loader_call=train_loader_call,
            max_epochs=args.max_epochs,
            warmup_steps=args.warmup_steps,
            start_decay=args.start_decay,
            n_features_deconfounder=args.n_features_deconfounder,
            exclude_surrogate_variables=args.exclude_surrogate_variables,
            label_smoothing=args.label_smoothing,
            mixup_alpha=args.mixup_alpha,
            partial_mixup=args.partial_mixup,
        )

        if args.checkpoint is not None:
            if len(args.checkpoint) > 1:
                checkpoint = args.checkpoint[val_fold]
            else:
                checkpoint = args.checkpoint
            load_checkpoint_to_model(
                network, checkpoint, args.exclude_from_state_dict
            )

        conditional_parameter_freezing(
            network, args.freeze_regex, args.not_freeze_regex
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

        if args.swa is True:
            swa_callback = StochasticWeightAveraging(
                network_config["learning_rate"],
                swa_epoch_start=args.warmup_steps,
            )
            callbacks.append(swa_callback)

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

        if args.correct_classification_bias is True and n_classes == 2:
            pos = len([x for x in classes if x in positive_labels])
            neg = len(classes) - pos
            set_classification_layer_bias(pos, neg, network)

        trainer = Trainer(
            accelerator=accelerator,
            devices=devices,
            logger=logger,
            callbacks=callbacks,
            max_epochs=args.max_epochs,
            enable_checkpointing=ckpt,
            gradient_clip_val=args.gradient_clip_val,
            precision=args.precision,
            strategy=strategy,
            accumulate_grad_batches=args.accumulate_grad_batches,
            sync_batchnorm=True if strategy is not None else None,
            check_val_every_n_epoch=1,
        )

        trainer.fit(
            network, train_loader, train_val_loader, ckpt_path=ckpt_path
        )

        # assessing performance on validation set
        print("Validating...")

        ckpt_list = ["last", "best"] if ckpt is True else ["last"]
        for ckpt_key in ckpt_list:
            test_metrics = trainer.test(
                network, validation_loader, ckpt_path=ckpt_key
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
                x = {
                    "metric": k,
                    "checkpoint": ckpt_key,
                    "val_fold": val_fold,
                    "idx": idx,
                    "value": value,
                    "n_train": len(train_pids),
                    "n_val": len(val_pids),
                }
                csv_logger.log(x)
                print(x)

        csv_logger.write()

        if args.delete_checkpoints is True:
            delete_checkpoints(trainer)
