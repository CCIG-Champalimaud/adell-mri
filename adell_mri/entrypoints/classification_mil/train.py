import random
import numpy as np
import torch
import monai
from sklearn.model_selection import train_test_split, StratifiedKFold

from lightning.pytorch import Trainer
from lightning.pytorch.callbacks import (
    EarlyStopping,
    StochasticWeightAveraging,
    RichProgressBar,
)

import sys
from ...entrypoints.assemble_args import Parser
from ...utils import (
    safe_collate,
    set_classification_layer_bias,
    ScaleIntensityAlongDimd,
    EinopsRearranged,
)
from ...utils.pl_utils import (
    get_ckpt_callback,
    get_logger,
    get_devices,
    GPULock,
)
from ...utils.batch_preprocessing import BatchPreprocessing
from ...utils.dataset import Dataset
from ...monai_transforms import get_transforms_classification as get_transforms
from ...monai_transforms import get_augmentations_class as get_augmentations
from ...modules.classification.pl import (
    TransformableTransformerPL,
    MultipleInstanceClassifierPL,
)
from ...modules.config_parsing import parse_config_2d_classifier_3d
from ...utils.parser import get_params, merge_args, parse_ids


def main(arguments):
    parser = Parser()

    parser.add_argument_by_key(
        [
            "params_from",
            "dataset_json",
            "image_keys",
            "t2_keys",
            "label_keys",
            "filter_on_keys",
            "fill_missing_with_placeholder",
            "fill_conditional",
            "possible_labels",
            "positive_labels",
            "label_groups",
            "cache_rate",
            "target_spacing",
            "resize_size",
            "pad_size",
            "crop_size",
            "subsample_size",
            "subsample_training_data",
            "batch_size",
            "learning_rate",
            "config_file",
            "mil_method",
            "module_path",
            "dev",
            "n_workers",
            "seed",
            "augment",
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
            "resume_from_last",
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
            "warmup_steps",
            "start_decay",
            "swa",
            "gradient_clip_val",
            "accumulate_grad_batches",
            "dropout_param",
            "weighted_sampling",
            "correct_classification_bias",
            "class_weights",
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
    gpu_lock = GPULock()
    if devices == "auto":
        is_auto = True
        gpu_lock.lock_first_available()
        devices = [int(i) for i in gpu_lock.locked_gpus]
    else:
        is_auto = False

    output_file = open(args.metric_path, "w")

    data_dict = Dataset(args.dataset_json, rng=rng)

    if args.excluded_ids_from_training_data is not None:
        excluded_ids_from_training_data = parse_ids(
            args.excluded_ids_from_training_data, output_format="list"
        )
    else:
        excluded_ids_from_training_data = []

    presence_keys = args.image_keys + [args.label_keys]
    data_dict.apply_filters(**vars(args), presence_keys=presence_keys)

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
                args.dataset_json, args.image_keys, args.label_keys()
            )
        )

    keys = args.image_keys
    t2_keys = args.t2_keys if args.t2_keys is not None else []
    adc_keys = []
    t2_keys = [k for k in t2_keys if k in keys]

    network_config, _ = parse_config_2d_classifier_3d(
        args.config_file, args.dropout_param, args.mil_method
    )

    if args.batch_size is not None:
        network_config["batch_size"] = args.batch_size
    if args.learning_rate is not None:
        network_config["learning_rate"] = args.learning_rate

    if "batch_size" not in network_config:
        network_config["batch_size"] = 1

    print("Setting up transforms...")
    label_mode = "binary" if n_classes == 2 and label_groups is None else "cat"
    transform_arguments = {
        "keys": keys,
        "adc_keys": adc_keys,
        "target_spacing": args.target_spacing,
        "target_size": args.resize_size,
        "crop_size": args.crop_size,
        "pad_size": args.pad_size,
        "possible_labels": args.possible_labels,
        "positive_labels": args.positive_labels,
        "label_groups": label_groups,
        "label_key": args.label_keys,
        "clinical_feature_keys": [],
        "label_mode": label_mode,
    }

    augment_arguments = {
        "augment": args.augment,
        "t2_keys": t2_keys,
        "all_keys": keys,
        "image_keys": keys,
        "flip_axis": None,
    }

    transforms_common = get_transforms("pre", **transform_arguments)
    transforms_train = monai.transforms.Compose(
        [
            get_augmentations(**augment_arguments, mask_key=None),
            *get_transforms("post", **transform_arguments),
            EinopsRearranged("image", "c h w d -> 1 h w (d c)"),
            ScaleIntensityAlongDimd("image", dim=-1),
        ]
    )
    transforms_val = monai.transforms.Compose(
        [
            *get_transforms("post", **transform_arguments),
            EinopsRearranged("image", "c h w d -> 1 h w (d c)"),
            ScaleIntensityAlongDimd("image", dim=-1),
        ]
    )
    transforms_train.set_random_state(args.seed)

    all_pids = [k for k in data_dict]
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

    full_dataset = monai.data.CacheDataset(
        [data_dict[pid] for pid in data_dict],
        transforms_common,
        cache_rate=args.cache_rate,
        num_workers=args.n_workers,
    )

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
        train_list = [data_dict[pid] for pid in train_pids]
        val_list = [data_dict[pid] for pid in val_pids]  # noqa

        train_dataset = monai.data.Dataset(
            [full_dataset[i] for i in train_idxs], transform=transforms_train
        )
        train_dataset_val = monai.data.Dataset(
            [full_dataset[i] for i in val_idxs], transform=transforms_val
        )
        validation_dataset = monai.data.Dataset(
            [full_dataset[i] for i in val_idxs], transform=transforms_val
        )

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
        class_weights = None
        if args.class_weights is not None:
            if args.class_weights[0] == "adaptive":
                if n_classes == 2:
                    pos = len(
                        [x for x in classes if x in args.positive_labels]
                    )
                    neg = len(classes) - pos
                    weight_neg = (1 / neg) * (len(classes) / 2.0)
                    weight_pos = (1 / pos) * (len(classes) / 2.0)
                    class_weights = weight_pos / weight_neg
                    class_weights = torch.as_tensor(
                        np.array(class_weights),
                        device=args.dev.split(":")[0],
                        dtype=torch.float32,
                    )
                else:
                    pos = {k: 0 for k in args.possible_labels}
                    for c in classes:
                        pos[c] += 1
                    pos = np.array([pos[k] for k in pos])
                    class_weights = (1 / pos) * (len(classes) / 2.0)
                    class_weights = torch.as_tensor(
                        np.array(class_weights),
                        device=args.dev.split(":")[0],
                        dtype=torch.float32,
                    )
            else:
                class_weights = [float(x) for x in args.class_weights]
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
        else:
            network_config["loss_fn"] = torch.nn.CrossEntropy(class_weights)

        n_workers = args.n_workers
        if network_config["batch_size"] < n_workers:
            n_workers = network_config["batch_size"]
        if isinstance(devices, list):
            n_workers = n_workers // len(devices)
        else:
            n_workers = n_workers

        def train_loader_call():
            return monai.data.ThreadDataLoader(
                train_dataset,
                batch_size=network_config["batch_size"],
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

        print("Loading data...")
        batch_preprocessing = BatchPreprocessing(
            args.label_smoothing,
            args.mixup_alpha,
            args.partial_mixup,
            args.seed,
        )
        n_slices = int(len(keys) * args.crop_size[-1])
        boilerplate_args = {
            "n_classes": n_classes,
            "training_dataloader_call": train_loader_call,
            "image_key": "image",
            "label_key": "label",
            "n_epochs": args.max_epochs,
            "warmup_steps": args.warmup_steps,
            "training_batch_preproc": batch_preprocessing,
            "start_decay": args.start_decay,
            "n_slices": n_slices,
        }

        network_config["module"] = torch.jit.load(args.module_path)
        network_config["module"].requires_grad = False
        network_config["module"].eval()
        network_config["module"] = torch.jit.freeze(network_config["module"])
        if "module_out_dim" not in network_config:
            print("2D module output size not specified, inferring...")
            input_example = torch.rand(
                1, 1, *[int(x) for x in args.crop_size][:2]
            ).to(args.dev.split(":")[0])
            output = network_config["module"](input_example)
            network_config["module_out_dim"] = int(output.shape[1])
            print(
                "2D module output size={}".format(
                    network_config["module_out_dim"]
                )
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
        if args.swa is True:
            swa_callback = StochasticWeightAveraging(
                network_config["learning_rate"],
                swa_epoch_start=args.warmup_steps,
            )
            callbacks.append(swa_callback)
        ckpt_callback, ckpt_path, status = get_ckpt_callback(
            checkpoint_dir=args.checkpoint_dir,
            checkpoint_name=args.checkpoint_name,
            max_epochs=args.max_epochs,
            resume_from_last=args.resume_from_last,
            val_fold=val_fold,
            monitor=args.monitor,
        )
        if ckpt_callback is not None:
            callbacks.append(ckpt_callback)
        ckpt = ckpt_callback is not None
        if status == "finished":
            continue

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

        if is_auto is True:
            # reload to correct device
            network_config["module"] = torch.jit.load(
                args.module_path, map_location=f"cuda:{devices[0]}"
            )
            network_config["module"].requires_grad = False
            network_config["module"].eval()
            network_config["module"] = torch.jit.freeze(
                network_config["module"]
            )

        if args.mil_method == "transformer":
            network = TransformableTransformerPL(
                **boilerplate_args, **network_config
            )
        elif args.mil_method == "standard":
            network = MultipleInstanceClassifierPL(
                **boilerplate_args, **network_config
            )

        if args.correct_classification_bias is True and n_classes == 2:
            pos = len([x for x in classes if x in args.positive_labels])
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
            strategy=strategy,
            accumulate_grad_batches=args.accumulate_grad_batches,
            check_val_every_n_epoch=1,
        )

        trainer.fit(
            network, train_loader, train_val_loader, ckpt_path=ckpt_path
        )

        # assessing performance on validation set
        print("Validating...")

        if ckpt is True:
            ckpt_list = ["last", "best"]
        else:
            ckpt_list = ["last"]
        for ckpt_key in ckpt_list:
            test_metrics = trainer.test(
                network, validation_loader, ckpt_path=ckpt_key
            )[0]
            for k in test_metrics:
                out = test_metrics[k]
                if n_classes == 2:
                    try:
                        value = float(out.detach().numpy())
                    except Exception:
                        value = float(out)
                    x = "{},{},{},{},{}".format(
                        k, ckpt_key, val_fold, 0, value
                    )
                    output_file.write(x + "\n")
                    print(x)
                else:
                    for i, v in enumerate(out):
                        x = "{},{},{},{},{}".format(
                            k, ckpt_key, val_fold, i, v
                        )
                        output_file.write(x + "\n")
                        print(x)
