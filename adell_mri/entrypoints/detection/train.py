import json
import random

import monai
import numpy as np
import torch
import yaml
from lightning.pytorch import Trainer
from lightning.pytorch.callbacks import RichProgressBar
from sklearn.model_selection import KFold, train_test_split

from adell_mri.utils.torch_utils import get_generator_and_rng

from ...entrypoints.assemble_args import Parser
from ...modules.object_detection import YOLONet3d
from ...monai_transforms import (get_transforms_detection_post,
                                 get_transforms_detection_pre)
from ...utils import load_anchors, safe_collate
from ...utils.dataset_filters import (filter_dictionary_with_filters,
                                      filter_dictionary_with_presence)
from ...utils.detection import anchors_from_nested_list
from ...utils.network_factories import get_detection_network
from ...utils.pl_utils import get_ckpt_callback, get_devices, get_logger
from ...utils.sitk_utils import spacing_from_dataset_json

torch.backends.cudnn.benchmark = True


def main(arguments):
    parser = Parser()

    parser.add_argument_by_key(
        [
            "dataset_json",
            "image_keys",
            "box_key",
            "box_class_key",
            "shape_key",
            "t2_keys",
            "adc_keys",
            "filter_on_keys",
            "mask_key",
            "mask_mode",
            "target_spacing",
            "pad_size",
            "crop_size",
            "n_classes",
            "anchor_csv",
            "min_anchor_area",
            "config_file",
            ("detection_net_type", "net_type"),
            "dev",
            "n_workers",
            "seed",
            (
                "augment",
                "augment",
                {
                    "help": "Augmentations (use rotate with care, not\
            well defined when objects are close to the border)"
                },
            ),
            "loss_gamma",
            "loss_comb",
            "max_epochs",
            "warmup_steps",
            "n_folds",
            "checkpoint_dir",
            "checkpoint_name",
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
            "class_weights",
            "subsample_size",
            "dropout_param",
            "iou_threshold",
        ]
    )

    args = parser.parse_args(arguments)

    g, rng = get_generator_and_rng(args.seed)

    if args.mask_key is not None:
        mode = "mask"
    else:
        mode = "boxes"

    accelerator, devices, strategy = get_devices(args.dev)

    with open(args.dataset_json, "r") as o:
        data_dict = json.load(o)
    if mode == "boxes":
        data_dict = filter_dictionary_with_presence(
            data_dict, args.image_keys + [args.box_key, args.box_class_key]
        )
    else:
        data_dict = filter_dictionary_with_presence(
            data_dict, args.image_keys + [args.mask_key]
        )
    if len(args.filter_on_keys) > 0:
        data_dict = filter_dictionary_with_filters(
            data_dict, args.filter_on_keys
        )
    if args.subsample_size is not None and len(data_dict) > args.subsample_size:
        data_dict = {
            k: data_dict[k]
            for k in np.random.choice(
                list(data_dict.keys()), args.subsample_size, replace=False
            )
        }

    crop_size = [int(i) for i in args.crop_size]
    pad_size = [int(i) for i in args.pad_size]

    keys = args.image_keys
    adc_keys = args.adc_keys if args.adc_keys else []
    t2_keys = args.t2_keys if args.t2_keys else []
    box_key = args.box_key
    box_class_key = args.box_class_key
    shape_key = args.shape_key
    mask_key = args.mask_key
    target_spacing = args.target_spacing

    with open(args.config_file, "r") as o:
        network_config = yaml.safe_load(o)

    all_pids = [k for k in data_dict]
    if args.n_folds > 1:
        fold_generator = KFold(
            args.n_folds, shuffle=True, random_state=args.seed
        ).split(all_pids)
    else:
        fold_generator = iter(
            [train_test_split(range(len(all_pids)), test_size=0.2)]
        )

    for val_fold in range(args.n_folds):
        train_idxs, val_idxs = next(fold_generator)
        train_idxs, train_val_idxs = train_test_split(train_idxs, test_size=0.2)
        train_pids = [all_pids[i] for i in train_idxs]
        train_val_pids = [all_pids[i] for i in train_val_idxs]
        val_pids = [all_pids[i] for i in val_idxs]
        path_list_train = [data_dict[pid] for pid in train_pids]
        path_list_train_val = [data_dict[pid] for pid in train_val_pids]
        path_list_val = [data_dict[pid] for pid in val_pids]

        if target_spacing[0] == "infer":
            target_spacing = spacing_from_dataset_json(
                dataset_dict={k: data_dict[k] for k in train_pids},
                key=keys[0],
                quantile=0.5,
                n_workers=args.n_workers,
            )
        else:
            target_spacing = [float(x) for x in target_spacing]

        transform_arguments_pre = {
            "keys": keys,
            "adc_keys": adc_keys,
            "crop_size": crop_size,
            "pad_size": pad_size,
            "target_spacing": target_spacing,
            "box_class_key": box_class_key,
            "shape_key": shape_key,
            "box_key": box_key,
            "mask_key": mask_key,
            "mask_mode": args.mask_mode,
        }

        transforms_train = get_transforms_detection_pre(
            **transform_arguments_pre
        )

        train_dataset = monai.data.CacheDataset(
            path_list_train, monai.transforms.Compose(transforms_train)
        )

        if args.anchor_csv == "infer":
            anchor_array = anchors_from_nested_list(
                train_dataset, box_key, shape_key, args.iou_threshold
            )
        else:
            anchor_array = load_anchors(args.anchor_csv)
        if args.min_anchor_area is not None:
            print(
                "Filtering anchor area (minimum area: {})".format(
                    args.min_anchor_area
                )
            )
            anchor_array = anchor_array[
                np.prod(anchor_array, 1) > args.min_anchor_area
            ]

        output_example = YOLONet3d(
            n_channels=1,
            n_classes=args.n_classes,
            adn_fn=torch.nn.Identity,
            anchor_sizes=anchor_array,
            dev=args.dev,
        )(torch.ones([1, 1, *crop_size]))
        output_size = output_example[0].shape[2:]

        transform_arguments_post = {
            "keys": keys,
            "t2_keys": t2_keys,
            "anchor_array": anchor_array,
            "crop_size": crop_size,
            "pad_size": pad_size,
            "output_size": output_size,
            "iou_threshold": args.iou_threshold,
            "box_class_key": box_class_key,
            "shape_key": shape_key,
            "box_key": box_key,
        }

        transforms_train_val = [
            *get_transforms_detection_pre(**transform_arguments_pre),
            *get_transforms_detection_post(
                **transform_arguments_post, augments=[]
            ),
        ]
        transforms_val = [
            *get_transforms_detection_pre(**transform_arguments_pre),
            *get_transforms_detection_post(
                **transform_arguments_post, augments=[]
            ),
        ]

        train_dataset = monai.data.CacheDataset(
            [x for x in train_dataset],
            monai.transforms.Compose(
                get_transforms_detection_post(
                    **transform_arguments_post, augments=args.augment
                )
            ),
        )
        train_dataset_val = monai.data.CacheDataset(
            path_list_train_val, monai.transforms.Compose(transforms_train_val)
        )
        validation_dataset = monai.data.CacheDataset(
            path_list_val, monai.transforms.Compose(transforms_val)
        )

        class_weights = torch.as_tensor(args.class_weights)
        class_weights = class_weights.to(args.dev)

        def train_loader_call():
            monai.data.ThreadDataLoader(
                train_dataset,
                batch_size=network_config["batch_size"],
                shuffle=True,
                num_workers=args.n_workers,
                generator=g,
                collate_fn=safe_collate,
                pin_memory=True,
                persistent_workers=args.n_workers > 0,
            )

        train_loader = train_loader_call()
        train_val_loader = monai.data.ThreadDataLoader(
            train_dataset_val,
            batch_size=network_config["batch_size"],
            shuffle=False,
            num_workers=args.n_workers,
            generator=g,
            collate_fn=safe_collate,
            persistent_workers=args.n_workers > 0,
        )
        validation_loader = monai.data.ThreadDataLoader(
            validation_dataset,
            batch_size=1,
            shuffle=False,
            num_workers=args.n_workers,
            generator=g,
            collate_fn=safe_collate,
            persistent_workers=args.n_workers > 0,
        )

        print("Setting up training...")
        yolo = get_detection_network(
            net_type=args.net_type,
            network_config=network_config,
            dropout_param=args.dropout_param,
            loss_gamma=args.loss_gamma,
            loss_comb=args.loss_comb,
            class_weights=class_weights,
            train_loader_call=train_loader,
            iou_threshold=args.iou_threshold,
            anchor_array=anchor_array,
            n_epochs=args.max_epochs,
            warmup_steps=args.warmup_steps,
            boxes_key=box_key,
            box_class_key=box_class_key,
            n_classes=args.n_classes,
            dev=devices,
        )

        callbacks = [RichProgressBar()]
        ckpt_callback, ckpt_path, status = get_ckpt_callback(
            checkpoint_dir=args.checkpoint_dir,
            checkpoint_name=args.checkpoint_name,
            max_epochs=args.max_epochs,
            resume_from_last=args.resume_from_last,
            val_fold=val_fold,
            monitor=args.monitor,
            metadata={
                "transform_arguments_pre": transform_arguments_pre,
                "transform_arguments_post": transform_arguments_post,
            },
        )
        if status == "finished":
            continue
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
                "network_config": network_config,
                "augment_arguments": None,
                "transform_arguments": {
                    "pre": transform_arguments_pre,
                    "post": transform_arguments_post,
                },
            },
        )

        trainer = Trainer(
            accelerator="gpu" if "cuda" in args.dev else "cpu",
            devices=devices,
            logger=logger,
            callbacks=callbacks,
            max_epochs=args.max_epochs,
            check_val_every_n_epoch=1,
            log_every_n_steps=10,
        )

        trainer.fit(yolo, train_loader, train_val_loader, ckpt_path=ckpt_path)

        # assessing performance on validation set
        print("Validating...")

        trainer.test(yolo, validation_loader)
        for k in yolo.test_metrics:
            out = yolo.test_metrics[k].compute()
            try:
                value = float(out.detach().numpy())
            except Exception:
                value = float(out)
            print("{},{},{},{}".format(k, val_fold, 0, value))

        torch.cuda.empty_cache()
