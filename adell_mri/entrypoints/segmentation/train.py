import random
import json
import numpy as np
import torch
import monai
import gc
import warnings
from sklearn.model_selection import KFold, train_test_split
from multiprocessing import Pool
from tqdm import tqdm

from lightning.pytorch import Trainer
from lightning.pytorch.callbacks import EarlyStopping
from lightning.pytorch.callbacks import RichProgressBar
from lightning.pytorch.utilities.combined_loader import CombinedLoader

from ...entrypoints.assemble_args import Parser
from ...utils import (
    GetAllCropsd,
    PartiallyRandomSampler,
    get_loss_param_dict,
    collate_last_slice,
    RandomSlices,
    SlicesToFirst,
    safe_collate,
    safe_collate_crops,
)
from ...utils.pl_utils import (
    get_ckpt_callback,
    get_logger,
    get_devices,
)
from ...utils.logging import CSVLogger
from ...monai_transforms import get_transforms_unet as get_transforms
from ...monai_transforms import get_augmentations_unet as get_augmentations
from ...monai_transforms import get_semi_sl_transforms
from ...utils.dataset_filters import filter_dictionary
from ...modules.layers import ResNet
from ...utils.network_factories import get_segmentation_network
from ...modules.config_parsing import parse_config_unet, parse_config_ssl
from ...utils.parser import parse_ids
from ...utils.sitk_utils import (
    spacing_values_from_dataset_json,
    get_spacing_quantile,
)
from ...utils.utils import return_classes

torch.backends.cudnn.benchmark = True

warnings.filterwarnings(
    "ignore",
    message=".*unable to generate class balanced samples.*",
)


def if_none_else(x, obj):
    if x is None:
        return obj
    return x


def inter_size(a, b):
    return len(set.intersection(set(a), set(b)))


def main(arguments):
    parser = Parser()

    parser.add_argument_by_key(
        [
            ("dataset_json", "dataset_json", {"nargs": "+"}),
            "image_keys",
            "mask_image_keys",
            ("mask_keys", "mask_keys", {"nargs": "+"}),
            "skip_keys",
            "skip_mask_keys",
            "feature_keys",
            "filter_on_keys",
            "filter_is_optional",
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
            ("segmentation_net_type", "net_type"),
            "res_config_file",
            "encoder_checkpoint",
            "lr_encoder",
            "dev",
            "n_workers",
            "seed",
            "augment",
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
            "optimizer",
            "learning_rate",
            "batch_size",
            "gradient_clip_val",
            "max_epochs",
            "dataset_iterations_per_epoch",
            "precision",
            "n_folds",
            "folds",
            "sliding_window_val",
            "check_val_every_n_epoch",
            "accumulate_grad_batches",
            "picai_eval",
            "metric_path",
            "early_stopping",
            "start_decay",
            "warmup_steps",
            ("class_weights", "class_weights", {"default": [1.0]}),
            "semi_supervised",
        ]
    )

    args = parser.parse_args(arguments)

    torch.manual_seed(args.seed)
    random.seed(args.seed)
    np.random.seed(args.seed)
    g = torch.Generator()
    g.manual_seed(args.seed)

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
    aux_keys = if_none_else(args.skip_keys, [])
    aux_mask_keys = if_none_else(args.skip_mask_keys, [])
    resize_keys = if_none_else(args.resize_keys, [])
    feature_keys = if_none_else(args.feature_keys, [])

    all_aux_keys = aux_keys + aux_mask_keys
    if len(all_aux_keys) > 0:
        aux_key_net = "aux_key"
    else:
        aux_key_net = None
    if len(feature_keys) > 0:
        feature_key_net = "tabular_features"
    else:
        feature_key_net = None

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
    intp.extend(
        [
            *["nearest"] * len(label_keys),
            *["area"] * len(aux_keys),
            *["nearest"] * len(aux_mask_keys),
        ]
    )
    intp_resampling_augmentations.extend(
        [
            *["nearest"] * len(label_keys),
            *["bilinear"] * len(aux_keys),
            *["nearest"] * len(aux_mask_keys),
        ]
    )
    all_keys = [*keys, *label_keys, *aux_keys, *aux_mask_keys]
    all_keys_t = [*all_keys, *feature_keys]
    if args.resize_size is not None:
        args.resize_size = [round(x) for x in args.resize_size]
    if args.crop_size is not None:
        args.crop_size = [round(x) for x in args.crop_size]
    if args.pad_size is not None:
        args.pad_size = [round(x) for x in args.pad_size]
    if args.random_crop_size is not None:
        args.random_crop_size = [round(x) for x in args.random_crop_size]
    label_mode = "binary" if n_classes == 2 else "cat"

    data_dict = {}
    for dataset_json in args.dataset_json:
        cur_dataset_dict = json.load(open(dataset_json, "r"))
        for k in cur_dataset_dict:
            data_dict[k] = cur_dataset_dict[k]
    if args.semi_supervised is True:
        data_dict_ssl = filter_dictionary(
            data_dict,
            filters_presence=args.image_keys,
            possible_labels=None,
            label_key=None,
            filters=args.filter_on_keys,
            filter_is_optional=args.filter_is_optional,
        )
    data_dict = filter_dictionary(
        data_dict,
        filters_presence=args.image_keys + args.mask_keys,
        possible_labels=None,
        label_key=None,
        filters=args.filter_on_keys,
        filter_is_optional=args.filter_is_optional,
    )

    data_dict = {
        k: data_dict[k]
        for k in data_dict
        if inter_size(data_dict[k], set(all_keys_t)) == len(all_keys_t)
    }

    for kk in feature_keys:
        data_dict = {
            k: data_dict[k]
            for k in data_dict
            if np.isnan(data_dict[k][kk]) == False
        }
    if args.subsample_size is not None:
        ss = np.random.choice(
            sorted(list(data_dict.keys())), args.subsample_size, replace=False
        )
        data_dict = {k: data_dict[k] for k in ss}
        if args.semi_supervised is True:
            data_dict_ssl = {k: data_dict_ssl[k] for k in ss}

    if args.excluded_ids is not None:
        args.excluded_ids = parse_ids(args.excluded_ids, output_format="list")
        print("Removing IDs specified in --excluded_ids")
        prev_len = len(data_dict)
        data_dict = {
            k: data_dict[k] for k in data_dict if k not in args.excluded_ids
        }
        print("\tRemoved {} IDs".format(prev_len - len(data_dict)))
        if args.semi_supervised is True:
            print(
                "Removing IDs specified in --excluded_ids from semi-supervision"
            )
            prev_len = len(data_dict_ssl)
            data_dict_ssl = {
                k: data_dict_ssl[k]
                for k in data_dict_ssl
                if k not in args.excluded_ids
            }
            print("\tRemoved {} IDs".format(prev_len - len(data_dict_ssl)))

    if args.target_spacing[0] == "infer":
        target_spacing_dict = spacing_values_from_dataset_json(
            data_dict, key=keys[0], n_workers=args.n_workers
        )

    all_pids = [k for k in data_dict]

    network_config, loss_keys = parse_config_unet(
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
            folds.append([train_idxs, val_idxs])
        args.n_folds = len(folds)
        fold_generator = iter(folds)

    csv_logger = CSVLogger(args.metric_path, not args.resume_from_last)
    for val_fold in range(args.n_folds):
        print("=" * 80)
        print("Starting fold={}".format(val_fold))

        train_idxs, val_idxs = next(fold_generator)
        if args.use_val_as_train_val == False:
            train_idxs, train_val_idxs = train_test_split(
                train_idxs, test_size=0.15
            )
        else:
            print("Using validation as training validation")
            train_val_idxs = val_idxs
        train_pids = [all_pids[i] for i in train_idxs]
        train_val_pids = [all_pids[i] for i in train_val_idxs]
        val_pids = [all_pids[i] for i in val_idxs]
        train_list = [data_dict[pid] for pid in train_pids]
        train_val_list = [data_dict[pid] for pid in train_val_pids]
        val_list = [data_dict[pid] for pid in val_pids]

        if args.semi_supervised is True:
            train_semi_sl_list = [
                data_dict_ssl[pid]
                for pid in data_dict_ssl
                if pid not in val_pids
            ]

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
            "all_aux_keys": all_aux_keys,
            "resize_keys": resize_keys,
            "feature_keys": feature_keys,
            "aux_key_net": aux_key_net,
            "feature_key_net": feature_key_net,
            "resize_size": args.resize_size,
            "pad_size": args.pad_size,
            "crop_size": args.crop_size,
            "random_crop_size": args.random_crop_size,
            "label_mode": label_mode,
            "fill_missing": False,
            "brunet": args.net_type == "brunet",
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
            "flip_axis": [0, 1, 2],
        }
        if args.random_crop_size:
            get_all_crops_transform = [
                GetAllCropsd(args.image_keys + ["mask"], args.random_crop_size)
            ]
        else:
            get_all_crops_transform = []
        transforms_train = [
            *get_transforms("pre", **transform_arguments),
            get_augmentations(**augment_arguments),
            *get_transforms("post", **transform_arguments),
        ]

        transforms_train_val = [
            *get_transforms("pre", **transform_arguments_val),
            *get_all_crops_transform,
            *get_transforms("post", **transform_arguments_val),
        ]

        if (
            args.net_type in ["unetr", "swin", "monai_unetr", "monai_swin"]
            or args.sliding_window_val
        ):
            transforms_val = [
                *get_transforms("pre", **transform_arguments_val),
                *get_all_crops_transform,
                *get_transforms("post", **transform_arguments_val),
            ]
        else:
            transforms_val = [
                *get_transforms("pre", **transform_arguments_val),
                *get_transforms("post", **transform_arguments_val),
            ]

        if args.semi_supervised is True:
            transforms_semi_sl = get_semi_sl_transforms(
                transform_arguments=transform_arguments,
                augment_arguments=augment_arguments,
                keys=keys,
            )

        if network_config["spatial_dimensions"] == 2:
            transforms_train.append(
                RandomSlices(["image", "mask"], "mask", n=8, base=0.05)
            )
            transforms_train_val.append(SlicesToFirst(["image", "mask"]))
            transforms_val.append(SlicesToFirst(["image", "mask"]))
            collate_fn_train = collate_last_slice
            collate_fn_train_semi_sl = collate_last_slice
            collate_fn_val = collate_last_slice
        elif args.random_crop_size is not None:
            collate_fn_train = safe_collate_crops
            collate_fn_train_semi_sl = safe_collate
            if args.net_type in ["unetr", "swin", "monai_unetr", "monai_swin"]:
                collate_fn_val = safe_collate_crops
            else:
                collate_fn_val = safe_collate
        else:
            collate_fn_train = safe_collate
            collate_fn_train_semi_sl = safe_collate
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
            metadata={"transform_arguments": transform_arguments},
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

        transforms_train = monai.transforms.Compose(transforms_train)
        transforms_train.set_random_state(args.seed)
        train_dataset = monai.data.CacheDataset(
            train_list,
            transforms_train,
            num_workers=args.n_workers,
            cache_rate=args.cache_rate,
        )
        train_dataset_val = monai.data.CacheDataset(
            train_val_list,
            monai.transforms.Compose(transforms_train_val),
            num_workers=args.n_workers,
            cache_rate=args.cache_rate,
        )
        validation_dataset = monai.data.Dataset(
            val_list, monai.transforms.Compose(transforms_val)
        )

        if args.semi_supervised is True:
            train_semi_sl_dataset = monai.data.CacheDataset(
                train_semi_sl_list,
                monai.transforms.Compose(transforms_semi_sl),
                num_workers=args.n_workers,
                cache_rate=args.cache_rate,
            )
            train_val_semi_sl_dataset = monai.data.CacheDataset(
                train_val_list,
                monai.transforms.Compose(transforms_semi_sl),
                num_workers=args.n_workers,
                cache_rate=args.cache_rate,
            )

        # calculate the mean/std of tabular features
        if feature_keys is not None:
            all_feature_params = {"mean": [], "std": []}
            for kk in feature_keys:
                f = np.array([x[kk] for x in train_list])
                all_feature_params["mean"].append(np.mean(f))
                all_feature_params["std"].append(np.std(f))
            all_feature_params["mean"] = torch.as_tensor(
                all_feature_params["mean"], dtype=torch.float32, device=dev
            )
            all_feature_params["std"] = torch.as_tensor(
                all_feature_params["std"], dtype=torch.float32, device=dev
            )
        else:
            all_feature_params = None

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
            all_masks = [
                [x[mask_key] for mask_key in label_keys] for x in train_list
            ]
            with Pool(args.n_workers) as pool:
                mapped_fn = pool.imap(return_classes, all_masks)
                with tqdm(mapped_fn, total=len(all_masks)) as t:
                    base = "Calculating positive pixel counts"
                    t.set_description(base)
                    n, n_nonzero = 0, 0
                    for x_classes in t:
                        n += 1
                        all_classes = {}
                        all_classes = {**all_classes, **x_classes}
                        total = []
                        for u, c in all_classes.items():
                            if u not in total:
                                total.append(u)
                            if u != 0:
                                n_nonzero += 1
                                pos_pixel_sum += c
                            total_pixel_sum += c
                        if len(total) > 1:
                            cl.append(1)
                        else:
                            cl.append(0)
                        t.set_description(base + f" ({n_nonzero}/{n})")
            adaptive_weights = len(cl) / np.sum(cl)
            adaptive_pixel_weights = total_pixel_sum / pos_pixel_sum
            if args.constant_ratio is not None:
                sampler = PartiallyRandomSampler(
                    cl, non_keep_ratio=args.constant_ratio, seed=args.seed
                )
                if args.dataset_iterations_per_epoch != 1.0:
                    sampler.set_n_samples(
                        int(sampler.n * args.dataset_iterations_per_epoch)
                    )
                if args.class_weights[0] == "adaptive":
                    adaptive_weights = 1 + args.constant_ratio
        # weights to tensor
        if args.class_weights[0] == "adaptive":
            network_config["loss_fn"]["weight"] = adaptive_weights
        elif args.class_weights[0] == "adaptive_pixel":
            network_config["loss_fn"]["weight"] = adaptive_pixel_weights
        else:
            w = torch.as_tensor(
                np.float32(np.array(args.class_weights)),
                dtype=torch.float32,
                device=dev,
            )
            network_config["loss_fn"].replace_item("weight", w)
        print("Loss specification:", network_config["loss_fn"])

        network_config["loss_fn"].loss_fns_and_kwargs = [
            (
                loss_fn,
                get_loss_param_dict(
                    loss_key=loss_key,
                    **kwargs,
                ),
            )
            for (loss_fn, kwargs), loss_key in zip(
                network_config["loss_fn"].loss_fns_and_kwargs, loss_keys
            )
        ]
        # get loss function parameters
        if "32" not in args.precision:
            network_config["loss_fn"].replace_item("eps", 1e-4)

        if isinstance(devices, list):
            nw = args.n_workers // len(devices)
        else:
            nw = args.n_workers

        def train_loader_call(batch_size):
            train_loader = monai.data.DataLoader(
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
            if args.semi_supervised is True:
                train_semi_sl_loader = monai.data.DataLoader(
                    dataset=train_semi_sl_dataset,
                    batch_size=batch_size,  # noqa: F821
                    num_workers=nw,
                    generator=g,
                    collate_fn=collate_fn_train_semi_sl,
                    pin_memory=True,
                    persistent_workers=True,
                    drop_last=True,
                )
                train_loader = CombinedLoader(
                    {
                        "supervised": train_loader,
                        "self_supervised": train_semi_sl_loader,
                    },
                    "min_size",
                )
            return train_loader

        train_loader = train_loader_call(network_config["batch_size"])
        train_val_loader = monai.data.DataLoader(
            train_dataset_val,
            batch_size=network_config["batch_size"],
            shuffle=False,
            num_workers=nw,
            collate_fn=collate_fn_train,
        )
        validation_loader = monai.data.DataLoader(
            validation_dataset,
            batch_size=1,
            shuffle=False,
            num_workers=args.n_workers,
            collate_fn=collate_fn_val,
        )

        if args.semi_supervised is True:
            train_val_semi_sl_loader = monai.data.DataLoader(
                train_val_semi_sl_dataset,
                batch_size=network_config["batch_size"],
                shuffle=False,
                collate_fn=collate_fn_train_semi_sl,
                num_workers=args.n_workers,
            )
            train_val_loader = CombinedLoader(
                {
                    "supervised": train_val_loader,
                    "self_supervised": train_val_semi_sl_loader,
                },
                "min_size",
            )

        max_steps_optim = len(iter(train_loader)) * args.max_epochs

        if args.res_config_file is not None:
            if args.net_type in ["unetr", "swin", "monai_unetr", "monai_swin"]:
                raise NotImplementedError(
                    "You can't use a ResNet backbone with\
                    a UNETR or SWINUNet model - what's the point?!"
                )
            _, network_config_ssl = parse_config_ssl(
                args.res_config_file, 0.0, len(keys)
            )
            for k in ["weight_decay", "learning_rate", "batch_size"]:
                if k in network_config_ssl:
                    del network_config_ssl[k]
            if args.net_type == "brunet":
                n = len(keys)
                nc = network_config_ssl["backbone_args"]["in_channels"]
                network_config_ssl["backbone_args"]["in_channels"] = nc // n
                res_net = [ResNet(**network_config_ssl) for _ in keys]
            else:
                res_net = [ResNet(**network_config_ssl)]
            if args.encoder_checkpoint is not None:
                for i in range(len(args.encoder_checkpoint)):
                    res_state_dict = torch.load(args.encoder_checkpoint[i])[
                        "state_dict"
                    ]
                    mismatched = res_net[i].load_state_dict(
                        res_state_dict, strict=False
                    )
            backbone = [x.backbone for x in res_net]
            network_config["depth"] = [
                backbone[0].structure[0][0],
                *[x[0] for x in backbone[0].structure],
            ]
            network_config["kernel_sizes"] = [
                3 for _ in network_config["depth"]
            ]
            # the last sum is for the bottleneck layer
            network_config["strides"] = [2]
            if "backbone_args" in network_config_ssl:
                mpl = network_config_ssl["backbone_args"]["maxpool_structure"]
            else:
                mpl = network_config_ssl["maxpool_structure"]
            network_config["strides"].extend(mpl)
            res_ops = [[x.input_layer, *x.operations] for x in backbone]
            res_pool_ops = [
                [x.first_pooling, *x.pooling_operations] for x in backbone
            ]
            if args.encoder_checkpoint is not None:
                # freezes training for resnet encoder
                for enc in res_ops:
                    for res_op in enc:
                        for param in res_op.parameters():
                            if args.lr_encoder == 0.0:
                                param.requires_grad = False

            encoding_operations = [torch.nn.ModuleList([]) for _ in res_ops]
            for i in range(len(res_ops)):
                A = res_ops[i]
                B = res_pool_ops[i]
                for a, b in zip(A, B):
                    encoding_operations[i].append(torch.nn.ModuleList([a, b]))
            encoding_operations = torch.nn.ModuleList(encoding_operations)
        else:
            encoding_operations = [None]

        unet = get_segmentation_network(
            net_type=args.net_type,
            network_config=network_config,
            bottleneck_classification=args.bottleneck_classification,
            clinical_feature_keys=feature_keys,
            all_aux_keys=aux_keys,
            clinical_feature_params=all_feature_params,
            clinical_feature_key_net=feature_key_net,
            aux_key_net=aux_key_net,
            max_epochs=args.max_epochs,
            encoding_operations=encoding_operations,
            picai_eval=args.picai_eval,
            lr_encoder=args.lr_encoder,
            start_decay=args.start_decay,
            warmup_steps=args.warmup_steps,
            encoder_checkpoint=args.bottleneck_classification,
            res_config_file=args.res_config_file,
            deep_supervision=args.deep_supervision,
            n_classes=n_classes,
            keys=keys,
            train_loader_call=train_loader_call,
            random_crop_size=args.random_crop_size,
            crop_size=args.crop_size,
            pad_size=args.pad_size,
            resize_size=args.resize_size,
            semi_supervised=args.semi_supervised,
            max_steps_optim=max_steps_optim,
        )

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

        # add all metadata here
        if ckpt_callback is not None:
            ckpt_callback.metadata = {
                "transform_arguments": transform_arguments,
                "network_config": network_config,
            }
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
            precision=args.precision,
            gradient_clip_val=args.gradient_clip_val,
            detect_anomaly=False,
        )

        trainer.fit(unet, train_loader, train_val_loader, ckpt_path=ckpt_path)

        print("Validating...")
        if ckpt is True:
            ckpt_list = ["last", "best"]
        else:
            ckpt_list = ["last"]
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
                        }
                        csv_logger.log(x)
                        print(x)

        csv_logger.write()
        print("=" * 80)

        # just for safety
        del trainer
        del train_dataset
        del train_loader
        del validation_dataset
        del validation_loader
        del train_dataset_val
        del train_val_loader
        del unet
        if args.semi_supervised:
            del train_semi_sl_dataset
            del train_val_semi_sl_loader
        gc.collect()
