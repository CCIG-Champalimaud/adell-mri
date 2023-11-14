import argparse
import json
import sys
import os
import numpy as np
import torch
import monai
import gc
from tqdm import trange

from ...entrypoints.assemble_args import Parser
from ...utils import collate_last_slice, SlicesToFirst, safe_collate
from ...monai_transforms import get_transforms_unet as get_transforms
from ...modules.layers import ResNet
from ...utils.network_factories import get_segmentation_network
from ...modules.config_parsing import parse_config_unet, parse_config_ssl
from ...modules.segmentation.pl import get_metric_dict
from ...utils.inference import SlidingWindowSegmentation, FlippedInference
from ...utils.parser import parse_ids

torch.backends.cudnn.benchmark = True


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
            "dataset_json",
            "image_keys",
            "mask_image_keys",
            "skip_keys",
            "skip_mask_keys",
            "mask_keys",
            "bottleneck_classification",
            "possible_labels",
            "positive_labels",
            "missing_to_empty",
            "adc_keys",
            "feature_keys",
            "test_ids",
            "excluded_ids",
            "resize_size",
            "resize_keys",
            "crop_size",
            "target_spacing",
            "sliding_window_size",
            "flip",
            "keep_largest_connected_component",
            "config_file",
            ("segmentation_net_type", "unet_model"),
            "res_config_file",
            "checkpoint",
            "one_to_one",
            "per_sample",
            "dev",
            "n_workers",
            "picai_eval",
            "metric_path",
        ]
    )

    args = parser.parse_args(arguments)

    if args.possible_labels == 2 or args.positive_labels is not None:
        n_classes = 2
    else:
        n_classes = args.possible_labels

    keys = args.image_keys
    label_keys = args.mask_keys

    mask_image_keys = if_none_else(args.mask_image_keys, [])
    adc_keys = if_none_else(args.adc_keys, [])
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
    intp.extend(["nearest"] * len(label_keys))
    intp.extend(["area"] * len(aux_keys))
    intp.extend(["nearest"] * len(aux_mask_keys))
    intp_resampling_augmentations.extend(["nearest"] * len(label_keys))
    intp_resampling_augmentations.extend(["bilinear"] * len(aux_keys))
    intp_resampling_augmentations.extend(["nearest"] * len(aux_mask_keys))
    all_keys = [*keys, *label_keys, *aux_keys, *aux_mask_keys]
    all_keys_t = [*all_keys, *feature_keys]
    if args.resize_size is not None:
        args.resize_size = [round(x) for x in args.resize_size]

    data_dict = json.load(open(args.dataset_json, "r"))
    if args.excluded_ids is not None:
        excluded_ids = parse_ids(args.excluded_ids, "list")
        data_dict = {
            k: data_dict[k] for k in data_dict if k not in excluded_ids
        }

    if args.missing_to_empty is None:
        data_dict = {
            k: data_dict[k]
            for k in data_dict
            if inter_size(data_dict[k], set(all_keys_t)) == len(all_keys_t)
        }
    else:
        if "image" in args.missing_to_empty:
            obl_keys = [*aux_keys, *aux_mask_keys, *feature_keys]
            opt_keys = keys
            data_dict = {
                k: data_dict[k]
                for k in data_dict
                if inter_size(data_dict[k], obl_keys) == len(obl_keys)
            }
            data_dict = {
                k: data_dict[k]
                for k in data_dict
                if inter_size(data_dict[k], opt_keys) > 0
            }
        if "mask" in args.missing_to_empty:
            data_dict = {
                k: data_dict[k]
                for k in data_dict
                if inter_size(data_dict[k], set(mask_image_keys)) >= 0
            }

    for kk in feature_keys:
        data_dict = {
            k: data_dict[k]
            for k in data_dict
            if np.isnan(data_dict[k][kk]) == False
        }

    network_config, loss_key = parse_config_unet(
        args.config_file, len(keys), n_classes
    )

    label_mode = "binary" if n_classes == 2 else "cat"
    transform_arguments = {
        "all_keys": all_keys,
        "image_keys": keys,
        "label_keys": label_keys,
        "non_adc_keys": non_adc_keys,
        "adc_keys": adc_keys,
        "target_spacing": args.target_spacing,
        "intp": intp,
        "intp_resampling_augmentations": intp_resampling_augmentations,
        "possible_labels": args.possible_labels,
        "positive_labels": args.positive_labels,
        "adc_factor": 1 / 3,
        "all_aux_keys": all_aux_keys,
        "resize_keys": resize_keys,
        "feature_keys": feature_keys,
        "aux_key_net": aux_key_net,
        "feature_key_net": feature_key_net,
        "resize_size": args.resize_size,
        "crop_size": args.crop_size,
        "pad_size": args.sliding_window_size,
        "label_mode": label_mode,
        "fill_missing": args.missing_to_empty is not None,
        "brunet": args.net_type == "brunet",
    }

    transforms = [
        *get_transforms("pre", **transform_arguments),
        *get_transforms("post", **transform_arguments),
    ]

    if network_config["spatial_dimensions"] == 2:
        transforms.append(SlicesToFirst(["image", "mask"]))
        collate_fn = collate_last_slice
    else:
        collate_fn = safe_collate

    torch.cuda.empty_cache()

    if ":" in args.dev:
        dev = args.dev.split(":")[0]
    else:
        dev = args.dev

    # calculate the mean/std of tabular features
    if feature_keys is not None:
        all_feature_params = {}
        all_feature_params["mean"] = torch.zeros(
            len(feature_keys), dtype=torch.float32, device=dev
        )
        all_feature_params["std"] = torch.ones(
            len(feature_keys), dtype=torch.float32, device=dev
        )
    else:
        all_feature_params = None

    if args.res_config_file is not None:
        _, network_config_ssl = parse_config_ssl(
            args.res_config_file, 0.0, len(keys), network_config["batch_size"]
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
        backbone = [x.backbone for x in res_net]
        network_config["depth"] = [
            backbone[0].structure[0][0],
            *[x[0] for x in backbone[0].structure],
        ]
        network_config["kernel_sizes"] = [3 for _ in network_config["depth"]]
        # the last sum is for the bottleneck layer
        network_config["strides"] = [2]
        network_config["strides"].extend(
            network_config_ssl["backbone_args"]["maxpool_structure"]
        )
        res_ops = [[x.input_layer, *x.operations] for x in backbone]
        res_pool_ops = [
            [x.first_pooling, *x.pooling_operations] for x in backbone
        ]

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
        encoding_operations=encoding_operations,
        network_config=network_config,
        loss_params={},
        bottleneck_classification=args.bottleneck_classification,
        clinical_feature_keys=feature_keys,
        all_aux_keys=aux_keys,
        clinical_feature_params=all_feature_params,
        clinical_feature_key_net=feature_key_net,
        aux_key_net=aux_key_net,
        max_epochs=100,
        picai_eval=args.picai_eval,
        lr_encoder=None,
        cosine_decay=False,
        encoder_checkpoint=args.bottleneck_classification,
        res_config_file=args.res_config_file,
        deep_supervision=False,
        n_classes=n_classes,
        keys=keys,
        train_loader_call=None,
        random_crop_size=None,
        crop_size=args.crop_size,
        pad_size=args.sliding_window_size,
        resize_size=args.resize_size,
    )

    test_ids = parse_ids(args.test_ids)
    n_ckpt = len(args.checkpoint)
    n_data = len(args.test_ids)
    all_metrics = []
    for test_idx in range(n_data):
        test_ids = [
            k for k in args.test_ids[test_idx].split(",") if k in data_dict
        ]
        curr_dict = {k: data_dict[k] for k in test_ids}
        data_list = [curr_dict[k] for k in curr_dict]
        dataset = monai.data.CacheDataset(
            data_list,
            monai.transforms.Compose(transforms),
            num_workers=args.n_workers,
        )

        if args.one_to_one == True:
            checkpoint_list = [args.checkpoint[test_idx]]
        else:
            checkpoint_list = args.checkpoint

        for checkpoint in checkpoint_list:
            state_dict = torch.load(checkpoint)["state_dict"]
            state_dict = {
                k: state_dict[k]
                for k in state_dict
                if "deep_supervision_ops" not in k
            }
            unet.load_state_dict(state_dict, strict=False)
            unet = unet.eval()
            unet = unet.to(args.dev)

            inference_function = unet.predict_step
            if args.sliding_window_size is not None:
                inference_function = SlidingWindowSegmentation(
                    sliding_window_size=args.sliding_window_size,
                    inference_function=inference_function,
                    n_classes=n_classes if n_classes > 2 else 1,
                    stride=[x // 2 for x in args.sliding_window_size],
                )
            if args.flip == True:
                flips = [(1,), (2,), (3,)]
                inference_function = FlippedInference(
                    inference_function=inference_function,
                    flips=flips,
                    flip_keys=["image"],
                    ndim=network_config["spatial_dimensions"],
                    inference_batch_size=len(flips),
                )

            metrics = get_metric_dict(
                n_classes, False, prefix="T_", dev=args.dev
            )
            metrics_global = get_metric_dict(
                n_classes, False, prefix="T_", dev=args.dev
            )
            metrics_dict = {"global_metrics": {}, "checkpoint": checkpoint}
            if args.per_sample is True:
                metrics_dict["metrics"] = {}
            for i in trange(len(dataset)):
                data_element = dataset[i]
                data_element = {
                    k: data_element[k].to(args.dev) for k in data_element
                }
                test_id = test_ids[i]
                pred = inference_function(
                    data_element, 0, return_only_segmentation=True
                )
                pred = pred.round().long().squeeze()
                y = data_element["mask"].round().int().squeeze()
                if args.per_sample is True:
                    metrics_dict["metrics"][test_id] = {}
                    for k in metrics:
                        metrics[k].update(pred, y)
                        v = metrics[k].compute().cpu().numpy().tolist()
                        metrics_dict["metrics"][test_id][k] = v
                        metrics[k].reset()
                        metrics_global[k].update(pred, y)
                for k in metrics_global:
                    metrics_global[k].update(pred, y)
            for k in metrics_global:
                v = metrics_global[k].compute().cpu().numpy().tolist()
                metrics_dict["global_metrics"][k] = v
            all_metrics.append(metrics_dict)
    metrics_json = json.dumps(all_metrics, indent=2)
    with open(args.metric_path, "w") as out_file:
        out_file.write(metrics_json)
    gc.collect()
