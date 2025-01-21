import gc
import json
import os
from copy import deepcopy

import monai
import numpy as np
import torch
from tqdm import tqdm

from ...entrypoints.assemble_args import Parser
from ...modules.config_parsing import parse_config_ssl, parse_config_unet
from ...modules.layers import ResNet
from ...monai_transforms import get_transforms_unet as get_transforms
from ...utils.utils import collate_last_slice, safe_collate
from ...utils.monai_transforms import SlicesToFirst
from ...utils.dataset import Dataset
from ...utils.inference import SegmentationInference, TensorListReduction
from ...utils.network_factories import get_segmentation_network
from ...utils.parser import parse_ids
from ...utils.sitk_writer import SitkWriter

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
            "adc_keys",
            "feature_keys",
            "prediction_ids",
            "excluded_ids",
            "filter_on_keys",
            "filter_is_optional",
            "target_spacing",
            "resize_size",
            "resize_keys",
            "pad_size",
            "crop_size",
            "bottleneck_classification",
            "possible_labels",
            "positive_labels",
            "missing_to_empty",
            "config_file",
            ("segmentation_net_type", "net_type"),
            ("segmentation_prediction_mode", "prediction_mode"),
            "res_config_file",
            "checkpoint",
            "one_to_one",
            "dev",
            "n_workers",
            "ensemble",
            "sliding_window_size",
            "flip",
            (
                "output_path",
                "output_path",
                {
                    "help": "Path to output masks (if prediction_mode == image) or json if\
          prediction_mode in [deep_features,bounding_box]",
                    "required": True,
                },
            ),
            "monte_carlo_dropout_iterations",
        ]
    )

    args = parser.parse_args(arguments)

    if args.possible_labels == 2 or args.positive_labels is not None:
        n_classes = 2
    else:
        n_classes = args.possible_labels

    keys = args.image_keys

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
    intp.extend(["area"] * len(aux_keys))
    intp.extend(["nearest"] * len(aux_mask_keys))
    intp_resampling_augmentations.extend(["bilinear"] * len(aux_keys))
    intp_resampling_augmentations.extend(["nearest"] * len(aux_mask_keys))
    all_keys = [*keys, *aux_keys, *aux_mask_keys]
    all_keys_t = [*all_keys, *feature_keys]
    if args.resize_size is not None:
        args.resize_size = [round(x) for x in args.resize_size]

    data_dict = Dataset(args.dataset_json, "r", verbose=True)
    data_dict.filter_dictionary(
        filters_presence=args.image_keys,
        possible_labels=None,
        label_key=None,
        filters=args.filter_on_keys,
        filter_is_optional=args.filter_is_optional,
    )
    if args.excluded_ids is not None:
        data_dict.subsample_dataset(excluded_key_list=args.excluded_ids)

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
        data_dict.dataset = {
            k: data_dict[k]
            for k in data_dict
            if np.isnan(data_dict[k][kk]) == False  # noqa
        }

    network_config, loss_key = parse_config_unet(
        args.config_file, len(keys), n_classes
    )

    label_mode = "binary" if n_classes == 2 else "cat"
    transform_arguments = {
        "all_keys": all_keys,
        "image_keys": keys,
        "label_keys": None,
        "non_adc_keys": non_adc_keys,
        "adc_keys": adc_keys,
        "target_spacing": args.target_spacing,
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
        "crop_size": args.crop_size,
        "label_mode": label_mode,
        "fill_missing": args.missing_to_empty is not None,
        "brunet": args.net_type == "brunet",
        "track_meta": False,
        "convert_to_tensor": False,
    }

    transforms = [
        *get_transforms("pre", **transform_arguments),
        *get_transforms("post", **transform_arguments),
    ]

    if network_config["spatial_dimensions"] == 2:
        transforms.append(SlicesToFirst(["image"]))
        collate_fn = collate_last_slice
    else:
        collate_fn = safe_collate  # noqa

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

    unet_base = get_segmentation_network(
        net_type=args.net_type,
        encoding_operations=encoding_operations,
        network_config=network_config,
        bottleneck_classification=args.bottleneck_classification,
        clinical_feature_keys=feature_keys,
        all_aux_keys=aux_keys,
        clinical_feature_params=all_feature_params,
        clinical_feature_key_net=feature_key_net,
        aux_key_net=aux_key_net,
        max_epochs=100,
        picai_eval=False,
        lr_encoder=None,
        encoder_checkpoint=args.bottleneck_classification,
        res_config_file=args.res_config_file,
        deep_supervision=False,
        n_classes=n_classes,
        keys=keys,
        train_loader_call=None,
        random_crop_size=None,
        crop_size=args.crop_size,
        pad_size=args.pad_size,
        resize_size=args.resize_size,
        semi_supervised=False,
    )

    if args.prediction_ids is not None:
        args.prediction_ids = parse_ids(args.prediction_ids)
    else:
        args.prediction_ids = [[k for k in data_dict]]
    n_data = len(args.prediction_ids)
    output = {}
    sitk_writer = SitkWriter(2)
    pred_mode = args.prediction_mode
    for pred_idx in range(n_data):
        pred_ids = [k for k in args.prediction_ids[pred_idx] if k in data_dict]
        curr_dict = {k: data_dict[k] for k in pred_ids}

        transform_input = transforms[0]
        transforms_preprocess = monai.transforms.Compose(transforms[1:])
        transforms_postprocess = monai.transforms.Invertd(
            "image", transforms_preprocess
        )

        if args.one_to_one is True:
            checkpoint_list = [args.checkpoint[pred_idx]]
        else:
            checkpoint_list = args.checkpoint

        networks = []
        for checkpoint in checkpoint_list:
            state_dict = torch.load(checkpoint)["state_dict"]
            state_dict = {
                k: state_dict[k]
                for k in state_dict
                if all(
                    [
                        "deep_supervision_ops" not in k,
                        "ema." not in k,
                        "linear_transformation.weight" not in k,
                        "linear_transformation.bias" not in k,
                    ]
                )
            }
            unet = deepcopy(unet_base)
            unet.load_state_dict(state_dict)
            unet = unet.to(args.dev)
            unet.eval()
            if args.monte_carlo_dropout_iterations is not None:
                assert args.prediction_mode == [
                    "probs",
                    "logits",
                ], "monte_carlo_dropout_iterations only supported for \
                    prediction_mode probs or logits"
                for mod in unet.modules():
                    if mod.__class__.__name__.startswith("Dropout"):
                        mod.train()

            networks.append(unet)

        if pred_mode in ["image", "probs"]:
            postproc_fn = unet.final_layer[-1]
        else:
            postproc_fn = None
        inference_fns = [
            SegmentationInference(
                base_inference_function=network.predict_step,
                sliding_window_size=args.sliding_window_size,
                stride=0.5,
                n_classes=n_classes if n_classes > 2 else 1,
                flip=args.flip,
                flip_keys=["image"],
                ndim=network_config["spatial_dimensions"],
                mc_iterations=args.monte_carlo_dropout_iterations,
                reduction=TensorListReduction(postproc_fn=postproc_fn),
            )
            for network in networks
        ]
        for key in tqdm(curr_dict):
            data_element = curr_dict[key]
            data_element = transform_input(data_element)
            data_element = transforms_preprocess(data_element)
            for k in all_keys:
                data_element[k] = data_element[k].to(args.dev).unsqueeze(0)
            pred_id = key

            if pred_mode in ["image", "probs", "bounding_box", "logits"]:
                pred = [
                    inference_fn(
                        data_element,
                        0,
                        return_logits=True,
                        return_only_segmentation=True,
                    )[0]
                    for inference_fn in inference_fns
                ]
                pred = [
                    transforms_postprocess({"image": p})["image"] for p in pred
                ]
            elif pred_mode == "deep_features":
                pred = [
                    inference_fn(
                        data_element,
                        0,
                        return_bottleneck=True,
                        return_only_segmentation=True,
                    )[2][0]
                    for inference_fn in inference_fns
                ]
                pred = [x.flatten(start_dim=1).max(1).values for x in pred]

            pred = torch.stack(pred, -1)
            if args.ensemble is not None:
                pred = pred.mean(-1)
            if pred_mode in ["image", "probs"]:
                pred = postproc_fn(pred)
            pred = pred.detach().cpu()
            if pred_mode == "image":
                pred = np.int32(np.round(pred))
                pred = np.transpose(pred, [2, 1, 0]).to(torch.int16)
                output_path = os.path.join(
                    args.output_path, pred_id + ".nii.gz"
                )
                t_image = curr_dict[pred_id]["image"]
                sitk_writer.put(output_path, pred, t_image)
            elif pred_mode in ["probs", "logits"]:
                sh = pred.shape
                if len(sh) == 3:
                    pred = np.transpose(pred, [2, 1, 0])
                elif len(sh) == 4:
                    pred = np.transpose(pred, [3, 2, 1, 0])
                elif len(sh) == 5:
                    pred = np.transpose(pred, [4, 0, 3, 2, 1])
                output_path = os.path.join(
                    args.output_path, pred_id + ".nii.gz"
                )
                t_image = curr_dict[pred_id]["image"]

                sitk_writer.put(output_path, pred, t_image)
            elif pred_mode == "deep_features":
                pred = pred
                output[pred_id] = pred.tolist()
            elif pred_mode == "bounding_box":
                pred = monai.transforms.KeepLargestConnectedComponent()(pred)
                coords = np.where(pred > 0)
                bb_min = [x.min() for x in coords]
                bb_max = [x.max() for x in coords]
                output[pred_id] = [*bb_min, *bb_max]
            gc.collect()

    sitk_writer.close()

    if pred_mode in ["deep_features", "bounding_box"]:
        with open(args.output_path, "w") as o:
            json.dump(output, o)
