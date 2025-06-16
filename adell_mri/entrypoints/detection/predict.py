import json
import sys

import monai
import torch
import yaml

from ...entrypoints.assemble_args import Parser
from ...modules.object_detection import YOLONet3d
from ...transform_factory.transforms import DetectionTransforms
from ...utils.dataset_filters import (
    filter_dictionary_with_filters,
    filter_dictionary_with_presence,
)
from ...utils.network_factories import get_detection_network
from ...utils.pl_utils import get_devices
from ...utils.utils import load_anchors

sys.path.append(r"..")

torch.backends.cudnn.benchmark = True


def main(arguments):
    parser = Parser()

    parser.add_argument_by_key(
        [
            (
                "dataset_json",
                "dataset_json",
                {"default": None, "required": False},
            ),
            "sequence_paths",
            "image_keys",
            "shape_key",
            "adc_keys",
            "filter_on_keys",
            "target_spacing",
            "pad_size",
            "crop_size",
            "n_classes",
            "anchor_csv",
            "config_file",
            ("detection_net_type", "net_type"),
            "dev",
            "n_workers",
            "seed",
            "checkpoint",
            "iou_threshold",
        ]
    )

    args = parser.parse_args(arguments)

    accelerator, devices, strategy = get_devices(args.dev)

    anchor_array = load_anchors(args.anchor_csv)
    if args.dataset_json is not None:
        with open(args.dataset_json, "r") as o:
            data_dict = json.load(o)
        data_dict = filter_dictionary_with_presence(data_dict, args.image_keys)
    elif args.sequence_paths is not None:
        if len(args.sequence_paths) != len(args.image_keys):
            raise ValueError(
                "sequence_paths and image_keys must have the same length"
            )
        data_dict = {k: v for k, v in zip(args.image_keys, args.sequence_paths)}
    else:
        raise TypeError("one of [dataset_json,sequence_paths] must be defined")
    if len(args.filter_on_keys) > 0:
        data_dict = filter_dictionary_with_filters(
            data_dict, args.filter_on_keys
        )

    pad_size = [int(i) for i in args.pad_size]
    crop_size = [int(i) for i in args.crop_size]

    keys = args.image_keys
    adc_keys = args.adc_keys if args.adc_keys else []

    with open(args.config_file, "r") as o:
        network_config = yaml.safe_load(o)

    output_example = YOLONet3d(
        in_channels=1,
        n_c=2,
        adn_fn=torch.nn.Identity,
        anchor_sizes=anchor_array,
        dev=args.dev,
    )(torch.ones([1, 1, *crop_size]))
    output_size = output_example[0].shape[2:]

    print("Setting up transforms...")
    transform_arguments = {
        "keys": keys,
        "adc_keys": adc_keys,
        "pad_size": pad_size,
        "crop_size": crop_size,
        "target_spacing": args.target_spacing,
        "box_class_key": None,
        "shape_key": None,
        "box_key": None,
        "mask_key": None,
        "mask_mode": None,
        "t2_keys": None,
        "anchor_array": anchor_array,
        "pad_size": pad_size,
        "crop_size": crop_size,
        "output_size": output_size,
        "iou_threshold": args.iou_threshold,
        "predict": False,
    }
    transforms_predict = DetectionTransforms(**transform_arguments).transforms()

    path_list = [data_dict[k] for k in data_dict]
    predict_dataset = monai.data.Dataset(
        path_list, monai.transforms.Compose(transforms_predict)
    )

    print("Setting up training...")
    yolo = get_detection_network(
        net_type=args.net_type,
        network_config=network_config,
        dropout_param=0.0,
        loss_gamma=None,
        loss_comb=None,
        class_weights=None,
        train_loader_call=None,
        iou_threshold=args.iou_threshold,
        anchor_array=anchor_array,
        n_epochs=100,
        warmup_steps=10,
        dev=devices,
    )

    yolo.load_from_checkpoint(args.checkpoint)
    yolo.eval()

    print("Predicting...")
    with torch.no_grad():
        for instance in predict_dataset:
            instance = instance.unsqueeze(0)
            # y_hat = yolo(instance)
