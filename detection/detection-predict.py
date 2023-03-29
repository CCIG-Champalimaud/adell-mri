import argparse
import random
import yaml
import numpy as np
import torch
import monai
import json

import sys
sys.path.append(r"..")
from lib.utils import (
    load_anchors)
from lib.monai_transforms import get_transforms_detection
from lib.modules.object_detection import YOLONet3d
from lib.utils.pl_utils import get_devices
from lib.utils.dataset_filters import (
    filter_dictionary_with_filters,
    filter_dictionary_with_presence)
from lib.utils.network_factories import get_detection_network

torch.backends.cudnn.benchmark = True

if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    # data
    parser.add_argument(
        '--sequence_paths',dest='sequence_paths',type=str,nargs="+",
        help="Path to sequence.",default=None)
    parser.add_argument(
        '--dataset_json',dest='dataset_json',type=str,
        help="JSON containing dataset information",default=None)
    parser.add_argument(
        '--image_keys',dest='image_keys',type=str,nargs="+",
        help="Image keys in dataset JSON",required=True)
    parser.add_argument(
        '--input_size',dest='input_size',type=float,nargs='+',
        help="Input size for network (for resize options",required=True)
    parser.add_argument(
        '--anchor_csv',dest='anchor_csv',type=str,
        help="Path to CSV file containing anchors",required=True)
    parser.add_argument(
        '--filter_on_keys',dest='filter_on_keys',type=str,default=[],nargs="+",
        help="Filters the dataset based on a set of specific key:value pairs.")
    parser.add_argument(
        '--adc_keys',dest='adc_keys',type=str,nargs="+",
        help="Image keys corresponding to ADC in dataset JSON")
    
    # network
    parser.add_argument(
        '--config_file',dest="config_file",
        help="Path to network configuration file (yaml)",
        required=True)
    parser.add_argument(
        '--net_type',dest="net_type",
        help="Network type",choices=["yolo"],required=True)
    
    # training
    parser.add_argument('--dev',dest='dev',default="cpu",
        help="Device for PyTorch training",type=str)
    parser.add_argument(
        '--checkpoint',dest='checkpoint',type=str,default=None,
        help='Path to checkpoint.')
    parser.add_argument(
        '--iou_threshold',dest='iou_threshold',type=float,
        help="IoU threshold for pred-gt overlaps.",default=0.5)

    args = parser.parse_args()
        
    accelerator,devices,strategy = get_devices(args.dev)

    anchor_array = load_anchors(args.anchor_csv)
    n_anchors = anchor_array.shape[0]
    if args.dataset_json is not None:
        with open(args.dataset_json,"r") as o:
            data_dict = json.load(o)
        data_dict = filter_dictionary_with_presence(
            data_dict,args.image_keys)
    elif args.sequence_paths is not None:
        if len(args.sequence_paths) != len(args.image_keys):
            raise ValueError(
                "sequence_paths and image_keys must have the same length")
        data_dict = {k:v for k,v in zip(args.image_keys,
                                        args.sequence_paths)}
    else:
        raise TypeError(
            "one of [dataset_json,sequence_paths] must be defined")
    if len(args.filter_on_keys) > 0:
        data_dict = filter_dictionary_with_filters(
            data_dict,args.filter_on_keys)
    
    input_size = [int(i) for i in args.input_size]
    
    keys = args.image_keys
    adc_keys = args.adc_keys if args.adc_keys else []

    with open(args.config_file,'r') as o:
        network_config = yaml.safe_load(o)

    output_example = YOLONet3d(
        n_channels=1,n_c=2,adn_fn=torch.nn.Identity,
        anchor_sizes=anchor_array,dev=args.dev)(
            torch.ones([1,1,*input_size]))
    output_size = output_example[0].shape[2:]

    print("Setting up transforms...")
    transform_arguments = {
        "keys":keys,
        "adc_keys":adc_keys,
        "t2_keys":[],
        "anchor_array":anchor_array,
        "input_size":args.input_size,
        "output_size":output_size,
        "iou_threshold":args.iou_threshold,
        "box_class_key":None,
        "shape_key":None,
        "box_key":None}
    transforms_predict = get_transforms_detection(
        **transform_arguments,augments=[],predict=True)

    path_list = [data_dict[k] for k in data_dict]
    predict_dataset = monai.data.Dataset(
        path_list,
        monai.transforms.Compose(transforms_predict))

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
        dev=devices)
        
    yolo.load_from_checkpoint(args.checkpoint)
    yolo.eval()

    print("Predicting...")
    with torch.no_grad():
        for instance in predict_dataset:
            instance = instance.unsqueeze(0)
            y_hat = yolo(instance)