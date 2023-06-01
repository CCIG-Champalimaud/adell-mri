import sys
sys.path.append(r"..")
from lib.modules.self_supervised.pl import (
    UNet,ConvNeXt,ResNet)
from lib.modules.config_parsing import parse_config_ssl,parse_config_unet
import inspect
from copy import deepcopy
import argparse
import torch

torch.backends.cudnn.benchmark = True

def force_cudnn_initialization():
    """Convenience function to initialise CuDNN (and avoid the lazy loading
    from PyTorch).
    """
    s = 16
    dev = torch.device('cuda')
    torch.nn.functional.conv2d(torch.zeros(s,s,s,s,device=dev), 
                               torch.zeros(s,s,s,s,device=dev))

def filter_dicom_dict_on_presence(data_dict,all_keys):
    def check_intersection(a,b):
        return len(set.intersection(set(a),set(b))) == len(set(b))
    for k in data_dict:
        for kk in data_dict[k]:
            data_dict[k][kk] = [
                element for element in data_dict[k][kk]
                if check_intersection(element.keys(),all_keys)]
    return data_dict

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Converts a SelfSL model to torchscript")

    # network + training
    parser.add_argument(
        '--net_type',dest='net_type',
        choices=["resnet","unet_encoder","convnext"],
        help="Which network should be trained.")
    parser.add_argument(
        "--input_shape",dest="input_shape",required=True,
        help="Unbatched input shape",type=int,nargs="+")
    parser.add_argument(
        '--config_file',dest="config_file",
        help="Path to network configuration file (yaml)",
        required=True)
    parser.add_argument(
        '--checkpoint',dest='checkpoint',action="store",
        help="Loads this checkpoint")
    parser.add_argument(
        '--output_model_path',dest='output_model_path',required=True,
        help="Output path the .pt model")
    
    # training
    parser.add_argument(
        '--dev',dest='dev',type=str,default="cuda",
        help="Device for model allocation")

    args = parser.parse_args()
    
    if args.net_type == "unet_encoder":
        network_config,_ = parse_config_unet(
            args.config_file,args.input_shape[0],2)
        network_config_correct = deepcopy(network_config)
        for k in network_config:
            if k in ["loss_fn"]:
                del network_config_correct[k]
    else:
        network_config,network_config_correct = parse_config_ssl(
            args.config_file,0.0,args.input_shape[0])

    network_config_correct = {k:network_config_correct[k]
                              for k in network_config_correct
                              if k not in ["prediction_head_args",
                                           "projection_head_args"]}
    network_config_correct["projection_head_args"] = {}
    if args.net_type == "unet_encoder":
        network_config_correct["encoder_only"] = True
        fn_args = [k for k in inspect.signature(UNet).parameters]
        network_config_correct = {k:network_config_correct[k] 
                                  for k in network_config_correct
                                  if k in fn_args}
        ssl = UNet(**network_config_correct)
    elif args.net_type == "convnext":
        fn_args = [k for k in inspect.signature(ConvNeXt).parameters]
        network_config_correct = {k:network_config_correct[k] 
                                  for k in network_config_correct
                                  if k in fn_args}
        network_config_correct["backbone_args"] = {
            k:network_config_correct["backbone_args"][k] 
            for k in network_config_correct["backbone_args"]
            if k not in ["res_type"]}
        ssl = ConvNeXt(**network_config_correct)
    else:
        fn_args = [k for k in inspect.signature(ResNet).parameters]
        network_config_correct = {k:network_config_correct[k] 
                                  for k in network_config_correct
                                  if k in fn_args}
        ssl = ResNet(**network_config_correct)

    ssl = ssl.to(args.dev)
    state_dict = torch.load(
        args.checkpoint,
        map_location=args.dev.split(":")[0])['state_dict']
    state_dict = {k:state_dict[k] for k in state_dict
                  if "prediction_head" not in k}
    state_dict = {k:state_dict[k] for k in state_dict
                  if "projection_head" not in k}
    state_dict = {k:state_dict[k] for k in state_dict
                  if "ema" not in k}
    inc = ssl.load_state_dict(state_dict)
    ssl.eval()
    
    ssl.forward = ssl.forward_representation
    example = torch.rand(1,*args.input_shape).to(args.dev)
    traced_ssl = torch.jit.trace(ssl,example_inputs=example)
    
    traced_ssl.save(args.output_model_path)
