from ...modules.config_parsing import parse_config_ssl,parse_config_unet
from ...utils.network_factories import get_ssl_network_no_pl
from copy import deepcopy
import argparse
import numpy as np
import torch

torch.backends.cudnn.benchmark = True

def unpack_shape(X):
    if isinstance(X,torch.Tensor):
        return X.shape
    else:
        return [unpack_shape(x) for x in X]

def force_cudnn_initialization():
    """Convenience function to initialise CuDNN (and avoid the lazy loading
    from PyTorch).
    """
    s = 16
    dev = torch.device('cuda')
    torch.nn.functional.conv2d(torch.zeros(s,s,s,s,device=dev), 
                               torch.zeros(s,s,s,s,device=dev))

def main(arguments):
    parser = argparse.ArgumentParser(
        description="Converts a SelfSL model to torchscript")

    # network + training
    parser.add_argument(
        '--net_type',dest='net_type',
        choices=["resnet","unet_encoder","convnext","vit"],
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
    parser.add_argument(
        '--ssl_method',dest="ssl_method",default="simclr")
    parser.add_argument(
        '--forward_method_name',dest="forward_method_name",
        default="forward")
    parser.add_argument(
        "--ema",dest="ema",action="store_true",
        help="Triggers exponential moving average (redundant, for \
            compatibility)")
    
    # training
    parser.add_argument(
        '--dev',dest='dev',type=str,default="cuda",
        help="Device for model allocation")

    args = parser.parse_args(arguments)
        
    if args.net_type == "unet_encoder":
        network_config,_ = parse_config_unet(
            args.config_file,args.input_shape[0],2)
        network_config_correct = deepcopy(network_config)
        for k in network_config:
            if k in ["loss_fn"]:
                del network_config_correct[k]
    else:
        network_config,network_config_correct = parse_config_ssl(
            args.config_file,0.0,args.input_shape[0],is_ijepa=args.ssl_method=="ijepa")
    
    if args.ssl_method == "ijepa":
        image_size = args.input_shape[1:]
        patch_size = network_config_correct["backbone_args"]["patch_size"]
        feature_map_size = [i//pi for i,pi in zip(image_size,patch_size)]
        network_config_correct["backbone_args"]["image_size"] = image_size
        network_config_correct["feature_map_dimensions"] = feature_map_size

    network_config_correct = {k:network_config_correct[k]
                              for k in network_config_correct
                              if k not in ["prediction_head_args",
                                           "projection_head_args"]}
    network_config_correct["projection_head_args"] = None
    network_config_correct = {k:network_config_correct[k]
                              for k in network_config_correct
                              if k not in ["learning_rate",
                                           "batch_size",
                                           "weight_decay"]}
    ssl = get_ssl_network_no_pl(
        ssl_method=args.ssl_method,
        net_type=args.net_type,
        network_config_correct=network_config_correct)

    train_loader_call = None
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
    state_dict = {k:state_dict[k] for k in state_dict
                  if "patch_masker" not in k}
    state_dict = {k:state_dict[k] for k in state_dict
                  if "predictor" not in k}
    inc = ssl.load_state_dict(state_dict,strict=False)
    print(f"\t{inc}")
    ssl.eval()

    print("Number of parameters:",sum([np.prod(x.shape) for x in ssl.parameters()]))

    print(f"ssl.{args.forward_method_name}")
    ssl.forward = eval(f"ssl.{args.forward_method_name}")
    example = torch.rand(1,*args.input_shape).to(args.dev)
    print(f"For input shape: {example.shape}")
    print(f"Expected output shape: {unpack_shape(ssl(example))}")
    traced_ssl = torch.jit.trace(ssl,example_inputs=example)

    print("Testing traced module...")
    print(f"Traced module output shape: {unpack_shape(traced_ssl(example))}")
    
    traced_ssl.save(args.output_model_path)
