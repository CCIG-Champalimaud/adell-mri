import yaml
from .layers import get_adn_fn
from ..modules.activations import activation_factory
from ..utils import loss_factory

unet_args = [
    "spatial_dimensions","encoding_operations","conv_type","link_type",
    "upscale_type","interpolation","norm_type","dropout_type",
    "padding","dropout_param","activation_fn","n_channels",
    "n_classes","depth","kernel_sizes","strides",
    "bottleneck_classification",
    "skip_conditioning"
]

def parse_config_unet(config_file,n_keys,n_classes):
    with open(config_file,'r') as o:
        network_config = yaml.safe_load(o)

    if "activation_fn" in network_config:
        network_config["activation_fn"] = activation_factory[
            network_config["activation_fn"]]

    if "loss_fn" in network_config:
        loss_key = network_config["loss_fn"]
        k = "binary" if n_classes == 2 else "categorical"
        network_config["loss_fn"] = loss_factory[k][
            network_config["loss_fn"]]

    if "spatial_dimensions" not in network_config:
        network_config["spatial_dimensions"] = 3

    if "batch_size" not in network_config:
        network_config["batch_size"] = 1
    
    network_config["n_channels"] = n_keys * network_config["n_channels"]
    return network_config,loss_key

def parse_config_cat(config_file):
    with open(config_file,'r') as o:
        network_config = yaml.safe_load(o)
    return network_config

def parse_config_ssl(config_file:str,dropout_param:float,n_keys:int,
                     is_ijepa=False):
    with open(config_file,'r') as o:
        network_config = yaml.safe_load(o)

    if "batch_size" not in network_config:
        network_config["batch_size"] = 1

    if is_ijepa is False:
        sd = network_config["backbone_args"]["spatial_dim"]
    else:
        sd = len(network_config["backbone_args"]["patch_size"])
    
    if is_ijepa is False:
        network_config["backbone_args"]["adn_fn"] = get_adn_fn(
            sd,network_config["norm_fn"],network_config["act_fn"],
            dropout_param=dropout_param)

        network_config["projection_head_args"]["adn_fn"] = get_adn_fn(
            1,network_config["norm_fn"],network_config["act_fn"],
            dropout_param=dropout_param)

    if "prediction_head_args" in network_config:
        network_config["prediction_head_args"]["adn_fn"] = get_adn_fn(
            1,network_config["norm_fn"],network_config["act_fn"],
            dropout_param=dropout_param)
    network_config_correct = {
        k:network_config[k] for k in network_config
        if k not in ["norm_fn","act_fn"]
    }
    if is_ijepa is False:
        n_c = network_config["backbone_args"]["in_channels"]
        network_config["backbone_args"]["in_channels"] = n_keys * n_c
    else:
        network_config["backbone_args"]["n_channels"] = n_keys

    return network_config,network_config_correct

def parse_config_2d_classifier_3d(config_file:str,
                                  dropout_param:float,
                                  mil_method:str="standard"):
    with open(config_file,'r') as o:
        network_config = yaml.safe_load(o)

    if "batch_size" not in network_config:
        network_config["batch_size"] = 1

    if "norm_fn" in network_config:
        norm_fn = network_config["norm_fn"]
    else:
        norm_fn = "layer"
    if "act_fn" in network_config:
        act_fn = network_config["act_fn"]
    else:
        act_fn = "gelu"
    
    if "classification_adn_fn" in network_config:
        network_config["classification_adn_fn"] = get_adn_fn(
            1,**network_config["classification_adn_fn"])
    
    network_config["adn_fn"] = get_adn_fn(1,norm_fn=norm_fn,act_fn=act_fn,
                                            dropout_param=dropout_param)

    network_config_correct = {
        k:network_config[k] for k in network_config
        if k not in ["norm_fn","act_fn"]}

    return network_config,network_config_correct

def parse_config_diffusion_unet(config_file,n_keys,n_classes):
    with open(config_file,'r') as o:
        network_config = yaml.safe_load(o)

    if "activation_fn" in network_config:
        network_config["activation_fn"] = activation_factory[
            network_config["activation_fn"]]

    network_config["classifier_classes"] = n_classes
    network_config["classifier_free_guidance"] = n_classes is not None

    if "spatial_dimensions" not in network_config:
        network_config["spatial_dimensions"] = 3

    if "batch_size" not in network_config:
        network_config["batch_size"] = 1
    
    network_config["n_channels"] = n_keys * network_config["n_channels"]
    if "loss_fn" in network_config:
        del network_config["loss_fn"]
    return network_config
