import yaml

from ..modules.activations import activation_factory
from ..modules.losses import CompoundLoss
from ..utils import loss_factory
from .layers import get_adn_fn

unet_args = [
    "spatial_dimensions",
    "encoding_operations",
    "conv_type",
    "link_type",
    "upscale_type",
    "interpolation",
    "norm_type",
    "dropout_type",
    "padding",
    "dropout_param",
    "activation_fn",
    "n_channels",
    "n_classes",
    "depth",
    "kernel_sizes",
    "strides",
    "bottleneck_classification",
    "skip_conditioning",
]


def parse_config_unet(config_file, n_keys, n_classes):
    with open(config_file, "r") as o:
        network_config = yaml.safe_load(o)

    if "activation_fn" in network_config:
        network_config["activation_fn"] = activation_factory[
            network_config["activation_fn"]
        ]

    k = "binary" if n_classes == 2 else "categorical"
    loss_fns_and_kwargs = []
    loss_keys = []
    for key in network_config["loss_fn"]:
        loss_fn = loss_factory[k][key]
        loss_params = network_config["loss_fn"][key]
        loss_fns_and_kwargs.append((loss_fn, loss_params))
        loss_keys.append(key)
    network_config["loss_fn"] = CompoundLoss(
        loss_fns_and_kwargs, network_config.get("loss_weights")
    )

    if "spatial_dimensions" not in network_config:
        network_config["spatial_dimensions"] = 3

    if "batch_size" not in network_config:
        network_config["batch_size"] = 1

    network_config["n_channels"] = n_keys * network_config["n_channels"]
    return network_config, loss_keys


def parse_config_cat(config_file):
    with open(config_file, "r") as o:
        network_config = yaml.safe_load(o)
    return network_config


def parse_config_ensemble(config_file: str, n_classes: int):
    with open(config_file, "r") as o:
        network_config = yaml.safe_load(o)

    if "head_adn_fn" in network_config:
        network_config["head_adn_fn"] = get_adn_fn(
            spatial_dim=1, **network_config["head_adn_fn"]
        )
    return network_config


def parse_config_ssl(config_file: str, dropout_param: float, n_keys: int, is_vit=False):
    with open(config_file, "r") as o:
        network_config = yaml.safe_load(o)

    if "batch_size" not in network_config:
        network_config["batch_size"] = 1

    if is_vit is False:
        sd = network_config["backbone_args"]["spatial_dim"]
    else:
        sd = len(network_config["backbone_args"]["patch_size"])

    if is_vit is False:
        network_config["backbone_args"]["adn_fn"] = get_adn_fn(
            sd,
            network_config["norm_fn"],
            network_config["act_fn"],
            dropout_param=dropout_param,
        )

        network_config["projection_head_args"]["adn_fn"] = get_adn_fn(
            1,
            network_config["norm_fn"],
            network_config["act_fn"],
            dropout_param=dropout_param,
        )

    if "prediction_head_args" in network_config:
        network_config["prediction_head_args"]["adn_fn"] = get_adn_fn(
            1,
            network_config["norm_fn"],
            network_config["act_fn"],
            dropout_param=dropout_param,
        )
    if "projection_head_args" in network_config:
        network_config["projection_head_args"]["adn_fn"] = get_adn_fn(
            1,
            network_config["norm_fn"],
            network_config["act_fn"],
            dropout_param=dropout_param,
        )
    network_config_correct = {
        k: network_config[k] for k in network_config if k not in ["norm_fn", "act_fn"]
    }
    if is_vit is False:
        n_c = network_config["backbone_args"]["in_channels"]
        network_config["backbone_args"]["in_channels"] = n_keys * n_c
    else:
        network_config["backbone_args"]["n_channels"] = n_keys

    return network_config, network_config_correct


def parse_config_2d_classifier_3d(
    config_file: str, dropout_param: float, mil_method: str = "standard"
):
    with open(config_file, "r") as o:
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
            1, **network_config["classification_adn_fn"]
        )

    network_config["adn_fn"] = get_adn_fn(
        1, norm_fn=norm_fn, act_fn=act_fn, dropout_param=dropout_param
    )

    network_config_correct = {
        k: network_config[k] for k in network_config if k not in ["norm_fn", "act_fn"]
    }

    return network_config, network_config_correct


def parse_config_gan(
    config_file: str,
    target_keys: list[str],
    input_keys: list[str] = None,
    **kwargs,
):
    with open(config_file, "r") as o:
        network_config = yaml.safe_load(o)

    if "batch_size" not in network_config:
        network_config["batch_size"] = 1

    generator_config = network_config["generator"]
    discriminator_config = network_config["discriminator"]

    del network_config["generator"]
    del network_config["discriminator"]

    if input_keys is None:
        generator_config["in_channels"] = len(target_keys)
        disc_channels = len(target_keys)
    else:
        generator_config["in_channels"] = len(input_keys)
        disc_channels = len(target_keys) + len(input_keys)
    generator_config["out_channels"] = len(target_keys)
    generator_config["spatial_dims"] = 2
    discriminator_config["spatial_dim"] = 2
    discriminator_config["n_channels"] = disc_channels

    for k in kwargs:
        if kwargs[k] is not None:
            network_config[k] = kwargs[k]

    return network_config, generator_config, discriminator_config
