"""
Helpers for identifying and freezing pretrained encoder parameters.
"""

import torch

ENCODER_PARAMETER_KEYS = (
    "feature_extraction",
    "encoder",
    "encoding",
    "network.embedding",
    "network.tbs",
    "network.res_net",
    "embedding",
    "tbs",
)


def is_encoder_parameter(name: str) -> bool:
    """
    Checks whether a module or parameter name belongs to a pretrained encoder.

    Args:
        name (str): module or parameter name.

    Returns:
        bool: whether the name belongs to an encoder.
    """
    return any(key in name for key in ENCODER_PARAMETER_KEYS)


def freeze_encoder_parameters(model: torch.nn.Module) -> None:
    """
    Disables gradients for all encoder parameters.

    Args:
        model (torch.nn.Module): model containing an encoder.
    """
    for name, parameter in model.named_parameters():
        if is_encoder_parameter(name):
            parameter.requires_grad = False


def set_encoder_eval(model: torch.nn.Module) -> None:
    """
    Keeps encoder modules in evaluation mode.

    Args:
        model (torch.nn.Module): model containing an encoder.
    """
    for name, module in model.named_modules():
        if name and is_encoder_parameter(name):
            module.eval()


def encoder_parameter_ids(model: torch.nn.Module) -> set[int]:
    """
    Returns the ids of all encoder parameters.

    Args:
        model (torch.nn.Module): model containing an encoder.

    Returns:
        set[int]: parameter ids.
    """
    return {
        id(parameter)
        for name, parameter in model.named_parameters()
        if is_encoder_parameter(name)
    }
