import torch


OPTIMIZER_MATCH = {
    "adam": torch.optim.Adam,
    "adamw": torch.optim.AdamW,
    "adamax": torch.optim.Adamax,
    "sgd": torch.optim.SGD,
    "adagrad": torch.optim.Adagrad,
    "nadam": torch.optim.NAdam,
    "radam": torch.optim.RAdam,
    "rmsprop": torch.optim.RMSprop,
}


def get_optimizer(
    optimizer_str: str, *args, **kwargs
) -> torch.optim.Optimizer:
    """
    Instantiates torch optimizers based on a string.

    Args:
        optimizer_str (str): string corresponding to an optimizer. Currently
            supports those specified in `OPTIMIZER_MATCH`.
        args, kwargs: arguments/keyword arguments for optimizer.

    Returns:
        torch.optim.Optimizer: a torch-ready optimizer.
    """
    if optimizer_str in OPTIMIZER_MATCH:
        return OPTIMIZER_MATCH[optimizer_str](*args, **kwargs)
