import torch

OPTIMIZER_EPS_DEFAULT = 1e-8

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


def get_optimizer(optimizer_str: str, *args, **kwargs) -> torch.optim.Optimizer:
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


def optimizer_eps_from_precision(precision: str) -> float:
    """
    Infers the optimizer epsilon from the training precision.

    When training in 16-bit (float16) precision, gradients and optimizer
    states are stored in low precision, which can lead to underflow with the
    default ``1e-8`` epsilon. In that case a larger epsilon of ``1e-4`` is
    recommended. Bfloat16 (``bf16``) and 32-bit precision keep the default.

    Args:
        precision (str): the training precision string, e.g. ``"32"``,
            ``"16"``, ``"16-mixed"`` or ``"bf16"``.

    Returns:
        float: ``1e-4`` if the precision is 16-bit float but not bfloat16,
            otherwise :data:`OPTIMIZER_EPS_DEFAULT` (``1e-8``).
    """
    if precision is None:
        return OPTIMIZER_EPS_DEFAULT
    p = str(precision).lower()
    if "16" in p and "bf16" not in p:
        return 1e-4
    return OPTIMIZER_EPS_DEFAULT
