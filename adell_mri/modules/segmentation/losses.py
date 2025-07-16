"""
Segmentation losses for binary and multiclass classification cases.
"""

from typing import Any, Union

import torch

eps = 1e-6
FOCAL_DEFAULT = {"alpha": None, "gamma": 1}
TVERSKY_DEFAULT = {"alpha": 1, "beta": 1, "gamma": 1}


def generalised_dice_score(
    pred: torch.Tensor,
    target: torch.Tensor,
    weight: torch.Tensor = None,
    smooth: torch.Tensor = 0.0,
    scale: torch.Tensor = 1.0,
    eps: torch.Tensor = eps,
) -> torch.Tensor:
    """
    Calculates the generalised dice score. Assumes pred and target have
    the same shape.

    Args:
        pred (torch.Tensor): probabilistic tensor (between 0 and 1).
        target (torch.Tensor): class tensor (0 or 1).
        weight (torch.Tensor): weight for the positive classes. Defaults to
            None (no weight).
        smooth (torch.Tensor): smoothing factor. Defaults to 0.0 (no
            smoothing).
        scale (float, optional): factor to scale result before reducing.
            Defaults to 1.0.

    Returns:
        torch.Tensor: generalised dice score.
    """
    if weight is None:
        weight = torch.ones([]).to(pred)
    elif len(weight.shape) == 1:
        weight = weight.unsqueeze(0)
    numerator = torch.sum(
        weight * torch.clip((target * pred) * scale, 0).sum(-1),
        -1,
    )
    denominator = torch.sum(
        weight * torch.clip((target + pred + smooth) * scale, eps).sum(-1),
        -1,
    )
    return torch.divide(
        numerator,
        denominator,
    )


def pt(
    pred: torch.Tensor, target: torch.Tensor, threshold: float = 0.5
) -> torch.Tensor:
    """
    Convenience function to convert probabilities of predicting
    the positive class to probability of predicting the corresponding
    target.

    Args:
        pred (torch.Tensor): prediction probabilities.
        target (torch.Tensor): target class.
        threshold (float, optional): threshold for the positive class in the
        focal loss. Helpful in cases where one is trying to model the
            probability explictly. Defaults to 0.5.

    Returns:
        torch.Tensor: prediction of element i in `pred` predicting class
        i in `target.`
    """
    return torch.where(target > threshold, pred, 1 - pred)


def binary_cross_entropy(
    pred: torch.Tensor,
    target: torch.Tensor,
    weight: float = 1.0,
    scale: float = 1.0,
    label_smoothing: float = 0.0,
    eps: float = eps,
) -> torch.Tensor:
    """
    Standard implementation of binary cross entropy.

    Args:
        pred (torch.Tensor): prediction probabilities.
        target (torch.Tensor): target class.
        weight (float, optional): weight for the positive
            class. Defaults to 1.
        scale (float, optional): factor to scale loss before reducing.
        label_smoothing (float, optional): smoothing factor. Defaults to 0.0.
        eps (float, optional): epsilon factor to avoid floating-point
            imprecisions.

    Returns:
        torch.Tensor: a tensor with size equal to the batch size (first
        dimension of `pred`).
    """
    target = target * (1 - label_smoothing) + label_smoothing / 2
    pred = torch.flatten(pred, start_dim=1)
    target = torch.flatten(target, start_dim=1)
    a = weight * target * torch.log(pred + eps)
    b = (1 - target) * torch.log(1 - pred + eps)
    return -torch.mean((a + b) * scale, dim=1)


def binary_focal_loss(
    pred: torch.Tensor,
    target: torch.Tensor,
    gamma: float,
    alpha: float = 1.0,
    threshold: float = 0.5,
    scale: float = 1.0,
    label_smoothing: float = 0.0,
    eps: float = eps,
) -> torch.Tensor:
    """
    Binary focal loss. Uses `alpha` to weight the positive class and
    `lambda` to suppress examples which are easy to classify (given that
    `lambda`>1). `lambda` is also known as the focusing parameter. In essence,
    `focal_loss = (1-pt(pred,target))**gamma*bce`, where `bce` is the binary
    cross entropy [1].

    [1] https://arxiv.org/abs/1708.02002

    Args:
        pred (torch.Tensor): prediction probabilities.
        target (torch.Tensor): target class.
        alpha (float): positive class weight.
        gamma (float): focusing parameter.
        threshold (float, optional): threshold for the positive class in the
            focal loss. Helpful in cases where one is trying to model the
            probability explictly. Defaults to 0.5.
        scale (float, optional): factor to scale loss before reducing.
        label_smoothing (float, optional): smoothing factor. Defaults to 0.0.
        eps (float, optional): epsilon factor to avoid floating-point
            imprecisions.

    Returns:
        torch.Tensor: a tensor with size equal to the batch size (first
            dimension of `pred`).
    """
    alpha = torch.as_tensor(alpha).type_as(pred)
    gamma = torch.as_tensor(gamma).type_as(pred)
    eps = torch.as_tensor(eps).type_as(pred)

    pred = torch.maximum(pred, eps).flatten(start_dim=2)
    pred_inv = torch.maximum(1 - pred, eps)
    target = (target > threshold).long().flatten(start_dim=2)
    target = target * (1 - label_smoothing) + label_smoothing / 2
    return (
        torch.add(
            alpha * (pred**gamma) * torch.log(pred) * target,
            (pred_inv**gamma) * torch.log(pred_inv) * (1 - target),
        )
        .negative()
        .multiply(scale)
        .mean(-1)
    )


def binary_focal_loss_(
    pred: torch.Tensor,
    target: torch.Tensor,
    gamma: float,
    alpha: float = 1.0,
    scale: float = 1.0,
    eps: float = eps,
) -> torch.Tensor:
    """
    Implementation of binary focal loss. Uses `alpha` to weight
    the positive class and `lambda` to suppress examples which are easy to
    classify (given that `lambda`>1). `lambda` is also known as the focusing
    parameter. In essence, `focal_loss = (1-pt(pred,target))**gamma*bce`, where
    `bce` is the binary cross entropy [1].

    Inspired in [2].

    [1] https://arxiv.org/abs/1708.02002
    [2] https://github.com/ultralytics/yolov5/blob/c23a441c9df7ca9b1f275e8c8719c949269160d1/utils/loss.py#L35

    Args:
        pred (torch.Tensor): prediction probabilities.
        target (torch.Tensor): target class.
        alpha (float): positive class weight.
        gamma (float): focusing parameter.
        scale (float, optional): factor to scale loss before reducing.
        eps (float, optional): epsilon factor to avoid floating-point
            imprecisions.

    Returns:
         torch.Tensor: a tensor with size equal to the batch size (first
        dimension of `pred`).
    """
    alpha = torch.as_tensor(alpha).type_as(pred)
    gamma = torch.as_tensor(gamma).type_as(pred)

    if len(pred.shape) > 2:
        pred = torch.flatten(pred, start_dim=1)

    target = target.reshape(pred.shape)
    loss = -(
        target * torch.log(pred + eps)
        + (1 - target) * torch.log(1 - pred + eps)
    )
    target_bin = (target > 0).float()
    alpha_factor = target_bin * alpha + (1 - target_bin) * (1 - alpha)
    modulating_factor = torch.pow(torch.abs(target - pred) + eps, gamma)
    loss *= alpha_factor * modulating_factor
    return torch.mean(loss * scale, 1)


def weighted_mse(
    pred: torch.Tensor,
    target: torch.Tensor,
    alpha: float,
    threshold: float = 0.5,
) -> torch.Tensor:
    """
    Weighted MSE. Useful for object detection tasks.

    Args:
        pred (torch.Tensor): prediction probabilities.
        target (torch.Tensor): target class.
        alpha (float): positive class weight.
        gamma (float): focusing parameter.
        threshold (float, optional): threshold for the positive class in the
        focal loss. Helpful in cases where one is trying to model the
        probability explictly. Defaults to 0.5.

    Returns:
         torch.Tensor: a tensor with size equal to the batch size (first
        dimension of `pred`).
    """
    alpha = torch.as_tensor(alpha).type_as(pred)

    pred = torch.flatten(pred, start_dim=1)
    target = torch.flatten(target, start_dim=1)
    mse = torch.square(pred - target)
    positive_mse = mse[target >= threshold].mean(-1)
    negative_mse = mse[target < threshold].mean(-1) / alpha

    return positive_mse + negative_mse


def binary_generalized_dice_loss(
    pred: torch.Tensor,
    target: torch.Tensor,
    weight: float = 1.0,
    smooth: float = 1.0,
    scale: float = 1.0,
    eps: float = eps,
) -> torch.Tensor:
    """
    Dice loss adapted to cases of very high class imbalance. In essence
    it adds class weights to the calculation of the Dice loss [1]. If
    `weights=1` it defaults to the regular Dice loss. This implementation
    works for the binary case.

    [1] https://arxiv.org/pdf/1707.03237.pdf

    Args:
        pred (torch.Tensor): prediction probabilities.
        target (torch.Tensor): target class.
        weight (float, optional): class weights. Defaults to 1.
        smooth (float, optional): term preventing the loss from being
            undefined.
        scale (float, optional): factor to scale loss before reducing.
            Defaults to 1.0.
        eps (float, optional): small constant value used to clamp predictions
            and targets. Defaults to 1e-8.

    Returns:
         torch.Tensor: a tensor with size equal to the batch size (first
        dimension of `pred`).
    """
    weight = torch.as_tensor(weight).type_as(pred)
    eps = torch.as_tensor(eps).type_as(pred)

    if pred.shape != target.shape:
        target = classes_to_one_hot(target)
        weight = unsqueeze_to_shape(weight, [1, 1], 1)

    target = torch.flatten(target, start_dim=2)
    pred = torch.flatten(pred, start_dim=2)
    cl_dice = generalised_dice_score(pred, target, weight, smooth, scale, eps)
    return 1 - 2 * cl_dice


def binary_focal_tversky_loss(
    pred: torch.Tensor,
    target: torch.Tensor,
    alpha: float,
    beta: float,
    gamma: float = 1,
) -> torch.Tensor:
    """
    Binary focal Tversky loss. Very similar to the original Tversky loss
    but features a `gamma` term similar to the focal loss to focus the
    learning on harder/rarer examples [1]. The Tversky loss itself is very
    similar to the Dice score loss but weighs false positives and false
    negatives differently (set as the `alpha` and `beta` hyperparameter,
    respectively).

    [1] https://arxiv.org/abs/1810.07842

    Args:
        pred (torch.Tensor): prediction probabilities.
        target (torch.Tensor): target class.
        alpha (float): weight for false positives.
        beta (float): weight for false negatives.
        gamma (float): focusing parameter.

    Returns:
         torch.Tensor: a tensor with size equal to the batch size (first
        dimension of `pred`).
    """
    alpha = torch.as_tensor(alpha).type_as(pred)
    beta = torch.as_tensor(beta).type_as(pred)
    gamma = torch.as_tensor(gamma).type_as(pred)

    p_fore = torch.flatten(pred, start_dim=1)
    p_back = 1 - p_fore
    t_fore = torch.flatten(target, start_dim=1)
    t_back = 1 - t_fore
    tp = torch.sum(p_fore * t_fore, dim=1)
    fn = torch.sum(p_fore * t_back, dim=1)
    fp = torch.sum(p_back * t_fore, dim=1)
    nd = (tp + 1) / (tp + alpha * fn + beta * fp + 1)

    return 1 - (nd) ** gamma


def combo_loss(
    pred: torch.Tensor,
    target: torch.Tensor,
    alpha: float = 0.5,
    weight: float = 1,
    gamma: float = 1.0,
    scale: float = 1.0,
    eps: float = eps,
) -> torch.Tensor:
    """
    Combo loss. Simply put, it is a weighted combination of the Dice loss
    and the weighted focal loss [1]. `alpha` is the weight for each loss
    and weight is the weight for the binary cross entropy.

    [1] https://arxiv.org/abs/1805.02798

    Args:
        pred (torch.Tensor): prediction probabilities.
        target (torch.Tensor): target class.
        alpha (float, optional): weight term for both losses. Binary cross
            entropy is scaled by `alpha` and the Dice score is scaled by
            `1-alpha`. Defaults to 0.5.
        weight (float, optional): weight for the cross entropy. Defaults to 1.
        gamma (float, optional): focusing parameter. Setting this to 0 leads
            this loss to be the weighted sum of generalized Dice loss and
            BCE. Defaults to 1.0.
        scale (float, optional): factor to scale focal loss before reducing.
            Defaults to 1.0.

    Returns:
        torch.Tensor: a tensor with size equal to the batch size (first
            dimension of `pred`).
    """
    alpha = torch.as_tensor(alpha).type_as(pred)
    weight = torch.as_tensor(weight).type_as(pred)

    bdl = binary_generalized_dice_loss(pred, target, weight, eps) * scale
    bce = binary_focal_loss(
        pred=pred,
        target=target,
        alpha=weight,
        gamma=gamma,
        scale=scale,
    )
    return (alpha) * bce + (1 - alpha) * bdl


def hybrid_focal_loss(
    pred: torch.Tensor,
    target: torch.Tensor,
    lam: float = 0.5,
    focal_params: dict = {},
    tversky_params: dict = {},
) -> torch.Tensor:
    """
    Hybrid focal loss. A combination of the focal loss and the focal
    Tversky loss. In essence, a weighted sum of both, where `lam` defines
    how both losses are combined.

    Args:
        pred (torch.Tensor): prediction probabilities.
        target (torch.Tensor): target class.
        lam (float, optional): weight term for both losses. Focal loss is
        scaled by `lam` and the Tversky focal loss is scaled by `1-lam`.
        Defaults to 0.5.
        focal_params (dict, optional): dictionary with parameters for the
        focal loss. Defaults to {}.
        tversky_params (dict, optional): dictionary with parameters for the
            Tversky focal loss. Defaults to {}.

    Returns:
        torch.Tensor: a tensor with size equal to the batch size (first
        dimension of `pred`).
    """
    a = focal_params["alpha"]
    if a is None or isinstance(a, int) is True or isinstance(a, float) is True:
        focal_params["alpha"] = torch.ones([1])
    bfl = binary_focal_loss(pred, target, **focal_params)
    bftl = binary_focal_tversky_loss(pred, target, **tversky_params)
    return lam * bfl + (1 - lam) * bftl


def unified_focal_loss(
    pred: torch.Tensor,
    target: torch.Tensor,
    weight: float,
    gamma: float,
    lam: float = 0.5,
    threshold: float = 0.5,
    scale: float = 1.0,
) -> torch.Tensor:
    """
    Unified focal loss. A combination of the focal loss and the focal
    Tversky loss. In essence, a weighted sum of both, where `lam` defines
    how both losses are combined but with fewer parameters than the hybrid
    focal loss (`gamma` and `weight` parametrize different aspects of each
    loss).

    Args:
        pred (torch.Tensor): prediction probabilities.
        target (torch.Tensor): target class.
        weight (float): equivalent to `alpha` in focal loss
            and `alpha` and `1-beta` in Tversky focal loss
        gamma (float): equivalent to `gamma` in both
        losses
        lam (float, optional): weight term for both losses. Focal loss is
        scaled by `lam` and the Tversky focal loss is scaled by `1-lam`.
        Defaults to 0.5.
        threshold (float, optional): threshold for the positive class in the
            focal loss. Helpful in cases where one is trying to model the
            probability explictly. Defaults to 0.5.
        scale (float, optional): factor to scale focal loss before reducing.

    Returns:
        torch.Tensor: a tensor with size equal to the batch size (first
        dimension of `pred`).
    """
    weight = torch.as_tensor(weight).type_as(pred)
    gamma = torch.as_tensor(gamma).type_as(pred)

    bfl = binary_focal_loss(pred, target, weight, 1 - gamma, threshold, scale)
    bftl = binary_focal_tversky_loss(pred, target, weight, 1 - weight, gamma)
    return lam * bfl + (1 - lam) * bftl


def mc_pt(pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    """
    Convenience function to convert probabilities of each class
    to probability of predicting the corresponding target. Also works with
    one hot-encoded classes.

    Args:
        pred (torch.Tensor): prediction probabilities.
        target (torch.Tensor): target class.

    Returns:
        torch.Tensor: prediction of element i in `pred` predicting class
        i in `target.`
    """
    return torch.where(target > 0.5, pred, 1 - pred).to(pred.device)


def classes_to_one_hot(X: torch.Tensor) -> torch.Tensor:
    """
    Converts classes to one-hot and permutes the dimension so that the
    channels (classes) are in second.

    Args:
        X (torch.Tensor): an indicator tensor.

    Returns:
        torch.Tensor: one-hot encoded tensor.
    """
    n_dim = len(list(X.shape))
    out_dim = [0, n_dim]
    out_dim.extend([i for i in range(1, n_dim)])
    return (
        torch.nn.functional.one_hot(X.long(), num_classes=3)
        .permute(out_dim)
        .to(X.device)
    )


def unsqueeze_to_shape(
    X: torch.Tensor, target_shape: Union[list, tuple], dim: int = 1
) -> torch.Tensor:
    """
    Unsqueezes a vector `X` into a shape that is castable to a given
    target_shape.

    Args:
        X (torch.Tensor): a one-dimensional tensor
        target_shape (Union[list,tuple]): shape to which `X` will be castable.
        dim (int, optional): dimension corresponding to the vector in the
        output. Defaults to 1.

    Returns:
        torch.Tensor: a tensor.
    """
    X = torch.flatten(X)
    output_shape = []
    for i, x in enumerate(target_shape):
        if i == dim:
            output_shape.append(int(X.shape[0]))
        else:
            output_shape.append(1)
    return X.reshape(output_shape)


def cat_cross_entropy(
    pred: torch.Tensor,
    target: torch.Tensor,
    weight: Union[float, torch.Tensor] = 1.0,
    scale: float = 1.0,
    label_smoothing: float = 0.0,
    eps: float = eps,
) -> torch.Tensor:
    """
    Standard implementation of categorical cross entropy.

    Args:
        pred (torch.Tensor): prediction probabilities.
        target (torch.Tensor): target class.
        weight (Union[float,torch.Tensor], optional): class weights.
            Should be the same shape as the number of classes. Defaults to 1.
        scale (float, optional): factor to scale loss before reducing.
        label_smoothing (float, optional): smoothing factor. Defaults to 0.0.
        eps (float, optional): epsilon factor to avoid floating-point
            imprecisions.

    Returns:
        torch.Tensor: a tensor with size equal to the batch size (first
        dimension of `pred`).
    """
    weight = torch.as_tensor(weight).type_as(pred)

    if pred.shape != target.shape:
        target = classes_to_one_hot(target)
    target = target * (1 - label_smoothing) + 1 / target.shape[1]
    if isinstance(weight, torch.Tensor) is True:
        weight = unsqueeze_to_shape(weight, pred.shape, 1)
    out = -target * torch.log(pred + eps)
    out = torch.flatten(out * weight, start_dim=1)
    return torch.mean(out * scale, dim=1)


def mc_focal_loss(
    pred: torch.Tensor,
    target: torch.Tensor,
    alpha: torch.Tensor,
    gamma: Union[float, torch.Tensor],
    scale: float = 1.0,
    label_smoothing: float = 0.0,
    eps: float = eps,
) -> torch.Tensor:
    """
    Categorical focal loss. Uses `alpha` to weight classes and `lambda` to
    suppress examples which are easy to classify (given that `lambda`>1).
    `lambda` is also known as the focusing parameter. In essence,
    `focal_loss = (1-mc_pt(pred,target))**gamma*ce`, where `ce` is the
    categorical cross entropy [1].

    [1] https://arxiv.org/abs/1708.02002

    Args:
        pred (torch.Tensor): prediction probabilities.
        target (torch.Tensor): target class.
        alpha (torch.Tensor): class weights.
        gamma (Union[float,torch.Tensor]): focusing parameter.
        scale (float, optional): factor to scale loss before reducing.
        label_smoothing (float, optional): smoothing factor. Defaults to 0.0.
        eps (float, optional): epsilon factor to avoid floating-point
            imprecisions.

    Returns:
         torch.Tensor: a tensor with size equal to the batch size (first
        dimension of `pred`).
    """
    alpha = torch.as_tensor(alpha).type_as(pred)
    gamma = torch.as_tensor(gamma).type_as(pred)

    alpha = unsqueeze_to_shape(alpha, pred.shape, dim=1)
    if pred.shape != target.shape:
        target = classes_to_one_hot(target)
    p = mc_pt(pred, target)
    target = target * (1 - label_smoothing) + 1 / target.shape[1]
    ce = -target * torch.log(pred + eps)
    out = torch.flatten(alpha * ((1 - p + eps) ** gamma) * ce, start_dim=1)
    return torch.mean(out * scale, dim=1)


def mc_generalized_dice_loss(
    pred: torch.Tensor,
    target: torch.Tensor,
    weight: float = 1.0,
    smooth: float = 1.0,
    scale: float = 1.0,
    eps: float = eps,
) -> torch.Tensor:
    """
    Dice loss adapted to cases of very high class imbalance. In essence
    it adds class weights to the calculation of the Dice loss [1]. If
    `weights=1` it defaults to the regular Dice loss. This implementation
    works for both the binary and categorical cases.

    [1] https://arxiv.org/pdf/1707.03237.pdf

    Args:
        pred (torch.Tensor): prediction probabilities.
        target (torch.Tensor): target class.
        weight (float, optional): class weights. Defaults to 1.
        smooth (float, optional): term preventing the loss from being
            undefined.
        scale (float, optional): factor to scale loss before reducing.
            Defaults to 1.0.
        eps (float, optional): small constant value used to clamp predictions
            and targets. Defaults to 1e-8.

    Returns:
         torch.Tensor: a tensor with size equal to the batch size (first
        dimension of `pred`).
    """
    weight = torch.as_tensor(weight).type_as(pred)
    eps = torch.as_tensor(eps).type_as(pred)
    scaling_term = 1

    if pred.shape != target.shape:
        target = classes_to_one_hot(target)
        weight = unsqueeze_to_shape(weight, [1, 1], 1)

    target = torch.flatten(target, start_dim=2)
    pred = torch.flatten(pred, start_dim=2)
    smooth = smooth * scaling_term
    cl_dice = generalised_dice_score(pred, target, weight, smooth, scale, eps)
    return 1 - 2 * cl_dice


def mc_focal_tversky_loss(
    pred: torch.Tensor,
    target: torch.Tensor,
    alpha: torch.Tensor,
    beta: torch.Tensor,
    gamma: Union[torch.Tensor, float] = 1.0,
) -> torch.Tensor:
    """
    Categorical focal Tversky loss. Very similar to the original Tversky
    loss but features a `gamma` term similar to the focal loss to focus the
    learning on harder/rarer examples [1]. The Tversky loss itself is very
    similar to the Dice score loss but weighs false positives and false
    negatives differently (set as the `alpha` and `beta` hyperparameter,
    respectively).

    [1] https://arxiv.org/abs/1810.07842

    Args:
        pred (torch.Tensor): prediction probabilities.
        target (torch.Tensor): target class.
        alpha (torch.Tensor): weight for false positives.
        beta (torch.Tensor): weight for false negatives.
        gamma (Union[torch.Tensor,float]): focusing parameter.

    Returns:
         torch.Tensor: a tensor with size equal to the batch size (first
        dimension of `pred`).
    """
    alpha = torch.as_tensor(alpha).type_as(pred)
    beta = torch.as_tensor(beta).type_as(pred)
    gamma = torch.as_tensor(gamma).type_as(pred)

    if pred.shape != target.shape:
        target = classes_to_one_hot(target)
    p_fore = torch.flatten(pred, start_dim=2)
    p_back = 1 - p_fore
    t_fore = torch.flatten(target, start_dim=2)
    t_back = 1 - t_fore
    n = torch.sum(p_fore * t_fore, dim=-1) + 1
    d_1 = alpha * torch.sum(p_fore * t_back, dim=-1)
    d_2 = beta * torch.sum(p_back * t_fore, dim=-1)
    d = n + d_1 + d_2 + 1
    return torch.mean(1 - torch.pow(n / d, gamma), dim=-1)


def mc_combo_loss(
    pred: torch.Tensor,
    target: torch.Tensor,
    alpha: float = 0.5,
    weight: Union[float, torch.Tensor] = 1,
    scale: float = 1.0,
) -> torch.Tensor:
    """
    Combo loss. Simply put, it is a weighted combination of the Dice loss
    and the weighted cross entropy [1]. `alpha` is the weight for each loss
    and weights is the weight for the binary cross entropy.

    [1] https://arxiv.org/abs/1805.02798

    Args:
        pred (torch.Tensor): prediction probabilities.
        target (torch.Tensor): target class.
        alpha (float, optional): weight term for both losses. Cross
            entropy is scaled by `alpha` and the Dice score is scaled by
            `1-alpha`. Defaults to 0.5.
        weight (Union[float,torch.Tensor], optional): weight for the cross
            entropy. Defaults to 1.
        scale (float, optional): factor to scale CE loss before reducing.

    Returns:
         torch.Tensor: a tensor with size equal to the batch size (first
        dimension of `pred`).
    """
    alpha = torch.as_tensor(alpha).type_as(pred)
    weight = torch.as_tensor(weight).type_as(pred)

    bdl = mc_generalized_dice_loss(pred, target, weight, scale)
    bce = cat_cross_entropy(pred, target, weight, scale)
    return (alpha) * bce + (1 - alpha) * bdl


def mc_hybrid_focal_loss(
    pred: torch.Tensor,
    target: torch.Tensor,
    lam: float = 1.0,
    focal_params: dict = {},
    tversky_params: dict = {},
) -> torch.Tensor:
    """
    Hybrid focal loss. A combination of the focal loss and the focal
    Tversky loss. In essence, a weighted sum of both, where `lam` defines
    how both losses are combined.

    Args:
        pred (torch.Tensor): prediction probabilities.
        target (torch.Tensor): target class.
        lam (float, optional): weight term for both losses. Focal loss is
            scaled by `lam` and the Tversky focal loss is scaled by `1-lam`.
            Defaults to 0.5.
        focal_params (dict, optional): dictionary with parameters for the
        focal loss. Defaults to {}.
        tversky_params (dict, optional): dictionary with parameters for the
            Tversky focal loss. Defaults to {}.

    Returns:
        torch.Tensor: a tensor with size equal to the batch size (first
        dimension of `pred`).
    """
    a = focal_params["alpha"]
    if a is None or isinstance(a, int) is True or isinstance(a, float) is True:
        focal_params["alpha"] = torch.ones(pred.shape[1])
    fl = mc_focal_loss(pred, target, **focal_params)
    ftl = mc_focal_tversky_loss(pred, target, **tversky_params)
    return lam * fl + (1 - lam) * ftl


def mc_unified_focal_loss(
    pred: torch.Tensor,
    target: torch.Tensor,
    delta: torch.Tensor,
    gamma: Union[torch.Tensor, float],
    lam: float,
    scale: float = 1.0,
) -> torch.Tensor:
    """
    Unified focal loss. A combination of the focal loss and the focal
    Tversky loss. In essence, a weighted sum of both, where `lam` defines
    how both losses are combined but with fewer parameters than the hybrid
    focal loss (`gamma` and `delta` parametrize different aspects of each
    loss).

    Args:
        pred (torch.Tensor): prediction probabilities.
        target (torch.Tensor): target class.
        delta (torch.Tensor): equivalent to `alpha` in focal loss
            and `alpha` and `1-beta` in Tversky focal loss
        gamma (Union[torch.Tensor,float]): equivalent to `gamma` in both
            losses
        lam (float, optional): weight term for both losses. Focal loss is
            scaled by `lam` and the Tversky focal loss is scaled by `1-lam`.
            Defaults to 0.5.
        scale (float, optional): factor to scale focal loss before reducing.

    Returns:
        torch.Tensor: a tensor with size equal to the batch size (first
        dimension of `pred`).
    """
    delta = torch.as_tensor(delta).type_as(pred)
    gamma = torch.as_tensor(gamma).type_as(pred)

    fl = mc_focal_loss(pred, target, delta, 1 - gamma, scale)
    ftl = mc_focal_tversky_loss(pred, target, delta, 1 - delta, gamma)
    return lam * fl + (1 - lam) * ftl


class CompoundLoss(torch.nn.Module):
    """
    Calculates using the same targets different loss function values. Using
    loss_fns_and_kwargs.
    """

    def __init__(
        self,
        loss_fns_and_kwargs: list[tuple[callable, list[dict[str, Any]]]],
        loss_weights: list[float] = None,
    ) -> list[torch.Tensor]:
        """
        Args:
            loss_fns_and_kwargs (list[tuple[callable, list[dict[str, Any]]]]): a
                list of tuples containing a loss function and a set of kwargs for
                it (a dictionary with values).
            loss_weights (list[float], optional): list containing the relative
                weights of each loss function. Defaults to None (equal
                weights).
        """
        super().__init__()
        self.loss_fns_and_kwargs = loss_fns_and_kwargs
        self.loss_weights = loss_weights

        if self.loss_weights is None:
            self.loss_weights = [1.0 for _ in self.loss_fns_and_kwargs]

        if len(self.loss_weights) != len(self.loss_fns_and_kwargs):
            raise Exception(
                "loss_weights and loss_fns_and_kwargs should have same length"
            )

        for i in range(len(self.loss_fns_and_kwargs)):
            loss_fn, kwargs = self.loss_fns_and_kwargs[i]
            if kwargs is None:
                kwargs = {}
            self.loss_fns_and_kwargs[i] = (loss_fn, kwargs)

    def __setitem__(self, key, value):
        for i in range(len(self.loss_fns_and_kwargs)):
            self.loss_fns_and_kwargs[i][1][key] = value

    def replace_item(self, key, value):
        for i in range(len(self.loss_fns_and_kwargs)):
            if key in self.loss_fns_and_kwargs[i][1]:
                self.loss_fns_and_kwargs[i][1][key] = value

    def convert_args(self, fn: callable):
        for i in range(len(self.loss_fns_and_kwargs)):
            self.loss_fns_and_kwargs[i][1] = fn(self.loss_fns_and_kwargs[i][1])

    def forward(
        self, pred: torch.Tensor, target: torch.Tensor
    ) -> list[torch.Tensor]:
        """
        Calculates loss function.

        Args:
            pred (torch.Tensor): prediction probabilities.
            target (torch.Tensor): target class.

        Returns:
            torch.Tensor: list of tensors, each containing a value for the
                loss functions in self.loss_fns_and_kwargs.
        """
        loss_values = []
        for (loss_fn, kwargs), w in zip(
            self.loss_fns_and_kwargs, self.loss_weights
        ):
            if kwargs is None:
                loss_value = loss_fn(pred, target)
            else:
                loss_value = loss_fn(pred, target, **kwargs)
            loss_values.append(loss_value * w)
        return loss_values
