import os
import random
import re
from multiprocessing import Pool
from typing import Any

import monai
import numpy as np
import torch
from tqdm import tqdm

from adell_mri.utils.python_logging import get_logger
from adell_mri.utils.utils import return_classes

logger = get_logger(__name__)


def meta_tensors_to_tensors(batch: dict) -> dict:
    """
    Converts any MetaTensor instances in a batch to regular PyTorch tensors.

    Args:
        batch (dict): A dictionary containing tensors, where some values may be
            MONAI MetaTensor instances.

    Returns:
        dict: The input batch with all MetaTensor instances converted to regular
            PyTorch tensors.
    """
    for key in batch:
        if isinstance(batch[key], monai.data.MetaTensor):
            batch[key] = batch[key].as_tensor()
    return batch


def log_current_lr(module, key: str = "lr", sync_dist: bool = False) -> None:
    """
    Logs the current learning rate taken from the module's active LR
    scheduler (falls back to ``module.learning_rate`` when unavailable).

    Args:
        module (pl.LightningModule): Lightning module with configured
            schedulers.
        key (str, optional): string under which the learning rate is logged.
            Defaults to "lr".
        sync_dist (bool, optional): whether to sync the logged value across
            processes. Defaults to False.
    """
    sch = module.lr_schedulers().state_dict()
    lr = module.learning_rate
    last_lr = sch["_last_lr"][0] if "_last_lr" in sch else lr
    module.log(key, last_lr, sync_dist=sync_dist)


def calculate_sn_weights(state_dict: dict) -> dict:
    """
    Calculates the spectral-normalized weights in a given state dictionary
    (if any).

    Args:
        state_dict (dict): a state dictionary.

    Returns:
        dict: the state dictionary with SN parameters.
    """
    sd_keys = list(state_dict.keys())
    sn_keys = []
    for key in sd_keys:
        if key.endswith(".original"):
            base_key = key[:-9]
            u_key = base_key + ".0._u"
            v_key = base_key + ".0._v"
            if (u_key in sd_keys) and (v_key in sd_keys):
                sn_keys.append(base_key)

    logger.debug(f"Found {len(sn_keys)} parameters with spectral normalization")
    for key in sn_keys:
        out_name = key.replace(".parametrizations", "")
        orig_key, u_key, v_key = (
            f"{key}.original",
            f"{key}.0._u",
            f"{key}.0._v",
        )
        orig_w = state_dict[orig_key]
        u = state_dict[u_key]
        v = state_dict[v_key]
        w_mat = orig_w.view(orig_w.size(0), -1)
        sigma = torch.dot(u, torch.mv(w_mat, v))
        w_normalized = orig_w / sigma
        state_dict[out_name] = w_normalized
        del state_dict[orig_key]
        del state_dict[u_key]
        del state_dict[v_key]
    return state_dict


def load_checkpoint_to_model(
    model: torch.nn.Module,
    checkpoint: str | dict[str, torch.Tensor],
    exclude_from_state_dict: list[str] | None = None,
    weights_only: bool = False,
    *args,
    **kwargs,
) -> None:
    """
    Loads a checkpoint into a PyTorch model.

    Loads state dictionary from a checkpoint file or dict into the model. Can
    optionally exclude keys from the state dict by regex. Checks that there are
    no extra keys in state dict not in model.

    Args:
        model (torch.nn.Module): PyTorch model to load state dict into.
        checkpoint (str | dict[str, torch.Tensor]): Checkpoint file path or
            dict containing state dict.
        exclude_from_state_dict (list[str], optional): List of regex patterns to
            exclude from state dict. Defaults to None.

    Returns:
        None: the model is updated in place.

    Raises:
      Exception: If state dict contains keys not in model.
    """
    if checkpoint is None:
        return
    elif isinstance(checkpoint, str):
        logger.info("Loading checkpoint from %s", checkpoint)
        sd = torch.load(checkpoint, weights_only=False, *args, **kwargs)
    else:
        sd = checkpoint
    if "state_dict" in sd:
        sd = sd["state_dict"]

    sd = calculate_sn_weights(sd)

    if exclude_from_state_dict is not None:
        for pattern in exclude_from_state_dict:
            n = len(sd)
            sd = {k: sd[k] for k in sd if re.search(pattern, k) is None}
            n_now = len(sd)
            logger.info(
                f"Removed {n} - {n_now} = {n - n_now} keys with {pattern}"
            )
    output = model.load_state_dict(sd, strict=False)

    if len(output.unexpected_keys) > 0:
        raise Exception(
            "State dict contains more keys than it should:"
            + str(output.unexpected_keys)
        )
    logger.debug("Missing keys: %s", output.missing_keys)


def get_class_weights(
    class_weights: list[float | str],
    n_classes: int,
    classes: list[Any],
    positive_labels: list[Any],
    possible_labels: list[Any],
    label_groups: list[list[Any]] = None,
) -> list[float]:
    """
    Computes class weights for imbalanced datasets.

    Supports class weights passed directly, computed adaptively from class
    frequencies, or computed from label groups.

    Args:
      class_weights (list[float | str]): List of weights or "adaptive"
        string.
      n_classes (int): Number of classes.
      classes (list[Any]): List of class labels.
      positive_labels (list[Any]): Labels treated as the positive class.
      possible_labels (list[Any]): Superset of labels.
      label_groups (list[list[Any]]): Groups of labels to compute weights
        together (merges classes together if they belong to the same
        label_group).

    Returns:
      List of class weights, one per class
    """
    if class_weights is not None:
        if class_weights[0] == "adaptive":
            if n_classes == 2:
                pos = len([x for x in classes if x in positive_labels])
                neg = len(classes) - pos
                weight_neg = (1 / neg) * (len(classes) / 2.0)
                weight_pos = (1 / pos) * (len(classes) / 2.0)
                class_weights = weight_pos / weight_neg
            else:
                pos = {k: 0 for k in possible_labels}
                for c in classes:
                    pos[c] += 1
                if label_groups is not None:
                    new_pos = {i: 0 for i in range(len(label_groups))}
                    for i in range(len(label_groups)):
                        label_group = label_groups[i]
                        for label in label_group:
                            new_pos[i] += pos[label]
                    pos = new_pos
                pos = np.array([pos[k] for k in pos])
                class_weights = (1 / pos) * (len(classes) / 2.0)
        else:
            class_weights = [float(x) for x in class_weights]

    return class_weights


def conditional_parameter_freezing(
    network: torch.nn.Module,
    freeze_regex: list[str] = None,
    do_not_freeze_regex: list[str] = None,
    state_dict: dict[str, torch.Tensor] = None,
):
    """
    Freezes (or not) parameters according to a list of regex and loads an
    optional state dict if frozen keys match dictionary.

    Args:
        network (torch.nn.Module): torch module with a named_parameters
            attribute.
        freeze_regex (list[str], optional): regex for parameter names that
            should be frozen. Defaults to None.
        do_not_freeze_regex (list[str], optional): regex for parameter names
            that should not be frozen (overrides freeze_regex). Defaults to
            None.
        state_dict (dict[str,torch.Tensor], optional): state dict that replaces
            frozen values. Defaults to None.
    """
    keys_to_load = []
    freeze_regex_list = []
    do_not_freeze_regex_list = []

    if freeze_regex is not None:
        freeze_regex_list = [re.compile(fr) for fr in freeze_regex]
    if do_not_freeze_regex is not None:
        do_not_freeze_regex_list = [
            re.compile(dnfr) for dnfr in do_not_freeze_regex
        ]

    for key, param in network.named_parameters():
        freeze = False
        if any([fr.search(key) is not None for fr in freeze_regex_list]):
            freeze = True
        if any(
            [dnfr.search(key) is not None for dnfr in do_not_freeze_regex_list]
        ):
            freeze = False
        if freeze is True:
            param.requires_grad = False
            if state_dict is not None:
                if key in state_dict:
                    keys_to_load.append(key)
    if state_dict is not None:
        with torch.no_grad():
            network.load_state_dict({k: state_dict[k] for k in keys_to_load})


def set_classification_layer_bias(
    pos: float,
    neg: float,
    network: torch.nn.Module,
    class_substr: str = "classification",
):
    """
    Sets the classification layer bias according to class prevalence in the
    binary classification setting.

    Args:
        pos (float): number of positive cases.
        neg (float): number of negative cases.
        network (torch.nn.Module): network.
        class_substr (str, optional): class substring corresponding to bias.
            Defaults to "classification".
    """
    value = torch.as_tensor(np.log(pos / neg))
    for k, v in network.named_parameters():
        if class_substr in k:
            if list(v.shape) == [1]:
                with torch.no_grad():
                    v[0] = value


def get_segmentation_sample_weights(
    data_list: list[dict],
    label_keys: list[str],
    n_workers: int = 1,
    base: str = "Calculating positive pixel counts",
) -> tuple[list[int], float, float]:
    """
    Calculates sample weights for the segmentation masks in a data list. The
    data list is composed of a list of dictionaries, each containing label_keys
    which correspond to paths to SimpleITK-readable segmentation masks.

    Args:
        data_list (list[dict]): list of data elements.
        label_keys (list[str]): keys corresponding to segmentation masks.
        n_workers (int, optional): number of parallel workers. Defaults to 1.
        base (str, optional): base for the tqdm progress bar. Defaults to
            "Calculating positive pixel counts".

    Returns:
        list[int]: list with the same length as data_list where each value is
            set to 1 if the corresponding segmentation mask has at least one
            positive element and 0 otherwise.
        float: number of elements divided by the number of elements with at
            least a positive pixel.
        float: a number of pixels divided by the number of positive pixels.
    """
    cl = []
    pos_pixel_sum = 0
    total_pixel_sum = 0
    all_masks = [[x[mask_key] for mask_key in label_keys] for x in data_list]
    with Pool(n_workers) as pool:
        mapped_fn = pool.imap(return_classes, all_masks)
        with tqdm(mapped_fn, total=len(all_masks)) as t:
            t.set_description(base)
            n, n_nonzero = 0, 0
            for x_classes in t:
                n += 1
                all_classes = {}
                all_classes = {**all_classes, **x_classes}
                total = []
                for u, c in all_classes.items():
                    if u not in total:
                        total.append(u)
                    if u != 0:
                        n_nonzero += 1
                        pos_pixel_sum += c
                    total_pixel_sum += c
                if len(total) > 1:
                    cl.append(1)
                else:
                    cl.append(0)
                t.set_description(base + f" ({n_nonzero}/{n})")
    adaptive_weights = len(cl) / np.sum(cl)
    adaptive_pixel_weights = total_pixel_sum / pos_pixel_sum

    return cl, adaptive_weights, adaptive_pixel_weights


def get_global_rank() -> int:
    """
    Returns the rank of the current process.

    The rank is read from the environment variables set by the process launcher
    (``RANK``, ``LOCAL_RANK``, ``SLURM_PROCID``, ``SLURM_LOCALID``), falling
    back to ``torch.distributed`` when it is already initialized. Returns 0 for
    single-process runs.

    Returns:
        int: the global rank of the current process.
    """
    for var in ("RANK", "LOCAL_RANK", "SLURM_PROCID", "SLURM_LOCALID"):
        value = os.environ.get(var)
        if value is not None:
            return int(value)
    if torch.distributed.is_available() and torch.distributed.is_initialized():
        return torch.distributed.get_rank()
    return 0


def get_generator_and_rng(
    seed: int,
) -> tuple[torch.Generator, np.random.Generator]:
    """
    Returns a torch generator and a numpy RNG.

    The torch generator is seeded with ``seed + global rank`` so that, in
    multi-GPU training, each process draws a different sequence of samples
    (data order, weighted sampling, shuffling) instead of every GPU training on
    the exact same batches. The numpy RNG and the global torch/random/numpy
    seeds are kept identical across processes so that dataset construction and
    fold assignments stay consistent between ranks.

    Args:
        seed (int): seed to use.

    Returns:
        torch.Generator: torch generator.
        np.random.Generator: numpy random number generator.
    """

    torch.manual_seed(seed)
    random.seed(seed)
    np.random.seed(seed)
    g = torch.Generator()
    g.manual_seed(seed + get_global_rank())
    rng = np.random.default_rng(seed)

    return g, rng
