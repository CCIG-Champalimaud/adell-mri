"""
Shared utilities for classification prediction entrypoints.

Consolidates the common prediction logic duplicated across
``classification``, ``classification_deconfounder``, ``classification_ensemble``
and ``classification_mil`` entrypoints.
"""

from dataclasses import dataclass, field
from typing import Any

import torch

from adell_mri.modules.classification.losses import OrdinalSigmoidalLoss
from adell_mri.utils.utils import make_json_serializable


def predict_sample(
    network,
    element,
    dev,
    post_proc_fn,
    extra_args,
    *,
    include_tabular=False,
    process_features=True,
):
    """
    Runs prediction on a single sample and post-processes the output.

    Args:
        network: Any object exposing a ``predict_step`` method (e.g.
            classification PL module or ensemble wrapper).
        element (dict): A single dataset element, typically produced by a
            MONAI transform chain. Must contain an ``"image"`` key.
        dev (str or torch.device): Target device for the forward pass.
        post_proc_fn (callable): Post-processing function applied to the
            ``"prediction"`` key of the output (e.g. ``Softmax``, ``Sigmoid``
            or ``Identity``).
        extra_args (dict): Additional keyword arguments forwarded to
            ``predict_step`` (e.g. ``return_features``,
            ``return_only_pre_bias``, ``return_attention``).
        include_tabular (bool, optional): If ``True`` and the element contains
            a ``"tabular"`` key, include it in the batch dict. Defaults to
            ``False``.
        process_features (bool, optional): If ``True`` and the output contains
            a ``"features"`` key, flatten and max-pool the features (global max
            pooling across spatial dimensions). Defaults to ``True``.

    Returns:
        dict: JSON-serializable prediction output. Keys typically include
        ``"prediction"`` and optionally ``"features"``, ``"attention"``, etc.
    """
    batch = {"image": element["image"].unsqueeze(0).to(dev)}
    if include_tabular and "tabular" in element:
        batch["tabular"] = element["tabular"].unsqueeze(0).to(dev)
    output = network.predict_step(batch, 0, **extra_args)
    if process_features and "features" in output:
        output["features"] = (
            output["features"].flatten(start_dim=2).max(-1).values
        )
    if "prediction" in output:
        output["prediction"] = post_proc_fn(output["prediction"])
    return make_json_serializable(output)


@dataclass
class ClassificationPredictionAccumulator:
    """
    Collects per-identifier prediction outputs into a structured dict.

    Replaces the ad-hoc :code:`output_dict` merge pattern (checking
    ``if key not in output_dict``) with a typed container that enforces
    the output schema.

    Args:
        iteration (int): The current prediction-ids iteration index.
        prediction_ids (list[str]): Ordered list of prediction identifiers
            for this iteration.
        checkpoint (str or None): Path to the loaded checkpoint, or ``None``
            when running in test mode (randomly initialised weights).
    """

    iteration: int
    prediction_ids: list[str]
    checkpoint: str | None
    _data: dict[str, dict[str, Any]] = field(default_factory=dict)

    def add(self, identifier: str, output: dict):
        """
        Register a prediction output for a single identifier.

        Args:
            identifier (str): The prediction identifier (e.g. patient ID).
            output (dict): The JSON-serializable prediction output from
                :func:`predict_sample`. Each top-level key becomes a
                namespace in the accumulator (e.g. ``"prediction"``,
                ``"features"``).
        """
        for key, value in output.items():
            if key not in self._data:
                self._data[key] = {}
            self._data[key][identifier] = value

    def as_dict(self) -> dict:
        """
        Return the accumulator contents as a flat dictionary.

        The returned dict has the same structure as the original
        :code:`output_dict`:

        - ``"iteration"``
        - ``"prediction_ids"``
        - ``"checkpoint"``
        - one key per prediction namespace, mapping identifiers to values.

        Returns:
            dict: Accumulated predictions in the legacy output format.
        """
        return {
            "iteration": self.iteration,
            "prediction_ids": self.prediction_ids,
            "checkpoint": self.checkpoint,
            **self._data,
        }


def resolve_checkpoint_list(
    checkpoints, one_to_one, ensemble, iteration, *, caller_logger
):
    """
    Determine the list of checkpoints to process for an iteration.

    Args:
        checkpoints (list[str] or None): User-supplied checkpoint paths.
        one_to_one (bool): Whether to pair each checkpoint with exactly one
            prediction-ids group (one-to-one mode).
        ensemble (str or None): Ensemble method string if ensemble prediction
            is requested, else ``None``.
        iteration (int): Current prediction-ids iteration index.
        caller_logger (logging.Logger): Logger from the calling module, used
            for the test-mode warning so that it is attributed to the correct
            entrypoint.

    Returns:
        list[str or None]: List of checkpoint paths to iterate over. Returns
        ``[None]`` when no checkpoints are supplied (test mode).
    """
    if checkpoints is None:
        caller_logger.warning(
            "No checkpoint specified through the CLI; test mode "
            "triggered (no checkpoint is loaded) and predictions will "
            "be produced with randomly initialised weights."
        )
        return [None]
    if one_to_one and ensemble is None:
        return [checkpoints[iteration]]
    return checkpoints


def configure_loss_fn(config, net_type, n_classes):
    """
    Assign a loss function to a network or ensemble config dict.

    The loss is stored in-place under ``config["loss_fn"]``.

    Args:
        config (dict): Network or ensemble configuration. Mutated in-place.
        net_type (str or None): Network type string (e.g. ``"cat"``,
            ``"ord"``, ``"unet"``). If ``None``, ordinal loss is never
            selected.
        n_classes (int): Number of output classes.
    """
    if n_classes == 2:
        config["loss_fn"] = torch.nn.BCEWithLogitsLoss()
    elif net_type == "ord":
        config["loss_fn"] = OrdinalSigmoidalLoss(n_classes=n_classes)
    else:
        config["loss_fn"] = torch.nn.CrossEntropyLoss()


def resolve_postprocessing(
    prediction_type, net_type, n_classes, *, caller_logger
):
    """
    Select the post-processing function and extra predict-step arguments.

    Args:
        prediction_type (str): Desired prediction output type. One of
            ``"probability"``, ``"logit"``, ``"pre_bias"``, ``"features"``,
            or ``"attention"``.
        net_type (str or None): Network type string. Only used to validate
            the ``"pre_bias"`` prediction type (which requires ``"ord"``).
        n_classes (int): Number of output classes (determines Softmax vs
            Sigmoid for ``"probability"``).
        caller_logger (logging.Logger): Logger from the calling module, used
            for the ``"pre_bias"`` incompatibility warning.

    Returns:
        tuple[callable, dict]: A ``(post_proc_fn, extra_args)`` pair.

        - *post_proc_fn*: ``Softmax(-1)`` when ``n_classes > 2`` and
          ``prediction_type`` is ``"probability"``; ``Sigmoid()`` for
          binary probabilities; ``Identity()`` otherwise.
        - *extra_args*: Keyword arguments forwarded to
          :meth:`~predict_step`. Empty for ``"probability"`` and
          ``"logit"``; ``{"return_only_pre_bias": True}`` for ``"pre_bias"``
          (only when ``net_type == "ord"``); ``{"return_features": True}``
          for ``"features"``; ``{"return_attention": True}`` for
          ``"attention"``.
    """
    extra_args = {}
    if prediction_type == "probability":
        if n_classes > 2:
            post_proc_fn = torch.nn.Softmax(-1)
        else:
            post_proc_fn = torch.nn.Sigmoid()
    else:
        post_proc_fn = torch.nn.Identity()
        if prediction_type == "pre_bias":
            if net_type == "ord":
                extra_args = {"return_only_pre_bias": True}
            else:
                caller_logger.warning(
                    "Net type must be ord for pre_bias, using probability instead"
                )
        elif prediction_type == "features":
            extra_args = {"return_features": True}
        elif prediction_type == "attention":
            extra_args = {"return_attention": True}
    return post_proc_fn, extra_args
