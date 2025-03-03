import numpy as np
from typing import Any


def get_ensemble_prediction(
    output: list[dict[str, Any]], ensemble_type: str
) -> dict[str, Any]:
    """
    Calculates the ensemble prediction from a list of predictions.

    Args:
        output (list[dict[str, Any]]): list of dictionaries, each of which
            should have an entry with "predictions" corresponding to a
            dictionary where keys are prediction identifiers and values are
            predictions.
        ensemble_type (str): type of ensemble to use. Should be one of "mean"
            or "majority".

    Raises:
        ValueError: if ensemble_type is not "mean" or "majority".

    Returns:
        dict[str, Any]: dictionary with keys "iteration" (always 0),
            "prediction_ids" (corresponding to the prediction IDs), "checkpoint"
            (always ensemble), "predictions" and "n_predictions"
            corresponding to the ensemble prediction and the number of
            predictions used to calculate the ensemble prediction.
    """
    if ensemble_type not in ["mean", "median"]:
        raise ValueError("Unknown ensemble type")
    output_dict_ensemble = {
        "iteration": 0,
        "prediction_ids": [],
        "checkpoint": "ensemble",
        "predictions": {},
        "n_predictions": {},
    }
    for output_dict in output:
        for k in output_dict["predictions"]:
            value = np.array(output_dict["predictions"][k])
            if k not in output_dict_ensemble["predictions"]:
                output_dict_ensemble["predictions"][k] = []
                output_dict_ensemble["n_predictions"][k] = 0
            output_dict_ensemble["predictions"][k].append(value)
            output_dict_ensemble["n_predictions"][k] += 1
    for k in output_dict_ensemble["predictions"]:
        n = output_dict_ensemble["predictions"][k]
        d = output_dict_ensemble["n_predictions"][k]
        if ensemble_type == "mean":
            output_dict_ensemble["predictions"][k] = float(sum(n) / d)
        elif ensemble_type == "majority":
            u, c = np.unique(n, return_counts=True)
            maj = u[np.argmax(c)]
            output_dict_ensemble["predictions"][k] = maj
    return output_dict_ensemble
