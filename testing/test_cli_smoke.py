import os
import sys

sys.path.append(os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))

import pytest

from adell_mri.entrypoints.cli_utils import run_main

TASK_MODES = {
    "classification": [
        "train",
        "test",
        "predict",
        "explain",
        "model_to_torchscript",
    ],
    "classification_mil": ["train", "test", "predict"],
    "classification_ensemble": ["train", "test", "predict"],
    "segmentation": ["train", "test", "predict", "test_from_predictions"],
    "ssl": ["train_2d", "train_3d", "model_to_torchscript", "predict_folder"],
    "detection": ["train", "predict"],
    "generative": ["train_2d", "train_3d", "generate"],
    "segmentation_from_2d_module": ["train"],
}


@pytest.mark.parametrize(
    "task,mode",
    [(t, m) for t, modes in TASK_MODES.items() for m in modes],
)
def test_cli_help_smoke(task, mode, capsys):
    package_name = f"adell_mri.entrypoints.{task}"
    with pytest.raises(SystemExit) as exc_info:
        run_main([mode, "--help"], package_name, TASK_MODES[task])
    assert exc_info.value.code == 0
    captured = capsys.readouterr()
    assert "usage" in captured.out.lower()


def test_top_level_lists_supported_modes(capsys, monkeypatch):
    from adell_mri.__main__ import main as top_main

    monkeypatch.setattr(sys, "argv", ["adell"])
    top_main()
    captured = capsys.readouterr()
    for task in TASK_MODES:
        assert task in captured.out


def test_top_level_rejects_unknown_mode(monkeypatch):
    from adell_mri.__main__ import main as top_main

    monkeypatch.setattr(sys, "argv", ["adell", "not_a_mode"])
    with pytest.raises(NotImplementedError):
        top_main()
