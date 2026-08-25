import importlib
import sys

from adell_mri.utils.python_logging import get_logger

logger = get_logger(__name__)

TASK_PACKAGES = {
    "classification": "adell_mri.entrypoints.classification",
    "classification_mil": "adell_mri.entrypoints.classification_mil",
    "classification_ensemble": (
        "adell_mri.entrypoints.classification_ensemble"
    ),
    "generative": "adell_mri.entrypoints.generative",
    "segmentation": "adell_mri.entrypoints.segmentation",
    "segmentation_from_2d_module": (
        "adell_mri.entrypoints.segmentation_from_2d_module"
    ),
    "ssl": "adell_mri.entrypoints.ssl",
    "detection": "adell_mri.entrypoints.detection",
    "utils": "adell_mri.entrypoints.utils",
}

supported_modes = list(TASK_PACKAGES.keys())


def set_threading_env_vars(args: list[str]):
    if len(args) > 1 and "train" in args[1]:
        import os

        logger.info("Detected training mode.")
        os.environ["OMP_NUM_THREADS"] = "1"
        os.environ["MKL_NUM_THREADS"] = "1"
        os.environ["OPENBLAS_NUM_THREADS"] = "1"
        os.environ["VECLIB_MAXIMUM_THREADS"] = "1"
        os.environ["NUMEXPR_NUM_THREADS"] = "1"


def main():
    arguments = sys.argv[1:]

    if len(arguments) == 0 or arguments[0] == "help":
        print(f"\n\tSupported modes: {supported_modes}")
    elif arguments[0] in TASK_PACKAGES:
        if arguments[0] != "utils":
            set_threading_env_vars(arguments)
        module = importlib.import_module(
            TASK_PACKAGES[arguments[0]] + ".__main__"
        )
        module.main(arguments[1:])
    else:
        raise NotImplementedError(
            f"\n\tMode {arguments[0]} not supported\n\tSupported modes: {supported_modes}"
        )


if __name__ == "__main__":
    main()
