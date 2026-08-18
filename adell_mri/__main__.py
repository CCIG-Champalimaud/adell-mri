import sys

from adell_mri.utils.python_logging import get_logger

logger = get_logger(__name__)

supported_modes = [
    "classification",
    "classification_mil",
    "classification_ensemble",
    "generative",
    "segmentation",
    "segmentation_from_2d_module",
    "ssl",
    "detection",
    "utils",
]


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

    if len(arguments) == 0:
        print(f"\n\tSupported modes: {supported_modes}")
    elif arguments[0] == "help":
        print(f"\n\tSupported modes: {supported_modes}")

    # classification modes
    elif arguments[0] == "classification":
        set_threading_env_vars(arguments)
        from adell_mri.entrypoints.classification.__main__ import main

        main(arguments[1:])
    elif arguments[0] == "classification_mil":
        set_threading_env_vars(arguments)
        from adell_mri.entrypoints.classification_mil.__main__ import main

        main(arguments[1:])
    elif arguments[0] == "classification_ensemble":
        set_threading_env_vars(arguments)
        from adell_mri.entrypoints.classification_ensemble.__main__ import main

        main(arguments[1:])

    # generation modes
    elif arguments[0] == "generative":
        set_threading_env_vars(arguments)
        from adell_mri.entrypoints.generative.__main__ import main

        main(arguments[1:])

    # segmentation modes
    elif arguments[0] == "segmentation":
        set_threading_env_vars(arguments)
        from adell_mri.entrypoints.segmentation.__main__ import main

        main(arguments[1:])
    elif arguments[0] == "segmentation_from_2d_module":
        set_threading_env_vars(arguments)
        from adell_mri.entrypoints.segmentation_from_2d_module.__main__ import (
            main,
        )

        main(arguments[1:])

    # ssl modes
    elif arguments[0] == "ssl":
        set_threading_env_vars(arguments)
        from adell_mri.entrypoints.ssl.__main__ import main

        main(arguments[1:])

    # detection modes
    elif arguments[0] == "detection":
        set_threading_env_vars(arguments)
        from adell_mri.entrypoints.detection.__main__ import main

        main(arguments[1:])

    # utils modes
    elif arguments[0] == "utils":
        from adell_mri.entrypoints.utils.__main__ import main

        main(arguments[1:])

    else:
        raise NotImplementedError(
            f"\n\tMode {arguments[0]} not supported\n\tSupported modes: {supported_modes}"
        )


if __name__ == "__main__":
    train_loader_call = None
    main()
