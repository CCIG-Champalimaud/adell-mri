import importlib
import sys


def run_main(
    arguments: list[str], package_name: str, supported_modes: list[str]
):
    if len(arguments) == 0:
        print(f"\n\tSupported modes: {supported_modes}")
    elif arguments[0] == "help":
        print(f"\n\tSupported modes: {supported_modes}")

    elif arguments[0] in supported_modes:
        main = getattr(
            importlib.import_module(
                package_name + "." + arguments[0], package=package_name
            ),
            "main",
        )
        main(arguments[1:])

    else:
        raise NotImplementedError(
            f"\n\tMode {arguments[0]} not supported\n\tSupported modes: {supported_modes}"
        )


def fail(message: str) -> None:
    """
    Logs an error message and terminates the process with a non-zero exit
    code.

    Args:
        message (str): the error message to log.
    """
    from adell_mri.utils.python_logging import get_logger

    get_logger("adell_mri.entrypoints").error("%s", message)
    sys.exit(1)
