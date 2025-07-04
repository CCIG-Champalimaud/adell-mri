import importlib


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
