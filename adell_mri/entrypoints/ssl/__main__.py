import sys

supported_modes = ["train_2d", "train_3d", "model_to_torchscript"]


def main(arguments):
    if len(arguments) == 0:
        print(f"\n\tSupported modes: {supported_modes}")
    elif arguments[0] == "help":
        print(f"\n\tSupported modes: {supported_modes}")

    elif arguments[0] == "train_2d":
        from .train_2d import main

        main(arguments[1:])
    elif arguments[0] == "train_3d":
        from .train_3d import main

        main(arguments[1:])
    elif arguments[0] == "model_to_torchscript":
        from .model_to_torchscript import main

        main(arguments[1:])
    else:
        raise NotImplementedError(
            f"\n\tMode {arguments[0]} not supported\n\tSupported modes: {supported_modes}"
        )


if __name__ == "__main__":
    main(sys.argv[1:])
