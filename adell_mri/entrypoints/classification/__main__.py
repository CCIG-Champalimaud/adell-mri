import sys

supported_modes = ["train", "test", "predict", "model_to_torchscript"]


def main(arguments):
    if len(arguments) == 0:
        print(f"\n\tSupported modes: {supported_modes}")
    elif arguments[0] == "help":
        print(f"\n\tSupported modes: {supported_modes}")

    elif arguments[0] == "train":
        from adell_mri.entrypoints.classification.train import main

        main(arguments[1:])
    elif arguments[0] == "test":
        from adell_mri.entrypoints.classification.test import main

        main(arguments[1:])
    elif arguments[0] == "predict":
        from adell_mri.entrypoints.classification.predict import main

        main(arguments[1:])

    elif arguments[0] == "model_to_torchscript":
        from adell_mri.entrypoints.classification.model_to_torchscript import main

        main(arguments[1:])
    else:
        raise NotImplementedError(
            f"\n\tMode {arguments[0]} not supported\n\tSupported modes: {supported_modes}"
        )


if __name__ == "__main__":
    main(sys.argv[1:])
