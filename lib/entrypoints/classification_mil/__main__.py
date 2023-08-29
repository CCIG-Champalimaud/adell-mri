import sys

supported_modes = [
    "train",
    "test",
    "predict"
]

def main(arguments):

    if len(arguments) == 0:
        print(f"\n\tSupported modes: {supported_modes}")
    elif arguments[0] == "help":
        print(f"\n\tSupported modes: {supported_modes}")

    elif arguments[0] == "train":
        from .train import main
        main(arguments[1:])
    elif arguments[0] == "test":
        from .test import main
        main(arguments[1:])
    elif arguments[0] == "predict":
        from .predict import main
        main(arguments[1:])
    else:
        raise NotImplementedError(
            f"\n\tMode {arguments[0]} not supported\n\tSupported modes: {supported_modes}"
        )
if __name__ == "__main__":
    main(sys.argv[1:])