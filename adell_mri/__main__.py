import sys

supported_modes = [
    "classification",
    "classification_deconfounder",
    "classification_mil",
    "classification_ensemble",
    "generative",
    "generative_gan",
    "segmentation",
    "segmentation_from_2d_module",
    "ssl",
    "detection",
    "utils",
]


def main():
    arguments = sys.argv[1:]

    if len(arguments) == 0:
        print(f"\n\tSupported modes: {supported_modes}")
    elif arguments[0] == "help":
        print(f"\n\tSupported modes: {supported_modes}")

    # classification modes
    elif arguments[0] == "classification":
        from .entrypoints.classification.__main__ import main

        main(arguments[1:])
    elif arguments[0] == "classification_deconfounder":
        from .entrypoints.classification_deconfounder.__main__ import main

        main(arguments[1:])
    elif arguments[0] == "classification_mil":
        from .entrypoints.classification_mil.__main__ import main

        main(arguments[1:])
    elif arguments[0] == "classification_ensemble":
        from .entrypoints.classification_ensemble.__main__ import main

        main(arguments[1:])

    # generation modes
    elif arguments[0] == "generative":
        from .entrypoints.generative.__main__ import main

        main(arguments[1:])

    # generation modes
    elif arguments[0] == "generative_gan":
        from .entrypoints.generative_gan.__main__ import main

        main(arguments[1:])

    # segmentation modes
    elif arguments[0] == "segmentation":
        from .entrypoints.segmentation.__main__ import main

        main(arguments[1:])
    elif arguments[0] == "segmentation_from_2d_module":
        from .entrypoints.segmentation_from_2d_module.__main__ import main

        main(arguments[1:])

    # ssl modes
    elif arguments[0] == "ssl":
        from .entrypoints.ssl.__main__ import main

        main(arguments[1:])

    # detection modes
    elif arguments[0] == "detection":
        from .entrypoints.detection.__main__ import main

        main(arguments[1:])

    # utils modes
    elif arguments[0] == "utils":
        from .entrypoints.utils.__main__ import main

        main(arguments[1:])

    else:
        raise NotImplementedError(
            f"\n\tMode {arguments[0]} not supported\n\tSupported modes: {supported_modes}"
        )


if __name__ == "__main__":
    train_loader_call = None
    main()
