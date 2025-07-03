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
        from adell_mri.entrypoints.classification.__main__ import main

        main(arguments[1:])
    elif arguments[0] == "classification_deconfounder":
        from adell_mri.entrypoints.classification_deconfounder.__main__ import main

        main(arguments[1:])
    elif arguments[0] == "classification_mil":
        from adell_mri.entrypoints.classification_mil.__main__ import main

        main(arguments[1:])
    elif arguments[0] == "classification_ensemble":
        from adell_mri.entrypoints.classification_ensemble.__main__ import main

        main(arguments[1:])

    # generation modes
    elif arguments[0] == "generative":
        from adell_mri.entrypoints.generative.__main__ import main

        main(arguments[1:])

    # generation modes
    elif arguments[0] == "generative_gan":
        from adell_mri.entrypoints.generative_gan.__main__ import main

        main(arguments[1:])

    # segmentation modes
    elif arguments[0] == "segmentation":
        from adell_mri.entrypoints.segmentation.__main__ import main

        main(arguments[1:])
    elif arguments[0] == "segmentation_from_2d_module":
        from adell_mri.entrypoints.segmentation_from_2d_module.__main__ import main

        main(arguments[1:])

    # ssl modes
    elif arguments[0] == "ssl":
        from adell_mri.entrypoints.ssl.__main__ import main

        main(arguments[1:])

    # detection modes
    elif arguments[0] == "detection":
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
