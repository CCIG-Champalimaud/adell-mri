import sys
import importlib
import re

supported_modes = {
    # preprocessing
    "bias_field_correction": ".preprocessing.bias_field_correction",
    "merge_masks": ".preprocessing.merge_masks",
    "resample_image": ".preprocessing.resample_image",
    "resample_volumes_and_masks": ".preprocessing.resample_volumes_and_masks",
    # statistics
    "compare_masks": ".statistics.compare_masks",
    "get_label_size": ".statistics.get_label_size",
    "match_to_mask": ".statistics.match_to_mask",
    # dataset
    "bb_to_anchors": ".dataset.bb_to_anchors",
    "bb_to_distances": ".dataset.bb_to_distances",
    "fill_with_condition": ".dataset.fill_with_condition",
    "generate_dataset_json": ".dataset.generate_dataset_json",
    "generate_dicom_dataset_json": ".dataset.generate_dicom_dataset_json",
    "generate_image_dataset_json": ".dataset.generate_image_dataset_json",
    "generate_json_from_csv": ".dataset.generate_json_from_csv",
    "get_test_set_and_folds": ".dataset.get_test_set_and_folds",
    "get_temporal_test_set_and_folds": ".dataset.get_temporal_test_set_and_folds",
    "inspect_dicom_dataset": ".dataset.inspect_dicom_dataset",
    "merge_json_datasets": ".dataset.merge_json_datasets",
    "remove_constant_masks": ".dataset.remove_constant_masks",
    # other
    "random_image_panel": ".other.random_image_panel",
    "test_traced_model": ".other.test_traced_model",
}


def print_supported_modes(supported_modes: dict[str, str], print_help=True):
    print("\n\tSupported modes:")
    for mode in supported_modes:
        if print_help is True:
            desc = getattr(
                importlib.import_module(
                    supported_modes[mode], package="lib.entrypoints.utils"
                ),
                "desc",
            )
            desc = re.sub("[ ]+", " ", desc)
            print(f"\t\t{mode} - {desc}")
        else:
            print(f"\t\t{mode}")


def test_import_supported_modes(supported_modes: dict[str, str]):
    print("\n\tSupported modes:")
    for mode in supported_modes:
        print(f"\t\t{mode}")
        main = getattr(
            importlib.import_module(
                supported_modes[mode], package="lib.entrypoints.utils"
            ),
            "main",
        )
        desc = getattr(
            importlib.import_module(
                supported_modes[mode], package="lib.entrypoints.utils"
            ),
            "desc",
        )
        desc = re.sub("[ ]+", " ", desc)

        print(f"\t\t\t{desc}")


def main(arguments):
    if len(arguments) == 0:
        print_supported_modes(supported_modes)

    elif arguments[0] == "help":
        print_supported_modes(supported_modes)

    elif arguments[0] == "test":
        test_import_supported_modes(supported_modes)

    elif arguments[0] in supported_modes:
        main = getattr(
            importlib.import_module(
                supported_modes[arguments[0]], package="lib.entrypoints.utils"
            ),
            "main",
        )
        main(arguments[1:])

    else:
        raise NotImplementedError(
            f"\n\tMode {arguments[0]} not supported\n\tSupported modes: {supported_modes}"
        )


if __name__ == "__main__":
    main(sys.argv[1:])
