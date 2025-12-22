import sys

from adell_mri.entrypoints.cli_utils import run_main

package_name = "adell_mri.entrypoints.segmentation"
supported_modes = ["train", "test", "predict", "test_from_predictions"]


def main(arguments):
    run_main(arguments, package_name, supported_modes)


if __name__ == "__main__":
    main(sys.argv[1])
