import sys
from adell_mri.entrypoints.cli_utils import run_main

package_name = "adell_mri.entrypoints.ssl"
supported_modes = [
    "train_2d",
    "train_3d",
    "model_to_torchscript",
    "predict_folder",
]


def main(arguments):
    run_main(arguments, package_name, supported_modes)


if __name__ == "__main__":
    main(sys.argv[1:])
