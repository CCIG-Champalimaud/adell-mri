import argparse

import torch

from adell_mri.utils.python_logging import get_logger

desc = "Tests a jit-traced model with an input of a given shape"


def main(arguments):
    logger = get_logger(__name__)
    parser = argparse.ArgumentParser(description=desc)

    parser.add_argument(
        "--model_path",
        required=True,
        help="Path to torch jit-traced model (torchscript format)",
    )
    parser.add_argument(
        "--input_shape",
        required=True,
        nargs="+",
        type=int,
        help="Input shape used for testing the model",
    )
    parser.add_argument("--dev", default="cuda", help="Device for test")
    args = parser.parse_args(arguments)

    logger.info("Loading TorchScript model")
    model = torch.jit.load(args.model_path)
    logger.info(
        "Defining example with shape %s on device %s",
        [1] + args.input_shape,
        args.dev,
    )
    example = torch.rand(1, *args.input_shape).to(args.dev)
    logger.info("Running example")
    out = model(example)
    logger.info("Output shape = %s", list(out.shape))
