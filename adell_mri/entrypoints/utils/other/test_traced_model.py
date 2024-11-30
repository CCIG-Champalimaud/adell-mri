import argparse

import torch

desc = "Tests a jit-traced model with an input of a given shape"


def main(arguments):
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

    print("Loading TorchScript model")
    model = torch.jit.load(args.model_path)
    print(
        "Defining example with shape {} on device {}".format(
            [1] + args.input_shape, args.dev
        )
    )
    example = torch.rand(1, *args.input_shape).to(args.dev)
    print("Running example")
    out = model(example)
    print("Output shape = {}".format(list(out.shape)))
