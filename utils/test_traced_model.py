import argparse
import torch

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Tests a jit-traced model with an input of a given shape"
    )

    parser.add_argument("--model_path", required=True)
    parser.add_argument("--input_shape", required=True, nargs="+", type=int)
    parser.add_argument("--dev", default="cuda")
    args = parser.parse_args()

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
