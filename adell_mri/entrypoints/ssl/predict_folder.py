import json
import numpy as np
import torch
import torch.nn.functional as F
from pathlib import Path
from tqdm import tqdm
from pydicom import dcmread
from ...entrypoints.assemble_args import Parser

torch.set_num_threads(8)


def main(arguments):
    parser = Parser()

    parser.add_argument(
        "--module",
        required=True,
        help="Path to feature extraction module",
    )
    parser.add_argument(
        "--dicom_dir",
        required=True,
        help="Path to DICOM directory. Will be recursively searched for DICOM \
            files.",
    )
    parser.add_argument(
        "--output_path",
        required=True,
        help="Output path for JSON containing features",
    )
    parser.add_argument(
        "--dev",
        default="cpu",
        help="Device for inference",
    )
    parser.add_argument(
        "--resize_size",
        nargs="+",
        type=int,
        help="Target size for DICOM file resizing",
    )

    args = parser.parse_args(arguments)

    # Load the module
    model = torch.jit.load(args.module).to(args.dev)
    model = model.eval()

    output = []
    for dcm_path in tqdm(Path(args.dicom_dir).rglob("*dcm")):
        try:
            t = torch.as_tensor(
                dcmread(str(dcm_path)).pixel_array.astype(np.float32)
            ).to(args.dev)
        except:
            continue
        if len(t.shape) > 2:
            continue
        t = t[None, None] / t.max()
        if args.resize_size is not None:
            t = F.interpolate(
                t,
                size=args.resize_size,
                mode="bilinear",
                align_corners=False,
            )
        f = (
            model(t)
            .flatten(start_dim=2)
            .max(-1)
            .values.squeeze()
            .detach()
            .cpu()
            .numpy()
        )
        output.append(
            {
                "file_name": str(dcm_path),
                "features": [round(x, 4) for x in f.tolist()],
            }
        )

    Path(args.output_path).parent.mkdir(parents=True, exist_ok=True)
    with open(args.output_path, "w") as o:
        json.dump(output, o)
