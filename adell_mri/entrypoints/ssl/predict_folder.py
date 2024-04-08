import json
import numpy as np
import torch
import torch.nn.functional as F
from pathlib import Path
from tqdm import tqdm
from pydicom import dcmread
from ...entrypoints.assemble_args import Parser

torch.set_num_threads(8)


def crop_or_pad(img: np.ndarray, target_size: tuple[int, int]) -> np.ndarray:
    """
    Crops or pads an image to a target size.

    Args:
      img: Input image as a numpy array.
      target_size: Tuple of target height and width.

    Returns:
      Image cropped or padded to target size.
    """

    h, w = img.shape[2:]

    new_h = target_size[0]
    new_w = target_size[1]

    if h > new_h:
        # Crop height
        start = (h - new_h) // 2
        img = img[:, :, start : start + new_h, :]
    elif h < new_h:
        # Pad height
        pad_h = new_h - h
        pad_top = pad_h // 2
        pad_bottom = new_h - pad_top
        img = F.pad(img, (pad_top, pad_bottom, 0, 0), mode="constant")

    if w > new_w:
        # Crop width
        start = (w - new_w) // 2
        img = img[:, :, :, start : start + new_w]
    elif w < new_w:
        # Pad width
        pad_w = new_w - w
        pad_left = pad_w // 2
        pad_right = new_w - pad_left
        img = F.pad(img, (0, 0, pad_left, pad_right), mode="constant")

    return img


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
    parser.add_argument(
        "--crop_size",
        nargs="+",
        type=int,
        help="Target size for DICOM file cropping",
    )
    parser.add_argument(
        "--reduce",
        type=str,
        default="max",
        help="Reduction mode",
        choices=["max", "mean"],
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
        except Exception:
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

        if args.crop_size is not None:
            t = crop_or_pad(t, args.crop_size)
        f = model(t).flatten(start_dim=2)
        if args.reduce == "max":
            f = f.max(-1).values
        elif args.reduce == "mean":
            f = f.mean(-1)
        f = f.squeeze().detach().cpu().numpy()
        output.append(
            {
                "file_name": str(dcm_path),
                "features": [round(x, 4) for x in f.tolist()],
            }
        )

    Path(args.output_path).parent.mkdir(parents=True, exist_ok=True)
    with open(args.output_path, "w") as o:
        json.dump(output, o)
