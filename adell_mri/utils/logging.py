"""
Includes a logging class which can store data in different formats.
"""

import os
import torch
from dataclasses import dataclass
from typing import Any


def make_grid(image_tensor: torch.Tensor) -> torch.Tensor:
    """
    Creates a grid of images from a batch of images.

    Args:
        image_tensor (torch.Tensor): A tensor of shape (batch_size, channels,
            height, width).

    Returns:
        torch.Tensor: A tensor of shape (channels, height, width) containing
            the grid.
    """
    sh = image_tensor.shape
    b, c, h, w = sh[0], sh[1], sh[2], sh[3]
    side_x = int(b**0.5)
    grid = []
    for i in range(b):
        if i % side_x == 0:
            grid.append([])
        grid[-1].append(image_tensor[i])
    out_size = len(grid[0])
    for g in range(len(grid)):
        while len(grid[g]) < out_size:
            grid[g].append(torch.zeros(c, h, w))

    output = torch.cat([torch.cat(x, -2) for x in grid], -1).detach()
    output = (output - output.min()) / (output.max() - output.min())
    output = output.multiply(255).to(torch.uint8)
    return output


@dataclass
class CSVLogger:
    """
    CSV logger class for failure-aware metric logging.

    Args:
        file_path (str): path to log file.
        overwrite (bool, optional): whether to overwrite folds in the log CSV
            file.
    """

    try:
        import pandas as pd
    except Exception:
        raise ImportError(
            "Pandas is required to parse parquet files. ",
            "Please install it with `pip install pandas`.",
        )

    file_path: str
    overwrite: bool = False

    def __post_init__(self):
        if os.path.exists(self.file_path) and (self.overwrite is False):
            self.history = pd.read_csv(self.file_path).to_dict("records")
        else:
            self.history = []

    def log(self, data_dict: dict[str, Any]):
        self.history.append(data_dict)

    def write(self):
        pd.DataFrame(self.history).to_csv(self.file_path, index=False)
