import os
import pandas as pd
from dataclasses import dataclass
from typing import Any


@dataclass
class CSVLogger:
    """
    CSV logger class for failure-aware metric logging.

    Args:
        file_path (str): path to log file.
        overwrite (bool, optional): whether to overwrite folds in the log CSV
            file.
    """

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
