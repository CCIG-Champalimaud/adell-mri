from copy import deepcopy
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import SimpleITK as sitk
import torch
from multiprocess import Process, Queue

from .sitk_utils import copy_information_nd


@dataclass
class SitkWriter:
    """
    Class that writes files using SimpleITK in the background and parallely.

    Args:
        n_workers (int): number of workers. Defaults to 1.
    """

    n_workers: int = 1

    def __post_init__(self):
        """
        Initialises the queues and processes.
        """
        self.queue = Queue()
        self.workers = [
            Process(target=self.worker) for _ in range(self.n_workers)
        ]
        for worker in self.workers:
            worker.start()

    def worker(self):
        """
        Performs all image writing operations.
        """
        while True:
            element = self.queue.get()
            if element is None:
                break
            path, image, source_image = element
            if isinstance(image, torch.Tensor):
                image = image.cpu().numpy()
            if isinstance(image, np.ndarray):
                image = sitk.GetImageFromArray(image)
            if source_image is not None:
                source_image_original = deepcopy(source_image)
                if isinstance(source_image, str):
                    source_image = sitk.ReadImage(source_image)
                image = copy_information_nd(image, source_image)
                # checks for differences in size
                if isinstance(image, str):
                    print(
                        f"error for image {path} and source_image"
                        "{source_image_original}",
                        image,
                    )
                    return
            Path(path).parent.mkdir(parents=True, exist_ok=True)
            sitk.WriteImage(image, path)

    def put(
        self,
        path: str,
        image: sitk.Image | np.ndarray | torch.Tensor,
        source_image: sitk.Image | str = None,
    ):
        """
        Adds image to the queue.

        Args:
            path (str): path where the image will be saved.
            image (sitk.Image | np.ndarray | torch.Tensor): image to be saved.
                Can be a numpy array, a SimpleITK image, or a PyTorch tensor.
            source_image (sitk.Image | str, optional): source image used to
                define metadata. Defaults to None.
        """
        self.queue.put((path, image, source_image))

    def close(self):
        """
        Terminates all processes.
        """
        for _ in self.workers:
            self.queue.put(None)
        for worker in self.workers:
            worker.join()
