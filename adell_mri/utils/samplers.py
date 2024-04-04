import numpy as np
import torch

from typing import List, Iterable


class PartiallyRandomSampler(torch.utils.data.Sampler):
    """
    PartiallyRandomSampler samples indices from keep_classes with
    probability keep_prob and the remaining indices with probability
    1 - keep_prob.

    This allows sampling from an imbalanced dataset while controlling the
    class balance.
    """

    def __init__(
        self,
        classes: Iterable,
        keep_classes: List[int] = [1],
        non_keep_ratio: float = 1.0,
        num_samples: int = None,
        seed: int = None,
    ) -> None:
        """
        Args:
            classes (Iterable): possible classes.
            keep_classes (List[int], optional): elements whose class is in
                keep classes are  always included when __iter__ is called.
                Defaults to [1].
            non_keep_ratio (float, optional): ratio of classes not specified in
                keep_classes. Defaults to 1.0.
            num_samples (int, optional): number of samples. Defaults to None.
            seed (int, optional): seed for sampling of elements not in
                keep_classes. Defaults to None.
        """
        self.classes = classes
        self.keep_classes = keep_classes
        self.non_keep_ratio = non_keep_ratio
        self.num_samples = num_samples
        self.seed = seed

        self.keep_list = []
        self.non_keep_list = []

        for x, c in enumerate(self.classes):
            if c in self.keep_classes:
                self.keep_list.append(x)
            else:
                self.non_keep_list.append(x)
        self.n_keep = len(self.keep_list)
        self.n = len(self.keep_list) + int(self.n_keep * (self.non_keep_ratio))

        if self.seed is None:
            self.seed = np.random.randint(1e6)
        self.rng = np.random.default_rng(self.seed)

    def __iter__(self) -> Iterable[int]:
        """
        Iters a random set of indices in the dataset.

        Yields:
            Iterable[int]: random indices corresponding to elements in the
                dataset.
        """
        rand_list = [
            *self.keep_list,
            *self.rng.choice(
                self.non_keep_list, int(self.n_keep * (self.non_keep_ratio))
            ),
        ]
        if self.num_samples is not None:
            rand_list = self.rng.choice(
                rand_list,
                self.num_samples,
                replace=self.num_samples > len(rand_list),
            )
        self.rng.shuffle(rand_list)
        yield from iter(rand_list)

    def set_n_samples(self, n: int) -> None:
        """
        Sets the number of samples to be drawn.

        Args:
            n (int): number of samples.
        """
        self.n = n
        self.num_samples = n

    def __len__(self) -> int:
        """
        Returns the number of elements retrieved during a complete iteration of
        the dataset.

        Returns:
            int: number of iterations.
        """
        return self.n
