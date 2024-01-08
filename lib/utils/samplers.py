import numpy as np
import torch

from typing import List, Iterable


class PartiallyRandomSampler(torch.utils.data.Sampler):
    def __init__(
        self,
        classes: Iterable,
        keep_classes: List[int] = [1],
        non_keep_ratio: float = 1.0,
        num_samples: int = None,
        seed: int = None,
    ) -> None:
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

    def __iter__(self):
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

    def set_n_samples(self, n: int):
        self.n = n
        self.num_samples = n

    def __len__(self) -> int:
        return self.n
