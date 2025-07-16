"""
Semi-supervised learning loss modules.
"""

from math import prod
from queue import Queue
from typing import Sequence

import numpy as np
import torch
import torch.nn.functional as F


def swap(xs: Sequence, a: int, b: int) -> None:
    """
    Swaps xs element a with element b.

    Args:
        xs (Sequence): sequence (list, tuple, etc).
        a (int): first index.
        b (int): second index.
    """
    xs[a], xs[b] = xs[b], xs[a]


def derangement(
    n: int, rng: np.random.Generator = None, seed: int = 42
) -> list[int]:
    """
    Generate a derangement of n elements.

    A derangement is a permutation of the elements 0..n-1 such that no element
    appears in its original position.

    Args:
        n: Number of elements to derange.
        rng: Random number generator.
        seed: Seed for random number generator.

    Returns:
        List of deranged elements.
    """
    if rng is None:
        rng = np.random.default_rng(seed)
    xs = [i for i in range(n)]
    for a in range(1, n):
        b = rng.choice(range(0, a))
        swap(xs, a, b)
    return xs


def anchors_from_derangement(
    X: torch.Tensor, rng: np.random.Generator | None = None
) -> torch.Tensor:
    """
    Generate anchors from a derangement of the elements of X.

    Args:
        X (torch.Tensor): tensor.
        rng (np.random.Generator | None): random number generator. Defaults to
            None (instantiates a default_rng).

    Returns:
        torch.Tensor: deranged X.
    """
    if rng is None:
        rng = np.random.default_rng()
    anchors = []
    for idx in derangement(X.shape[0], rng=rng):
        anchors.append(X[idx])
    anchors = torch.stack(anchors)
    return anchors


class AnatomicalContrastiveLoss(torch.nn.Module):
    """
    Implementation of the anatomical loss method suggested in "Bootstrapping
    Semi-supervised Medical Image Segmentation with Anatomical-aware Contrastive
    Distillation". Allow for both local and global KL divergence calculation.

    This was altered to extract a fixed number of hard examples (unlike what was
    specified in the original paper, which uses a threshold to define hard
    examples).
    """

    def __init__(
        self,
        n_classes: int,
        n_features: int,
        batch_size: int,
        top_k: int = 100,
        ema_theta: float = 0.9,
        tau: float = 0.1,
    ):
        """
        Args:
            n_classes (int): number of classes.
            n_features (int): number of features.
            batch_size (int): size of batch.
            top_k (int, optional): how many hard examples should be extracted.
                Defaults to 100.
            ema_theta (float, optional): theta for exponential moving average.
                Defaults to 0.9.
            tau (float, optional): temperature. Defaults to 0.1.
        """
        super().__init__()
        self.n_classes = n_classes
        self.n_features = n_features
        self.batch_size = batch_size
        self.top_k = top_k
        self.ema_theta = ema_theta
        self.tau = tau

        self.average_representations = torch.zeros(
            [1, self.n_classes, self.n_features]
        )

        self.hard_examples = torch.zeros([batch_size, self.top_k, n_features])
        self.hard_example_class = torch.zeros([batch_size, self.top_k, 1])

    def update_average_class_representation(
        self,
        pred: torch.Tensor,
        b: torch.LongTensor,
        c: torch.LongTensor,
        v: torch.LongTensor,
    ):
        """
        Updates exponential moving average class representation.

        Args:
            pred (torch.Tensor): prediction.
            b (torch.LongTensor): batch indices.
            c (torch.LongTensor): class indices.
            v (torch.LongTensor): feature indices.
        """
        for i in range(self.n_classes):
            rep = pred[b[c == i], :, v[c == i]]
            if prod(rep.shape) > 0:
                rep = rep.permute(1, 0)
                self.average_representations[:, i, :] = torch.add(
                    self.average_representations[:, i, :]
                    * (1 - self.ema_theta),
                    rep.mean(1) * self.ema_theta,
                )

    def update_hard_examples(
        self,
        proba: torch.Tensor,
        embeddings: torch.Tensor,
        labels: torch.LongTensor,
    ):
        """
        Updates the hard example list. ``proba``, ``embeddings`` and ``labels``
        are expected to have shame shape (excluding the number of channels).

        Args:
            proba (torch.Tensor): probability tensor.
            embeddings (torch.Tensor): embeddings tensor.
            labels (torch.LongTensor): labels.
        """
        weights = proba.prod(1)
        for i in range(self.batch_size):
            top_k = weights[i].topk(self.top_k)
            self.hard_examples[i] = embeddings[i, :, top_k.indices].permute(
                1, 0
            )
            self.hard_example_class[i] = labels[i, top_k.indices].unsqueeze(-1)

    def delete(self, X: torch.Tensor, idx: int) -> torch.Tensor:
        """
        Deletes an index from a tensor in axis=1.

        Args:
            X (torch.Tensor): tensor.
            idx (int): index to be deleted.

        Returns:
            torch.Tensor: tensor without index idx.
        """
        return torch.cat([X[:, :idx], X[:, (idx + 1) :]], 1)

    def l_anco(self) -> torch.Tensor:
        """
        Anatomical contrastive loss between hard examples and average
        representations.

        Returns:
            torch.Tensor: loss value for anatomical contrastive loss.
        """
        output = torch.zeros([self.batch_size, self.n_classes]).to(
            self.average_representations
        )
        for batch in range(self.batch_size):
            for nc in range(self.n_classes):
                idx, _ = torch.where(self.hard_example_class[batch] == nc)
                he = self.hard_examples[batch, idx]
                par = self.average_representations[:, nc, :]
                nar = self.delete(self.average_representations, nc)
                num = torch.exp(he * par / self.tau)
                neg_den = torch.sum(
                    torch.exp(he[:, None] * nar / self.tau), 1, keepdim=True
                )
                out = -torch.log(num / (num + neg_den))
                if prod(out.shape) > 0:
                    output[batch, nc] = torch.mean(out.flatten(start_dim=1))
        return -output.sum() / self.batch_size

    def forward(
        self, proba: torch.Tensor, y: torch.Tensor, embeddings: torch.Tensor
    ) -> torch.Tensor:
        """
        Forward method for anatomical contrastive loss.

        Args:
            proba (torch.Tensor): probabilities tensor.
            y (torch.Tensor): ground truth.
            embeddings (torch.Tensor): embeddings tensor.

        Returns:
            torch.Tensor: loss value.
        """
        # expects y to be one hot encoded
        proba = proba.flatten(start_dim=2)
        y = y.flatten(start_dim=2)
        embeddings = embeddings.flatten(start_dim=2)
        labels = torch.argmax(y, dim=1)

        # get indices for y
        b, c, v = torch.where(y > 0)

        # update average class representation
        self.update_average_class_representation(embeddings, b, c, v)

        # mine hard examples
        self.update_hard_examples(
            proba=proba, embeddings=embeddings, labels=labels
        )

        # calculate the anatomical contrastive loss
        return self.l_anco()


class NearestNeighbourLoss(torch.nn.Module):
    """
    Nearest neighbour loss:

    1. For a given FIFO queue of past elements and a new sample, use the oldest
        elements from the queue to calculate the distances between the new sample
        and the old elements
    2. Maximise the cosine similarity between queue elements and elements from
        the new sample belonging to the same class.

    Based on Frosst 2019 [1].

    [1] https://proceedings.mlr.press/v97/frosst19a.html
    """

    def __init__(
        self,
        maxsize: int,
        n_classes: int,
        max_elements_per_batch: int,
        n_samples_per_class: int,
        temperature: float = 0.1,
        seed: int = 42,
    ):
        """
        Args:
            maxsize (int): maximum queue size.
            n_classes (int): number of classes.
            max_elements_per_batch (int): maximum number of elements to be
                retrieved for each batch.
            n_samples_per_class (int): number of samples to be retrieved for
                each class.
            temperature (float, optional): temperature for softmax. Defaults to
                0.1.
            seed (int, optional): random seed. Defaults to 42.
        """
        super().__init__()
        self.maxsize = maxsize
        self.n_classes = n_classes
        self.max_elements_per_batch = max_elements_per_batch
        self.n_samples_per_class = n_samples_per_class
        self.temperature = temperature
        self.seed = seed

        self.q = [Queue(maxsize=self.maxsize) for _ in range(self.n_classes)]
        self.rng = np.random.default_rng(seed)

    def put(self, X: torch.Tensor, y: torch.Tensor):
        """
        Adds elements to queue stratified by class.

        Args:
            X (torch.Tensor): embeddings tensor.
            y (torch.Tensor): ground truth.
        """
        X = X.flatten(start_dim=2)
        y = y.flatten(start_dim=2)
        b, c, v = torch.where(y > 0)
        for cl in range(self.n_classes):
            idx = c == cl
            elements = X[b[idx], :, v[idx]]
            n_elements = elements.shape[0]
            if n_elements > self.max_elements_per_batch:
                elements = elements[
                    self.rng.choice(n_elements, self.max_elements_per_batch)
                ]
            if n_elements > 0:
                self.q[cl].put(elements.detach())

    def get_from_class(self, n: int, cl: int) -> torch.Tensor:
        """
        Retrieves ``n`` elements from queue with class ``cl``.


        Args:
            n (int): number of elements to be retrieved.
            cl (int): class.

        Returns:
            torch.Tensor: ``n`` elements from class ``c``.
        """
        q = self.q[cl]
        n_elements = q.qsize()
        return [q.get() for _ in self.rng.choice(n_elements, n)]

    def get(self, n: int, cl: int | None = None) -> None:
        """
        Gets a set of elements from the queue. If ``cl`` is specified, retrieves
        elements from each class, otherwise retrieves a random set of elements.

        Args:
            n (int): number of elements to be retrieved.
            cl (int | None, optional): class. Defaults to None.

        Returns:
            torch.Tensor: ``n`` elements from class ``c`` or ``n`` random
                elements.
        """
        if cl is not None:
            output = self.get_from_class(n, cl)
        else:
            output = []
            sample = self.rng.choice(self.n_classes, size=n)
            un, count = np.unique(sample, return_counts=True)
            for cl, n in zip(un, count):
                output.append(self.get_from_class(n, cl))
        return torch.cat(output, 0)

    def __len__(self) -> int:
        """
        Returns the size of the queue.

        Returns:
            int: size of queue.
        """
        return sum([q.qsize() for q in self.q])

    def get_past_samples(
        self, device="cuda"
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Retrieves elements from queue (the same number of elements for each
        class) and their respective one hot labels.

        Args:
            device (str, optional): device. Defaults to "cuda".

        Returns:
            torch.Tensor: past samples.
            torch.Tensor: past sample classes.
        """
        n_samples = [
            np.minimum(self.n_samples_per_class, self.q[cl].qsize())
            for cl in range(self.n_classes)
        ]
        past_samples = [
            self.get(n, cl) for cl, n in zip(range(self.n_classes), n_samples)
        ]
        past_sample_labels = torch.as_tensor(
            np.concatenate(
                [
                    np.repeat(cl, past_sample.shape[0])
                    for cl, past_sample in zip(
                        range(self.n_classes), past_samples
                    )
                ],
                0,
            ),
            device=device,
        )
        past_sample_labels = F.one_hot(past_sample_labels, self.n_classes)
        past_samples = torch.cat(past_samples)
        return past_samples, past_sample_labels

    def forward(self, X: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        """
        Forward method.

        Args:
            X (torch.Tensor): output tensor.
            y (torch.Tensor): ground truth.

        Returns:
            torch.Tensor: value for nearest neighbour loss.
        """
        X = X.flatten(start_dim=2).permute(0, 2, 1)
        y = y.flatten(start_dim=2).permute(0, 2, 1)
        b, c, v = torch.where(y > 0)
        past_samples, past_sample_labels = self.get_past_samples(X.device)
        distances = 1 - F.cosine_similarity(
            X[:, :, None], past_samples[None, None, :], -1
        )
        is_same = torch.sum(
            y[:, :, None] * past_sample_labels[None, None, :], -1
        )
        same_class_distances = torch.exp(
            -distances * is_same / self.temperature
        )
        other_class_distances = torch.exp(
            -distances * (1 - is_same) / self.temperature
        )
        return torch.nanmean(
            same_class_distances.nansum(-1) / other_class_distances.nansum(-1)
        )


class PseudoLabelCrossEntropy(torch.nn.Module):
    """
    Calculates cross-entropy between probability map p_1 and pseudo-labels
    calculated from probability map p_2 given a probability threshold.

    Useful for distillation, semi-supervised learning, etc.
    """

    def __init__(self, threshold: float, *args, **kwargs):
        """
        Args:
            threshold (float): threshold for probability threshold.
        """
        super().__init__()
        self.threshold = threshold

        self.ce = torch.nn.CrossEntropyLoss(*args, **kwargs)

    def forward(self, pred: torch.Tensor, proba: torch.Tensor) -> torch.Tensor:
        """
        Forward method.

        Args:
            pred (torch.Tensor): predictions.
            proba (torch.Tensor): probabilities.

        Returns:
            torch.Tensor: cross entropy between predictions and probabilities.
        """
        pseudo_y = proba > self.threshold
        return self.ce(pred, pseudo_y.float())


class LocalContrastiveLoss(torch.nn.Module):
    """
    Implements a local contrastive loss function.
    """

    def __init__(self, temperature: float = 0.1, seed: int = 42):
        """
        Args:
            temperature (float, optional): temperature for cross entropy.
                Defaults to 0.1.
            seed (int, optional): random seed. Defaults to 42.
        """
        super().__init__()
        self.temperature = temperature
        self.seed = seed
        self.rng = np.random.default_rng(seed)
        self.eps = torch.as_tensor(1e-8)

    def forward(
        self,
        X_1: torch.Tensor,
        X_2: torch.Tensor,
        anchors: torch.Tensor = None,
    ) -> torch.Tensor:
        """
        Forward method. Based on original LoCo paper [1].

        [1] https://proceedings.neurips.cc/paper/2020/file/7fa215c9efebb3811a7ef58409907899-Paper.pdf

        Args:
            X_1 (torch.Tensor): output for first view of tensor.
            X_2 (torch.Tensor): output for second view of tensor.
            anchors (torch.Tensor, optional): for compatibility. Defaults to
                None.

        Returns:
            torch.Tensor: loss value.
        """
        X_1 = X_1.flatten(start_dim=2)[None, :, :, :]
        X_2 = X_2.flatten(start_dim=2)[:, None, :, :]
        sim = F.softmax(
            F.cosine_similarity(X_1, X_2, dim=2) / self.temperature, dim=1
        )
        loss = -torch.log(
            torch.max(sim.diagonal().permute(1, 0), self.eps)
        ).mean(-1)
        return loss


class LocalContrastiveLossWithAnchors(torch.nn.Module):
    """
    Implements a local contrastive loss function with anchors.
    """

    def __init__(self, temperature: float = 0.1, seed: int = 42):
        """
        Args:
            temperature (float, optional): temperature for cross validation.
                Defaults to 0.1.
            seed (int, optional): random seed. Defaults to 42.
        """
        super().__init__()
        self.temperature = temperature
        self.seed = seed
        self.rng = np.random.default_rng(seed)
        self.eps = torch.as_tensor(1e-8)

    def anchors_from_derangement(self, X: torch.Tensor) -> torch.Tensor:
        """
        Generates anchors from derangement of X.

        Args:
            X (torch.Tensor): tensor.

        Returns:
            torch.Tensor: deranged X.
        """
        anchors = []
        for idx in derangement(X.shape[0], rng=self.rng):
            anchors.append(X[idx])
        anchors = torch.stack(anchors)
        return anchors

    def forward(
        self,
        X: torch.Tensor,
        anchors_1: torch.Tensor | None = None,
        anchors_2: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """
        Forward method.

        Args:
            X (torch.Tensor): embeddings tensor.
            anchors_1 (torch.Tensor, optional): anchor embeddings. Defaults to
                None (generated from X through derangement).
            anchors_2 (torch.Tensor, optional): anchor embeddings. Defaults to
                None (generated from X through derangement).

        Returns:
            torch.Tensor: loss value.
        """
        anchors_1 = (
            anchors_from_derangement(X, self.rng)
            if anchors_1 is None
            else anchors_1
        )
        anchors_2 = (
            anchors_from_derangement(X, self.rng)
            if anchors_2 is None
            else anchors_2
        )
        X = X.flatten(start_dim=2)
        anchors_1 = anchors_1.flatten(start_dim=2)
        anchors_2 = anchors_2.flatten(start_dim=2)
        sim_1 = F.cosine_similarity(X, anchors_1, dim=1) / self.temperature
        sim_2 = F.cosine_similarity(X, anchors_2, dim=1) / self.temperature
        return F.kl_div(
            F.softmax(sim_1, dim=1),
            F.softmax(sim_2, dim=1),
            reduction="none",
        ).sum(-1)
