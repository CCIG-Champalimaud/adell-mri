from copy import deepcopy
from typing import Any, List

import numpy as np
import torch


def mode(X: np.ndarray):
    u, c = np.unique(X, return_counts=True)
    out = u[c.argmax()]
    return out


class CategoricalEmbedder(torch.nn.Module):
    def __init__(
        self,
        cat_feats: List[int | List[Any]],
        embedding_size: int = 512,
        max_norm: float | None = None,
    ):
        """
        Embeds categorical features. If cat_feats is specified as a list of
        n integers, alocates n embeddings, one for each integer with input size
        identical to each embedding. If cat_feats is specified as a list of
        lists or tuples, first alocates a dictionary converting between
        list/tuple elements to a number and then allocates embeddings for each
        one.

        Args:
            cat_feats (List[int  |  List[Any]]): list of categorical
                specifications.
            embedding_size (int, optional): size of the embedding. Defaults to
                512.
            max_norm (float, optional): maximum norm for the embeddings.
                Defaults to None.
        """
        super().__init__()
        self.cat_feats = cat_feats
        self.embedding_size = embedding_size
        self.max_norm = max_norm

        self.output_length = len(self.cat_feats) * self.embedding_size

        self.init_embeddings()

    def init_embeddings(self):
        self.conversions = []
        self.embedders = torch.nn.ModuleList([])
        self.convert = False
        for cat_feat in self.cat_feats:
            if isinstance(cat_feat, int):
                self.conversions.append(None)
                self.embedders.append(
                    torch.nn.Embedding(
                        cat_feat, self.embedding_size, max_norm=self.max_norm
                    )
                )
            elif isinstance(cat_feat, (list, tuple)):
                self.conversions.append(
                    {str(k): i for i, k in enumerate(cat_feat)}
                )
                self.embedders.append(
                    torch.nn.Embedding(
                        len(cat_feat),
                        self.embedding_size,
                        max_norm=self.max_norm,
                    )
                )
                self.convert = True

    @property
    def device(self):
        return next(self.embedders[-1].parameters()).device

    def forward(self, X: List[torch.Tensor], return_X: bool = False):
        X_conv = deepcopy(X)
        if isinstance(X_conv, torch.Tensor):
            ndim = len(X_conv.shape)
            if ndim == 1:
                X_conv = X_conv[:, None]
            elif ndim != 2:
                raise ValueError(
                    f"If X is a tensor it should have 1 or 2 dimensions but X has {ndim} ({X.shape})"
                )
        elif self.convert is True:
            for i in range(len(X_conv)):
                X_conv[i] = [
                    (
                        conversion[str(x[0])]
                        if isinstance(x, np.ndarray)
                        else conversion[str(x)]
                    )
                    for x, conversion in zip(X_conv[i], self.conversions)
                ]
            X_conv = torch.as_tensor(X_conv, device=self.device)
        out = []
        for x, embedder in zip(
            X_conv.permute(1, 0).contiguous(), self.embedders
        ):
            out.append(embedder(x))
        out = torch.cat(out, axis=1)
        if len(out.shape) == 2:
            out = out[:, None, :]
        if return_X is True:
            return out, X_conv
        return out


class Embedder(torch.nn.Module):
    def __init__(
        self,
        cat_feat: List[int | List[Any]] = None,
        n_num_feat: int = None,
        max_norm: float | None = None,
        embedding_size: int = 512,
        max_queue_size: int = 512,
        numerical_moments: int = None,
        device: torch.device = None,
    ):
        """
        Embedder for categorical and numerical features. For the categorical
        feature specification (cat_feat) check CategoricalEmbedder, the
        numerical embedding is simply performed with a linear layer
        transforming n features to embedding_size * n features.

        This also features a queue which is used to define values for the
        generative approach, the size of the queue is defined using
        max_queue_size.

        All embeddings are concatenated and a linear layer is applied to them
        such that the output of the forward pass is always [B, embedding_size],
        where B is the batch size.

        Args:
            cat_feat (List[int  |  List[Any]], optional): categorical feature
                specification. Defaults to None.
            n_num_feat (int, optional): number of numerical features. Defaults
                to None.
            max_norm (float, optional): maximum norm for the categorical
                embeddings. Defaults to None.
            embedding_size (int, optional): size of the embedding. Defaults to
                512.
            max_queue_size (int, optional): maximum queue size. Defaults to
                512.
            numerical_moments (tuple[list[float], list[float]] | None,
                optional): list of means and standard deviations for numerical
                normalisation of the regression targets. Defaults to None (no
                normalisation).
            device (torch.device, optional): device. Defaults to None.
        """
        super().__init__()
        self.cat_feat = cat_feat
        self.n_num_feat = n_num_feat
        self.max_norm = max_norm
        self.embedding_size = embedding_size
        self.max_queue_size = max_queue_size
        self.numerical_moments = numerical_moments
        self.device = device

        self.rng = np.random.default_rng(42)

        self.init_embeddings()
        if self.cat_feat:
            self.cat_distributions = [[] for _ in self.cat_feat]
        else:
            self.cat_distributions = []
        if self.n_num_feat:
            self.num_distributions = [[] for _ in range(n_num_feat)]
        else:
            self.num_distributions = []

    def init_embeddings(self):
        """
        Initialises all embeddings.

        Raises:
            ValueError: if the number of means and standard deviations in
                ``numerical_moments`` is different from the number of numerical
                features.
        """
        self.final_n_features = 0
        if self.cat_feat is not None:
            self.cat_embedder = CategoricalEmbedder(
                self.cat_feat, self.embedding_size, max_norm=self.max_norm
            )
            self.final_n_features += self.cat_embedder.output_length
        if self.n_num_feat is not None:
            nnf = self.n_num_feat
            self.num_embedder = torch.nn.ModuleList(
                [torch.nn.Linear(1, self.embedding_size) for _ in range(nnf)]
            )
            self.final_n_features += self.embedding_size
            if self.numerical_moments is not None:
                means, stds = self.numerical_moments
                if (len(means) != nnf) or (len(stds) != nnf):
                    raise ValueError(
                        "Number of means and stds in numerical_moments does not\
                            match number of numerical features."
                    )
                self.means = torch.nn.Parameter(
                    torch.as_tensor(means, dtype=torch.float32),
                    requires_grad=False,
                )
                self.stds = torch.nn.Parameter(
                    torch.as_tensor(stds, dtype=torch.float32),
                    requires_grad=False,
                )

        self.final_embedding = torch.nn.Linear(
            self.final_n_features, self.embedding_size
        )
        self.unconditioned_embeddings = torch.nn.Embedding(
            1, self.embedding_size, max_norm=self.max_norm
        )

    def update_queues(
        self, X_cat: List[torch.LongTensor] = None, X_num: torch.Tensor = None
    ):
        """
        Updates numerical and categorical distribution queues.

        Args:
            X_cat (List[torch.LongTensor], optional): categorical features.
                Defaults to None.
            X_num (torch.Tensor, optional): numerical features. Defaults to
                None.
        """
        if len(self.cat_distributions) > 0:
            if len(self.cat_distributions[0]) > self.max_queue_size:
                self.cat_distributions = [
                    d[-512:] for d in self.cat_distributions[-512:]
                ]
        if len(self.num_distributions) > 0:
            if len(self.num_distributions[0]) > self.max_queue_size:
                self.num_distributions = [
                    d[-512:] for d in self.num_distributions[-512:]
                ]
        if X_cat is not None:
            for i in range(len(self.cat_distributions)):
                self.cat_distributions[i].extend([x[i] for x in X_cat])
        if X_num is not None:
            X_num = X_num.cpu().numpy()
            for i in range(len(self.num_distributions)):
                self.num_distributions[i].extend(X_num[:, i])

    def get_expected_cat(self, n: int = 1) -> list[np.ndarray] | None:
        """
        Get expected value (mode) for all categorical features.

        Args:
            n (int, optional): number of samples. Defaults to 1.

        Returns:
            list[np.ndarray] | None: expected value for all categorical
                features.
        """
        # returns mode for all categorical features
        if self.cat_feat is None:
            return
        output = [[] for _ in range(n)]
        for i in range(len(self.cat_distributions)):
            curr = self.cat_distributions[i]
            if len(curr) == 0:
                tmp = self.cat_feat[i][0]
            else:
                tmp = str(mode(curr))
            for j in range(n):
                output[j].append(tmp)
        output = [np.array(x) for x in output]
        return output

    def get_random_cat(self, n: int = 1) -> list[np.ndarray] | None:
        """
        Returns random categorical features.

        Args:
            n (int, optional): sample size. Defaults to 1.

        Returns:
            list[np.ndarray] | None: random categorical features.
        """
        # returns random categorical features
        if self.cat_feat is None:
            return
        output = [[] for _ in range(n)]
        for i in range(len(self.cat_distributions)):
            curr = self.cat_distributions[i]
            tmp = self.rng.choice(curr, n)
            for j in range(n):
                output[j].append(tmp[j])
        output = [np.array(x) for x in output]
        return output

    def get_expected_num(self, n: int = 1) -> torch.Tensor | None:
        """
        Returns expected numerical features (average).

        Args:
            n (int, optional): number of samples. Defaults to 1.

        Returns:
            torch.Tensor | None: expected numerical features.
        """
        # returns mean for all numerical features
        if self.n_num_feat is None:
            return
        output = []
        for i in range(len(self.num_distributions)):
            curr = self.num_distributions[i]
            tmp = np.mean(curr)
            output.append([tmp for _ in range(n)])
        output = torch.as_tensor(
            output, device=self.device, dtype=torch.float32
        ).T
        return output

    def get_random_num(self, n: int = 1) -> torch.Tensor | None:
        """
        Returns random numerical features.

        Args:
            n (int, optional): number of samples. Defaults to 1.

        Returns:
            torch.Tensor | None: random numerical features.
        """
        # returns mean for all numerical features
        if self.n_num_feat is None:
            return
        output = []
        for i in range(len(self.num_distributions)):
            curr = self.num_distributions[i]
            output.append(self.rng.choice(curr, n))
        output = torch.as_tensor(
            output, device=self.device, dtype=torch.float32
        ).T
        return output

    def normalize_numeric_features(self, X: torch.Tensor) -> torch.Tensor:
        """
        Normalizes the numerical features of a tensor X with the provided
        numerical moments (``numerical_moments``).

        Args:
            X (torch.Tensor): a batched tensor.

        Returns:
            torch.Tensor: the normalized tensor.
        """
        if self.numerical_moments is None:
            return X
        else:
            return (X - self.means[None, :]) / self.stds[None, :]

    def unconditional_like(self, X: torch.Tensor) -> torch.Tensor:
        """
        Returns the unconditioned representation (the same for all) with the
        same batch size as X.

        Args:
            X (torch.Tensor): a batched tensor.

        Returns:
            torch.Tensor: the repeated unconditioned embedding.
        """
        return self.unconditioned_embeddings(
            torch.zeros(X.shape[0], device=X.device, dtype=torch.int32)
        )

    def embed_categorical(
        self,
        X: List[torch.LongTensor],
        uncondition_idx: int | list[int] | str | None = None,
    ) -> torch.Tensor:
        """
        Embeds the categorical features of a tensor X.

        Args:
            X (List[torch.LongTensor], optional): categorical features.
            uncondition_idx (int | list[int] | str | None, optional): indices of
                the categorical features to be unconditioned (replaced by the
                unconditioned representation). Defaults to None.

        Returns:
            torch.Tensor: the embedded categorical features.
        """
        categorical_embeddings, converted_class_X = self.cat_embedder(
            X, return_X=True
        )
        if uncondition_idx is not None:
            if uncondition_idx == "all":
                uncondition_idx = range(
                    categorical_embeddings.shape[-1] // self.embedding_size
                )
            elif isinstance(uncondition_idx, int):
                uncondition_idx = [uncondition_idx]
            for idx in uncondition_idx:
                ncf = self.cat_embedder.embedding_size
                start_idx, stop_idx = ncf * idx, ncf * (idx + 1)
                categorical_embeddings[
                    :, :, start_idx:stop_idx
                ] = self.unconditional_like(categorical_embeddings)[:, None]

        return categorical_embeddings, converted_class_X

    def embed_numerical(
        self, X: torch.Tensor, uncondition_idx: int | list[int] | None = None
    ) -> torch.Tensor:
        """
        Embeds the numerical features of a tensor X.

        Args:
            X (torch.Tensor): numerical features.
            uncondition_idx (int | list[int] | None, optional): indices of the
                numerical features to be unconditioned (replaced by the
                unconditioned representation). Defaults to None.

        Returns:
            torch.Tensor: the embedded numerical features.
        """
        X_norm = self.normalize_numeric_features(X)
        numerical_embeddings = [
            self.num_embedder[i](X_norm[:, i].unsqueeze(1))
            for i in range(self.n_num_feat)
        ]
        if uncondition_idx is not None:
            if uncondition_idx == "all":
                uncondition_idx = range(len(numerical_embeddings))
            elif isinstance(uncondition_idx, int):
                uncondition_idx = [uncondition_idx]
            for idx in uncondition_idx:
                numerical_embeddings[idx] = self.unconditional_like(X)

        numerical_embeddings = torch.stack(numerical_embeddings, dim=0)
        numerical_embeddings = numerical_embeddings.sum(0)
        return numerical_embeddings, X_norm

    def forward(
        self,
        X_cat: List[torch.LongTensor] | None = None,
        X_num: torch.Tensor | None = None,
        batch_size: int = 1,
        update_queues: bool = True,
        return_X: bool = False,
        uncondition_cat_idx: int | list[int] | str | None = None,
        uncondition_num_idx: int | list[int] | str | None = None,
    ) -> torch.Tensor:
        """
        Forward method for embeddings. If X_cat is None, the expected value
        (mode) is produced as a placeholder. If X_num is None, the expected
        value (mean) is produced as a placeholder. The size of the batch
        corresponds to batch_size.

        Args:
            X_cat (List[torch.LongTensor], optional): categorical features.
                Defaults to None.
            X_num (torch.Tensor, optional): numerical features. Defaults to
                None.
            batch_size (int, optional): size of the batch. Defaults to 1.
            update_queues (bool, optional): whether queues should be updated.
                Defaults to True.
            return_X (bool, optional): whether the converted categorical input
                should be returned.
            uncondition_cat_idx (int | list[int] | str | bool, optional):
                unconditions the categorical features at the specified indicies.
                If "all" then conditions on all features. Defaults to None.
            uncondition_num_idx (int | list[int] | str  | bool, optional):
                unconditions the numerical features at the specified indicies.
                If "all" then conditions on all features.  Defaults to None.

        Returns:
            torch.Tensor: final embedding.
        """
        if update_queues is True:
            self.update_queues(X_cat=X_cat, X_num=X_num)
        embeddings = []
        converted_class_X = None
        converted_reg_X = None
        if self.cat_feat is not None:
            if X_cat is None:
                X_cat = self.get_random_cat(batch_size)
            embedded_X, converted_class_X = self.embed_categorical(
                X_cat, uncondition_idx=uncondition_cat_idx
            )
            embeddings.append(embedded_X)
            if self.device is None:
                self.device = embeddings[-1].device

        if self.n_num_feat is not None:
            if X_num is None:
                X_num = self.get_random_num(batch_size)
            X_num, converted_reg_X = self.embed_numerical(
                X_num, uncondition_idx=uncondition_num_idx
            )
            embeddings.append(X_num[:, None, :])
            converted_reg_X = X_num
            if self.device is None:
                self.device = embeddings[-1].device
        out = self.final_embedding(torch.cat(embeddings, -1))
        if return_X:
            return out, converted_class_X, converted_reg_X
        return out
