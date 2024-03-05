import numpy as np
import torch

from typing import List, Any


def mode(X: np.ndarray):
    u, c = np.unique(X, return_counts=True)
    out = u[c.argmax()]
    return out


class CategoricalEmbedder(torch.nn.Module):
    def __init__(
        self, cat_feats: List[int | List[Any]], embedding_size: int = 512
    ):
        super().__init__()
        self.cat_feats = cat_feats
        self.embedding_size = embedding_size

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
                    torch.nn.Embedding(cat_feat, self.embedding_size)
                )
            elif isinstance(cat_feat, (list, tuple)):
                self.conversions.append(
                    {str(k): i for i, k in enumerate(cat_feat)}
                )
                self.embedders.append(
                    torch.nn.Embedding(len(cat_feat), self.embedding_size)
                )
                self.convert = True

    @property
    def device(self):
        return next(self.embedders[-1].parameters()).device

    def forward(self, X: List[torch.Tensor]):
        if self.convert is True:
            for i in range(len(X)):
                X[i] = [
                    conversion[str(x[0])]
                    if isinstance(x, np.ndarray)
                    else conversion[str(x)]
                    for x, conversion in zip(X[i], self.conversions)
                ]
            X = torch.as_tensor(X, device=self.device)
        out = []
        for x, embedder in zip(X.permute(1, 0).contiguous(), self.embedders):
            out.append(embedder(x))
        out = torch.cat(out, axis=1)
        return out


class Embedder(torch.nn.Module):
    def __init__(
        self,
        cat_feat: List[int | List[Any]] = None,
        n_num_feat: int = None,
        embedding_size: int = 512,
        max_queue_size: int = 512,
        device: torch.device = None,
    ):
        super().__init__()
        self.cat_feat = cat_feat
        self.n_num_feat = n_num_feat
        self.embedding_size = embedding_size
        self.max_queue_size = max_queue_size
        self.device = device

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
        self.final_n_features = 0
        if self.cat_feat is not None:
            self.cat_embedder = CategoricalEmbedder(
                self.cat_feat, self.embedding_size
            )
            self.final_n_features += self.cat_embedder.output_length
        if self.n_num_feat is not None:
            self.num_embedder = torch.nn.Linear(
                self.n_num_feat, self.embedding_size
            )
            self.final_n_features += self.embedding_size

        self.final_embedding = torch.nn.Linear(
            self.final_n_features, self.embedding_size
        )

    def update_queues(
        self, X_cat: List[torch.LongTensor] = None, X_num: torch.Tensor = None
    ):
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

    def get_expected_cat(self, n: int = 1):
        # returns mode for all categorical features
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

    def get_expected_num(self, n: int = 1):
        # returns mean for all numerical features
        output = []
        for i in range(len(self.num_distributions)):
            curr = self.num_distributions[i]
            tmp = np.mean(curr)
            output.append([tmp for _ in range(n)])
        output = torch.as_tensor(
            output, device=self.device, dtype=torch.float32
        ).T
        return output

    def forward(
        self,
        X_cat: List[torch.LongTensor] = None,
        X_num: torch.Tensor = None,
        batch_size: int = 1,
        update_queues: bool = True,
    ):
        if update_queues is True:
            self.update_queues(X_cat=X_cat, X_num=X_num)
        embeddings = []
        if self.cat_feat is not None:
            if X_cat is None:
                X_cat = self.get_expected_cat(batch_size)
            embeddings.append(self.cat_embedder(X_cat))
            if self.device is None:
                self.device = embeddings[-1].device

        if self.n_num_feat is not None:
            if X_num is None:
                X_num = self.get_expected_num(batch_size)
            embeddings.append(self.num_embedder(X_num))
            if self.device is None:
                self.device = embeddings[-1].device
        out = self.final_embedding(torch.cat(embeddings, 1))
        return out
