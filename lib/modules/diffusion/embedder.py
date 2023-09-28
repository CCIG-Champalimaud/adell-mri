import torch

from typing import List, Any

class CategoricalEmbedder(torch.nn.Module):
    def __init__(self,
                 cat_feats: List[int | List[Any]],
                 embedding_size: int=512):
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
            if isinstance(cat_feat,int):
                self.conversions.append(None)
                self.embedders.append(
                    torch.nn.Embedding(cat_feat, self.embedding_size))
            elif isinstance(cat_feat,(list,tuple)):
                self.conversions.append({str(k):i for i,k in enumerate(cat_feat)})
                self.embedders.append(
                    torch.nn.Embedding(len(cat_feat), self.embedding_size))
                self.convert = True
    
    @property
    def device(self):
        return next(self.embedders[-1].parameters()).device

    def forward(self, X: List[torch.Tensor]):
        if self.convert is True:
            for i in range(len(X)):
                print(self.conversions)
                X[i] = [conversion[x] 
                        for x,conversion in zip(X[i],self.conversions)]
            X = torch.as_tensor(X,device=self.device)
        out = []
        for x,embedder in zip(X.permute(1,0),self.embedders):
            out.append(embedder(x))
        out = torch.cat(out,axis=1)
        return out

class Embedder(torch.nn.Module):
    def __init__(self,
                 cat_feat: List[int | List[Any]]=None,
                 n_num_feat: int=None,
                 embedding_size: int=512):
        super().__init__()
        self.cat_feat = cat_feat
        self.n_num_feat = n_num_feat
        self.embedding_size = embedding_size

        self.init_embeddings()

    def init_embeddings(self):
        self.final_n_features = 0
        if self.cat_feat is not None:
            self.cat_embedder = CategoricalEmbedder(self.cat_feat, 
                                                     self.embedding_size)
            self.final_n_features += self.cat_embedder.output_length
        if self.n_num_feat is not None:
            self.num_embedder = torch.nn.Linear(self.n_num_feat,
                                                  self.embedding_size)
            self.final_n_features += self.embedding_size

        self.final_embedding = torch.nn.Linear(self.final_n_features,
                                               self.embedding_size)
    
    def forward(self, 
                X_cat: List[torch.LongTensor]=None,
                X_num: torch.Tensor=None):
        embeddings = []
        if self.cat_feat is not None:
            embeddings.append(self.cat_embedder(X_cat))
        if self.n_num_feat is not None:
            embeddings.append(self.num_embedder(X_num))
        return self.final_embedding(torch.cat(embeddings,1))
