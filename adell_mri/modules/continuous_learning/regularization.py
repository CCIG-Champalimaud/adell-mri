from copy import deepcopy
from typing import List

import torch


class ElasticWeightConsolidation(torch.nn.Module):
    """
    Stores model parameters to calculate a elastic weight consolidation
    term (the p-norm of the difference between the model parameters at the
    current and first epoch/step). This mostly works to reduce a messy
    implementation with other modules. In essence, to use this EWC module
    all one has to do is:

    1. Initialize `ewc = ElasticWeightConsolidation(model,keys)`
    2. Call it as a term in the loss function, i.e. `loss = loss + ewc(model)`
    """

    def __init__(
        self, model: torch.nn.Module, keys: List[str] = None, p: int = 2
    ):
        """
        Args:
            model (torch.nn.Module): input model.
            keys (List[str], optional): names of parameters which will be
                regularized. Defaults to None (regularizes all parameters).
            p (int, optional): order of the norm. Defaults to 2.
        """
        super().__init__()
        self.model = deepcopy(model)
        self.keys = keys
        self.p = p

        self.named_parameters = dict(self.model.named_parameters())
        if self.keys is None:
            self.keys = list(self.named_parameters.keys())

    def __call__(self, model):
        output = torch.zeros([1])
        for k, param in model.named_parameters():
            if k in self.keys:
                output = output.add(
                    torch.norm(param - self.named_parameters[k], p=self.p)
                )
        return output
