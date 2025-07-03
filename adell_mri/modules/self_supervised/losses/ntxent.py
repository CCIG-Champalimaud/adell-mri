"""
Implementation of the NTXent loss.
"""

import torch
import torch.nn.functional as F

from adell_mri.modules.self_supervised.losses.functional import cos_sim


class NTXentLoss(torch.nn.Module):
    """
    Quick and simple implementation of the NT-Xent loss used in the
    SimCLR paper.
    """

    def __init__(self, temperature: float = 1.0, apply_relu: bool = True):
        """
        Args:
            temperature (float, optional): temperature for the scaled
                cross-entropy calculation. Defaults to 1.0.
            apply_relu (bool, optional): applies ReLU before calculating cosine
                similarity between features.
        """
        super().__init__()
        self.temperature = temperature
        self.apply_relu = apply_relu

    def forward(self, X1: torch.Tensor, X2: torch.Tensor):
        if self.apply_relu is True:
            X1 = F.relu(X1)
            X2 = F.relu(X2)
        dev = X1.device
        # calculate denominator
        X_cat = torch.cat([X1, X2], 0)
        n2 = X_cat.shape[0]
        sim_mat = cos_sim(X_cat, X_cat) / self.temperature
        mask = torch.eye(n2, device=dev, dtype=torch.bool)
        sim = sim_mat[mask.roll(shifts=mask.shape[0] // 2, dims=0)]
        mask = torch.zeros_like(sim_mat).masked_fill(mask, -torch.inf)
        desim = torch.logsumexp(sim_mat + mask, dim=-1)

        loss = -sim + desim
        loss = loss.mean()

        return loss
