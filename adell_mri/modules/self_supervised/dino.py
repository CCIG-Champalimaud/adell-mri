from typing import Any, Dict, List, Tuple

import torch

from ..layers.linear_blocks import MLP
from ..layers.vit import ViT

TensorList = List[torch.Tensor]
DINOOut = Tuple[torch.Tensor, TensorList]


class DINO(torch.nn.Module):
    """
    Implementation of the DINO network from META.

    Based on I-JEPA, https://arxiv.org/abs/2104.14294
    """

    def __init__(
        self,
        backbone_args: Dict[str, Any],
        projection_head_args: Dict[str, Any],
        out_dim: int,
    ):
        """
        Args:
            backbone_args (Dict[str, Any]): arguments for the backbone encoder.
            projection_head_args (Dict[str, Any]): arguments for the projection
                head.
            out_dim (int): output dimension for DINO.
        """
        super().__init__()
        self.backbone_args = backbone_args
        self.projection_head_args = projection_head_args
        self.out_dim = out_dim

        self.initialize_encoder()
        self.initialize_projection()
        self.initialize_last_layer()

    def forward_encoder(self, X: torch.Tensor) -> torch.Tensor:
        out = self.encoder_(X)[0]
        if self.encoder_.n_registers > 0:
            out = out[:, self.encoder_.n_registers :]
        if self.encoder_.use_class_token is True:
            out = out[:, 0]
        else:
            out = out.mean(-1)
        return out

    def initialize_encoder(self):
        self.encoder_ = ViT(**self.backbone_args)

    def initialize_projection(self):
        if "structure" not in self.projection_head_args:
            raise KeyError("`structure` must be specified in `projection_head_args`.")
        self.mlp_out_dim = self.projection_head_args["structure"][-1]
        self.projection_head_args["structure"] = self.projection_head_args["structure"][
            :-1
        ]
        self.projection_ = MLP(
            input_dim=self.encoder_.attention_dim,
            output_dim=self.mlp_out_dim,
            **self.projection_head_args,
        )

    def initialize_last_layer(self):
        self.last_layer_ = torch.nn.utils.parametrizations.weight_norm(
            torch.nn.Linear(self.mlp_out_dim, self.out_dim, bias=False)
        )
        self.last_layer_.parametrizations.weight.original0.data.fill_(1)
        self.last_layer_.parametrizations.weight.original0.requires_grad = False

    def forward(self, X: torch.Tensor) -> DINOOut:
        # encode full image and return
        return self.last_layer_(
            torch.nn.functional.normalize(
                self.projection_(self.forward_encoder(X)), dim=-1, p=2
            )
        )

    def forward_representation(self, X: torch.Tensor) -> torch.Tensor:
        # return self.encoder_(X)[0].permute(0, 2, 1)
        return self.forward_encoder(X)
