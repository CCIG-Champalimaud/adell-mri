from typing import Any

import torch

from ...utils.masking import get_masker
from ..layers.linear_blocks import MLP
from ..layers.vit import ViT

iBOTOut = torch.Tensor


class iBOT(torch.nn.Module):
    """
    Implementation of the iBOT network from META.

    Based on https://arxiv.org/abs/2111.07832
    """

    def __init__(
        self,
        backbone_args: dict[str, Any],
        projection_head_args: dict[str, Any],
        out_dim: int,
        feature_map_dimensions: list[int],
        n_encoder_features: int,
        min_patch_size: list[int],
        max_patch_size: list[int],
        n_patches: int = 4,
        reduce_fn: str = "mean",
        seed: int = 42,
    ):
        """
        Args:
            backbone_args (Dict[str, Any]): arguments for the backbone encoder.
            projection_head_args (Dict[str, Any]): arguments for the projection
                head.
            out_dim (int): output dimension for iBOT.
            feature_map_dimensions (List[int]): dimension of the feature map.
            n_encoder_features (int): number of output features from the
                encoder.
            min_patch_size (List[int]): minimum patch size.
            max_patch_size (List[int]): maximum patch size.
            n_patches (int, optional): number of masked patches.
                Defaults to 1.
            reduce_fn (str, optional): function for reduction before prediction.
                Defaults to "mean".
            seed (int, optional): random seed. Defaults to 42.
        """
        super().__init__()
        self.backbone_args = backbone_args
        self.projection_head_args = projection_head_args
        self.out_dim = out_dim
        self.feature_map_dimensions = feature_map_dimensions
        self.n_encoder_features = n_encoder_features
        self.min_patch_size = min_patch_size
        self.max_patch_size = max_patch_size
        self.n_patches = n_patches
        self.reduce_fn = reduce_fn
        self.seed = seed

        self.initialize_masker()
        self.initialize_mask_token()
        self.initialize_encoder()
        self.initialize_projection()
        self.initialize_last_layer()

    def initialize_masker(self):
        self.patch_masker_ = get_masker(
            model_type="generic_transformer",
            image_dimensions=self.feature_map_dimensions,
            min_patch_size=self.min_patch_size,
            max_patch_size=self.max_patch_size,
            n_patches=self.n_patches,
            n_features=self.n_encoder_features,
            seed=self.seed,
        )

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

    def initialize_mask_token(self):
        self.mask_token_ = torch.nn.Parameter(torch.rand(self.n_encoder_features))

    def reduce(self, X: torch.Tensor) -> torch.Tensor:
        if self.encoder_.use_class_token is True:
            class_token, X = X[:, 0], X[:, 1:]
        if self.reduce_fn == "cls":
            return class_token, X
        if self.reduce_fn == "mean":
            return X.mean(1), X
        elif self.reduce_fn == "max":
            return X.max(1), X

    @property
    def extra_tokens(self):
        return (self.encoder_.n_registers + self.encoder_.use_class_token,)

    def forward_training(self, X: torch.Tensor, mask: bool = False) -> torch.Tensor:
        # embed image
        X = self.encoder_.embedding(X)
        # mask image if necessary with mask_token
        if mask is True:
            X, mask_coords = self.patch_masker_(
                X,
                mask_vector=self.mask_token_,
                n_patches=self.n_patches,
                skip_n=self.extra_tokens,
            )
        # encode masked image
        for _, block in enumerate(self.encoder_.tbs.transformer_blocks):
            X = block(X)
        if self.encoder_.n_registers > 0:
            X = X[:, self.encoder_.n_registers :]
        X = self.last_layer_(
            torch.nn.functional.normalize(self.projection_(X), dim=-1, p=2)
        )
        reduced_X, X = self.reduce(X)
        if mask is True:
            return reduced_X, X, mask_coords
        else:
            return reduced_X, X

    def forward(self, X: torch.Tensor) -> torch.Tensor:
        # encode full image and return
        return self.forward_representation(X)

    def forward_representation(self, X: torch.Tensor) -> torch.Tensor:
        return self.encoder_(X)[0].permute(0, 2, 1)
