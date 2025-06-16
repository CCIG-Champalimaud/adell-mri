from typing import Any, Dict, List, Tuple

import numpy as np
import torch

from ...custom_types import Size2dOr3d
from ..layers.conv_next import ConvNeXtV2Backbone
from ..layers.vit import LinearEmbedding, TransformerBlockStack


def random_masking(
    x: torch.Tensor, mask_ratio: float, rng: np.random.RandomState
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Perform per-sample random masking by per-sample shuffling.
    Per-sample shuffling is done by argsort random noise.

    Adapted from https://github.com/facebookresearch/mae/blob/main/models_mae.py

    Args:
        x (torch.Tensor): tensor with shape [batch, n_tokens, embedding_size].
        mask_ratio (float): ratio of tokens to mask.
        rng (np.random.RandomState): random number generator.
    """
    N, L, D = x.shape  # batch, length, dim
    len_keep = int(L * (1 - mask_ratio))

    noise = torch.as_tensor(
        rng.uniform(size=[N, L]), device=x.device
    )  # noise in [0, 1]

    # sort noise for each sample
    ids_shuffle = torch.argsort(
        noise, dim=1, stable=True
    )  # ascend: small is keep, large is remove
    ids_restore = torch.argsort(ids_shuffle, dim=1, stable=True)
    ids_restore = ids_restore.to(x.device)  # Ensure it's on the same device as input

    # keep the first subset
    ids_keep = ids_shuffle[:, :len_keep]
    x_masked = torch.gather(
        x, dim=1, index=ids_keep.unsqueeze(-1).repeat(1, 1, D)
    )

    # generate the binary mask: 0 is keep, 1 is remove
    mask = torch.ones([N, L], device=x.device)
    mask[:, :len_keep] = 0
    # unshuffle to get the binary mask
    mask = torch.gather(mask, dim=1, index=ids_restore)

    return x_masked, mask, ids_restore


class ConvNeXtAutoEncoder(torch.nn.Module):
    def __init__(
        self,
        image_size: Size2dOr3d,
        in_channels: int,
        encoder_structure: List[Tuple[int, int, int, int]],
        decoder_structure: List[Tuple[int, int, int, int]],
        spatial_dim: int = 2,
        batch_ensemble: int = 0,
    ):
        super().__init__()
        self.image_size = image_size
        self.in_channels = in_channels
        self.encoder_structure = encoder_structure
        self.decoder_structure = decoder_structure
        self.maxpool_structure = [2 for _ in self.encoder_structure]
        self.spatial_dim = spatial_dim
        self.batch_ensemble = batch_ensemble

        self.patch_size = [
            x // (2 ** len(self.maxpool_structure)) for x in self.image_size
        ]

    def init_encoder(self):
        self.encoder = ConvNeXtV2Backbone(
            spatial_dim=self.spatial_dim,
            in_channels=self.in_channels,
            structure=self.encoder_structure,
            maxpool_structure=self.maxpool_structure,
            batch_ensemble=self.batch_ensemble,
        )

    def conv(self, *args, **kwargs):
        if self.spatial_dim == 2:
            return torch.nn.Conv2d(*args, **kwargs)
        if self.spatial_dim == 3:
            return torch.nn.Conv3d(*args, **kwargs)

    def init_proj(self):
        input_channels = self.encoder_structure[-1][0]
        output_channels = self.decoder_structure[0][0]
        self.proj = self.conv(input_channels, output_channels, 1)

    def init_decoder(self):
        self.encoder = ConvNeXtV2Backbone(
            spatial_dim=self.spatial_dim,
            in_channels=self.in_channels,
            structure=self.decoder_structure,
            maxpool_structure=[1 for _ in self.decoder_structure],
            batch_ensemble=self.batch_ensemble,
        )

    def init_pred(self):
        input_channels = self.decoder_structure[-1][0]
        output_channels = int(np.prod(self.patch_size))
        self.pred = self.conv(input_channels, output_channels, 1)

    def forward(self, X):
        return X


class ViTAutoEncoder(torch.nn.Module):
    def __init__(
        self,
        image_size: int,
        patch_size: int,
        n_channels: int,
        input_dim_size: int,
        encoder_args: Dict[str, Any],
        decoder_args: Dict[str, Any],
        embed_method: str = "linear",
        dropout_rate: float = 0.0,
    ):
        super().__init__()
        self.image_size = image_size
        self.patch_size = patch_size
        self.n_channels = n_channels
        self.input_dim_size = input_dim_size
        self.encoder_args = encoder_args
        self.decoder_args = decoder_args
        self.embed_method = embed_method
        self.dropout_rate = dropout_rate

        self.init_projection()
        self.init_encoder()
        self.init_positional_embedding()
        self.init_decoder()
        self.init_decoder_pred()

    def init_projection(self):
        self.proj = LinearEmbedding(
            self.image_size,
            self.patch_size,
            n_channels=self.n_channels,
            dropout_rate=self.dropout_rate,
            embed_method=self.embed_method,
            use_class_token=False,
        )
        self.n_patches = self.proj.n_patches
        self.n_features = self.proj.true_n_features

    def init_encoder(self):
        self.encoder = TransformerBlockStack(
            number_of_blocks=self.encoder_args["num_layers"],
            input_dim_primary=self.n_features,
            attention_dim=self.n_features,
            hidden_dim=self.encoder_args["mlp_dim"],
            n_heads=self.encoder_args["num_heads"],
            mlp_structure=[self.encoder_args["mlp_dim"]],
            dropout_rate=self.encoder_args.get("dropout", 0.0),
        )

    def init_positional_embedding(self):
        self.positional_embedding = torch.nn.Parameter(
            torch.rand(1, self.n_patches, self.n_features)
        )
        torch.nn.init.trunc_normal_(
            self.positional_embedding, mean=0.0, std=0.02, a=-2.0, b=2.0
        )

    def init_decoder(self):
        self.decoder = TransformerBlockStack(
            number_of_blocks=self.decoder_args["num_layers"],
            input_dim_primary=self.n_features,
            attention_dim=self.n_features,
            hidden_dim=self.decoder_args["mlp_dim"],
            n_heads=self.decoder_args["num_heads"],
            mlp_structure=[self.decoder_args["mlp_dim"]],
            dropout_rate=self.decoder_args.get("dropout", 0.0),
        )

    def init_decoder_pred(self):
        self.decoder_pred = torch.nn.Linear(
            self.n_features, int(np.prod(self.patch_size)) * self.n_channels
        )

    def forward(self, X: torch.Tensor) -> torch.Tensor:
        X = self.proj(X)
        X = self.encoder(X) + self.positional_embedding
        X = self.decoder(X)
        return X


class ViTMaskedAutoEncoder(ViTAutoEncoder):
    """
    A ViT masked autoencoder.
    """

    def __init__(
        self,
        image_size: int,
        patch_size: int,
        n_channels: int,
        input_dim_size: int,
        encoder_args: Dict[str, Any],
        decoder_args: Dict[str, Any],
        embed_method: str = "linear",
        dropout_rate: float = 0.0,
        mask_fraction: float = 0.3,
        seed: int = 42,
    ):
        super().__init__(
            image_size=image_size,
            patch_size=patch_size,
            n_channels=n_channels,
            input_dim_size=input_dim_size,
            encoder_args=encoder_args,
            decoder_args=decoder_args,
            embed_method=embed_method,
            dropout_rate=dropout_rate,
        )
        self.image_size = image_size
        self.patch_size = patch_size
        self.n_channels = n_channels
        self.input_dim_size = input_dim_size
        self.encoder_args = encoder_args
        self.decoder_args = decoder_args
        self.embed_method = embed_method
        self.dropout_rate = dropout_rate
        self.mask_fraction = mask_fraction
        self.seed = seed

        self.init_projection()
        self.init_encoder()
        self.init_positional_embedding()
        self.init_mask_token()
        self.init_decoder()
        self.init_decoder_pred()

        self.rng = np.random.default_rng(self.seed)

    def init_mask_token(self):
        self.mask_token = torch.nn.Parameter(torch.rand(1, 1, self.n_features))
        torch.nn.init.trunc_normal_(
            self.mask_token, mean=0.0, std=0.02, a=-2.0, b=2.0
        )

    def forward(self, X: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        # based on https://github.com/facebookresearch/mae/blob/main/models_mae.py

        # Project input to embedding space
        X_embed = self.proj(X)  # [batch_size, seq_len, embed_dim]

        # Add positional embeddings
        X_embed = X_embed + self.positional_embedding

        # Apply random masking
        X_masked, mask, ids_restore = random_masking(
            X_embed, self.mask_fraction, self.rng
        )

        # Encode visible tokens
        encoder_output = self.encoder(X_masked)  # [batch_size, num_visible, embed_dim]
        # Handle case where encoder returns a tuple (output, attention_weights)
        if isinstance(encoder_output, tuple):
            X_encoded = encoder_output[0]
        else:
            X_encoded = encoder_output

        mask_tokens = self.mask_token.repeat(
            X_encoded.shape[0], ids_restore.shape[1] - X_encoded.shape[1], 1
        )

        # Concatenate encoded visible tokens with mask tokens
        X_full = torch.cat([X_encoded, mask_tokens], dim=1)

        # Unshuffle to original order
        X_unshuffled = torch.gather(
            X_full, 
            dim=1, 
            index=ids_restore.unsqueeze(-1).expand(-1, -1, X_full.shape[2])
        )

        # Decode
        decoder_output = self.decoder(X_unshuffled)
        # Handle case where decoder returns a tuple (output, attention_weights)
        if isinstance(decoder_output, tuple):
            X_decoded = decoder_output[0]
        else:
            X_decoded = decoder_output
        X_reconstructed = self.decoder_pred(X_decoded)  # [batch_size, seq_len, n_features]

        # Reshape to [batch_size, n_channels, height, width]
        batch_size = X_reconstructed.shape[0]
        height = width = int(np.sqrt(X_reconstructed.shape[1]))  # Assuming square patches
        n_patches = height * width

        # Get patch dimensions
        if isinstance(self.patch_size, tuple):
            patch_h, patch_w = self.patch_size
        else:
            patch_h = patch_w = self.patch_size

        # Reshape to [batch_size, n_patches, patch_h, patch_w, n_channels]
        X_reconstructed = X_reconstructed.view(
            batch_size, 
            n_patches, 
            patch_h, 
            patch_w, 
            self.n_channels
        )

        # Reshape to [batch_size, n_channels, height * patch_h, width * patch_w]
        X_reconstructed = X_reconstructed.permute(0, 4, 1, 2, 3).contiguous()
        X_reconstructed = X_reconstructed.view(
            batch_size,
            self.n_channels,
            height * patch_h,
            width * patch_w
        )

        return X_reconstructed, mask
