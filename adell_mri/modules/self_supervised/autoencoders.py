"""
Implements ViT-based autoencoder and masked autoencoder.
"""

from typing import Any, Dict, List, Tuple

import numpy as np
import torch

from ...custom_types import Size2dOr3d
from ..layers.conv_next import ConvNeXtV2Backbone
from ..layers.vit import LinearEmbedding, TransformerBlockStack


def random_masking(
    x: torch.Tensor, mask_ratio: float, rng: np.random.Generator
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Perform per-sample random masking by per-sample shuffling.
    Per-sample shuffling is done by argsort random noise.

    Adapted from https://github.com/facebookresearch/mae/blob/main/models_mae.py

    Args:
        x (torch.Tensor): tensor with shape [batch, n_tokens, embedding_size].
        mask_ratio (float): ratio of tokens to mask.
        rng (np.random.Generator): random number generator.
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
    """
    A ConvNeXt-based autoencoder.
    """
    def __init__(
        self,
        image_size: Size2dOr3d,
        in_channels: int,
        encoder_structure: List[Tuple[int, int, int, int]],
        decoder_structure: List[Tuple[int, int, int, int]],
        spatial_dim: int = 2,
        batch_ensemble: int = 0,
    ):
        """
        Args:
            image_size (Size2dOr3d): size of the image.
            in_channels (int): number of input channels.
            encoder_structure (List[Tuple[int, int, int, int]]): structure of the encoder.
            decoder_structure (List[Tuple[int, int, int, int]]): structure of the decoder.
            spatial_dim (int, optional): number of dimensions. Defaults to 2.
            batch_ensemble (int, optional): number of batch ensemble modules. Defaults to 0.
        """
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
        """
        Initialize the encoder.
        """
        self.encoder = ConvNeXtV2Backbone(
            spatial_dim=self.spatial_dim,
            in_channels=self.in_channels,
            structure=self.encoder_structure,
            maxpool_structure=self.maxpool_structure,
            batch_ensemble=self.batch_ensemble,
        )

    def conv(self, *args, **kwargs):
        """
        Initialize the convolutional layer.
        """
        if self.spatial_dim == 2:
            return torch.nn.Conv2d(*args, **kwargs)
        if self.spatial_dim == 3:
            return torch.nn.Conv3d(*args, **kwargs)

    def init_proj(self):
        """
        Initialize the projection layer.
        """
        input_channels = self.encoder_structure[-1][0]
        output_channels = self.decoder_structure[0][0]
        self.proj = self.conv(input_channels, output_channels, 1)

    def init_decoder(self):
        """
        Initialize the decoder.
        """
        self.encoder = ConvNeXtV2Backbone(
            spatial_dim=self.spatial_dim,
            in_channels=self.in_channels,
            structure=self.decoder_structure,
            maxpool_structure=[1 for _ in self.decoder_structure],
            batch_ensemble=self.batch_ensemble,
        )

    def init_pred(self):
        """
        Initialize the prediction layer.
        """
        input_channels = self.decoder_structure[-1][0]
        output_channels = int(np.prod(self.patch_size))
        self.pred = self.conv(input_channels, output_channels, 1)

    def forward(self, X):
        return X


class ViTAutoEncoder(torch.nn.Module):
    """
    ViT autoencoder.
    """

    def __init__(
        self,
        image_size: Size2dOr3d,
        patch_size: Size2dOr3d,
        in_channels: int,
        input_dim_size: int,
        encoder_args: Dict[str, Any],
        decoder_args: Dict[str, Any],
        embed_method: str = "linear",
        dropout_rate: float = 0.0,
        decoder_pred_ratio: float = 4.0,
    ):
        """
        Args:
            image_size (Size2dOr3d): size of the image.
            patch_size (Size2dOr3d): size of the patch.
            in_channels (int): number of input channels.
            input_dim_size (int): size of the input dimension.
            encoder_args (Dict[str, Any]): arguments for the encoder. Follows
                the signature for class:`TransformerBlockStack`.
            decoder_args (Dict[str, Any]): arguments for the decoder. Follows
                the signature for class:`TransformerBlockStack`.
            embed_method (str, optional): embedding method. Defaults to
                "linear".
            dropout_rate (float, optional): dropout rate. Defaults to 0.0.
            decoder_pred_ratio (float, optional): ratio of the decoder
                prediction layer to the decoder output. Defaults to 4.0.
        """
        super().__init__()
        self.image_size = image_size
        self.patch_size = patch_size
        self.in_channels = in_channels
        self.input_dim_size = input_dim_size
        self.encoder_args = encoder_args
        self.decoder_args = decoder_args
        self.embed_method = embed_method
        self.dropout_rate = dropout_rate
        self.decoder_pred_ratio = decoder_pred_ratio

        self.init_projection()
        self.init_encoder()
        self.init_positional_embedding()
        self.init_decoder()
        self.init_decoder_pred()

    def init_projection(self):
        """
        Initialize the projection layer.
        """
        self.proj = LinearEmbedding(
            self.image_size,
            self.patch_size,
            in_channels=self.in_channels,
            dropout_rate=self.dropout_rate,
            embed_method=self.embed_method,
            use_class_token=False,
        )
        self.n_patches = self.proj.n_patches
        self.n_features = self.proj.true_n_features

    def init_encoder(self):
        """
        Initialize the encoder.
        """
        self.encoder = TransformerBlockStack(
            number_of_blocks=self.encoder_args["number_of_blocks"],
            input_dim_primary=self.n_features,
            attention_dim=self.n_features,
            hidden_dim=self.encoder_args["hidden_dim"],
            n_heads=self.encoder_args["n_heads"],
            mlp_structure=self.encoder_args["mlp_structure"],
            dropout_rate=self.encoder_args.get("dropout_rate", 0.0),
        )

    def init_positional_embedding(self):
        """
        Defines the positional embedding as the embedding from the patching
        layer.
        """
        self.positional_embedding = self.proj.positional_embedding

    def init_decoder(self):
        """
        Initialize the decoder.
        """
        self.decoder = TransformerBlockStack(
            number_of_blocks=self.decoder_args["number_of_blocks"],
            input_dim_primary=self.n_features,
            attention_dim=self.n_features,
            hidden_dim=self.decoder_args["hidden_dim"],
            n_heads=self.decoder_args["n_heads"],
            mlp_structure=self.decoder_args["mlp_structure"],
            dropout_rate=self.decoder_args.get("dropout_rate", 0.0),
        )

    def init_decoder_pred(self):
        """
        Initialize the decoder prediction layer.
        """
        dps = int(self.decoder_pred_ratio * self.n_features)
        self.decoder_pred = torch.nn.Sequential(
            torch.nn.Linear(self.n_features, dps),
            torch.nn.GELU(),
            torch.nn.Linear(
                dps, int(np.prod(self.patch_size)) * self.in_channels
            ),
        )

    def forward(self, X: torch.Tensor) -> torch.Tensor:
        """
        Forward pass.

        Args:
            X (torch.Tensor): Input tensor.

        Returns:
            torch.Tensor: Output tensor.
        """
        X = self.proj(X)
        X = self.encoder(X) + self.positional_embedding
        X = self.decoder(X)
        X = self.decoder_pred(X)
        return X


class ViTMaskedAutoEncoder(ViTAutoEncoder):
    """
    A ViT masked autoencoder.
    """

    def __init__(
        self,
        image_size: Size2dOr3d,
        patch_size: Size2dOr3d,
        in_channels: int,
        input_dim_size: int,
        encoder_args: Dict[str, Any],
        decoder_args: Dict[str, Any],
        embed_method: str = "linear",
        dropout_rate: float = 0.0,
        decoder_pred_ratio: float = 4.0,
        mask_fraction: float = 0.3,
        seed: int = 42,
    ):
        """
        Args:
            image_size (Size2dOr3d): size of the image.
            patch_size (Size2dOr3d): size of the patch.
            in_channels (int): number of input channels.
            input_dim_size (int): size of the input dimension.
            encoder_args (Dict[str, Any]): arguments for the encoder. Follows
                the signature for class:`TransformerBlockStack`.
            decoder_args (Dict[str, Any]): arguments for the decoder. Follows
                the signature for class:`TransformerBlockStack`.
            embed_method (str, optional): embedding method. Defaults to
                "linear".
            dropout_rate (float, optional): dropout rate. Defaults to 0.0.
            decoder_pred_ratio (float, optional): ratio of the decoder
                prediction layer to the decoder output. Defaults to 4.0.
            mask_fraction (float, optional): fraction of patches to mask.
                Defaults to 0.3.
            seed (int, optional): random seed. Defaults to 42.
        """
        super().__init__(
            image_size=image_size,
            patch_size=patch_size,
            in_channels=in_channels,
            input_dim_size=input_dim_size,
            encoder_args=encoder_args,
            decoder_args=decoder_args,
            embed_method=embed_method,
            dropout_rate=dropout_rate,
            decoder_pred_ratio=decoder_pred_ratio,
        )
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
        """
        Initialize the mask token.
        """
        self.mask_token_scale = torch.nn.Parameter(torch.ones(1))
        self.mask_token = torch.nn.Parameter(
            torch.randn(1, 1, self.n_features) * 0.02
        )

    def forward(self, X: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Forward pass.

        Args:
            X (torch.Tensor): Input tensor.

        Returns:
            tuple[torch.Tensor, torch.Tensor]: Tuple of encoded and decoded
                tensors.
        """
        # based on https://github.com/facebookresearch/mae/blob/main/models_mae.py

        batch_size, _, height, width = X.shape

        # Project input to embedding space
        X_embed = self.proj(X)  # [batch_size, seq_len, embed_dim]

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

        mask_tokens = self.mask_token_scale * self.mask_token.repeat(
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

        X_unshuffled = X_unshuffled + self.positional_embedding

        # Decode
        decoder_output = self.decoder(X_unshuffled)
        # Handle case where decoder returns a tuple (output, attention_weights)
        if isinstance(decoder_output, tuple):
            X_decoded = decoder_output[0]
        else:
            X_decoded = decoder_output
        X_reconstructed = self.decoder_pred(X_decoded)  # [batch_size, seq_len, n_features]

        # Reshape to [batch_size, in_channels, height, width]
        batch_size = X_reconstructed.shape[0]
        n_patches = (height // self.patch_size[0]) * (
            width // self.patch_size[1]
        )

        # Reshape to [batch_size, n_patches, patch_h, patch_w, in_channels]
        X_reconstructed = X_reconstructed.view(
            batch_size,
            n_patches,
            self.patch_size[0],
            self.patch_size[1],
            self.in_channels,
        )

        # Reshape to [batch_size, in_channels, height, width]
        X_reconstructed = X_reconstructed.permute(0, 4, 1, 2, 3).contiguous()
        X_reconstructed = X_reconstructed.view(
            batch_size,
            self.in_channels,
            height,
            width,
        )

        return X_reconstructed, mask
