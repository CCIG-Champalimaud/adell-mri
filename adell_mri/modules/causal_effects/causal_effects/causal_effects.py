from typing import List

import torch

from ...layers.standard_blocks import VGGConvolution3d, VGGDeconvolution3d


class VGGAutoencoder(torch.nn.Module):
    """
    Very simple and naive VGG-like autoencoder.
    """

    def __init__(
        self,
        spatial_dimensions: int = 3,
        n_channels: int = 1,
        feature_extraction=None,
        resnet_structure=None,
        maxpool_structure=None,
        adn_fn=None,
        res_type=None,
        classification_structure: List[int] = [512, 512, 512],
        batch_ensemble: int = 0,
    ):
        """
        Args:
            spatial_dimensions (int, optional): number of spatial dimensions.
                Defaults to 3.
            n_channels (int, optional): number of input channels. Defaults to
                1.
            n_classes (int, optional): number of classes. Defaults to 2.
            classification_structure (List[int], optional): structure of the
                classifier. Defaults to [512,512,512].
            batch_ensemble (int, optional): number of batch ensemble modules.
                Defautls to 0.
        """
        super().__init__()
        self.in_channels = n_channels

        self.encoder_block = self.encoder(in_channels=self.in_channels)
        self.decoder_block = self.decoder(in_channels=self.in_channels)

    def encoder(self, in_channels):
        encoder = torch.nn.Sequential(
            VGGConvolution3d(input_channels=in_channels, first_depth=64),
            VGGConvolution3d(input_channels=128, first_depth=128),
            VGGConvolution3d(input_channels=256, first_depth=256),
        )
        return encoder

    def decoder(self, in_channels):
        decoder = torch.nn.Sequential(
            VGGDeconvolution3d(input_channels=512, first_depth=256),
            VGGDeconvolution3d(input_channels=256, first_depth=128),
            VGGDeconvolution3d(
                input_channels=128,
                first_depth=64,
                last=True,
                last_channels=in_channels,
            ),
        )
        return decoder

    def forward_features(
        self,
        X: torch.Tensor,
    ) -> torch.Tensor:
        """Forward method for features only.

        Args:
            X (torch.Tensor): input tensor
            batch_idx (int, optional): uses a specific batch ensemble
                transformation. Defaults to None (usually random).

        Returns:
            torch.Tensor: output (latent features)
        """
        return self.forward(X, return_features=True)

    def forward(
        self,
        X: torch.Tensor,
        return_features: bool = False,
    ) -> torch.Tensor:
        """Forward method.

        Args:
            X (torch.Tensor): input tensor
            return_features (bool, optional): returns the features before
                applying classification layer. Defaults to False.
            batch_idx (int, optional): uses a specific batch ensemble
                transformation. Defaults to None (usually random).

        Returns:
            torch.Tensor: output (reconstruction)
        """
        X = self.encoder_block(X)
        if return_features is True:
            return X

        return self.decoder_block(X)
