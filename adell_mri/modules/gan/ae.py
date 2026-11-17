from collections.abc import Sequence

import torch

from adell_mri.modules.gan.autoencoder_kl import StridedAutoencoderKL


class AutoEncoder(StridedAutoencoderKL):
    """
    Deterministic strided autoencoder.
    """

    def encode(self, x: torch.Tensor) -> torch.Tensor:
        """
        Encodes an image into the latent representation.

        Args:
            x (torch.Tensor): input tensor.

        Returns:
            torch.Tensor: latent mean.
        """
        return super().encode(x)[0]

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Runs the encoder and decoder.

        Args:
            x (torch.Tensor): input tensor.

        Returns:
            torch.Tensor: reconstruction.
        """
        return self.decode(self.encode(x))

    def reconstruct(self, x: torch.Tensor) -> torch.Tensor:
        """
        Reconstructs an input image.

        Args:
            x (torch.Tensor): input tensor.

        Returns:
            torch.Tensor: reconstruction.
        """
        return self.forward(x)

    @torch.inference_mode()
    def generate(self, shape: Sequence[int]) -> torch.Tensor:
        """
        Generates an image from a latent sample.

        Args:
            shape (Sequence[int]): latent shape, including the batch dimension.

        Returns:
            torch.Tensor: generated image.
        """
        device = next(self.parameters()).device
        z = torch.randn(*shape, device=device)
        return self.decode(z)
