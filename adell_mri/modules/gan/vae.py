from collections.abc import Sequence

import torch

from adell_mri.modules.gan.autoencoder_kl import StridedAutoencoderKL


class VariationalAutoEncoder(StridedAutoencoderKL):
    """
    Variational strided autoencoder returning posterior log-variance.
    """

    def encode_distribution(
        self, x: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Encodes an image and returns the posterior parameters.

        Args:
            x (torch.Tensor): input tensor.

        Returns:
            tuple[torch.Tensor, torch.Tensor]: mean and log-variance.
        """
        mu, sigma = super().encode(x)
        logvar = 2 * torch.log(sigma)
        return mu, logvar

    def encode(self, x: torch.Tensor) -> torch.Tensor:
        """
        Encodes an image into the latent representation.

        Args:
            x (torch.Tensor): input tensor.

        Returns:
            torch.Tensor: latent mean.
        """
        return self.encode_distribution(x)[0]

    def sample(self, mu: torch.Tensor, logvar: torch.Tensor) -> torch.Tensor:
        """
        Samples from the posterior using the reparameterisation trick.

        Args:
            mu (torch.Tensor): posterior mean.
            logvar (torch.Tensor): posterior log-variance.

        Returns:
            torch.Tensor: sampled latent.
        """
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std

    def forward(
        self, x: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Runs the encoder and decoder.

        Args:
            x (torch.Tensor): input tensor.

        Returns:
            tuple[torch.Tensor, torch.Tensor, torch.Tensor]: reconstruction,
                posterior mean and posterior log-variance.
        """
        mu, logvar = self.encode_distribution(x)
        z = self.sample(mu, logvar)
        return self.decode(z), mu, logvar

    def reconstruct(self, x: torch.Tensor) -> torch.Tensor:
        """
        Reconstructs an input image using the posterior mean.

        Args:
            x (torch.Tensor): input tensor.

        Returns:
            torch.Tensor: reconstruction.
        """
        return self.decode(self.encode(x))

    @torch.inference_mode()
    def generate(self, shape: Sequence[int]) -> torch.Tensor:
        """
        Generates an image from a sample of the prior.

        Args:
            shape (Sequence[int]): latent shape, including the batch dimension.

        Returns:
            torch.Tensor: generated image.
        """
        device = next(self.parameters()).device
        z = torch.randn(*shape, device=device)
        return self.decode(z)
