from collections.abc import Callable

import torch

from adell_mri.modules.gan.generator import Generator


class VariationalAutoEncoder(Generator):
    """
    A variational autoencoder built on top of :class:`Generator`.

    The posterior is predicted from the bottleneck with channel-wise linear
    layers. When ``latent_channels`` is specified, the sampled bottleneck is
    projected to a smaller number of channels with a 1x1 convolution, which
    keeps the latent compact for downstream generative modelling.
    """

    def __init__(
        self, *args, latent_channels: int | None = None, **kwargs
    ) -> None:
        """
        Args:
            *args: positional arguments passed to :class:`Generator`.
            latent_channels (int | None, optional): number of channels in the
                compact latent. If None, the bottleneck is used directly.
                Defaults to None.
            **kwargs: keyword arguments passed to :class:`Generator`.

        Raises:
            ValueError: if ``latent_channels`` is specified and
                ``no_skip_connection`` is False.
        """
        super().__init__(*args, **kwargs)
        if latent_channels is not None and self.no_skip_connection is False:
            raise ValueError("latent_channels requires no_skip_connection=True")
        self.latent_channels = latent_channels
        self.bottleneck_dim = self.block_out_channels[-1]
        self.predict_mu = torch.nn.Linear(
            self.bottleneck_dim, self.bottleneck_dim
        )
        self.predict_logvar = torch.nn.Linear(
            self.bottleneck_dim, self.bottleneck_dim
        )
        self.quant_conv = None
        self.post_quant_conv = None
        if latent_channels is not None:
            conv_op = (
                torch.nn.Conv2d if self.spatial_dims == 2 else torch.nn.Conv3d
            )
            self.quant_conv = conv_op(self.bottleneck_dim, latent_channels, 1)
            self.post_quant_conv = conv_op(
                latent_channels, self.bottleneck_dim, 1
            )

    def sample(self, mu: torch.Tensor, logvar: torch.Tensor) -> torch.Tensor:
        """
        Samples from the posterior using the reparameterisation trick.

        Args:
            mu (torch.Tensor): mean tensor.
            logvar (torch.Tensor): log variance tensor.

        Returns:
            torch.Tensor: sampled tensor.
        """
        std = torch.exp(0.5 * logvar)
        eps = torch.normal(0, 1, std.shape).to(std)
        return mu + eps * std

    def apply_to_channels_as_last(
        self, x: torch.Tensor, fn: Callable
    ) -> torch.Tensor:
        """
        Applies function to the permuted version of a tensor such that the 1st
        channel is placed at the last position, the function is applied and the
        input format is returned.

        Args:
            x (torch.Tensor): input tensor.
            fn (Callable): callable function.

        Returns:
            torch.Tensor: output tensor.
        """
        dims = [i for i in range(len(x.shape))]
        a = [0, *dims[2:], 1]
        b = [0, -1, *dims[1:-1]]
        return torch.permute(fn(torch.permute(x, a)), b)

    def posterior(self, h: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Predicts the posterior parameters from the bottleneck.

        Args:
            h (torch.Tensor): bottleneck features.

        Returns:
            tuple[torch.Tensor, torch.Tensor]: mean and log variance.
        """
        mu = self.apply_to_channels_as_last(h, self.predict_mu)
        logvar = self.apply_to_channels_as_last(h, self.predict_logvar)
        return mu, logvar

    def encode_distribution(
        self,
        x: torch.Tensor,
        context: torch.Tensor | None = None,
        class_labels: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Encodes an image and returns the posterior parameters.

        Args:
            x (torch.Tensor): input tensor (N, C, SpatialDims).
            context (torch.Tensor | None, optional): context tensor.
            class_labels (torch.Tensor | None, optional): class labels.

        Returns:
            tuple[torch.Tensor, torch.Tensor]: mean and log variance.
        """
        h = self.forward_bottleneck(
            x, context=context, class_labels=class_labels
        )
        return self.posterior(h)

    def encode(
        self,
        x: torch.Tensor,
        context: torch.Tensor | None = None,
        class_labels: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """
        Encodes an image into the latent representation.

        The posterior mean is used, so that repeated calls return the same
        latent for the same input. Use :meth:`encode_distribution` if the
        posterior parameters are needed.

        Args:
            x (torch.Tensor): input tensor (N, C, SpatialDims).
            context (torch.Tensor | None, optional): context tensor.
            class_labels (torch.Tensor | None, optional): class labels.

        Returns:
            torch.Tensor: latent tensor.
        """
        mu, _ = self.encode_distribution(
            x, context=context, class_labels=class_labels
        )
        if self.quant_conv is not None:
            mu = self.quant_conv(mu)
        return mu

    def decode(
        self,
        z: torch.Tensor,
        context: torch.Tensor | None = None,
        class_labels: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """
        Decodes a latent representation into an image.

        Args:
            z (torch.Tensor): latent tensor.
            context (torch.Tensor | None, optional): context tensor.
            class_labels (torch.Tensor | None, optional): class labels.

        Returns:
            torch.Tensor: reconstructed image.
        """
        if self.post_quant_conv is not None:
            z = self.post_quant_conv(z)
        return super().decode(z, context=context, class_labels=class_labels)

    def forward(
        self,
        x: torch.Tensor,
        context: torch.Tensor | None = None,
        class_labels: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Runs the encoder and decoder and returns the reconstruction and the
        posterior parameters.

        Args:
            x (torch.Tensor): input tensor (N, C, SpatialDims).
            context (torch.Tensor | None, optional): context tensor.
            class_labels (torch.Tensor | None, optional): class labels.

        Returns:
            tuple[torch.Tensor, torch.Tensor, torch.Tensor]: reconstructed
                image, posterior mean and posterior log variance.
        """
        if context is not None and self.with_conditioning is False:
            raise ValueError(
                "model should have with_conditioning = True if context is provided"
            )
        # 1. class
        class_emb = None
        if self.num_class_embeds is not None:
            if class_labels is None:
                raise ValueError(
                    "class_labels should be provided when num_class_embeds > 0"
                )
            class_emb = self.class_embedding(class_labels)
            class_emb = class_emb.to(dtype=x.dtype)

        # 2. initial convolution
        h = self.conv_in(x)

        # 3. down
        down_block_res_samples: list[torch.Tensor] = [h]
        for downsample_block in self.down_blocks:
            h, res_samples = downsample_block(
                hidden_states=h, emb=class_emb, context=context
            )
            down_block_res_samples.extend(res_samples)

        # 4. mid
        h = self.middle_block(hidden_states=h, emb=class_emb, context=context)

        # VAE-specific
        mu, logvar = self.posterior(h)
        h = self.sample(mu, logvar)
        self.last_bottleneck_shape = h.shape
        if self.quant_conv is not None:
            h = self.quant_conv(h)
            h = self.post_quant_conv(h)

        # 5. up
        h, _ = self._decode_blocks(
            h,
            class_emb,
            context,
            down_block_res_samples=down_block_res_samples,
        )

        return h, mu, logvar

    @torch.inference_mode()
    def generate(
        self,
        shape: list[int] | tuple[int],
        class_labels: torch.Tensor | None = None,
        context: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """
        Generates an image from a sample of the prior.

        Args:
            shape (list[int] | tuple[int]): shape of the latent sample,
                including the batch dimension. If ``latent_channels`` is
                specified, the channel dimension refers to the compact latent.
                If None, the shape of the last forward pass is used.
            class_labels (torch.Tensor, optional): class conditioning labels.
                Defaults to None.
            context (torch.Tensor, optional): additional conditioning tensor.
                Defaults to None.

        Returns:
            torch.Tensor: generated image.

        Raises:
            NotImplementedError: if ``no_skip_connection`` is False.
            ValueError: if ``shape`` is None and no forward pass has been run.
        """
        if self.no_skip_connection is False:
            raise NotImplementedError(
                f"no_skip_connection must be True for "
                f"{self.__class__.__name__} generation"
            )
        if shape is None:
            if self.last_bottleneck_shape is None:
                raise ValueError(
                    "At least one forward pass required for generation with shape=None"
                )
            shape = self.last_bottleneck_shape
        shape = list(shape)
        if self.latent_channels is not None:
            shape[1] = self.bottleneck_dim
        device = next(self.parameters()).device
        mu = torch.zeros(*shape, device=device)
        logvar = torch.zeros(*shape, device=device)
        h = self.sample(mu, logvar)
        class_emb = self.get_class_embeddings(h, class_labels)
        h, _ = self._decode_blocks(h, class_emb, context)
        return h
