import torch

from adell_mri.modules.gan.generator import Generator


class AutoEncoder(Generator):
    """
    A Generator subclass that decodes directly from a latent code without
    encoder skip connections.

    When ``latent_channels`` is specified, the bottleneck is projected to a
    smaller number of channels with a 1x1 convolution.
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
        self.latent_proj = None
        self.latent_unproj = None
        if latent_channels is not None:
            conv_op = (
                torch.nn.Conv2d if self.spatial_dims == 2 else torch.nn.Conv3d
            )
            self.latent_proj = conv_op(
                self.block_out_channels[-1], latent_channels, 1
            )
            self.latent_unproj = conv_op(
                latent_channels, self.block_out_channels[-1], 1
            )

    def encode(
        self,
        x: torch.Tensor,
        context: torch.Tensor | None = None,
        class_labels: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """
        Encodes an image into the latent representation.

        Args:
            x (torch.Tensor): input tensor (N, C, SpatialDims).
            context (torch.Tensor | None, optional): context tensor.
            class_labels (torch.Tensor | None, optional): class labels.

        Returns:
            torch.Tensor: latent tensor.
        """
        h = self.forward_bottleneck(
            x, context=context, class_labels=class_labels
        )
        if self.latent_proj is not None:
            h = self.latent_proj(h)
        return h

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
            torch.Tensor: generated image.
        """
        if self.latent_unproj is not None:
            z = self.latent_unproj(z)
        return super().decode(z, context=context, class_labels=class_labels)

    def forward(
        self,
        x: torch.Tensor,
        context: torch.Tensor | None = None,
        class_labels: torch.Tensor | None = None,
        **kwargs,
    ) -> torch.Tensor:
        """
        Runs the encoder and decoder.

        Args:
            x (torch.Tensor): input tensor (N, C, SpatialDims).
            context (torch.Tensor | None, optional): context tensor.
            class_labels (torch.Tensor | None, optional): class labels.
            **kwargs: additional arguments passed to :class:`Generator` when
                no compact latent is used.

        Returns:
            torch.Tensor: reconstructed image.
        """
        if self.latent_proj is None:
            return super().forward(
                x, context=context, class_labels=class_labels, **kwargs
            )
        z = self.encode(x, context=context, class_labels=class_labels)
        return self.decode(z, context=context, class_labels=class_labels)

    @torch.inference_mode()
    def generate(
        self,
        h: torch.Tensor,
        class_labels: torch.Tensor | None = None,
        context: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """
        Generates an image from a latent code.

        Args:
            h (torch.Tensor): latent code.
            class_labels (torch.Tensor, optional): class conditioning labels.
                Defaults to None.
            context (torch.Tensor, optional): additional conditioning tensor.
                Defaults to None.

        Returns:
            torch.Tensor: generated image.

        Raises:
            NotImplementedError: if ``no_skip_connection`` is False.
        """
        if self.no_skip_connection is False:
            raise NotImplementedError(
                f"no_skip_connection must be True for "
                f"{self.__class__.__name__} generation"
            )
        return self.decode(h, context=context, class_labels=class_labels)
