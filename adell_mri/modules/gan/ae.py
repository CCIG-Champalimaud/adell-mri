import torch

from adell_mri.modules.gan.generator import Generator


class AutoEncoder(Generator):
    """
    A Generator subclass that decodes directly from a latent code without
    encoder skip connections.
    """

    @torch.inference_mode()
    def generate(
        self,
        h: torch.Tensor,
        class_labels: torch.Tensor | None = None,
        context: torch.Tensor | None = None,
    ):
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
        class_emb = self.get_class_embeddings(h, class_labels)
        for upsample_block in self.up_blocks:
            res_samples = None
            h = upsample_block(
                hidden_states=h,
                emb=class_emb,
                res_hidden_states_list=res_samples,
                context=context,
            )
        h = self.out(h)
        h = self.out_activation(h)

        return h
