from typing import Callable

import torch

from adell_mri.modules.gan.generator import Generator


class VariationalAutoEncoder(Generator):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.bottleneck_dim = self.block_out_channels[-1]
        self.predict_mu = torch.nn.Linear(
            self.bottleneck_dim, self.bottleneck_dim
        )
        self.predict_logvar = torch.nn.Linear(
            self.bottleneck_dim, self.bottleneck_dim
        )

    def sample(self, mu: torch.Tensor, logvar: torch.Tensor) -> torch.Tensor:
        """
        Args:
            mu (torch.Tensor): mean tensor (N, C).
            logvar (torch.Tensor): log variance tensor (N, C).
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

    def forward(
        self,
        x: torch.Tensor,
        context: torch.Tensor | None = None,
        class_labels: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """
        Args:
            x: input tensor (N, C, SpatialDims).
            timesteps: timestep tensor (N,).
            context: context tensor (N, 1, ContextDim).
            class_labels: context tensor (N, ).
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
            for residual in res_samples:
                down_block_res_samples.append(residual)

        # 4. mid
        h = self.middle_block(hidden_states=h, emb=class_emb, context=context)

        # VAE-specific
        bottleneck = h
        mu = self.apply_to_channels_as_last(bottleneck, self.predict_mu)
        logvar = self.apply_to_channels_as_last(bottleneck, self.predict_logvar)
        h = self.sample(mu, logvar)
        self.last_bottleneck_shape = h.shape

        # 5. up
        for upsample_block in self.up_blocks:
            res_samples = down_block_res_samples[-len(upsample_block.resnets) :]
            down_block_res_samples = down_block_res_samples[
                : -len(upsample_block.resnets)
            ]
            if self.no_skip_connection:
                res_samples = None
            h = upsample_block(
                hidden_states=h,
                emb=class_emb,
                res_hidden_states_list=res_samples,
                context=context,
            )

        # 6. output block
        h = self.out(h)
        h = self.out_activation(h)

        return h, mu, logvar

    @torch.inference_mode()
    def generate(
        self,
        shape: list[int] | tuple[int],
        class_labels: torch.Tensor | None = None,
        context: torch.Tensor | None = None,
    ):
        if self.no_skip_connection is False:
            raise NotImplementedError(
                f"no_skip_connection must be False for {self.__name__} generation"
            )
        if shape is None:
            if self.last_bottleneck_shape is None:
                raise ValueError(
                    "At least one forward pass required for generation with shape=None"
                )
            shape = self.last_bottleneck_shape
        mu = torch.zeros(*shape)
        logvar = torch.zeros(*shape)
        h = self.sample(mu, logvar)
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
