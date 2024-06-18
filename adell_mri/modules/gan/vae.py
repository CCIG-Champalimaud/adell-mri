import torch
from .generator import Generator


class VAE(Generator):
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
        eps = torch.normal(0, 1, std.shape)
        return mu + eps * std

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

        # VAE-specific
        bottleneck = down_block_res_samples[-1]
        mu = self.predict_mu(bottleneck)
        logvar = self.predict_logvar(bottleneck)
        down_block_res_samples[-1] = self.sample(mu, logvar)

        # 4. mid
        h = self.middle_block(hidden_states=h, emb=class_emb, context=context)

        # 5. up
        for upsample_block in self.up_blocks:
            res_samples = down_block_res_samples[
                -len(upsample_block.resnets) :
            ]
            down_block_res_samples = down_block_res_samples[
                : -len(upsample_block.resnets)
            ]
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
