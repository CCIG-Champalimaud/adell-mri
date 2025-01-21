import math

import numpy as np
import torch

from ..layers.class_attention import EfficientConditioningAttentionBlock
from ..segmentation.unet import UNet, crop_to_size


def get_timestep_embedding(
    t: int, channels: int, max_period: int = 10000
) -> torch.Tensor:
    exponent = -math.log(max_period) * torch.arange(
        start=0, end=channels // 2, dtype=torch.float32, device=t.device
    )
    freqs = torch.exp(exponent / (channels // 2))

    args = t[:, None].float() * freqs[None, :]
    embedding = torch.cat([torch.cos(args), torch.sin(args)], dim=-1)

    # zero pad
    if channels % 2 == 1:
        embedding = torch.nn.functional.pad(embedding, (0, 1, 0, 0))

    return embedding


class DiffusionUNet(UNet):
    """
    U-Net for diffusion models

    Args:
        UNet (_type_): _description_
    """

    def __init__(
        self,
        t_dim: int = 256,
        classifier_free_guidance: bool = False,
        classifier_classes: int = 2,
        *args,
        **kwargs,
    ):
        if "n_channels" not in kwargs:
            raise Exception("n_channels must be defined")
        kwargs["n_classes"] = kwargs["n_channels"]
        kwargs["encoding_operations"] = None
        kwargs["bottleneck_classification"] = False
        kwargs["feature_conditioning"] = None
        super().__init__(*args, **kwargs)

        self.t_dim = t_dim
        self.classifier_free_guidance = classifier_free_guidance
        self.classifier_classes = classifier_classes

        self.init_eca()

    def init_eca(self):
        eca_op = EfficientConditioningAttentionBlock
        if self.classifier_free_guidance is True:
            # use this to map classes to self.t_dim vectors
            self.embedding = torch.nn.Embedding(self.classifier_classes, self.t_dim)
        else:
            self.embedding = None
        self.encoder_eca = torch.nn.ModuleList(
            [eca_op(self.t_dim, d, op_type="linear") for d in self.depth]
        )
        self.decoder_eca = torch.nn.ModuleList(
            [eca_op(self.t_dim, d, op_type="linear") for d in self.depth[:-1][::-1]]
        )
        self.link_eca = torch.nn.ModuleList(
            [eca_op(self.t_dim, d, op_type="linear") for d in self.depth[:-1][::-1]]
        )

    def get_final_layer(self, d: int) -> torch.nn.Module:
        """Returns the final layer.

        Args:
            d (int): depth.

        Returns:
            torch.nn.Module: final classification layer.
        """
        if self.spatial_dimensions == 2:
            op = torch.nn.Conv2d
        elif self.spatial_dimensions == 3:
            op = torch.nn.Conv3d
        # maps back to [0,1] range
        return torch.nn.Sequential(
            op(d, d, 3, padding=1), self.adn_fn(d), op(d, self.n_channels, 1)
        )

    def forward(
        self, X: torch.Tensor, t: torch.Tensor, cls: torch.Tensor = None
    ) -> torch.Tensor:
        """Forward pass for this class.

        Args:
            X (torch.Tensor)
            t (int): time step.
            cls (torch.Tensor): classification for guidance.

        Returns:
            torch.Tensor
        """

        t = t.to(X.device)
        if len(t.shape) == 1:
            t = t[:, None]
        t = get_timestep_embedding(t, self.t_dim)
        if cls is not None:
            if self.classifier_free_guidance is False:
                raise Exception(
                    "cls can only be defined if classifier_free_guidance is \
                        True in the constructor"
                )
            cls = self.embedding(cls.long())[:, None, :]
            t = t + cls
        encoding_out = []
        curr = X
        for (op, op_ds), eca in zip(self.encoding_operations, self.encoder_eca):
            curr = op(curr)
            curr = eca(curr, t)
            encoding_out.append(curr)
            curr = op_ds(curr)

        deep_outputs = []
        for i in range(len(self.decoding_operations)):
            op = self.decoding_operations[i]
            link_op = self.link_ops[i]
            up = self.upscale_ops[i]
            link_op_input = encoding_out[-i - 2]
            eca_link = self.link_eca[i]
            eca_decoder = self.decoder_eca[i]
            encoded = eca_link(link_op(link_op_input), t)
            curr = up(curr)
            sh = list(curr.shape)[2:]
            sh2 = list(encoded.shape)[2:]
            if np.prod(sh) < np.prod(sh2):
                encoded = crop_to_size(encoded, sh)
            if np.prod(sh) > np.prod(sh2):
                curr = crop_to_size(curr, sh2)
            curr = torch.concat((curr, encoded), dim=1)
            curr = eca_decoder(op(curr), t)
            deep_outputs.append(curr)

        curr = self.final_layer(curr)

        if self.deep_supervision is True:
            for i in range(len(deep_outputs)):
                o = deep_outputs[i]
                op = self.deep_supervision_ops[i]
                deep_outputs[i] = op(o)
            return curr, deep_outputs

        return curr
