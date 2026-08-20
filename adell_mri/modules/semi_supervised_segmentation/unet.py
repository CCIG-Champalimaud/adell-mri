"""
UNet implementation with some minor modifications to support semi-supervised
learning (i.e. return features).
"""

import numpy as np
import torch
import torch.nn.functional as F

from adell_mri.modules.segmentation.unet import UNet, crop_to_size


class UNetSemiSL(UNet):
    """
    Identical to UNet but supports returning features of the last layer through
    `return_features`.
    """

    def __init__(self, *args, **kwargs):
        """
        Initializes the UNet module.
        """
        super().__init__(*args, **kwargs)

        self.init_linear_transformation()

    def init_linear_transformation(self, *args, **kwargs):
        """
        Initialise the linear transformation layer for predictions.
        """
        if self.spatial_dimensions == 2:
            self.linear_transformation = torch.nn.Conv2d(
                self.depth[0],
                self.depth[0],
                kernel_size=1,
            )
        elif self.spatial_dimensions == 3:
            self.linear_transformation = torch.nn.Conv3d(
                self.depth[0],
                self.depth[0],
                kernel_size=1,
            )

    def _run_encoder(
        self,
        X: torch.Tensor,
        X_skip_layer: torch.Tensor = None,
        X_feature_conditioning: torch.Tensor = None,
    ) -> tuple[
        list[torch.Tensor],
        torch.Tensor,
        torch.Tensor | None,
        torch.Tensor | None,
    ]:
        """
        Runs the encoder, normalising any conditioning inputs.

        Args:
            X (torch.Tensor): input tensor.
            X_skip_layer (torch.Tensor, optional): skip conditioning input.
            X_feature_conditioning (torch.Tensor, optional): feature
                conditioning input.

        Returns:
            tuple[list[torch.Tensor], torch.Tensor, torch.Tensor | None,
                torch.Tensor | None]: encoding outputs, bottleneck and the
                (possibly channel-expanded) conditioning tensors.
        """
        if X_skip_layer is not None and len(X_skip_layer.shape) < len(X.shape):
            X_skip_layer = X_skip_layer.unsqueeze(1)

        if X_feature_conditioning is not None:
            X_feature_conditioning = X_feature_conditioning - self.f_mean
            X_feature_conditioning = X_feature_conditioning / self.f_std

        encoding_out = []
        curr = X
        for op, op_ds in self.encoding_operations:
            curr = op(curr)
            encoding_out.append(curr)
            curr = op_ds(curr)
        return encoding_out, curr, X_skip_layer, X_feature_conditioning

    def _run_decoder(
        self,
        encoding_out: list[torch.Tensor],
        curr: torch.Tensor,
        X_skip_layer: torch.Tensor = None,
        X_feature_conditioning: torch.Tensor = None,
    ) -> tuple[torch.Tensor, list[torch.Tensor]]:
        """
        Runs the decoder given the encoding outputs and the bottleneck.

        Args:
            encoding_out (list[torch.Tensor]): encoder outputs at each level.
            curr (torch.Tensor): bottleneck tensor.
            X_skip_layer (torch.Tensor, optional): skip conditioning input.
            X_feature_conditioning (torch.Tensor, optional): feature
                conditioning input.

        Returns:
            tuple[torch.Tensor, list[torch.Tensor]]: final features and the
                list of deep outputs.
        """
        deep_outputs = []
        for i in range(len(self.decoding_operations)):
            op = self.decoding_operations[i]
            link_op = self.link_ops[i]
            up = self.upscale_ops[i]
            if X_skip_layer is not None:
                S = encoding_out[-i - 2].shape[2:]
                xfl = F.interpolate(X_skip_layer, S, mode="nearest")
                link_op_input = torch.cat([encoding_out[-i - 2], xfl], axis=1)
            else:
                link_op_input = encoding_out[-i - 2]
            encoded = link_op(link_op_input)
            if X_feature_conditioning is not None:
                feat_op = self.feature_conditioning_ops[i]
                transformed_features = feat_op(X_feature_conditioning)
                transformed_features = self.unsqueeze_to_dim(
                    transformed_features, encoded
                )
                encoded = torch.multiply(encoded, transformed_features)
            curr = up(curr)
            sh = list(curr.shape)[2:]
            sh2 = list(encoded.shape)[2:]
            if np.prod(sh) < np.prod(sh2):
                encoded = crop_to_size(encoded, sh)
            if np.prod(sh) > np.prod(sh2):
                curr = crop_to_size(curr, sh2)
            curr = torch.concat((curr, encoded), dim=1)
            curr = op(curr)
            deep_outputs.append(curr)
        return curr, deep_outputs

    def forward(
        self,
        X: torch.Tensor,
        X_skip_layer: torch.Tensor = None,
        X_feature_conditioning: torch.Tensor = None,
        return_features=False,
        return_bottleneck=False,
        return_logits=False,
    ) -> torch.Tensor:
        """
        Forward pass for this class.

        Args:
            X (torch.Tensor)

        Returns:
            torch.Tensor
        """
        (
            encoding_out,
            curr,
            X_skip_layer,
            X_feature_conditioning,
        ) = self._run_encoder(X, X_skip_layer, X_feature_conditioning)
        bottleneck = curr
        if return_bottleneck is True:
            return None, None, bottleneck
        elif self.encoder_only is True:
            return bottleneck

        curr, deep_outputs = self._run_decoder(
            encoding_out, curr, X_skip_layer, X_feature_conditioning
        )

        final_features = curr

        if return_logits is True:
            curr = self.final_layer[:-1](curr)
        else:
            curr = self.final_layer(curr)
        if return_features is True:
            return curr, final_features, bottleneck

        if self.bottleneck_classification is True:
            bottleneck = bottleneck.flatten(start_dim=2).max(-1).values
            bn_out = self.bottleneck_classifier(bottleneck)
        else:
            bn_out = None

        if self.deep_supervision is True:
            for i in range(len(deep_outputs)):
                o = deep_outputs[i]
                op = self.deep_supervision_ops[i]
                deep_outputs[i] = op(o)
            return curr, bn_out, deep_outputs

        return curr, bn_out

    def forward_features(
        self,
        X: torch.Tensor,
        X_skip_layer: torch.Tensor = None,
        X_feature_conditioning: torch.Tensor = None,
        apply_linear_transformation: bool = False,
    ) -> torch.Tensor:
        """
        Forward pass for this class.

                Args:
                    X (torch.Tensor)

                Returns:
                    torch.Tensor
        """
        (
            encoding_out,
            curr,
            X_skip_layer,
            X_feature_conditioning,
        ) = self._run_encoder(X, X_skip_layer, X_feature_conditioning)
        curr, _ = self._run_decoder(
            encoding_out, curr, X_skip_layer, X_feature_conditioning
        )

        if apply_linear_transformation is True:
            curr = self.linear_transformation(curr)

        return curr
