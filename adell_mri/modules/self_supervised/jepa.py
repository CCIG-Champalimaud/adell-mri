"""
Implements I-JEPA.

This implementation follows the algorithm described in [1]: a context
encoder (the "student") processes only the tokens belonging to a set of
context blocks, while a target encoder (the "teacher", an EMA of the
student) processes the full image. A predictor takes the context tokens,
appends learnable mask tokens at the positions of the target blocks and
predicts the (layer-normalised) target encoder features for those blocks.
The loss is a smooth-L1 loss between the predictions and the target
features.

[1] https://arxiv.org/abs/2301.08243
"""

from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn.functional as F

from adell_mri.constants import DEFAULT_SEED
from adell_mri.modules.layers.vit import TransformerBlockStack, ViT
from adell_mri.utils.masking import get_masker

TensorList = List[torch.Tensor]
IJEPAOut = Tuple[TensorList, TensorList]


class IJEPAPredictor(torch.nn.Module):
    """
    Predictor used in I-JEPA.

    Given a sequence of context tokens, it appends one learnable mask token
    (plus its positional embedding) for each target position and runs a stack
    of transformer blocks. The outputs at the appended positions are then
    projected back to the encoder feature dimension.

    Args:
        encoder_dim (int): dimension of the encoder features.
        predictor_dim (int): dimension of the predictor features.
        block_stack_args (Dict[str, Any]): arguments for the
            ``TransformerBlockStack`` (``input_dim_primary`` is overridden by
            ``predictor_dim``).
        n_tokens (int): number of token positions in the feature map.
    """

    def __init__(
        self,
        encoder_dim: int,
        predictor_dim: int,
        block_stack_args: Dict[str, Any],
        n_tokens: int,
    ):
        super().__init__()
        self.encoder_dim = encoder_dim
        self.predictor_dim = predictor_dim
        self.n_tokens = n_tokens

        self.predictor_embed_ = torch.nn.Linear(
            encoder_dim, predictor_dim, bias=True
        )
        self.predictor_pos_embed_ = torch.nn.Parameter(
            torch.rand(1, n_tokens, predictor_dim)
        )
        self.mask_token_ = torch.nn.Parameter(torch.rand(1, 1, predictor_dim))
        block_stack_args = dict(block_stack_args)
        block_stack_args["input_dim_primary"] = predictor_dim
        self.predictor_ = TransformerBlockStack(**block_stack_args)
        self.predictor_proj_ = torch.nn.Linear(
            predictor_dim, encoder_dim, bias=False
        )

    def forward(
        self,
        context: torch.Tensor,
        context_idx: np.ndarray,
        target_idx: np.ndarray,
    ) -> torch.Tensor:
        """
        Forward pass.

        Args:
            context (torch.Tensor): context tokens with shape
                [B, n_context, encoder_dim].
            context_idx (np.ndarray): token indices of the context blocks.
            target_idx (np.ndarray): token indices of the target block.

        Returns:
            torch.Tensor: predicted target block features with shape
                [B, len(target_idx), encoder_dim].
        """
        B = context.shape[0]
        x = self.predictor_embed_(context)
        x = x + self.predictor_pos_embed_[:, context_idx, :]
        # append the mask token (plus its positional embedding) for each of
        # the target positions
        mask_tokens = (
            self.mask_token_ + self.predictor_pos_embed_[:, target_idx, :]
        )
        mask_tokens = mask_tokens.expand(B, -1, -1)
        x = torch.cat([x, mask_tokens], dim=1)
        x, _ = self.predictor_(x)
        pred = self.predictor_proj_(x[:, -len(target_idx) :])
        return pred


class IJEPA(torch.nn.Module):
    """
    Implementation of the I-JEPA network from META.

    Based on https://github.com/facebookresearch/ijepa
    """

    def __init__(
        self,
        backbone_args: Dict[str, Any],
        projection_head_args: Dict[str, Any],
        feature_map_dimensions: List[int],
        n_encoder_features: int,
        min_patch_size: List[int],
        max_patch_size: List[int],
        n_patches: int = 1,
        n_masked_patches: int = 4,
        predictor_dim: Optional[int] = None,
        encoder_architecture: str = "vit",
        predictor_architecture: str = "vit",
        reduce_fn: str = "mean",
        seed: int = DEFAULT_SEED,
    ):
        """
        Args:
            backbone_args (Dict[str, Any]): arguments for the backbone encoder
                (must be a ``ViT``).
            projection_head_args (Dict[str, Any]): arguments for the predictor
                (a ``TransformerBlockStack``).
            feature_map_dimensions (List[int]): dimension of the feature map.
            n_encoder_features (int): number of output features from the
                encoder.
            min_patch_size (List[int]): minimum patch size.
            max_patch_size (List[int]): maximum patch size.
            n_patches (int, optional): number of context blocks (the tokens of
                these blocks are kept by the encoder). Defaults to 1.
            n_masked_patches (int, optional): number of target blocks that are
                masked and predicted by the predictor. Defaults to 4.
            predictor_dim (int, optional): dimension of the predictor. If
                None, defaults to the encoder embedding dimension. Defaults to
                None.
            encoder_architecture (str, optional): architecture of the encoder
                (only "vit" is supported). Defaults to "vit".
            predictor_architecture (str, optional): architecture for the
                predictor (only "vit" is supported). Defaults to "vit".
            reduce_fn (str, optional): kept for backwards compatibility (the
                training loss no longer uses a reduction function).
                Defaults to "mean".
            seed (int, optional): random seed. Defaults to 42.
        """
        super().__init__()
        self.backbone_args = backbone_args
        self.projection_head_args = projection_head_args
        self.feature_map_dimensions = feature_map_dimensions
        self.n_encoder_features = n_encoder_features
        self.min_patch_size = min_patch_size
        self.max_patch_size = max_patch_size
        self.n_patches = n_patches
        self.n_masked_patches = n_masked_patches
        self.predictor_dim = predictor_dim
        self.encoder_architecture = encoder_architecture
        self.predictor_architecture = predictor_architecture
        self.reduce_fn = reduce_fn
        self.seed = seed

        assert (
            self.encoder_architecture == "vit"
        ), "only 'vit' is supported as encoder architecture for I-JEPA"
        assert (
            self.predictor_architecture == "vit"
        ), "only 'vit' is supported as predictor architecture for I-JEPA"

        self.initialize_masker()
        self.initialize_encoder()
        self.initialize_predictor()

    @property
    def extra_tokens(self) -> int:
        """
        Returns:
            int: number of extra tokens (class token and registers) prepended
                to the patch tokens by the embedding layer.
        """
        return self.encoder_.n_registers + int(self.encoder_.use_class_token)

    def initialize_masker(self):
        self.patch_masker_ = get_masker(
            model_type="generic_transformer",
            image_dimensions=self.feature_map_dimensions,
            min_patch_size=self.min_patch_size,
            max_patch_size=self.max_patch_size,
            n_patches=self.n_patches,
            n_features=self.n_encoder_features,
            seed=self.seed,
        )

    def initialize_encoder(self):
        self.encoder_ = ViT(**self.backbone_args)
        self.encoder_dim_ = self.encoder_.input_dim_primary

    def initialize_predictor(self):
        predictor_dim = (
            self.predictor_dim
            if self.predictor_dim is not None
            else self.encoder_dim_
        )
        n_tokens = int(np.prod(self.feature_map_dimensions))
        self.predictor_ = IJEPAPredictor(
            encoder_dim=self.encoder_dim_,
            predictor_dim=predictor_dim,
            block_stack_args=self.projection_head_args,
            n_tokens=n_tokens,
        )

    def _get_teacher_encoder(self, teacher_model) -> torch.nn.Module:
        if teacher_model is None:
            teacher_model = self
        teacher = getattr(teacher_model, "shadow", teacher_model)
        return teacher.encoder_

    def _sample_mask_indices(self) -> Tuple[np.ndarray, List[np.ndarray]]:
        """
        Samples the context and target block token indices.

        Returns:
            Tuple[np.ndarray, List[np.ndarray]]: the (unique) context token
                indices and a list of target token index arrays. Any target
                token overlapping the context is removed so that the
                prediction is not trivial.
        """
        context_coords = self.patch_masker_.sample_patches(self.n_patches)
        context_idx = np.unique(
            np.concatenate(
                [self.patch_masker_.retrieve_patch(c) for c in context_coords]
            )
        ).astype(np.int64)
        target_coords = self.patch_masker_.sample_patches(self.n_masked_patches)
        target_idx = [
            np.setdiff1d(
                self.patch_masker_.retrieve_patch(c).astype(np.int64),
                context_idx,
            )
            for c in target_coords
        ]
        # drop target blocks that fully overlap the context
        target_idx = [t for t in target_idx if len(t) > 0]
        return context_idx, target_idx

    def forward_training(
        self, X: torch.Tensor, teacher_model: torch.nn.Module = None
    ) -> IJEPAOut:
        """
        Training forward pass.

        Args:
            X (torch.Tensor): input tensor with shape
                [B, C, *image_size].
            teacher_model (torch.nn.Module, optional): the teacher (EMA)
                model. If it is an ``ExponentialMovingAverage`` module, its
                ``shadow`` is used. Defaults to None (the module itself is
                used as the teacher).

        Returns:
            IJEPAOut: a tuple ``(predictions, targets)`` where each element
                is a list of one tensor per target block. Predictions have
                shape [B, n_target_tokens, encoder_dim] and targets have the
                same shape.
        """
        embedded = self.encoder_.embedding(X)
        skip_n = self.extra_tokens
        spatial = embedded[:, skip_n:]

        context_idx, target_idx = self._sample_mask_indices()

        # student: encode the context tokens only
        context = spatial[:, context_idx]
        for block in self.encoder_.tbs.transformer_blocks:
            context = block(context)

        # predictor: predict each target block
        predictions = [
            self.predictor_(context, context_idx, t_idx) for t_idx in target_idx
        ]

        # teacher: encode the full image, layer-normalise and extract the
        # target blocks
        teacher_encoder = self._get_teacher_encoder(teacher_model)
        with torch.no_grad():
            h = teacher_encoder(X)[0]
            h = F.layer_norm(h, (h.shape[-1],))
            h_spatial = h[:, skip_n:]
            targets = [h_spatial[:, t_idx].detach() for t_idx in target_idx]

        return predictions, targets

    def forward(self, X: torch.Tensor) -> IJEPAOut:
        # encode full image and return
        return self.forward_representation(X)

    def forward_representation(self, X: torch.Tensor) -> torch.Tensor:
        return self.encoder_(X)[0].permute(0, 2, 1)
