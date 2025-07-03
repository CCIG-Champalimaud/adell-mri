from typing import Any, Dict, List, Tuple

import torch

from adell_mri.utils.masking import get_masker
from adell_mri.modules.layers.conv_next import ConvNeXt
from adell_mri.modules.layers.res_net import ResNet
from adell_mri.modules.layers.vit import TransformerBlockStack, ViT

TensorList = List[torch.Tensor]
IJEPAOut = Tuple[torch.Tensor, TensorList]

encoder_architectures = {"vit": ViT, "resnet": ResNet, "convnext": ConvNeXt}
predictor_architectures = {
    "vit": TransformerBlockStack,
    "resnet": ResNet,
    "convnext": ConvNeXt,
}


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
        n_patches: int = 4,
        n_masked_patches: int = 1,
        encoder_architecture: str = "vit",
        predictor_architecture: str = "vit",
        reduce_fn: str = "mean",
        seed: int = 42,
    ):
        """
        Args:
            backbone_args (Dict[str, Any]): arguments for the backbone encoder.
            projection_head_args (Dict[str, Any]): arguments for the projection
                head.
            feature_map_dimensions (List[int]): dimension of the feature map.
            n_encoder_features (int): number of output features from the
                encoder.
            min_patch_size (List[int]): minimum patch size.
            max_patch_size (List[int]): maximum patch size.
            n_patches (int, optional): number of patches. Defaults to 4.
            n_masked_patches (int, optional): number of masked patches.
                Defaults to 1.
            encoder_architecture (str, optional): architecture of the encoder
                (supports "vit" (`ViT`), "resnet" (`ResNet`) and "convnext"
                (`ConvNeXt`)). Defaults to "vit".
            predictor_architecture (str, optional): architecture for the
                predictor (only "vit" is supported). Defaults to "vit".
            reduce_fn (str, optional): function for reduction before prediction.
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
        self.encoder_architecture = encoder_architecture
        self.predictor_architecture = predictor_architecture
        self.reduce_fn = reduce_fn
        self.seed = seed

        assert (
            self.encoder_architecture in encoder_architectures
        ), f"only {encoder_architectures.keys()} are supported for encoder"
        assert self.predictor_architecture in [
            "vit"
        ], f"only {predictor_architectures.keys()} are supported for predictor"

        self.initialize_masker()
        self.initialize_encoder()
        self.initialize_predictor()
        self.initialize_mask_token()

    def initialize_masker(self):
        if self.encoder_architecture in ["vit"]:
            self.model_type_ = "transformer"
        else:
            self.model_type_ = "convolutional"
        self.patch_masker_ = get_masker(
            model_type=self.model_type_,
            image_dimensions=self.feature_map_dimensions,
            min_patch_size=self.min_patch_size,
            max_patch_size=self.max_patch_size,
            n_patches=self.n_patches,
            n_features=self.n_encoder_features,
            seed=self.seed,
        )

    def initialize_encoder(self):
        self.encoder_ = encoder_architectures[self.encoder_architecture](
            **self.backbone_args
        )

    def initialize_predictor(self):
        if self.projection_head_args is not None:
            arch = predictor_architectures[self.predictor_architecture]
            self.predictor_ = arch(**self.projection_head_args)

    def initialize_mask_token(self):
        self.mask_token_ = torch.nn.Parameter(
            torch.rand(self.n_encoder_features)
        )

    def reduce(self, X: torch.Tensor) -> torch.Tensor:
        if self.model_type_ == "transformer":
            X = X.permute(0, 2, 1)
        X = X.flatten(start_dim=2)
        if self.reduce_fn == "mean":
            X = X.mean(-1)
        elif self.reduce_fn == "max":
            X = X.max(-1)
        return X

    def forward_training(
        self, X: torch.Tensor, teacher_model: torch.nn.Module = None
    ) -> IJEPAOut:
        if teacher_model is None:
            teacher_model = self
        # encode full image
        encoded_X, _ = self.encoder_(X)
        # encode target with teacher module
        encoded_X_target, _ = teacher_model.encoder_(X)
        # mask image with mask_token
        encoded_X, _, patch_coords = self.patch_masker_(
            encoded_X, mask_vector=self.mask_token_
        )
        # retrieve patches using same coordinates
        _, patches, _ = self.patch_masker_(
            encoded_X_target, None, patch_coords=patch_coords
        )
        # mask additional parts of the image with mask token
        encoded_X, _, _ = self.patch_masker_(
            encoded_X,
            mask_vector=self.mask_token_,
            n_patches=self.n_masked_patches,
        )
        # get predictions for both encoded_X and patches + reduce
        predicted_X = self.reduce(self.predictor_(encoded_X)[0])
        predicted_patches = [
            self.reduce(self.predictor_(patch)[0]) for patch in patches
        ]
        return predicted_X, predicted_patches

    def forward(self, X: torch.Tensor) -> IJEPAOut:
        # encode full image and return
        return self.forward_representation(X)

    def forward_representation(self, X: torch.Tensor) -> torch.Tensor:
        return self.encoder_(X)[0].permute(0, 2, 1)
