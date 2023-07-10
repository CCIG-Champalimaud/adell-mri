import torch
from ..layers.res_net import ResNet
from ..layers.conv_next import ConvNeXt
from ..layers.vit import ViT
from ...utils.masking import get_masker

from typing import List, Dict, Any, Tuple

TensorList = List[torch.Tensor]
IJEPAOut = Tuple[torch.Tensor,TensorList]

encoder_architectures = {
    "vit":ViT,
    "resnet":ResNet,
    "convnext":ConvNeXt}
predictor_architectures = {
    "vit":ViT,
    "resnet":ResNet,
    "convnext":ConvNeXt}

class IJEPA(torch.nn.Module):
    def __init__(self,
                 encoder_parameters:Dict[str,Any],
                 predictor_parameters:Dict[str,Any],
                 feature_map_dimensions:List[int],
                 n_encoder_features:int,
                 min_patch_size:List[int],
                 max_patch_size:List[int],
                 n_patches:int=4,
                 n_masked_patches:int=1,
                 encoder_architecture:str="vit",
                 predictor_architecture:str="vit",
                 seed:int=42):
        self.encoder_parameters = encoder_parameters
        self.predictor_parameters = predictor_parameters
        self.feature_map_dimensions = feature_map_dimensions
        self.n_encoder_features = n_encoder_features
        self.min_patch_size = min_patch_size
        self.max_patch_size = max_patch_size
        self.n_patches = n_patches
        self.n_masked_patches = n_masked_patches
        self.encoder_architecture = encoder_architecture
        self.predictor_architecture = predictor_architecture
        self.seed = seed

        assert self.encoder_architecture in encoder_architectures, \
        f"only {encoder_architectures.keys()} are supported for encoder"
        assert self.predictor_architecture in ["vit"], \
        f"only {predictor_architectures.keys()} are supported for predictor"

        self.initialize_masker()
        self.initialize_encoder()
        self.initialize_predictor()


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
            seed=self.seed)

        self.masker_ = get_masker(
            model_type=self.model_type_,
            image_dimensions=self.feature_map_dimensions,
            min_patch_size=self.min_patch_size,
            max_patch_size=self.max_patch_size,
            n_patches=self.n_masked_patches,
            seed=self.seed)

    def initialize_encoder(self):
        self.encoder_ = encoder_architectures[self.encoder_architecture](
            **self.encoder_parameters)

    def initialize_predictor(self):
        self.predictor_ = predictor_architectures[self.predictor_architecture](
            **self.predictor_parameters)
        
    def initialize_mask_token(self):
        self.mask_token_ = torch.nn.Parameter(
            torch.rand(self.n_encoder_features))
    
    def reduce(self,X:torch.Tensor)->torch.Tensor:
        if self.model_type_ == "transformer":
            X = X.permute(0,2,1)
        X = X.flatten(start_dim=2)
        if self.reduce_fn == "mean":
            X = X.mean(-1)
        elif self.reduce_fn == "max":
            X = X.max(-1)
        return X

    def forward_training(self,
                         X:torch.Tensor,
                         teacher_model:torch.nn.Module=None)->IJEPAOut:
        if teacher_model is None:
            teacher_model = self
        # encode full image
        encoded_X = self.encoder_(X)
        # encode target with teacher module
        encoded_X_target = teacher_model.encoder_(X)
        # mask image with mask_token
        encoded_X,_,patch_coords = self.patch_masker_(
            encoded_X,self.mask_token_)
        # retrieve patches using same coordinates
        _,patches,_ = self.patch_masker_(
            encoded_X_target,None,patch_coords)
        # mask additional parts of the image with mask token
        encoded_X,_ = self.masker_(encoded_X,self.mask_token_)
        # get predictions for both encoded_X and patches + reduce
        predicted_X = self.reduce(self.predictor_(encoded_X))
        predicted_patches = [self.reduce(self.predictor_(patch)) 
                             for patch in patches]
        return predicted_X,predicted_patches

    def forward(self,X:torch.Tensor)->IJEPAOut:
        # encode full image and return
        encoded_X = self.encoder_(X)
        return encoded_X
