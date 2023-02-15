import torch
from ..layers.conv_next import ConvNeXtV2Backbone

from typing import List,Union,Tuple

class ConvNeXtAutoEncoder(torch.nn.Module):
    def __init__(self,
                 in_channels:int,
                 encoder_structure:List[Tuple[int,int,int,int]],
                 decoder_structure:List[Tuple[int,int,int,int]],
                 spatial_dim:int=2,
                 batch_ensemble:int=0):
        super().__init__()
        self.in_channels = in_channels
        self.encoder_structure = encoder_structure
        self.decoder_structure = decoder_structure
        # TODO: define maxpool
        self.maxpool_structure = None
        self.spatial_dim = spatial_dim
        self.batch_ensemble = batch_ensemble
    
    def init_encoder(self):
        self.encoder = ConvNeXtV2Backbone(
            spatial_dim=self.spatial_dim,
            in_channels=self.in_channels,
            structure=self.encoder_structure,
            maxpool_structure=self.maxpool_structure,
            batch_ensemble=self.batch_ensemble)
    
    def init_proj(self):
        self.proj = torch.nn.Conv2d(self.encoder_structure[-1][0],
                                    self.decoder_structure[0][0],
                                    1)
    
    def init_decoder(self):
        self.encoder = ConvNeXtV2Backbone(
            spatial_dim=self.spatial_dim,
            in_channels=self.in_channels,
            structure=self.decoder_structure,
            maxpool_structure=[1 for _ in self.decoder_structure],
            batch_ensemble=self.batch_ensemble)
    
    def init_pred(self):
        self.encoder
    
    def forward(self,X):
        return X
