import os
import sys
sys.path.append(os.path.join(os.path.dirname(os.path.abspath(__file__)),'..'))
import pytest

import torch
from lib.modules.layers.res_net import ResNet
from lib.modules.layers.adn_fn import ActDropNormBuilder

h,w,d,c = 32,32,20,1

depths = [[16,32,64],[16,32,64,128]]
spatial_dims = [2,3]

resnet_args = {'backbone_args': 
    {'spatial_dim': 3,
     'in_channels': 1,
     'structure': [[1,32,3,1],[32,64,3,1],[32,64,3,1]],
     'maxpool_structure': None,
     'res_type': 'resnet',
     'adn_fn': ActDropNormBuilder(norm_fn=torch.nn.BatchNorm3d)},
    'projection_head_args': {
        'in_channels': 32, 
        'structure': [32, 64, 32],
        'adn_fn': ActDropNormBuilder(norm_fn=torch.nn.BatchNorm1d)},
    'prediction_head_args': {
        'in_channels': 32,
        'structure': [32, 64],
        'adn_fn': ActDropNormBuilder(norm_fn=torch.nn.BatchNorm1d)},
    }

@pytest.mark.parametrize("be",[0,2,4])
def test_resnet(be):
    resnet_args["backbone_args"]["batch_ensemble"] = be
    resnet = ResNet(**resnet_args)
    input_tensor = torch.ones([2,1,32,32,32])
    resnet(input_tensor)