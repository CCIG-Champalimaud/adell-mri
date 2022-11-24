import os
import sys
sys.path.append(os.path.join(os.path.dirname(os.path.abspath(__file__)),'..'))

import torch
from lib.modules.segmentation import UNet
from lib.modules.layers import ResNet,ActDropNormBuilder,resnet_to_encoding_ops

h,w,d,c = 32,32,20,1

depths = [[16,32,64],[16,32,64,128]]
spatial_dims = [2,3]

resnet_args = {'backbone_args': 
    {'spatial_dim': 3,
     'in_channels': 1,
     'structure': None,
     'maxpool_structure': None,
     'res_type': 'resnet',
     'adn_fn': ActDropNormBuilder()},
    'projection_head_args': {
        'in_channels': 512, 
        'structure': [1024, 512, 256],
        'adn_fn': ActDropNormBuilder(norm_fn=torch.nn.BatchNorm1d)},
    'prediction_head_args': {
        'in_channels': 256,
        'structure': [512, 256],
        'adn_fn': ActDropNormBuilder(norm_fn=torch.nn.BatchNorm1d)},
    }

def unet_base(D,sd,conv_type,strides="regular"):
    K = [3 for _ in D]
    if strides == "irregular":
        if sd == 2:
            S = [[2,2],*[[2,1] for _ in D[1:]]]
        elif sd == 3:
            S = [[2,2,2],*[[2,2,1] for _ in D[1:]]]
    else:
        S = [2 for _ in D]
    if sd == 2:
        i = torch.rand(size=[1,c,h,w])
        output_size = [1,1,h,w]
    elif sd == 3:
        i = torch.rand(size=[1,c,h,w,d])
        output_size = [1,1,h,w,d]
    a = UNet(sd,depth=D,upscale_type="transpose",padding=1,
             strides=S,kernel_sizes=K,conv_type=conv_type,link_type="identity")
    o,bb = a(i)
    assert list(o.shape) == output_size

def unet_skip(D,sd,conv_type):
    K = [3 for _ in D]
    S = [2 for _ in D]
    if sd == 2:
        i = torch.rand(size=[1,c,h,w])
        i_skip = torch.rand(size=[1,c,h,w])
        output_size = [1,1,h,w]
    elif sd == 3:
        i = torch.rand(size=[1,c,h,w,d])
        i_skip = torch.rand(size=[1,c,h,w,d])
        output_size = [1,1,h,w,d]
    a = UNet(sd,depth=D,upscale_type="transpose",padding=1,
            strides=S,kernel_sizes=K,skip_conditioning=1,
            link_type="conv",conv_type=conv_type)
    o,bb = a(i,X_skip_layer=i_skip)
    assert list(o.shape) == output_size

def unet_skip_feature(D,sd,conv_type):
    n_features = 4
    K = [3 for _ in D]
    S = [2 for _ in D]
    if sd == 2:
        i = torch.rand(size=[2,c,h,w])
        i_skip = torch.rand(size=[2,1,h,w])
        i_feat = torch.rand(size=[2,n_features])
        output_size = [2,1,h,w]
    elif sd == 3:
        i = torch.rand(size=[2,c,h,w,d])
        i_skip = torch.rand(size=[2,1,h,w,d])
        i_feat = torch.rand(size=[2,n_features])
        output_size = [2,1,h,w,d]
    a = UNet(sd,depth=D,upscale_type="transpose",padding=1,
            strides=S,kernel_sizes=K,feature_conditioning=n_features,
            skip_conditioning=1,
            feature_conditioning_params={
            "mean":torch.zeros_like(i_feat),
            "std":torch.ones_like(i_feat)},
            link_type="conv")
    o,bb = a(i,X_feature_conditioning=i_feat,X_skip_layer=i_skip)
    assert list(o.shape) == output_size

# TODO: finish this test
def unet_from_encoder(D,sd,strides):
    K = [3] + [3 for _ in D]
    if strides == "irregular":
        if sd == 2:
            S = [[2,2],*[[2,1] for _ in D[1:]]]
        elif sd == 3:
            S = [[2,2,2],*[[2,2,1] for _ in D[1:]]]
    else:
        S = [2 for _ in D]
    # first instantiate resnet
    resnet_args["backbone_args"]["spatial_dim"] = sd
    resnet_args["backbone_args"]["structure"] = [[d,d,3,1] for d in D]
    resnet_args["backbone_args"]["maxpool_structure"] = S
    if sd == 2: 
        resnet_args["backbone_args"]["adn_fn"] = ActDropNormBuilder(
            norm_fn=torch.nn.BatchNorm2d)
    if sd == 3:
        resnet_args["backbone_args"]["adn_fn"] = ActDropNormBuilder(
            norm_fn=torch.nn.BatchNorm3d)
    res_net = ResNet(**resnet_args)
    encoding_operations = resnet_to_encoding_ops(
        [res_net])[0]
    if sd == 2:
        i = torch.rand(size=[1,c,h,w])
        output_size = [1,1,h,w]
    elif sd == 3:
        i = torch.rand(size=[1,c,h,w,d])
        output_size = [1,1,h,w,d]
    S = [2,*resnet_args["backbone_args"]["maxpool_structure"]]
    D_unet = [D[0],*D]
    a = UNet(sd,encoding_operations=encoding_operations,depth=D_unet,
             upscale_type="transpose",padding=1,
             strides=S,kernel_sizes=K,link_type="identity")
    o,bb = a(i)
    print(o.shape,output_size)
    assert list(o.shape) == output_size

def test_unet_2d():
    for D in depths:
        for conv_type in ["regular","resnet"]:
            unet_base(D,2,conv_type)

def test_unet_2d_irregular_strides():
    for D in depths:
        for conv_type in ["regular","resnet"]:
            unet_base(D,2,conv_type,strides="irregular")

def test_unet_3d():
    for D in depths:
        for conv_type in ["regular","resnet"]:
            unet_base(D,3,conv_type)

def test_unet_3d_irregular_strides():
    for D in depths:
        for conv_type in ["regular","resnet"]:
            unet_base(D,3,conv_type,strides="irregular")

def test_unet_2d_skip():
    for D in depths:
        for conv_type in ["regular","resnet"]:
            unet_skip(D,2,conv_type)

def test_unet_3d_skip():
    for D in depths:
        for conv_type in ["regular","resnet"]:
            unet_skip(D,3,conv_type)

def test_unet_2d_skip_feature():
    for D in depths:
        for conv_type in ["regular","resnet"]:
            unet_skip_feature(D,2,conv_type)

def test_unet_3d_skip_feature():
    for D in depths:
        for conv_type in ["regular","resnet"]:
            unet_skip_feature(D,3,conv_type)

def test_unet_2d_resnet():
    for D in depths:
        unet_from_encoder(D,2,strides="regular")

def test_unet_2d_irregular_strides_resnet():
    for D in depths:
        unet_from_encoder(D,2,strides="irregular")

def test_unet_3d_resnet():
    for D in depths:
        unet_from_encoder(D,3,strides="regular")

def test_unet_3d_irregular_strides_resnet():
    for D in depths:
        unet_from_encoder(D,3,strides="irregular")
