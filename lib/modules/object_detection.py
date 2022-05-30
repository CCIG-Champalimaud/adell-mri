import torch
import torch.nn.functional as F
from math import floor
from typing import List,Dict

def split_int_into_n(i,n):
    r = i % n
    o = [floor(i/n) for _ in range(n)]
    idx = 0
    while r > 0:
        o[idx] += 1
        r -= 1
        idx += 1
    return o

class DepthWiseSeparableConvolution2d(torch.nn.Module):
    def __init__(self,in_channels:int,out_channels:int,
                 kernel_size:int=3,padding:int=1,
                 act_fn:torch.nn.Module=torch.nn.ReLU)->torch.nn.Module:
        super(self).__init__()
        self.input_channels = in_channels
        self.output_channels = out_channels
        self.kernel_size = kernel_size
        self.padding = padding
        self.act_fn = act_fn

    def init_layers(self):
        self.depthwise_op = torch.nn.Conv2d(
            self.input_channels,self.input_channels,
            kernel_size=self.kernel_size,padding=self.paddign,
            groups=self.input_channels)
        self.pointwise_op = torch.nn.Conv2d(
            self.input_channels,self.output_channels,
            kernel_size=1)
        self.act_op = self.act_fn(inplace=True)

    def forward(self,X:torch.Tensor)->torch.Tensor:
        X = self.depthwise_op(X)
        X = self.pointwise_op(X)
        X = self.act_op(X)
        return X

class DepthWiseSeparableConvolution3d(torch.nn.Module):
    def __init__(self,in_channels:int,out_channels:int,
                 kernel_size:int=3,padding:int=1,
                 act_fn:torch.nn.Module=torch.nn.ReLU)->torch.nn.Module:
        super(self).__init__()
        self.input_channels = in_channels
        self.output_channels = out_channels
        self.kernel_size = kernel_size
        self.padding = padding
        self.act_fn = act_fn

    def init_layers(self):
        self.depthwise_op = torch.nn.Conv3d(
            self.input_channels,self.input_channels,
            kernel_size=self.kernel_size,padding=self.paddign,
            groups=self.input_channels)
        self.pointwise_op = torch.nn.Conv3d(
            self.input_channels,self.output_channels,
            kernel_size=1)
        self.act_op = self.act_fn(inplace=True)

    def forward(self,X:torch.Tensor)->torch.Tensor:
        X = self.depthwise_op(X)
        X = self.pointwise_op(X)
        X = self.act_op(X)
        return X

class SpatialPyramidPooling2d(torch.nn.Module):
    def __init__(self,in_channels:int,out_channels:int,
                 filter_sizes:List[int],
                 act_fn:torch.nn.Module=torch.nn.ReLU)->torch.nn.Module:
        super(self).__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.filter_size = filter_sizes
        self.act_fn = act_fn

    def init_layers(self):
        self.layers = torch.nn.ModuleList([])
        for filter_size in self.filter_sizes:
            op = torch.nn.Sequential(
                torch.nn.Conv2d(
                    self.in_channels,self.out_channels,
                    kernel_size=filter_size,padding="same"),
                self.act_fn(inplace=True),
                DepthWiseSeparableConvolution2d(
                    self.out_channels,self.out_channels,
                    kernel_size=filter_size,padding="same"),
                self.act_fn(inplace=True))
            self.layers.append(op)
    
    def forward(self,X:torch.Tensor)->torch.Tensor:
        outputs = []
        for layer in self.layers:
            outputs.append(layer(X))
        output = torch.cat(output,dim=1)
        return output

class SpatialPyramidPooling3d(torch.nn.Module):
    def __init__(self,in_channels:int,out_channels:int,
                 filter_sizes:List[int],
                 act_fn:torch.nn.Module=torch.nn.ReLU)->torch.nn.Module:
        super(self).__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.filter_size = filter_sizes
        self.act_fn = act_fn

    def init_layers(self):
        self.layers = torch.nn.ModuleList([])
        for filter_size in self.filter_sizes:
            op = torch.nn.Sequential(
                torch.nn.Conv3d(
                    self.in_channels,self.out_channels,
                    kernel_size=filter_size,padding="same"),
                self.act_fn(inplace=True),
                DepthWiseSeparableConvolution3d(
                    self.out_channels,self.out_channels,
                    kernel_size=filter_size,padding="same"),
                self.act_fn(inplace=True))
            self.layers.append(op)
    
    def forward(self,X:torch.Tensor)->torch.Tensor:
        outputs = []
        for layer in self.layers:
            outputs.append(layer(X))
        output = torch.cat(output,dim=1)
        return output

class AtrousSpatialPyramidPooling2d(torch.nn.Module):
    def __init__(self,in_channels:int,out_channels:int,
                 rates:List[int],
                 act_fn:torch.nn.Module=torch.nn.ReLU)->torch.nn.Module:
        super(self).__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.rates = rates
        self.act_fn = act_fn

    def init_layers(self):
        self.layers = torch.nn.ModuleList([])
        for rate in self.rates:
            op = torch.nn.Sequential(
                torch.nn.Conv2d(
                    self.in_channels,self.out_channels,
                    dilation=rate,padding="same"),
                self.act_fn(inplace=True),
                DepthWiseSeparableConvolution2d(
                    self.out_channels,self.out_channels,
                    kernel_size=3,padding="same"),
                self.act_fn(inplace=True))
            self.layers.append(op)

    def forward(self,X:torch.Tensor)->torch.Tensor:
        outputs = []
        for layer in self.layers:
            outputs.append(layer(X))
        output = torch.cat(output,dim=1)
        return output

class AtrousSpatialPyramidPooling3d(torch.nn.Module):
    def __init__(self,in_channels:int,out_channels:int,
                 rates:List[int],
                 act_fn:torch.nn.Module=torch.nn.ReLU)->torch.nn.Module:
        super(self).__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.rates = rates
        self.act_fn = act_fn

    def init_layers(self):
        self.layers = torch.nn.ModuleList([])
        for rate in self.rates:
            op = torch.nn.Sequential(
                torch.nn.Conv3d(
                    self.in_channels,self.out_channels,
                    dilation=rate,padding="same"),
                self.act_fn(inplace=True),
                DepthWiseSeparableConvolution3d(
                    self.out_channels,self.out_channels,
                    kernel_size=3,padding="same"),
                self.act_fn(inplace=True))
            self.layers.append(op)

    def forward(self,X:torch.Tensor)->torch.Tensor:
        outputs = []
        for layer in self.layers:
            outputs.append(layer(X))
        output = torch.cat(output,dim=1)
        return output

class ReceptiveFieldBlock2d(torch.nn.Module):
    def __init__(self,in_channels:int,out_channels:int,
                 rates:List[int],
                 act_fn:torch.nn.Module=torch.nn.ReLU)->torch.nn.Module:
        # https://arxiv.org/pdf/1711.07767.pdf
        super(self).__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.rates = rates
        self.act_fn = act_fn

        self.out_c_list = split_int_into_n(
            self.out_channels,len(self.rates))

    def init_layers(self):
        self.layers = torch.nn.ModuleList([])
        for rate,o in zip(self.rates,self.out_c_list):
            if rate == 1:
                op = torch.nn.Sequential(
                    torch.nn.Conv2d(
                        self.in_channels,o,kernel_size=1,padding="same"),
                    torch.act_fn(inplace=True),
                    torch.nn.Conv2d(o,o,kernel_size=3,padding="same"),
                    torch.act_fn(inplace=True))
            else:
                op = torch.nn.Sequential(
                    torch.nn.Conv2d(
                        self.in_channels,o,kernel_size=1,padding="same"),
                    torch.act_fn(inplace=True),
                    torch.nn.Conv2d(o,o,kernel_size=rate,padding="same"),
                    torch.act_fn(inplace=True),
                    torch.nn.Conv2d(o,o,dilation=rate,kernel_size=3,
                                    padding="same"),
                    torch.act_fn(inplace=True))
            self.layers.append(op)
        self.final_op = torch.nn.Conv2d(
            self.out_channels,self.out_channels,1)

    def forward(self,X:torch.Tensor)->torch.Tensor:
        outputs = []
        for layer in self.layers:
            outputs.append(layer(X))
        output = torch.cat(output,dim=1)
        output = self.final_op(output)
        output = X + output
        return output

class ReceptiveFieldBlock3d(torch.nn.Module):
    def __init__(self,in_channels:int,out_channels:int,
                 rates:List[int],
                 act_fn:torch.nn.Module=torch.nn.ReLU)->torch.nn.Module:
        # https://arxiv.org/pdf/1711.07767.pdf
        super(self).__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.rates = rates
        self.act_fn = act_fn

        self.out_c_list = split_int_into_n(
            self.out_channels,len(self.rates))

    def init_layers(self):
        self.layers = torch.nn.ModuleList([])
        for rate,o in zip(self.rates,self.out_c_list):
            if rate == 1:
                op = torch.nn.Sequential(
                    torch.nn.Conv3d(
                        self.in_channels,o,kernel_size=1,padding="same"),
                    torch.act_fn(inplace=True),
                    torch.nn.Conv3d(o,o,kernel_size=3,padding="same"),
                    torch.act_fn(inplace=True))
            else:
                op = torch.nn.Sequential(
                    torch.nn.Conv3d(
                        self.in_channels,o,kernel_size=1,padding="same"),
                    torch.act_fn(inplace=True),
                    torch.nn.Conv3d(o,o,kernel_size=rate,padding="same"),
                    torch.act_fn(inplace=True),
                    torch.nn.Conv3d(o,o,dilation=rate,kernel_size=3,
                                    padding="same"),
                    torch.act_fn(inplace=True))
            self.layers.append(op)
        self.final_op = torch.nn.Conv3d(
            self.out_channels,self.out_channels,1)

    def forward(self,X:torch.Tensor)->torch.Tensor:
        outputs = []
        for layer in self.layers:
            outputs.append(layer(X))
        output = torch.cat(output,dim=1)
        output = self.final_op(output)
        output = X + output
        return output

class SpatialSqueezeAndExcite2d(torch.nn.Module):
    def __init__(self,input_channels:int)->torch.nn.Module:
        super(self).__init__()
        self.input_channels = input_channels
    
    def init_layers(self):
        self.op = torch.nn.Sequential(
            torch.nn.Conv2d(self.input_channels,1,kernel_size=1),
            torch.nn.Sigmoid())
    
    def forward(self,X:torch.Tensor)->torch.Tensor:
        spatial_squeeze = self.op(X)
        X = X * spatial_squeeze
        return X

class SpatialSqueezeAndExcite3d(torch.nn.Module):
    def __init__(self,input_channels:int)->torch.nn.Module:
        super(self).__init__()
        self.input_channels = input_channels
    
    def init_layers(self):
        self.op = torch.nn.Sequential(
            torch.nn.Conv3d(self.input_channels,1,kernel_size=1),
            torch.nn.Sigmoid())
    
    def forward(self,X:torch.Tensor)->torch.Tensor:
        spatial_squeeze = self.op(X)
        X = X * spatial_squeeze
        return X

class ChannelSqueezeAndExcite(torch.nn.Module):
    def __init__(self,input_channels:int)->torch.nn.Module:
        super(self).__init__()
        self.input_channels = input_channels
    
    def init_layers(self):
        I = self.input_channels
        self.op = torch.nn.Sequential(
            torch.nn.Linear(I,I),
            torch.nn.ReLU(),
            torch.nn.Linear(I,I),
            torch.nn.Sigmoid())
    
    def forward(self,X:torch.Tensor)->torch.Tensor:
        channel_average = torch.flatten(X,start_dim=2).mean(-1)
        channel_squeeze = self.op(channel_average)
        channel_squeeze = torch.unsqueeze(
            torch.unsqueeze(channel_squeeze,-1),-1)
        X = X * channel_squeeze
        return X

class SCSAE2d(torch.nn.Module):
    def __init__(self,input_channels:int)->torch.nn.Module:
        super(self).__init__()
        self.input_channels = input_channels
    
    def init_layers(self):
        self.spatial = SpatialSqueezeAndExcite2d(self.input_channels)
        self.channel = ChannelSqueezeAndExcite(self.input_channels)
    
    def forward(self,X):
        s = self.spatial(X)
        c = self.channel(X)
        output = s+c
        return output

class SCSAE3d(torch.nn.Module):
    def __init__(self,input_channels:int)->torch.nn.Module:
        super(self).__init__()
        self.input_channels = input_channels
    
    def init_layers(self):
        self.spatial = SpatialSqueezeAndExcite3d(self.input_channels)
        self.channel = ChannelSqueezeAndExcite(self.input_channels)
    
    def forward(self,X):
        s = self.spatial(X)
        c = self.channel(X)
        output = s+c
        return output

class YOLO90002d(torch.nn.Module):
    def __init__(self,in_channels:int,n_b:int,n_c:int,
                 act_fn:torch.nn.Module,anchor_sizes:List,
                 dev:str="cuda")->torch.nn.Module:
        super(self).__init__()
        self.in_channels = in_channels
        self.n_b = n_b
        self.n_c = n_c
        self.act_fn = act_fn
        self.anchor_sizes = anchor_sizes
        self.dev = dev

    def init_anchor_tensors(self):
        self.anchor_tensor = [
            torch.reshape(torch.Tensor(anchor_size),[1,3,1,1])
            for anchor_size in self.anchor_sizes]
        self.anchor_tensor = torch.cat(
            self.anchor_tensor,dim=1).to(self.dev)

    def cab(self,in_channels,out_channels,k,stride=1)->torch.Tensor:
        p = int(k/2)
        return torch.nn.Sequential(
            torch.nn.Conv2d(
                in_channels,out_channels,k,stride=stride,padding=[0,0,p]),
            self.act_fn(inplace=True),
            torch.nn.BatchNorm2d(out_channels))
    
    def cab_n(self,in_channels,out_channels,ks,stride=1):
        ops = torch.nn.ModuleList([])
        for i,o,k in zip(in_channels,out_channels,ks):
            op = self.cab(i,o,k,stride)
            ops.append(op)
        final_op = torch.nn.Sequential(ops)
        return final_op

    def init_layers(self):
        # loosely based on the original publication
        self.feature_extraction = torch.nn.Sequential(
            self.cab(self.in_channels,64,3),
            torch.nn.MaxPool2d(2), # /2
            self.cab_n([64,128,256],[128,256,128],[3,3,1]),
            torch.nn.MaxPool2d(2), # /4
            self.cab_n([128,256,512],[256,512,256],[3,3,1]),
            torch.nn.MaxPool2d(2), # /8
            self.cab_n([256,512,1024],[512,1024,512],[3,3,1]),
            torch.nn.MaxPool2d([2,2,1]), # /16
            self.cab_n([512,1024],[1024,512],[3,3]),
            torch.nn.MaxPool2d([2,2,1]), # /32
            self.cab_n([512,1024],[1024,1024],[3,3]))
        
        self.bb_size_layer = torch.nn.Conv2d(
            1024,self.n_b*3,1)
        self.bb_center_layer = torch.nn.Sequential(
            torch.nn.Conv2d(1024,self.n_b*3,1),
            torch.nn.Sigmoid())
        self.bb_objectness_layer = torch.nn.Sequential(
            torch.nn.Conv2d(1024,self.n_b,1),
            torch.nn.Sigmoid())
        if self.n_c == 2:
            self.classifiation_layer = torch.nn.Sequential(
                torch.nn.Conv2d(1024,1,1),
                torch.nn.Sigmoid())
        else:
            self.classifiation_layer = torch.nn.Sequential(
                torch.nn.Conv2d(1024,self.n_c,1),
                torch.nn.Softmax(dim=1))
    
    def forward(self,X:torch.Tensor)->torch.Tensor:
        features = self.feature_extraction(X)
        # size prediction
        bb_size_pred = self.bb_size_layer(features)
        bb_size_pred = self.anchor_tensor*torch.exp(bb_size_pred)
        # center prediction
        bb_center_pred = self.bb_center_layer(features)
        # objectness prediction
        bb_object_pred = self.bb_objectness_layer(features)
        # class prediction
        class_pred = self.class_pred(features)
        return bb_center_pred,bb_size_pred,bb_object_pred,class_pred

class YOLO90003d(torch.nn.Module):
    def __init__(self,in_channels:int,n_b:int,n_c:int,
                 act_fn:torch.nn.Module,anchor_sizes:List,
                 dev:str="cuda")->torch.nn.Module:
        super(self).__init__()
        self.in_channels = in_channels
        self.n_b = n_b
        self.n_c = n_c
        self.act_fn = act_fn
        self.anchor_sizes = anchor_sizes
        self.dev = dev

    def init_anchor_tensors(self):
        self.anchor_tensor = [
            torch.reshape(torch.Tensor(anchor_size),[1,3,1,1])
            for anchor_size in self.anchor_sizes]
        self.anchor_tensor = torch.cat(
            self.anchor_tensor,dim=1).to(self.dev)

    def cab(self,in_channels,out_channels,k,stride=1)->torch.Tensor:
        p = int(k/2)
        return torch.nn.Sequential(
            torch.nn.Conv3d(
                in_channels,out_channels,k,stride=stride,padding=[0,0,p]),
            self.act_fn(inplace=True),
            torch.nn.BatchNorm3d(out_channels))
    
    def cab_n(self,in_channels,out_channels,ks,stride=1):
        ops = torch.nn.ModuleList([])
        for i,o,k in zip(in_channels,out_channels,ks):
            op = self.cab(i,o,k,stride)
            ops.append(op)
        final_op = torch.nn.Sequential(ops)
        return final_op

    def init_layers(self):
        # loosely based on the original publication
        self.feature_extraction = torch.nn.Sequential(
            self.cab(self.in_channels,64,3),
            torch.nn.MaxPool3d(2), # /2
            self.cab_n([64,128,256],[128,256,128],[3,3,1]),
            torch.nn.MaxPool3d(2), # /4
            self.cab_n([128,256,512],[256,512,256],[3,3,1]),
            torch.nn.MaxPool3d(2), # /8
            self.cab_n([256,512,1024],[512,1024,512],[3,3,1]),
            torch.nn.MaxPool3d([2,2,1]), # /16
            self.cab_n([512,1024],[1024,512],[3,3]),
            torch.nn.MaxPool3d([2,2,1]), # /32
            self.cab_n([512,1024],[1024,1024],[3,3]))
        
        self.bb_size_layer = torch.nn.Conv3d(
            1024,self.n_b*3,1)
        self.bb_center_layer = torch.nn.Sequential(
            torch.nn.Conv3d(1024,self.n_b*3,1),
            torch.nn.Sigmoid())
        self.bb_objectness_layer = torch.nn.Sequential(
            torch.nn.Conv3d(1024,self.n_b,1),
            torch.nn.Sigmoid())
        if self.n_c == 2:
            self.classifiation_layer = torch.nn.Sequential(
                torch.nn.Conv3d(1024,1,1),
                torch.nn.Sigmoid())
        else:
            self.classifiation_layer = torch.nn.Sequential(
                torch.nn.Conv3d(1024,self.n_c,1),
                torch.nn.Softmax(dim=1))
    
    def forward(self,X:torch.Tensor)->torch.Tensor:
        features = self.feature_extraction(X)
        # size prediction
        bb_size_pred = self.bb_size_layer(features)
        bb_size_pred = self.anchor_tensor*torch.exp(bb_size_pred)
        # center prediction
        bb_center_pred = self.bb_center_layer(features)
        # objectness prediction
        bb_object_pred = self.bb_objectness_layer(features)
        # class prediction
        class_pred = self.class_pred(features)
        return bb_center_pred,bb_size_pred,bb_object_pred,class_pred
