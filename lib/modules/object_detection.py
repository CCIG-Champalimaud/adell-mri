import torch
import torchvision

from ..types import *
from .layers import *

class YOLONet2d(torch.nn.Module):
    def __init__(self,in_channels:int,n_c:int,
                 act_fn:torch.nn.Module,anchor_sizes:List,
                 dev:str="cuda"):
        """Implementation of YOLO9000 for 2d inputs [1], a model for object
        detection. Frequently known as YOLOv2.

        [1] https://arxiv.org/abs/1612.08242

        Args:
            in_channels (int): number of input channels.
            n_b (int): number of bounding boxes.
            n_c (int): number of classes.
            act_fn (torch.nn.Module): activation function.
            anchor_sizes (List): sizes of anchors.
            dev (str, optional): device for memory allocation. Defaults to 
            "cuda".
        """
        super().__init__()
        self.in_channels = in_channels
        self.n_c = n_c
        self.act_fn = act_fn
        self.anchor_sizes = anchor_sizes
        self.n_b = len(self.anchor_sizes)
        self.dev = dev

        self.init_anchor_tensors()

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
            self.act_fn(),
            torch.nn.BatchNorm2d(out_channels))
    
    def cab_n(self,in_channels,out_channels,ks,stride=1):
        ops = torch.nn.ModuleList([])
        for i,o,k in zip(in_channels,out_channels,ks):
            op = self.cab(i,o,k,stride)
            ops.append(op)
        final_op = torch.nn.Sequential(ops)
        return final_op

    def init_layers(self):
        # the key challenge here is not losing too much of the initial 
        # resolution, particularly because it is anysotropic in the depth axis,
        # requiring special attention. For this reason I focus here on using 
        # layers which can capture large range relationships between pixels
        self.feature_extraction = torch.nn.Sequential(
            self.cab(self.in_channels,64,7),
            torch.nn.MaxPool2d(2), # /2
            ReceptiveFieldBlock2d(64,128,[1,3,5,7]),
            self.act_fn(),
            self.cab(64,128,3),
            torch.nn.MaxPool2d([2,2,1]), # /4
            ReceptiveFieldBlock2d(128,256,[1,3,5,7]),
            self.act_fn(),
            self.cab(128,256,3),
            ReceptiveFieldBlock2d(256,512,[1,3,5,7]),
            self.act_fn(),
            self.cab(256,512,3))
        
        self.bb_size_layer = torch.nn.Conv2d(
            512,self.n_b*3,1)
        self.bb_center_layer = torch.nn.Sequential(
            torch.nn.Conv2d(512,self.n_b*3,1),
            torch.nn.Sigmoid())
        self.bb_objectness_layer = torch.nn.Sequential(
            torch.nn.Conv2d(512,self.n_b,1),
            torch.nn.Sigmoid())
        if self.n_c == 2:
            self.classifiation_layer = torch.nn.Sequential(
                torch.nn.Conv2d(512,1,1),
                torch.nn.Sigmoid())
        else:
            self.classifiation_layer = torch.nn.Sequential(
                torch.nn.Conv2d(512,self.n_c,1),
                torch.nn.Softmax(dim=1))
    
    def forward(self,X:torch.Tensor)->torch.Tensor:
        features = self.feature_extraction(X)
        # size prediction
        bb_size_pred = self.bb_size_layer(features)
        # center prediction
        bb_center_pred = self.bb_center_layer(features)
        # objectness prediction
        bb_object_pred = self.bb_objectness_layer(features)
        # class prediction
        class_pred = self.class_pred(features)
        return bb_center_pred,bb_size_pred,bb_object_pred,class_pred
    
    def recover_boxes(self,bb_size_pred:torch.Tensor,
                      bb_center_pred:torch.Tensor,
                      bb_object_pred:torch.Tensor)->torch.Tensor:
        c,h,w = bb_center_pred.shape
        mesh = torch.stack(
            torch.meshgrid([torch.arange(h),torch.arange(w)]))
        mesh = torch.unsqueeze(mesh,0)
        # back to original size
        bb_size_pred = self.anchor_tensor*torch.exp(bb_size_pred)
        # back to original center
        bb_center_pred = bb_center_pred + mesh
        # get indices for good predictions
        c,x,y = torch.where(bb_object_pred > 0.5)
        # get formatted boxes
        object_scores = bb_object_pred[0,x,y]
        long_sizes = bb_size_pred[:,x,y]
        long_centers = bb_center_pred[:,x,y]
        upper_corner = long_centers - long_sizes/2
        lower_corner = long_centers + long_sizes/2
        long_bb = torch.cat([upper_corner,lower_corner],1)
        bb_idxs = torchvision.ops.nms(long_bb,object_scores,0.5)
        
        return long_bb[bb_idxs],object_scores[bb_idxs]

class YOLONet3d(torch.nn.Module):
    def __init__(self,in_channels:int,n_c:int,
                 act_fn:torch.nn.Module,anchor_sizes:List,
                 dev:str="cuda"):
        super().__init__()
        self.in_channels = in_channels
        self.n_c = n_c
        self.act_fn = act_fn
        self.anchor_sizes = anchor_sizes
        self.n_b = len(self.anchor_sizes)
        self.dev = dev

        self.init_anchor_tensors()
        self.init_layers()

    def init_anchor_tensors(self):
        self.anchor_tensor = [
            torch.reshape(torch.Tensor(anchor_size),[3,1,1,1])
            for anchor_size in self.anchor_sizes]
        self.anchor_tensor = torch.cat(
            self.anchor_tensor,dim=0).to(self.dev)

    def cab(self,in_channels,out_channels,k,stride=1)->torch.Tensor:
        p = int(k/2)
        return torch.nn.Sequential(
            torch.nn.Conv3d(
                in_channels,out_channels,k,stride=stride,padding=[0,0,p]),
            self.act_fn(),
            torch.nn.BatchNorm3d(out_channels))
    
    def cab_n(self,in_channels,out_channels,ks,stride=1):
        ops = torch.nn.ModuleList([])
        for i,o,k in zip(in_channels,out_channels,ks):
            op = self.cab(i,o,k,stride)
            ops.append(op)
        final_op = torch.nn.Sequential(ops)
        return final_op

    def init_layers(self):
        # the key challenge here is not losing too much of the initial 
        # resolution, particularly because it is anysotropic in the depth axis,
        # requiring special attention. For this reason I focus here on using 
        # layers which can capture large range relationships between pixels
        self.feature_extraction = torch.nn.Sequential(
            self.cab(self.in_channels,64,7),
            torch.nn.MaxPool3d(2), # /2
            ReceptiveFieldBlock3d(64,128,[1,3,5]),
            self.act_fn(),
            AtrousSpatialPyramidPooling3d(64,64,[5,7,9,11]),
            self.act_fn(),
            self.cab(256,128,1),
            torch.nn.MaxPool3d([2,2,1]), # /4
            ReceptiveFieldBlock3d(128,256,[1,3,5]),
            self.act_fn(),
            AtrousSpatialPyramidPooling3d(128,64,[5,7,9,11]),
            self.act_fn(),
            self.cab(256,256,3),
            ReceptiveFieldBlock3d(256,512,[1,3,5]),
            self.act_fn(),
            AtrousSpatialPyramidPooling3d(256,128,[5,7,9,11]),
            self.act_fn(),
            self.cab(512,512,3))
        
        self.bb_size_layer = torch.nn.Conv3d(
            512,self.n_b*3,1)
        self.bb_center_layer = torch.nn.Sequential(
            torch.nn.Conv3d(512,self.n_b*3,1),
            torch.nn.Sigmoid())
        self.bb_objectness_layer = torch.nn.Sequential(
            torch.nn.Conv3d(512,self.n_b,1),
            torch.nn.Sigmoid())
        if self.n_c == 2:
            self.classifiation_layer = torch.nn.Sequential(
                torch.nn.Conv3d(512,1,1),
                torch.nn.Sigmoid())
        else:
            self.classifiation_layer = torch.nn.Sequential(
                torch.nn.Conv3d(512,self.n_c,1),
                torch.nn.Softmax(dim=1))
    
    def forward(self,X:torch.Tensor)->torch.Tensor:
        features = self.feature_extraction(X)
        # size prediction
        bb_size_pred = self.bb_size_layer(features)
        # center prediction
        bb_center_pred = self.bb_center_layer(features)
        # objectness prediction
        bb_object_pred = self.bb_objectness_layer(features)
        # class prediction
        class_pred = self.classifiation_layer(features)
        return bb_center_pred,bb_size_pred,bb_object_pred,class_pred

    def recover_boxes(self,bb_size_pred:torch.Tensor,
                      bb_center_pred:torch.Tensor,
                      bb_object_pred:torch.Tensor,
                      nms:bool=False)->torch.Tensor:
        c,h,w,d = bb_center_pred.shape
        mesh = torch.stack(
            torch.meshgrid([torch.arange(h),torch.arange(w),torch.arange(d)]))
        mesh = torch.cat([mesh for _ in self.anchor_sizes])
        # back to original size
        bb_size_pred = self.anchor_tensor*torch.exp(bb_size_pred)
        # back to original center
        bb_center_pred = bb_center_pred + mesh
        # get indices for good predictions
        c,x,y,z = torch.where(bb_object_pred > 0.5)
        # get formatted boxes
        object_scores = bb_object_pred[:,x,y,z]
        # unpack values from different anchors 
        long_sizes = torch.tensor_split(bb_size_pred[:,x,y,z],self.n_b,0)
        long_sizes = torch.cat(long_sizes,1).swapaxes(0,1)
        long_centers = torch.tensor_split(bb_center_pred[:,x,y,z],self.n_b,0)
        long_centers = torch.cat(long_centers,1).swapaxes(0,1)

        upper_corner = long_centers - long_sizes/2
        lower_corner = long_centers + long_sizes/2
        long_bb = torch.cat([upper_corner,lower_corner],1)
        if nms == True:
            bb_idxs = torchvision.ops.nms(
                long_bb,object_scores.reshape(-1),0.5)
            return long_bb[bb_idxs],object_scores[bb_idxs]
        else:
            return long_bb,object_scores