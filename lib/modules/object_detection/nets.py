import torch
import numpy as np

from .utils import resnet_default,maxpool_default,pyramid_default,nms_nd
from ..layers.adn_fn import ActDropNorm
from ..layers.res_net import ResNetBackbone
from ..layers.conv_next import ConvNeXtBackboneDetection
from ..layers.multi_resolution import AtrousSpatialPyramidPooling3d
from ..layers.self_attention import ConcurrentSqueezeAndExcite3d
from ...custom_types import List,Tuple

class YOLONet3d(torch.nn.Module):
    def __init__(self,
                 backbone_str:str="resnet",
                 n_channels:int=1,n_classes:int=2,
                 anchor_sizes:List=np.ones([1,6]),dev:str="cuda",
                 resnet_structure:List[Tuple[int,int,int,int]]=resnet_default,
                 maxpool_structure:List[Tuple[int,int,int]]=maxpool_default,
                 pyramid_layers:List[Tuple[int,int,int]]=pyramid_default,
                 adn_fn:torch.nn.Module=lambda s:ActDropNorm(
                     s,norm_fn=torch.nn.BatchNorm3d)):
        """Implementation of a YOLO network for object detection in 3d.

        Args:
            n_channels (int, optional): number of input channels. Defaults to 
                1.
            n_classes (int, optional): number of classes. Defaults to 2.
            anchor_sizes (List, optional): anchor sizes. Defaults to 
                np.ones([1,6]).
            dev (str, optional): device for memory allocation. Defaults to 
                "cuda".
            resnet_structure (List[Tuple[int,int,int,int]], optional): 
                structure for the ResNet backbone. Defaults to resnet_default.
            maxpool_structure (List[Tuple[int,int,int]], optional): structure
                for the maximum pooling operations. Defaults to 
                maxpool_default.
            pyramid_layers (List[Tuple[int,int,int]], optional): structure for
                the atrous spatial pyramid pooling layer. Defaults to 
                pyramid_default.
            adn_fn (torch.nn.Module, optional): function that is applied after
                each layer (activations, batch normalisation, dropout and etc. 
                should be specified here). Defaults to 
                lambda s:ActDropNorm(s,norm_fn=torch.nn.BatchNorm3d).
        """
        super().__init__()
        self.backbone_str = backbone_str
        self.in_channels = n_channels
        self.n_classes = n_classes
        self.anchor_sizes = anchor_sizes
        self.resnet_structure = resnet_structure
        self.maxpool_structure = maxpool_structure
        self.pyramid_layers = pyramid_layers
        self.adn_fn = adn_fn
        self.n_b = len(self.anchor_sizes)
        self.dev = dev

        self.init_anchor_tensors()
        self.init_layers()

    def init_anchor_tensors(self):
        self.anchor_tensor = [
            torch.reshape(torch.as_tensor(anchor_size),[3,1,1,1])
            for anchor_size in self.anchor_sizes]
        self.anchor_tensor = torch.cat(
            self.anchor_tensor,dim=0).to(self.dev)
        self.anchor_tensor = torch.nn.Parameter(
            self.anchor_tensor,requires_grad=False)

    def init_layers(self):
        if self.backbone_str == "resnet":
            self.res_net = ResNetBackbone(
                3,self.in_channels,self.resnet_structure,
                adn_fn=self.adn_fn,maxpool_structure=self.maxpool_structure)
        elif self.backbone_str == "convnext":
            self.res_net = ConvNeXtBackboneDetection(
                3,self.in_channels,self.resnet_structure,
                adn_fn=self.adn_fn,maxpool_structure=self.maxpool_structure)
        self.feature_extraction = self.res_net 
        last_size = self.resnet_structure[-1][0]
        
        self.pyramidal_feature_extraction = torch.nn.ModuleList(
            [self.adn_fn(last_size)])
        if self.pyramid_layers is not None and len(self.pyramid_layers) > 0:
            self.pyramidal_feature_extraction.extend(
                [AtrousSpatialPyramidPooling3d(
                    last_size,last_size,self.pyramid_layers,
                    adn_fn=torch.nn.Identity),
                 self.adn_fn(last_size)])
        self.pyramidal_feature_extraction.extend([
            ConcurrentSqueezeAndExcite3d(last_size),
            self.adn_fn(last_size)])
        self.pyramidal_feature_extraction = torch.nn.Sequential(
            *self.pyramidal_feature_extraction)
        
        last_size = last_size
        self.bb_size_layer = torch.nn.Sequential(
            torch.nn.Conv3d(last_size,last_size,1),
            self.adn_fn(last_size),
            torch.nn.Conv3d(last_size,self.n_b*3,1))
        self.bb_center_layer = torch.nn.Sequential(
            torch.nn.Conv3d(last_size,last_size,1),
            self.adn_fn(last_size),
            torch.nn.Conv3d(last_size,self.n_b*3,1),
            torch.nn.Tanh())
        self.bb_objectness_layer = torch.nn.Sequential(
            torch.nn.Conv3d(last_size,last_size,1),
            self.adn_fn(last_size),
            torch.nn.Conv3d(last_size,self.n_b,1),
            torch.nn.Sigmoid())
        if self.n_classes == 2:
            self.classifiation_layer = torch.nn.Sequential(
                torch.nn.Conv3d(last_size,last_size,1),
                self.adn_fn(last_size),
                torch.nn.Conv3d(last_size,1,1),
                torch.nn.Sigmoid())
        else:
            self.classifiation_layer = torch.nn.Sequential(
                torch.nn.Conv3d(last_size,last_size,1),
                self.adn_fn(last_size),
                torch.nn.Conv3d(last_size,self.n_classes,1),
                torch.nn.Softmax(dim=1))
    
    def forward(self,X:torch.Tensor)->torch.Tensor:
        features = self.feature_extraction(X)
        features = self.pyramidal_feature_extraction(features)
        # size prediction
        bb_size_pred = self.bb_size_layer(features)
        # center prediction
        bb_center_pred = self.bb_center_layer(features)
        # objectness prediction
        bb_object_pred = self.bb_objectness_layer(features)
        # class prediction
        class_pred = self.classifiation_layer(features)
        return bb_center_pred,bb_size_pred,bb_object_pred,class_pred

    def split(self,x,n_splits,dim):
        size = int(x.shape[dim]//n_splits)
        return torch.split(x,size,dim)

    def channels_to_anchors(self,prediction):
        prediction[:-1] = [torch.stack(self.split(x,self.n_b,1),-1)
                           for x in prediction[:-1]]
        return prediction

    def recover_boxes(self,
                      bb_center_pred:torch.Tensor,
                      bb_size_pred:torch.Tensor,
                      bb_object_pred:torch.Tensor,
                      class_pred:torch.Tensor,
                      nms:bool=False,
                      correction_factor:torch.Tensor=None)->torch.Tensor:
        """Converts the predictions from a single prediction from forward into 
        bounding boxes. The format of these boxes is 
        (uc_x,uc_y,uc_z,lc_x,lc_y,lc_z), where uc and lc refer to upper and
        lower corners, respectively.

        Args:
            bb_center_pred (torch.Tensor): center offset predictions.
            bb_size_pred (torch.Tensor): size predictions.
            bb_object_pred (torch.Tensor): objectness predictions.
            class_pred (torch.Tensor): class predictions.
            nms (bool, optional): whether to perform non-maximum suppression. 
                Defaults to False.
            correction_factor (torch.Tensor, optional): corrects the long 
                centres using a multiplicative factor. Defaults to None (no 
                correction).

        Returns:
            long_bb (torch.Tensor): bounding boxes in the corner format 
                specified above.
            object_scores (torch.Tensor): objectness scores (1d).
            long_classes (torch.Tensor): classes.
        """
        c,h,w,d,a = bb_center_pred.shape
        
        mesh = torch.stack(
            torch.meshgrid([torch.arange(h),torch.arange(w),torch.arange(d)],
            indexing='ij')).to(self.dev)
        mesh = torch.stack([mesh for _ in self.anchor_sizes],-1)
        # back to original size
        anc = torch.stack(torch.split(self.anchor_tensor,3,0),-1)
        bb_size_pred = anc*torch.exp(bb_size_pred)
        # back to original center
        bb_center_pred = bb_center_pred + mesh
        # get indices for good predictions
        c,x,y,z,a = torch.where(bb_object_pred > 0.5)
        # get formatted boxes
        object_scores = bb_object_pred[:,x,y,z,a].swapaxes(0,1)
        # unpack values from different anchors 
        long_sizes = bb_size_pred[:,x,y,z,a].swapaxes(0,1)
        long_centers = bb_center_pred[:,x,y,z,a].swapaxes(0,1)
        if correction_factor is not None:
            long_centers = long_centers * correction_factor.unsqueeze(0)
        if self.n_classes > 2:
            long_classes = class_pred[:,x,y,z].swapaxes(0,-1)
        else:
            long_classes = bb_object_pred[:,x,y,z,a].swapaxes(0,1)
        long_classes = long_classes.squeeze()
        object_scores = object_scores.squeeze(1)

        upper_corner = long_centers - long_sizes/2
        lower_corner = long_centers + long_sizes/2
        long_bb = torch.cat([upper_corner,lower_corner],1)
        if nms is True:
            bb_idxs = nms_nd(
                long_bb,object_scores.reshape(-1),0.7,0.5)
            return (
                long_bb[bb_idxs],
                object_scores[bb_idxs],
                long_classes[bb_idxs])
        else:
            if len(object_scores.shape) > 0:
                if object_scores.shape[0] > 1000:
                    top_1000 = torch.argsort(object_scores)
                    top_1000 = top_1000[top_1000 < 1000]
                    return (
                        long_bb[top_1000],
                        object_scores[top_1000],
                        long_classes[top_1000])

        return (long_bb,object_scores,long_classes)

    def recover_boxes_batch(self,
                            bb_size_pred:torch.Tensor,
                            bb_center_pred:torch.Tensor,
                            bb_object_pred:torch.Tensor,
                            class_pred:torch.Tensor,
                            nms:bool=False,
                            correction_factor:torch.Tensor=None,
                            to_dict:bool=False)->List[torch.Tensor]:
        """Generalises recover_boxes to a batch.

        Args:
            bb_center_pred (torch.Tensor): center offset predictions.
            bb_size_pred (torch.Tensor): size predictions.
            bb_object_pred (torch.Tensor): objectness predictions.
            class_pred (torch.Tensor): class predictions.
            nms (bool, optional): whether to perform non-maximum suppression. 
                Defaults to False.
            correction_factor (torch.Tensor, optional): corrects the long 
                centres using a multiplicative factor. Defaults to None (no 
                correction).
            to_dict (bool, optional): returns the output as a dict. Defaults
                to False.

        Returns:
            long_bb (torch.Tensor): bounding boxes in the corner format 
                specified above.
            object_scores (torch.Tensor): objectness scores (1d).
            long_classes (torch.Tensor): classes.
        """
        def convert_to_dict(x):
            return {'boxes':x[0],'scores':x[1],'labels':x[2]}

        output = []
        for b in range(bb_size_pred.shape[0]):
            o = self.recover_boxes(
                bb_size_pred[b],bb_center_pred[b],
                bb_object_pred[b],class_pred[b],nms=nms,
                correction_factor=correction_factor)
            # corrects labels shape
            
            if o[0].shape[0] == 1 and len(o[2].shape) == 1:
                o = o[0],o[1],o[2].unsqueeze(0)
            if to_dict is True:
                o = convert_to_dict(o)
            output.append(o)
        return output

class CoarseDetector3d(torch.nn.Module):
    def __init__(self,
                 n_channels:int=1,
                 anchor_sizes:List=np.ones([1,6]),dev:str="cuda",
                 resnet_structure:List[Tuple[int,int,int,int]]=resnet_default,
                 maxpool_structure:List[Tuple[int,int,int]]=maxpool_default,
                 pyramid_layers:List[Tuple[int,int,int]]=pyramid_default,
                 adn_fn:torch.nn.Module=lambda s:ActDropNorm(
                     s,norm_fn=torch.nn.BatchNorm3d)):
        """Implementation of a YOLO network for object detection in 3d.

        Args:
            n_channels (int, optional): number of input channels. Defaults to 
                1.
            anchor_sizes (List, optional): anchor sizes. Redundant (kept for
                compatibility purposes). Defaults to np.ones([1,6]).
            dev (str, optional): device for memory allocation. Defaults to 
                "cuda".
            resnet_structure (List[Tuple[int,int,int,int]], optional): 
                structure for the ResNet backbone. Defaults to resnet_default.
            maxpool_structure (List[Tuple[int,int,int]], optional): structure
                for the maximum pooling operations. Defaults to 
                maxpool_default.
            pyramid_layers (List[Tuple[int,int,int]], optional): structure for
                the atrous spatial pyramid pooling layer. Defaults to None.
            adn_fn (torch.nn.Module, optional): function that is applied after
                each layer (activations, batch normalisation, dropout and etc. 
                should be specified here). Defaults to 
                lambda s:ActDropNorm(s,norm_fn=torch.nn.BatchNorm3d).
        """

        super().__init__()
        self.in_channels = n_channels
        self.anchor_sizes = anchor_sizes
        self.resnet_structure = resnet_structure
        self.maxpool_structure = maxpool_structure
        self.pyramid_layers = pyramid_layers
        self.adn_fn = adn_fn
        self.dev = dev

        self.init_layers()

    def init_layers(self):
        self.res_net = ResNetBackbone(
            3,self.in_channels,self.resnet_structure,adn_fn=self.adn_fn,
            maxpool_structure=self.maxpool_structure)
        self.feature_extraction = self.res_net 
        last_size = self.resnet_structure[-1][0]
        self.feature_extraction = torch.nn.Sequential(
            self.feature_extraction,
            self.adn_fn(last_size),
            AtrousSpatialPyramidPooling3d(
                last_size,last_size,self.pyramid_layers,
                adn_fn=lambda s: torch.nn.Identity()),
            self.adn_fn(last_size * (len(self.pyramid_layers))),
            ConcurrentSqueezeAndExcite3d(last_size * (len(self.pyramid_layers))),
            self.adn_fn(last_size * (len(self.pyramid_layers))))
        last_size = last_size * (len(self.pyramid_layers))
        self.object_prediction_layer = torch.nn.Sequential(
            torch.nn.Conv3d(last_size,last_size,1,padding="same"),
            self.adn_fn(last_size),
            torch.nn.Conv3d(last_size,1,1),
            torch.nn.Sigmoid())
    
    def forward(self,X:torch.Tensor)->torch.Tensor:
        features = self.feature_extraction(X)
        # size prediction
        objectness = self.object_prediction_layer(features)
        return objectness
