import torch
import torchvision
import torchmetrics

from ..types import *
from .layers import *

resnet_default = [
    (64,128,5,2),(128,256,3,5),
    #(128,256,3,2),(256,512,3,1)
    ]
maxpool_default = [
    (2,2,2),(1,1,1),(2,2,1),(2,2,1)]
pyramid_default = [3,5,[7,7,5],[9,9,5],[11,11,5]]

def check_overlap(bb1:torch.Tensor,bb2:torch.Tensor,ndim:int=3)->torch.Tensor:
    return torch.logical_and(
        torch.any(bb1[:,ndim:] > bb2[:,:ndim],axis=1),
        torch.any(bb1[:,:ndim] < bb2[:,ndim:],axis=1))

def bb_volume(bb:torch.Tensor,ndim:int=3)->torch.Tensor:
    return torch.prod(bb[:,ndim:] - bb[:,:ndim]+1,axis=1)

def calculate_iou(bb1:torch.Tensor,bb2:torch.Tensor,ndim:int=3)->torch.Tensor:
    inter_tl = torch.maximum(
        bb1[:,:ndim],bb2[:,:ndim])
    inter_br = torch.minimum(
        bb1[:,ndim:],bb2[:,ndim:])
    inter_volume = torch.prod(inter_br - inter_tl + 1,axis=1)
    union_volume = bb_volume(bb1,ndim)+bb_volume(bb2,ndim)-inter_volume
    return inter_volume/union_volume

def nms_nd(bb:torch.Tensor,scores:torch.Tensor,
           score_threshold:float,iou_threshold:float=0.5)->torch.Tensor:
    # first we sort the scores and boxes according to the scores and 
    # remove boxes with a score below `score_threshold`
    n,ndim = bb.shape
    ndim = int(ndim//2)
    original_idx = torch.arange(n,dtype=torch.long)
    scores_idxs = scores > score_threshold
    scores,bb = scores[scores_idxs],bb[scores_idxs]
    original_idx = original_idx[scores_idxs]
    score_order = torch.argsort(scores).flip(0)
    scores = scores[score_order]
    bb = bb[score_order]
    original_idx = original_idx[score_order]
    excluded = torch.zeros_like(scores,dtype=bool)
    idxs = torch.arange(scores.shape[0],dtype=torch.long)
    # iteratively remove boxes which have a high overlap with other boxes,
    # keeping those with higher confidence
    for i in range(bb.shape[0]):
        if excluded[i] == False:
            cur_bb = torch.unsqueeze(bb[i],0)
            cur_excluded = excluded[(i+1):]
            cur_idxs = idxs[(i+1):][~cur_excluded]
            remaining_bb = bb[cur_idxs]
            overlap = check_overlap(cur_bb,remaining_bb,ndim)
            remaining_bb = remaining_bb[overlap]
            cur_idxs = cur_idxs[overlap]
            iou = calculate_iou(cur_bb,remaining_bb,ndim)
            cur_idxs = cur_idxs[iou>iou_threshold]
            if cur_idxs.shape[0] > 0:
                excluded[cur_idxs] = True
    return original_idx[~excluded]

class mAP(torch.nn.Module):
    def __init__(self,ndim=3,score_threshold=0.5,iou_threshold=0.5,n_classes=2):
        super().__init__()
        self.ndim = ndim
        self.iou_threshold = iou_threshold
        self.score_threshold = score_threshold
        self.n_classes = n_classes
        # will do most of the AP heavylifting
        nc = None if n_classes==2 else n_classes
        self.average_precision = torchmetrics.AveragePrecision(nc)
        
        self.pred_list = []
        self.target_list = []
        self.pred_keys = ["boxes","scores","labels"]
        self.target_keys = ["boxes","labels"]
        
        self.hits = 0
        
    def check_input(self,x,what):
        if what == "pred":
            K = self.pred_keys
        elif what == "target":
            K = self.target_keys
        for y in x:
            keys = y.keys()
            if all([k in keys for k in K]):
                pass
            else:
                raise ValueError(
                    "{} should have the keys {}".format(what,K))
            if len(np.unique([y[k].shape[0] for k in K])) > 1:
                raise ValueError(
                    "Inputs in {} should have the same number of elements".format(what))

    def update(self,pred,target):
        self.check_input(pred,"pred")
        self.check_input(target,"target")
        
        self.pred_list.extend(pred)
        self.target_list.extend(target)

    def forward(self,pred,target):
        self.update(pred,target)
        
    def compute_image(self,pred,target):
        pbb,ps,pcp = [
            pred[k] for k in self.pred_keys]
        tbb,tc = [
            target[k] for k in self.target_keys]
        
        # step 1 - exclude low confidence predictions, sort by scores
        ps_v = ps > self.score_threshold
        pbb,ps,pcp = pbb[ps_v],ps[ps_v],pcp[ps_v]
        score_order = torch.argsort(ps)
        pbb = pbb[score_order]
        ps = ps[score_order]
        pcp = pcp[score_order]
        
        # step 2 - calculate iou
        n_pred = pbb.shape[0]
        n_target = tbb.shape[0]
        iou_array = torch.zeros([n_target,n_pred],device=pcp.device)
        any_hit = False
        for i in range(tbb.shape[0]):
            cur_bb = torch.unsqueeze(tbb[i],0)
            # start by calculating overlap
            overlap = check_overlap(cur_bb,pbb,self.ndim)
            if overlap.sum() > 0:
                cur_pbb = pbb[overlap]
                iou = calculate_iou(cur_bb,cur_pbb,self.ndim)
                iou_array[i,overlap] = iou
                any_hit = True
        
        if any_hit == True:
            # step 3 - filter by highest iou
            best_pred = torch.argmax(iou_array,1)

            # step 4 - threshold highest iou
            true_ious = iou_array[torch.arange(0,n_target),best_pred]
            hit = true_ious > self.iou_threshold

            # step 5 - update the precision recall curve 
            target_classes = tc[hit]
            pred_classes_proba = pcp[best_pred][hit]
            if hit.sum() > 0:
                self.average_precision.update(
                    pred_classes_proba,target_classes)
                self.hits += 1

    def compute(self):
        for pred,target in zip(self.pred_list,self.target_list):
            self.compute_image(pred,target)
        
        if self.hits > 0:
            return self.average_precision.compute()
        else:
            return torch.nan
    
    def reset(self):
        self.pred_list = []
        self.target_list = []
        self.hit = 0
        self.average_precision.reset()

class YOLONet3d(torch.nn.Module):
    def __init__(self,n_channels:int=1,n_c:int=2,
                 activation_fn:torch.nn.Module=torch.nn.ReLU,
                 anchor_sizes:List=np.ones([1,6]),dev:str="cuda",
                 resnet_structure:List[Tuple[int,int,int,int]]=resnet_default,
                 maxpool_structure:List[Tuple[int,int,int]]=maxpool_default,
                 pyramid_layers:List[Tuple[int,int,int]]=pyramid_default,
                 adn_fn:torch.nn.Module=lambda s:ActDropNorm(
                     s,norm_fn=torch.nn.BatchNorm3d)):
        super().__init__()
        self.in_channels = n_channels
        self.n_c = n_c
        self.act_fn = activation_fn
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
            torch.reshape(torch.Tensor(anchor_size),[3,1,1,1])
            for anchor_size in self.anchor_sizes]
        self.anchor_tensor = torch.cat(
            self.anchor_tensor,dim=0).to(self.dev)

    def init_layers(self):
        self.res_net = ResNetBackbone(
            3,self.in_channels,self.resnet_structure,adn_fn=self.adn_fn,
            maxpool_structure=self.maxpool_structure)
        self.feature_extraction = self.res_net 
        last_size = self.resnet_structure[-1][0]
        #self.feature_extraction = FeaturePyramidNetworkBackbone(
        #    self.res_net,3,self.resnet_structure,adn_fn=self.adn_fn)
        #last_size = self.resnet_structure[0][0]
        self.feature_extraction = torch.nn.Sequential(
            self.feature_extraction,
            self.adn_fn(last_size),
            AtrousSpatialPyramidPooling3d(
                last_size,last_size,self.pyramid_layers,
                act_fn=lambda: torch.nn.Identity()),
            self.adn_fn(last_size * (len(self.pyramid_layers))),
            ConcurrentSqueezeAndExcite3d(last_size * (len(self.pyramid_layers))),
            self.adn_fn(last_size * (len(self.pyramid_layers))))
        last_size = last_size * (len(self.pyramid_layers))
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
            torch.nn.ReLU(),
            torch.nn.Conv3d(last_size,self.n_b,1),
            torch.nn.Sigmoid())
        if self.n_c == 2:
            self.classifiation_layer = torch.nn.Sequential(
                torch.nn.Conv3d(last_size,last_size,1),
                self.adn_fn(last_size),
                torch.nn.Conv3d(last_size,1,1),
                torch.nn.Sigmoid())
        else:
            self.classifiation_layer = torch.nn.Sequential(
                torch.nn.Conv3d(last_size,last_size,1),
                self.adn_fn(last_size),
                torch.nn.Conv3d(last_size,self.n_c,1),
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
                      class_pred:torch.Tensor,
                      nms:bool=False)->torch.Tensor:
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
        object_scores = bb_object_pred[:,x,y,z,a].swapaxes(0,1).squeeze()
        # unpack values from different anchors 
        long_sizes = bb_size_pred[:,x,y,z,a].swapaxes(0,1)
        long_centers = bb_center_pred[:,x,y,z,a].swapaxes(0,1)
        if self.n_c > 2:
            long_classes = class_pred[:,:,x,y,z].swapaxes(0,-1).squeeze()
        else:
            long_classes = bb_object_pred[:,x,y,z,a].swapaxes(0,1).squeeze()

        upper_corner = long_centers - long_sizes/2
        lower_corner = long_centers + long_sizes/2
        long_bb = torch.cat([upper_corner,lower_corner],1)
        if nms == True:
            bb_idxs = nms_nd(
                long_bb,object_scores.reshape(-1),0.8,0.5)
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
                else:
                    return (long_bb,object_scores,long_classes)
            else:
                return (long_bb,object_scores,long_classes)
    
    def recover_boxes_batch(self,bb_size_pred:torch.Tensor,
                            bb_center_pred:torch.Tensor,
                            bb_object_pred:torch.Tensor,
                            class_pred:torch.Tensor,
                            nms:bool=False,
                            to_dict:bool=False)->List[torch.Tensor]:
        def convert_to_dict(x):
            return {'boxes':x[0],'scores':x[1],'labels':x[2]}

        output = []
        for b in range(bb_size_pred.shape[0]):
            o = self.recover_boxes(
                bb_size_pred[b],bb_center_pred[b],
                bb_object_pred[b],class_pred,nms=nms)
            if to_dict == True:
                o = convert_to_dict(o)
            output.append(o)
        return output

class CoarseDetector3d(torch.nn.Module):
    def __init__(self,n_channels:int=1,
                 activation_fn:torch.nn.Module=torch.nn.ReLU,
                 anchor_sizes:List=np.ones([1,6]),dev:str="cuda",
                 resnet_structure:List[Tuple[int,int,int,int]]=resnet_default,
                 maxpool_structure:List[Tuple[int,int,int]]=maxpool_default,
                 pyramid_layers:List[Tuple[int,int,int]]=pyramid_default,
                 adn_fn:torch.nn.Module=lambda s:ActDropNorm(
                     s,norm_fn=torch.nn.BatchNorm3d)):
        super().__init__()
        self.in_channels = n_channels
        self.act_fn = activation_fn
        self.anchor_sizes = anchor_sizes
        self.resnet_structure = resnet_structure
        self.maxpool_structure = maxpool_structure
        self.pyramid_layers = pyramid_layers
        self.adn_fn = adn_fn
        self.dev = dev
        self.n_b = len(self.anchor_sizes)

        self.init_layers()

    def init_layers(self):
        self.res_net = ResNetBackbone(
            3,self.in_channels,self.resnet_structure,adn_fn=self.adn_fn,
            maxpool_structure=self.maxpool_structure)
        self.feature_extraction = self.res_net 
        last_size = self.resnet_structure[-1][0]
        #self.feature_extraction = FeaturePyramidNetworkBackbone(
        #    self.res_net,3,self.resnet_structure,adn_fn=self.adn_fn)
        #last_size = self.resnet_structure[0][0]
        self.feature_extraction = torch.nn.Sequential(
            self.feature_extraction,
            self.adn_fn(last_size),
            AtrousSpatialPyramidPooling3d(
                last_size,last_size,self.pyramid_layers,
                act_fn=lambda: torch.nn.Identity()),
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
