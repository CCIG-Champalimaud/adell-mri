import time
import numpy as np
import torch
import torch.nn.functional as F
import einops

from ...custom_types import TensorList
from ..layers.adn_fn import ActDropNorm,get_adn_fn
from ..layers.batch_ensemble import BatchEnsembleWrapper
from ..layers.standard_blocks import GlobalPooling, simple_convolutional_block
from ..layers.res_net import ResNet,ResNetBackboneAlt,ProjectionHead
from ..layers.self_attention import (
    ConcurrentSqueezeAndExcite2d,
    ConcurrentSqueezeAndExcite3d)
from ..segmentation.unet import UNet
from ..layers.linear_blocks import MLP,SeqPool
from ..layers.vit import ViT,FactorizedViT,TransformerBlockStack

from typing import Union,Dict,List,Tuple,Callable

resnet_default = [(64,128,5,2),(128,256,3,5)]
maxpool_default = [(2,2,2),(2,2,2)]

def label_to_ordinal(label:torch.Tensor,
                     n_classes:int,
                     ignore_0:bool=True)->torch.Tensor:
    """
    Converts a label to an ordinal classification tensor. In essence, for a 
    given label L, this function returns an n-hot encoded version of L that
    is 0 for indices < L and 1 otherwise.

    Args:
        label (torch.Tensor): one dimensional tensor with labels.
        n_classes (int): number of classes.
        ignore_0 (bool, optional): whether 0 should not be considered a class.
            Defaults to True.

    Returns:
        torch.Tensor: ordinal classes.
    """
    label = torch.squeeze(label,1)
    if ignore_0 is True:
        label = torch.clamp(label - 1,min=0)
    one_hot = F.one_hot(label,n_classes)
    one_hot = one_hot.unsqueeze(1).swapaxes(1,-1).squeeze(-1)
    one_hot = torch.clamp(one_hot,max=1)
    one_hot_cumsum = torch.cumsum(one_hot,axis=1)
    output = torch.ones_like(one_hot_cumsum,device=one_hot_cumsum.device)
    return output - one_hot_cumsum

def ordinal_prediction_to_class(x:torch.Tensor)->torch.Tensor:
    """
    Converts an ordinal prediction to a specific class.

    Args:
        x (torch.Tensor): ordinal prediction tensor.

    Returns:
        torch.Tensor: categorical prediction.
    """
    x_thresholded = F.threshold(x,0.5,1)
    output = x_thresholded.argmax(dim=1)
    # consider 0 only when no class class reaches the threshold
    output[x_thresholded.sum(dim=1)>0] = 0
    return output

class VGG(torch.nn.Module):
    """
    Very simple and naive VGG net for standard categorical classification.
    """
    def __init__(self, spatial_dimensions: int=3,
                 n_channels: int=1,
                 n_classes: int=2,
                 feature_extraction: torch.nn.Module=None,
                 resnet_structure: List[Tuple[int,int,int,int]]=resnet_default,
                 maxpool_structure: List[Tuple[int,int,int]]=maxpool_default,
                 adn_fn: torch.nn.Module=None,
                 res_type: str="resnet",
                 batch_ensemble: bool=False,
                 ):
        """
        Args:
            spatial_dimensions (int, optional): number of spatial dimensions. 
                Defaults to 3.
            n_channels (int, optional): number of input channels. Defaults to 
                1.
            n_classes (int, optional): number of classes. Defaults to 2.
        """
        super().__init__()
        self.in_channels = n_channels
        self.n_classes = n_classes

        self.conv1 = simple_convolutional_block(self.in_channels, 64)
        self.conv2 = simple_convolutional_block(128, 128)
        self.conv3 = simple_convolutional_block(256, 256)

        final_n = 1
        self.last_act = torch.nn.Sigmoid()

        self.classification_layer = torch.nn.Sequential(
            GlobalPooling(),
            MLP(512,final_n,[512 for _ in range(3)],
                adn_fn=get_adn_fn(1,"batch","gelu")))
    
    def forward(self,X: torch.Tensor,
                return_features:bool=False)->torch.Tensor:
        """Forward method.

        Args:
            X (torch.Tensor): input tensor

        Returns:
            torch.Tensor: output (classification)
        """
        X =self.conv1(X)
        X = self.conv2(X)
        X = self.conv3(X)
        if return_features == True:
            return X
        return self.classification_layer(X)

class CatNet(torch.nn.Module):
    """
    Case class for standard categorical classification. Defaults to 
    feature extraction using ResNet.
    """
    def __init__(self,
                 spatial_dimensions: int=3,
                 n_channels: int=1,
                 n_classes: int=2,
                 feature_extraction: torch.nn.Module=None,
                 resnet_structure: List[Tuple[int,int,int,int]]=resnet_default,
                 maxpool_structure: List[Tuple[int,int,int]]=maxpool_default,
                 adn_fn: torch.nn.Module=None,
                 res_type: str="resnet",
                 batch_ensemble: bool=False):
        """
        Args:
            spatial_dimensions (int, optional): number of spatial dimensions. 
                Defaults to 3.
            n_channels (int, optional): number of input channels. Defaults to 
                1.
            n_classes (int, optional): number of classes. Defaults to 2.
            feature_extraction (torch.nn.Module, optional): module to use for
                feature extraction. Defaults to None (builds a ResNet using
                `resnet_structure` and `maxpool_structure`).
            resnet_structure (List[Tuple[int,int,int,int]], optional): 
                structure for ResNet should be a list of tuples with 4 
                elements, corresponding to input size, intermediate size,
                kernel size and number of consecutive residual operations. 
                Defaults to [(64,128,5,2),(128,256,3,5)].
            maxpool_structure (List[Tuple[int,int,int]], optional): structure
                for the max pooling operations. Must be a list with the same
                length as resnet_structure and defining the kernel size and 
                stride (these will be identical). Defaults to 
                [(2,2,2),(2,2,2)].
            adn_fn (torch.nn.Module, optional): activation dropout 
                normalization function. Must be a function that takes an 
                argument (number of channels) and returns a torch Module. 
                Defaults to None (batch normalization).
            res_type (str, optional): type of residual operation, can be either
                "resnet" or "resnext". Defaults to "resnet".
            batch_ensemble (bool, optional): uses batch ensemble layers. 
                Defaults to False.
        """
        super().__init__()
        self.spatial_dim = spatial_dimensions
        self.in_channels = n_channels
        self.n_classes = n_classes
        self.feature_extraction = feature_extraction
        self.resnet_structure = resnet_structure
        self.maxpool_structure = maxpool_structure
        self.adn_fn = adn_fn
        self.res_type = res_type
        self.batch_ensemble = batch_ensemble

        if self.adn_fn is None:
            if self.spatial_dim == 2:
                self.adn_fn = lambda s:ActDropNorm(
                    s,norm_fn=torch.nn.BatchNorm2d)
            if self.spatial_dim == 3:
                self.adn_fn = lambda s:ActDropNorm(
                    s,norm_fn=torch.nn.BatchNorm3d)

        self.init_layers()
        self.init_classification_layer()

    def init_layers(self):
        if self.feature_extraction is None:
            self.res_net = ResNetBackboneAlt(
                self.spatial_dim,self.in_channels,self.resnet_structure,
                adn_fn=self.adn_fn,maxpool_structure=self.maxpool_structure,
                res_type=self.res_type,batch_ensemble=self.batch_ensemble)
            self.feature_extraction = self.res_net
            self.last_size = self.resnet_structure[-1][0]
        else:
            input_shape = [2,self.in_channels,128,128]
            if self.spatial_dim == 3:
                input_shape.append(32)
            example_tensor = torch.ones(input_shape)
            self.last_size = self.feature_extraction(example_tensor).shape[1]

    def init_classification_layer(self):
        if self.n_classes == 2:
            final_n = 1
            self.last_act = torch.nn.Sigmoid()
        else:
            final_n = self.n_classes
            self.last_act = torch.nn.Softmax(1)
        self.gp = GlobalPooling()
        self.classification_layer = torch.nn.Sequential(
            MLP(self.last_size,final_n,[self.last_size for _ in range(3)],
                adn_fn=get_adn_fn(1,"batch","gelu")))
        if self.batch_ensemble > 0:
            self.classification_layer = BatchEnsembleWrapper(
                self.classification_layer,self.batch_ensemble,self.last_size,
                final_n,torch.nn.Identity)
    
    def forward(self,X:torch.Tensor,return_features:bool=False)->torch.Tensor:
        """Forward method.

        Args:
            X (torch.Tensor): input tensor
            return_features (bool, optional): returns the features rather than
                the classification_head output. Defaults to False.

        Returns:
            torch.Tensor: output (classification or features)
        """
        features = self.gp(self.feature_extraction(X))
        if return_features == True:
            return features
        classification = self.classification_layer(features)
        return classification

class OrdNet(CatNet):
    """
    Same as CatNet but the output is ordinal.
    """
    def __init__(self,*args,**kwargs):
        """
        Args:
            args, kwargs: same arguments as CatNet.
        """
        super().__init__(*args,**kwargs)

    def init_classification_layer(self):
        self.gp = GlobalPooling()
        self.classification_layer = torch.nn.Sequential(
            torch.nn.Linear(self.last_size,self.last_size),
            torch.nn.ReLU(),
            torch.nn.Linear(self.last_size,1))
        self.bias = torch.nn.parameter.Parameter(
            torch.zeros([1,self.n_classes-1]))
        self.last_act = torch.nn.Sigmoid()
    
    def forward(self,X:torch.Tensor,return_features:bool=False)->torch.Tensor:
        """Forward method.

        Args:
            X (torch.Tensor): input tensor
            return_features (bool, optional): returns the features rather than
                the classification_head output. Defaults to False.

        Returns:
            torch.Tensor: output (classification or features)
        """
        features = self.gp(self.feature_extraction(X))
        if return_features == True:
            return features
        p_general = self.classification_layer(features)
        p_ordinal = self.last_act(p_general + self.bias)
        return p_ordinal

class SegCatNet(torch.nn.Module):
    """
    Uses the bottleneck and final layer features from a U-Net module
    to train a classifier. The `u_net` module should have a `forward` 
    method that can take a `return_features` argument that, when set to
    True, returns a tuple of tensors: prediction, final layer features 
    (before prediction) and bottleneck features.
    """
    def __init__(self,
                 spatial_dim:int,
                 u_net:torch.nn.Module,
                 n_input_channels:int,
                 n_features_backbone:int,
                 n_features_final_layer:int,
                 n_classes:int):
        """
        Args:
            spatial_dim (int): number of spatial dimensions.
            u_net (torch.nn.Module): U-Net module.
            n_input_channels (int): number of input channels.
            n_features_backbone (int): number of channels in the U-Net 
                backbone.
            n_features_final_layer (int): number of features in the U-Net final
                layer.
            n_classes (int): number of classes.
        """
        super().__init__()
        self.spatial_dim = spatial_dim
        self.u_net = u_net
        self.n_input_channels = n_input_channels
        self.n_features_backbone = n_features_backbone
        self.n_features_final_layer = n_features_final_layer
        self.n_classes = n_classes

        if self.n_classes == 2: 
            self.nc = 1
        else: 
            self.nc = self.n_classes

        self.init_final_layer_classification()
        self.init_bottleneck_classification()
        self.init_weighted_average()

    def init_final_layer_classification(self):
        d = self.n_features_final_layer
        input_d = d
        inter_d = self.n_features_final_layer * 2
        structure = [[input_d,inter_d,3,2],
                     [d*2,d*2,3,2],
                     [d*4,d*4,3,2]]
        prediction_structure = [d*4,d*4,d*4]
        self.resnet_backbone_args = {
            "spatial_dim":self.spatial_dim,
            "in_channels":input_d,
            "structure":structure,
            "maxpool_structure":[2 for _ in structure],
            "res_type":"resnet",
            "adn_fn":get_adn_fn(self.spatial_dim,"batch","swish",0.1)}
        self.resnet_prediction_args = {
            "in_channels":structure[-1][0],
            "structure":prediction_structure,
            "adn_fn":get_adn_fn(1,"batch","swish",0.1)}
        self.final_layer_classifier = torch.nn.Sequential(
            ResNet(
                self.resnet_backbone_args,self.resnet_prediction_args),
            torch.nn.Linear(d*4,self.nc,bias=False))
    
    def init_bottleneck_classification(self):
        d = self.n_features_backbone
        self.bottleneck_prediction_structure = [d,d*2,d*4,d*2,d]
        
        self.bottleneck_classifier = torch.nn.Sequential(
            ProjectionHead(
                d,self.bottleneck_prediction_structure,
                adn_fn=get_adn_fn(1,"batch","swish",0.1)),
            torch.nn.Linear(d,self.nc,bias=False))
    
    def init_weighted_average(self):
        self.weighted_average = torch.nn.Linear(self.nc*2,self.nc,bias=False)

    def forward(self,X,**kwargs):
        times = {}
        times['a'] = time.time()
        with torch.no_grad():
            pred,final_layer,bottleneck = self.u_net.forward(
               X,return_features=True,**kwargs)
        times['b'] = time.time()

        class_fl = self.final_layer_classifier(
            torch.cat([final_layer],axis=1))
        times['c'] = time.time()
        class_bn = self.bottleneck_classifier(bottleneck)
        times['d'] = time.time()
        features = torch.cat([class_fl,class_bn],axis=1)
        classification = self.weighted_average(features)
        times['e'] = time.time()
        
        return classification

class GenericEnsemble(torch.nn.Module):
    """
    Generically combines multiple encoders to produce an ensemble model.
    """
    def __init__(self,
                 spatial_dimensions:int,
                 networks:List[torch.nn.Module],
                 n_features:Union[List[int],int],
                 head_structure:List[int],
                 n_classes:int,
                 head_adn_fn: Callable=None,
                 sae: bool=False):
        """
        Args:
            spatial_dimensions (int): spatial dimension of input.
            networks (List[torch.nn.Module]): list of Torch modules.
            n_features (List[int]): list of output sizes for networks.
            head_structure (List[int]): structure for the prediction head.
            n_classes (int): number of classes.
            head_adn_fn (Callable, optional): activation-dropout-normalization
                function for the prediction head. Defaults to None (no 
                function).
            sae (bool, optional): applies a squeeze and excite layer to the 
                output of each network. Defaults to False
        """
        super().__init__()
        self.spatial_dimensions = spatial_dimensions
        self.networks = torch.nn.ModuleList(networks)
        self.n_features = n_features
        self.head_structure = head_structure
        self.n_classes = n_classes
        self.head_adn_fn = head_adn_fn
        self.sae = sae

        if isinstance(self.n_features,int):
            self.n_features = [self.n_features for _ in self.networks]
        self.n_features_final = sum(self.n_features)
        self.initialize_sae_if_necessary()
        self.initialize_head()

    def initialize_head(self):
        if self.n_classes == 2:
            nc = 1
        else:
            nc = self.n_classes
        self.prediction_head = MLP(
            self.n_features_final,nc,
            self.head_structure,
            self.head_adn_fn)

    def initialize_sae_if_necessary(self):
        if self.sae is True:
            if self.spatial_dimensions == 2:
                self.preproc_method = torch.nn.ModuleList(
                    [ConcurrentSqueezeAndExcite2d(f)
                    for f in self.n_features])
            elif self.spatial_dimensions == 3:
                self.preproc_method = torch.nn.ModuleList(
                    [ConcurrentSqueezeAndExcite3d(f)
                    for f in self.n_features])
        else:
            self.preproc_method = torch.nn.ModuleList(
                [torch.nn.Identity() for _ in self.n_features])
    
    def forward(self,X):
        if isinstance(X,torch.Tensor):
            X = [X for _ in self.networks]
        outputs = []
        for x,network,pp in zip(X,self.networks,self.preproc_method):
            out = pp(network(x))
            out = out.flatten(start_dim=2).max(-1).values
            outputs.append(out)
        outputs = torch.concat(outputs,1)
        output = self.prediction_head(outputs)
        return output
    
class EnsembleNet(torch.nn.Module):
    """
    Creates an ensemble of networks which can be trained online. The 
    input of each network can be different and the forward method supports
    predictions with missing data (as the average of all networks).
    """
    def __init__(self,
                 cat_net_args:Union[Dict[str,int],List[Dict[str,int]]]):
        """
        Args:
            cat_net_args (Union[Dict[str,int],List[Dict[str,int]]], optional):
                dictionary or list of dictionaries containing arguments for a 
                CatNet models.
        """
        super().__init__()
        self.cat_net_args = cat_net_args
        
        self.coerce_cat_net_args_if_necessary()
        self.check_args()
        self.init_networks()
        self.define_final_activation()

    def check_args(self):
        n_classes = []
        for d in self.cat_net_args_:
            for k in d:
                if k == "n_classes":
                    n_classes.append(d[k])
        unique_classes = np.unique(n_classes)
        if len(unique_classes) != 1:
            raise Exception("Classes should be identical across CatNets")
        elif unique_classes[0] == 1:
            raise Exception("n_classes == 1 not supported. If the problem is \
                binary set n_classes == 2")
        else:
            self.n_classes_ = unique_classes[0]

    def coerce_cat_net_args_if_necessary(self):
        # coerce cat_net_args if necessary
        if isinstance(self.cat_net_args,dict):
            self.cat_net_args_ = [self.input_structure]
        elif isinstance(self.cat_net_args,list):
            self.cat_net_args_ = self.cat_net_args
        else:
            raise TypeError("cat_net_args must be dict or list of dicts")

    def init_networks(self):
        self.networks = torch.nn.ModuleList([])
        for c_n_a in zip(self.cat_net_args_):
            self.networks.append(CatNet(**c_n_a))
            
    def define_final_activation(self):
        if self.n_classes_ == 2:
            self.final_activation = torch.nn.Sigmoid()
        else:
            self.final_activation = torch.nn.Softmax(self.n_classes_,1)
        
    def forward(self,X:List[torch.Tensor]):
        predictions = []
        for n,x in zip(self.networks,X):
            if x is not None:
                predictions.append(self.final_activation(n(x)))
        return sum(predictions) / len(predictions)

class UNetEncoder(UNet):
    """
    U-Net encoder for classification.
    """
    def __init__(self,
                 head_structure:List[int],
                 n_classes:int,
                 head_adn_fn: Callable=None,
                 *args,**kwargs):
        """
        Args:
            head_structure (List[int]): structure for the prediction head.
            n_classes (int): number of classes.
            head_adn_fn (Callable, optional): activation-dropout-normalization
                function for the prediction head. Defaults to None (no 
                function).
        """
        self.head_structure = head_structure
        self.n_classes = n_classes
        self.head_adn_fn = head_adn_fn
        
        kwargs["encoder_only"] = True
        super().__init__(*args,**kwargs)

        self.n_features = self.depth[-1]
        if self.head_structure is not None:
            self.prediction_head = MLP(
                self.n_features,
                self.n_classes,
                self.head_structure,
                self.head_adn_fn)
        else:
            self.prediction_head = None
            
        self.initialize_head()

    def initialize_head(self):
        if self.n_classes == 2:
            nc = 1
        else:
            nc = self.n_classes
        if self.head_structure is not None:
            self.prediction_head = MLP(
                self.n_features,nc,
                self.head_structure,
                self.head_adn_fn)
        else:
            self.prediction_head = None
        
    def forward(self,X:torch.Tensor,return_features:bool=False)->torch.Tensor:
        """Forward pass for this class.

        Args:
            X (torch.Tensor): input tensor
            return_features (bool, optional): returns the features rather than
                the classification_head output. Defaults to False.

        Returns:
            torch.Tensor
        """
        encoding_out = []
        curr = X
        for op,op_ds in self.encoding_operations:
            curr = op(curr)
            encoding_out.append(curr)
            curr = op_ds(curr)
        if self.prediction_head is None:
            return curr
        out = curr.flatten(start_dim=2).max(-1).values
        if return_features == True:
            return out
        out = self.prediction_head(out)
        return out

class ViTClassifier(ViT):
    """
    Implementation of the vision transformer (ViT) as a classifier.
    """
    """
    Implementation of the vision transformer (ViT) as a classifier.
    """
    def __init__(self,
                 n_classes:int,
                 use_class_token:bool=False,
                 *args,**kwargs):
        """
        Args:
            n_classes (int): number of classses.
            use_class_token (Union[str,bool], optional): whether a class token 
                is being used. If "seqpool" uses the SeqPool method described 
                in the compact convolutional transformers paper. Defaults to 
                False.
            args, kwargs: args that are to be used to parametrize the ViT.
        """
        if use_class_token == "seqpool":
            kwargs["use_class_token"] = False
            self.use_seq_pool = True
        else:
            kwargs["use_class_token"] = use_class_token
            self.use_seq_pool = False
        super().__init__(*args,**kwargs)
        self.n_classes = n_classes

        if self.n_classes == 2:
            nc = 1
        else:
            nc = self.n_classes
        if self.use_seq_pool == True:
            self.seqpool = SeqPool(self.input_dim_primary)
        self.classification_layer = torch.nn.Sequential(
            MLP(
                self.input_dim_primary,nc,
                [self.input_dim_primary for _ in range(1)],
                adn_fn=get_adn_fn(1,"layer","gelu"))
        )

    def forward(
        self,
        X:torch.Tensor,
        return_features:bool=False)->Tuple[torch.Tensor,TensorList]:
        """Forward pass.

        Args:
            X (torch.Tensor): tensor of shape 
                [-1,self.n_channels,*self.image_size]
            return_features (bool, optional): returns the features rather than
                the classification_head output. Defaults to False.

        Returns:
            torch.Tensor: tensor of shape [...,self.input_dim_primary]
        """
        embeded_X,_ = super().forward(X)
        if self.use_seq_pool is True:
            embeded_X = self.seqpool(embeded_X).squeeze(1)
        elif self.use_class_token is True:
            embeded_X = embeded_X[:,0]
        else:
            embeded_X = embeded_X.mean(1)
        if return_features == True:
            return embeded_X
        classification = self.classification_layer(embeded_X)
        return classification

class FactorizedViTClassifier(FactorizedViT):
    """
    Implementation of a factorized ViT. The fundamental aspect of the 
    factorized ViT is its separate processing within and between slice 
    information. More concretely, for a given input tensor with shape
    [b,c,h,w,d]:
    
        1. The slices are linearly embeded as [b,d,(h/x*w/y),(x*y*c)]
        2. A transformer is applied to this (to the last two dimensions of the
            input)
        3. The information in the tensor is aggregated (using either a class 
            token or MAP) and [b,d,(h/x*w/y),(x*y*c)] -> [b,(h/x*w/y),(x*y*c)]
        4. A transformer is applied to this in a more standard fashion.
        
    Since the arguments are similar between VitClassifier and 
    FactorizedViTClassifier, the number of blocks (number_of_blocks) is split
    between the first transformer block (within slice) and second transformer
    block (between slices).
    """
    def __init__(self,
                 n_classes:int,
                 use_class_token:bool=False,
                 *args,**kwargs):
        """
        Args:
            n_classes (int): number of classses.
            use_class_token (Union[str,bool], optional): whether a class token 
                is being used. If "seqpool" uses the SeqPool method described 
                in the compact convolutional transformers paper. Defaults to 
                False.
            args, kwargs: args that are to be used to parametrize the ViT.
        """
        if use_class_token == "seqpool":
            kwargs["use_class_token"] = False
            self.use_seq_pool = True
        else:
            kwargs["use_class_token"] = use_class_token
            self.use_seq_pool = False
        super().__init__(*args,**kwargs)
        self.n_classes = n_classes

        if self.n_classes == 2:
            nc = 1
        else:
            nc = self.n_classes
        if self.use_seq_pool is True:
            self.seqpool = SeqPool(self.input_dim_primary)
        self.classification_layer = torch.nn.Sequential(
            MLP(
                self.input_dim_primary,nc,
                [self.input_dim_primary for _ in range(1)],
                adn_fn=get_adn_fn(1,"layer","gelu")))
    
    def forward(
        self,
        X:torch.Tensor,
        return_features:bool=False)->Tuple[torch.Tensor,TensorList]:
        """Forward pass.

        Args:
            X (torch.Tensor): tensor of shape 
                [-1,self.n_channels,*self.image_size]
            return_features (bool, optional): returns the features rather than
                the classification_head output. Defaults to False.

        Returns:
            torch.Tensor: tensor of shape [...,self.input_dim_primary]
        """
        embeded_X = super().forward(X)
        if self.use_seq_pool is True:
            embeded_X = self.seqpool(embeded_X).squeeze(1)
        elif self.use_class_token is True:
            embeded_X = embeded_X[:,0]
        else:
            embeded_X = embeded_X.mean(1)
        if return_features == True:
            return embeded_X
        classification = self.classification_layer(embeded_X)
        return classification

class MONAIViTClassifier(torch.nn.Module):
    """
    Small wraper around the MONAI ViT as a classifier with the same arguments
    as those used in ViT and FactorizedViT.
    """
    def __init__(self,
                 n_classes:int,
                 use_class_token:bool=False,
                 *args,**kwargs):
        import monai
        kwargs["use_class_token"] = use_class_token
        super().__init__()
        
        self.n_classes = n_classes
        if self.n_classes == 2:
            nc = 1
        else:
            nc = self.n_classes

        self.network = monai.networks.nets.vit.ViT(
            in_channels=kwargs["n_channels"],
            img_size=[int(x) for x in kwargs["image_size"]],
            patch_size=kwargs["patch_size"],
            hidden_size=kwargs["hidden_dim"],
            mlp_dim=kwargs["mlp_structure"][0],
            num_layers=kwargs["number_of_blocks"],
            num_heads=kwargs["n_heads"],
            pos_embed="conv",
            classification=True,
            num_classes=nc
        )
    
    def forward(
        self,
        X:torch.Tensor)->Tuple[torch.Tensor,TensorList]:
        """Forward pass.

        Args:
            X (torch.Tensor): tensor of shape 
                [-1,self.n_channels,*self.image_size]

        Returns:
            torch.Tensor: tensor of shape [...,self.input_dim_primary]
        """
        return self.network(X)[0]

class TransformableTransformer(torch.nn.Module):
    """
    Transformer that uses a 2D module to transform a 3D image along a given
    dimension dim into a transformer ready format. In essence, given an 
    input with size [b,c,h,w,d] and a module that processes 2D images with
    output size [b,o]:
    
        1. Iterate over a given dim of the input (0/1/2, corresponding to 
            h, w or d) producing (for dim=2) slices with shape [b,c,h,w]
        2. Apply module to each slice [b,c,h,w] -> [b,o]
        3. Concatenate modules along the channel dimension 
            [[b,o],...,[b,o]] -> [b,d,o]
        4. Apply transformer to this output and aggregate (shape [b,o])
        5. Apply MLP classifier to the transformer output
        
    In theory, this can also be applied to other Tensor shapes, but the tested
    use is as stated above.
    """
    def __init__(self,
                 module:torch.nn.Module,
                 module_out_dim:int,
                 n_classes:int,
                 classification_structure:torch.nn.Module,
                 input_dim:int=None,
                 classification_adn_fn:Callable=get_adn_fn(
                     1,"layer","gelu",0.1),
                 n_slices:int=None,
                 dim:int=2,
                 use_class_token:bool=True,
                 reduce_fn:str="max",
                 *args,**kwargs):
        """
        Args:
            module (torch.nn.Module): end-to-end module that takes a batched 
                set of 2D images and output a vector for each image.
            module_out_dim (int): output size for module.
            n_classes (int): number of output classes.
            classification_structure (torch.nn.Module): structure for the 
                classification MLP.
            input_dim (int, optional): input dimension for the transformer. If
                different from module_out_dim, applies linear layer to input.
                Defaults to None (same as module_out_dim).
            classification_adn_fn (Callable, optional): ADN function for the
                MLP module. Defaults to get_adn_fn( 1,"layer","gelu",0.1).
            n_slices (int, optional): number of slices. Used to initialize 
                positional embedding. Defaults to None (no positional 
                embedding).
            dim (int, optional): dimension along which the module is applied.
                Defaults to 2.
            use_class_token (bool, optional): whether a classification token
                should be used. Defaults to True.
            reduce_fn (str, optional): function used to reduce features coming
                from feature extraction module.
        """
        
        assert dim in [0,1,2], "dim must be one of [0,1,2]"
        super().__init__()
        self.module = module
        self.module_out_dim = module_out_dim
        self.n_classes = n_classes
        self.classification_structure = classification_structure
        self.input_dim = input_dim
        self.classification_adn_fn = classification_adn_fn
        self.n_slices = n_slices
        self.dim = dim
        self.use_class_token = use_class_token
        self.reduce_fn = reduce_fn
        
        self.vol_to_2d = einops.layers.torch.Rearrange(
            "b c h w s -> (b s) c h w")
        self.rep_to_emb = einops.layers.torch.Rearrange(
            "(b s) v -> b s v",s=self.n_slices)
        
        if input_dim is not None:
            self.transformer_input_dim = input_dim
            self.input_layer = torch.nn.Sequential(
                torch.nn.LayerNorm(module_out_dim),
                torch.nn.Linear(module_out_dim,input_dim),
                torch.nn.LayerNorm(input_dim))
        else:
            self.transformer_input_dim = module_out_dim
            self.input_layer = torch.nn.LayerNorm(module_out_dim)
            
        kwargs["input_dim_primary"] = self.transformer_input_dim
        kwargs["attention_dim"] = self.transformer_input_dim
        kwargs["hidden_dim"] = self.transformer_input_dim
        self.tbs = TransformerBlockStack(*args,**kwargs)
        self.classification_module = MLP(
            self.transformer_input_dim,
            1 if self.n_classes == 2 else self.n_classes,
            structure=self.classification_structure,
            adn_fn=self.classification_adn_fn)
        self.initialize_classification_token()
        self.init_positional_embedding()
    
    def initialize_classification_token(self):
        """
        Initializes the classification token.
        """
        if self.use_class_token is True:
            self.class_token = torch.nn.Parameter(
               torch.zeros([1,1,self.transformer_input_dim]))
    
    def init_positional_embedding(self):
        """
        Initializes the positional embedding.
        """
        self.positional_embedding = None
        if self.n_slices is not None:
            self.positional_embedding = torch.nn.Parameter(
                torch.zeros([1,self.n_slices,1]))
            torch.nn.init.trunc_normal_(
                self.positional_embedding,mean=0.0,std=0.02,a=-2.0,b=2.0)

    def iter_over_dim(self,X:torch.Tensor)->torch.Tensor:
        """Iterates a tensor along the dim specified in the constructor.

        Args:
            X (torch.Tensor): a tensor with shape [b,c,h,w,d].

        Yields:
            Iterator[torch.Tensor]: tensor slices along dim.
        """
        dim = 2 + self.dim
        for i in range(X.shape[dim]):
            curr_idx = [i if j == dim else slice(0,None) 
                        for j in range(len(X.shape))]
            yield X[tuple(curr_idx)]
        
    def extract_features(self,X:torch.Tensor)->torch.Tensor:
        if len(X.shape) > 2:
            X = X.flatten(start_dim=2)
        if self.reduce_fn == "mean":
            return X.mean(-1)
        elif self.reduce_fn == "max":
            return X.max(-1).values
    
    def v_module_old(self,X:torch.Tensor)->torch.Tensor:
        sh = X.shape
        n_slices = sh[2+self.dim]
        ssl_representation = torch.zeros(
            [sh[0],n_slices,self.module_out_dim],
            device=X.device)
        for i,X_slice in enumerate(self.iter_over_dim(X)):
            mod_out = self.module(X_slice)
            mod_out = self.extract_features(mod_out)
            ssl_representation[:,i,:] = mod_out
        return ssl_representation
    
    def v_module(self,X:torch.Tensor)->torch.Tensor:
        X = self.vol_to_2d(X)
        X = self.module(X)
        X = self.extract_features(X)
        X = self.rep_to_emb(X)
        return X

    def forward(self,X:torch.Tensor)->torch.Tensor:
        """Forward pass.

        Args:
            X (torch.Tensor): input tensor.

        Returns:
            torch.Tensor: output logits.
        """
        sh = X.shape
        batch_size = sh[0]
        # tried to replace this with vmap but it leads to OOM errors?
        ssl_representation = self.v_module(X)
        ssl_representation = self.input_layer(ssl_representation)
        if self.positional_embedding is not None:
            ssl_representation = ssl_representation + self.positional_embedding
        if self.use_class_token == True:
            if self.use_class_token is True:
                class_token = einops.repeat(self.class_token,
                                            '() n e -> b n e',
                                            b=batch_size)
                ssl_representation = torch.concat(
                    [class_token,ssl_representation],1)
        transformer_output = self.tbs(ssl_representation)[0]
        if self.use_class_token == True:
            transformer_output = transformer_output[:,0,:]
        else:
            transformer_output = transformer_output.mean(1)
        output = self.classification_module(transformer_output)
        return output

class TabularClassifier(torch.nn.Module):
    def __init__(self,
                 n_features:int,
                 mlp_structure:List[int],
                 mlp_adn_fn:Callable,
                 n_classes:int,
                 feature_means:torch.Tensor=None,
                 feature_stds:torch.Tensor=None):
        super().__init__()
        self.n_features = n_features
        self.mlp_structure = mlp_structure
        self.mlp_adn_fn = mlp_adn_fn
        self.n_classes = n_classes
        self.feature_means = feature_means
        self.feature_stds = feature_stds
        
        self.init_mlp()
        self.init_normalization_parameters()

    def init_mlp(self):
        nc = 1 if self.n_classes == 2 else self.n_classes
        self.mlp = MLP(input_dim=self.n_features,
                       output_dim=nc,
                       structure=self.mlp_structure,
                       adn_fn=self.mlp_adn_fn)
    
    def init_normalization_parameters(self):
        if self.feature_means is None:
            feature_means = torch.zeros([1,self.n_features])
        else:
            feature_means = torch.as_tensor(self.feature_means).reshape(1,-1)
        if self.feature_stds is None:
            feature_stds = torch.ones([1,self.n_features])
        else:
            feature_stds = torch.as_tensor(self.feature_stds).reshape(1,-1)
            
        self.mu = torch.nn.Parameter(feature_means.float(),
                                     requires_grad=False)
        self.sigma = torch.nn.Parameter(feature_stds.float(),
                                        requires_grad=False)
    
    def normalize(self,X:torch.Tensor)->torch.Tensor:
        return (X - self.mu) / self.sigma
    
    def forward(self,X)->torch.Tensor:
        return self.mlp(self.normalize(X))

class HybridClassifier(torch.nn.Module):
    def __init__(self,
                 convolutional_module:torch.nn.Module,
                 tabular_module:torch.nn.Module):
        super().__init__()
        self.convolutional_module = convolutional_module
        self.tabular_module = tabular_module
        
        self.set_n_classes()
        self.init_weight()
    
    def init_weight(self):
        self.raw_weight = torch.nn.Parameter(torch.ones([1]))
    
    def set_n_classes(self):
        n_classes_conv = None
        n_classes_tab = None
        if hasattr(self.convolutional_module,"n_classes"):
            n_classes_conv = self.convolutional_module.n_classes
        if hasattr(self.tabular_module,"n_classes"):
            n_classes_tab = self.tabular_module.n_classes
        
        if all([(n_classes_conv is not None),(n_classes_tab is not None)]):
            assert n_classes_conv == n_classes_tab,\
                "convolutional_module.n_classes should be the same as \
                    tabular_module.n_classes"
        if n_classes_conv is not None:
            self.n_classes = n_classes_conv
        elif n_classes_tab is not None:
            self.n_classes = n_classes_tab
        else:
            err_message = "{} or {} should be defined".format(
                "convolutional_module.n_classes",
                "tabular_module.n_classes")
            raise ValueError(err_message)
    
    def forward(self,X_conv:torch.Tensor,X_tab:torch.Tensor)->torch.Tensor:
        class_conv = self.convolutional_module(X_conv)
        class_tab = self.tabular_module(X_tab)
        
        weight = F.sigmoid(self.raw_weight)
        class_out = weight * class_conv + (1-weight)*class_tab

        return class_out