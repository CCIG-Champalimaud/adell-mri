import time
import torch
import torch.nn.functional as F
import einops

from ...custom_types import *
from ..layers.adn_fn import ActDropNorm,get_adn_fn
from ..layers.batch_ensemble import BatchEnsembleWrapper
from ..layers.standard_blocks import GlobalPooling
from ..layers.res_net import ResNet,ResNetBackboneAlt,ProjectionHead
from ..layers.self_attention import (
    ConcurrentSqueezeAndExcite2d,ConcurrentSqueezeAndExcite3d)
from ..segmentation.unet import UNet
from ..layers.linear_blocks import MLP
from ..layers.vit import ViT,FactorizedViT

resnet_default = [(64,128,5,2),(128,256,3,5)]
maxpool_default = [(2,2,2),(2,2,2)]

def label_to_ordinal(label:torch.Tensor,n_classes:int,ignore_0:bool=True):
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
    x_thresholded = F.threshold(x,0.5,1)
    output = x_thresholded.argmax(dim=1)
    # consider 0 only when no class class reaches the threshold
    output[x_thresholded.sum(dim=1)>0] = 0
    return output

class CatNet(torch.nn.Module):
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
        """Case class for standard categorical classification. Defaults to 
        feature extraction using ResNet.

        Args:
            spatial_dimensions (int, optional): number of spatial dimenssions. 
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
        self.classification_layer = torch.nn.Sequential(
            GlobalPooling(),
            MLP(self.last_size,final_n,[self.last_size for _ in range(3)],
                adn_fn=get_adn_fn(1,"batch","gelu")))
        if self.batch_ensemble > 0:
            self.classification_layer = BatchEnsembleWrapper(
                self.classification_layer,self.batch_ensemble,self.last_size,
                final_n,torch.nn.Identity)
    
    def forward(self,X:torch.Tensor)->torch.Tensor:
        features = self.feature_extraction(X)
        classification = self.classification_layer(features)
        return classification

class OrdNet(CatNet):
    def __init__(self,*args,**kwargs):
        """Same as CatNet but the output is ordinal.
        """
        super().__init__(*args,**kwargs)

    def init_classification_layer(self):
        self.classification_layer = torch.nn.Sequential(
            GlobalPooling(),
            torch.nn.Linear(self.last_size,self.last_size),
            torch.nn.ReLU(),
            torch.nn.Linear(self.last_size,1))
        self.bias = torch.nn.parameter.Parameter(
            torch.zeros([1,self.n_classes-1]))
        self.last_act = torch.nn.Sigmoid()
    
    def forward(self,X:torch.Tensor)->torch.Tensor:
        features = self.feature_extraction(X)
        p_general = self.classification_layer(features)
        p_ordinal = self.last_act(p_general + self.bias)
        return p_ordinal

class SegCatNet(torch.nn.Module):
    def __init__(self,
                 spatial_dim:int,
                 u_net:torch.nn.Module,
                 n_input_channels:int,
                 n_features_backbone:int,
                 n_features_final_layer:int,
                 n_classes:int):
        """Uses the bottleneck and final layer features from a U-Net module
        to train a classifier. The `u_net` module should have a `forward` 
        method that can take a `return_features` argument that, when set to
        True, returns a tuple of tensors: prediction, final layer features 
        (before prediction) and bottleneck features.

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

        if self.n_classes == 2: self.nc = 1
        else: self.nc = self.n_classes

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
    """Generically combines multiple encoders to produce an ensemble model.
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
    def __init__(self,
                 cat_net_args:Union[Dict[str,int],List[Dict[str,int]]]):
        """Creates an ensemble of networks which can be trained online. The 
        input of each network can be different and the forward method supports
        predictions with missing data (as the average of all networks)

        Args:
            cat_net_args (Union[Dict[str,int],List[Dict[str,int]]], optional): _description_. Defaults to .
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
    """U-Net encoder for classification.
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
        
    def forward(self,X:torch.Tensor)->torch.Tensor:
        """Forward pass for this class.

        Args:
            X (torch.Tensor)

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
        out = self.prediction_head(out)
        return out

class ViTClassifier(ViT):
    def __init__(self,
                 n_classes:int,
                 use_class_token:bool=False,
                 *args,**kwargs):
        kwargs["use_class_token"] = use_class_token
        super().__init__(*args,**kwargs)
        self.n_classes = n_classes

        if self.n_classes == 2:
            nc = 1
        else:
            nc = self.n_classes
        self.classification_layer = torch.nn.Sequential(
            torch.nn.LayerNorm(self.input_dim_primary),
            torch.nn.Linear(self.input_dim_primary,nc))

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
        embeded_X = self.embedding(X)
        for block in self.tbs.transformer_blocks:
            embeded_X = block(embeded_X)
        if self.use_class_token is True:
            embeded_X = embeded_X[:,0]
        else:
            embeded_X = embeded_X.mean(1)
        classification = self.classification_layer(embeded_X)
        return classification

class FactorizedViTClassifier(FactorizedViT):
    def __init__(self,
                 n_classes:int,
                 use_class_token:bool=False,
                 *args,**kwargs):
        kwargs["use_class_token"] = use_class_token
        super().__init__(*args,**kwargs)
        self.n_classes = n_classes
        
        if self.n_classes == 2:
            nc = 1
        else:
            nc = self.n_classes
        self.classification_layer = torch.nn.Sequential(
            torch.nn.LayerNorm(self.input_dim_primary),
            torch.nn.Linear(self.input_dim_primary,nc))
    
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
        embeded_X = self.embedding(X)
        embeded_X,_ = self.transformer_block_within(embeded_X)
        # extract the maximum value of each token for all slices
        if self.use_class_token is True:
            embeded_X = embeded_X[:,:,0]
            embeded_X = embeded_X + self.slice_positional_embedding
            class_token = einops.repeat(
                self.slice_class_token,'() n e -> b n e',b=X.shape[0])
            embeded_X = torch.concat([class_token,embeded_X],1)
        else:
            embeded_X = embeded_X.mean(-2)
            embeded_X = embeded_X + self.slice_positional_embedding
        embeded_X,_ = self.transformer_block_between(embeded_X)
        if self.use_class_token is True:
            embeded_X = embeded_X[:,0]
        else:
            embeded_X = embeded_X.mean(1)
        classification = self.classification_layer(embeded_X)
        return classification

class MONAIViTClassifier(torch.nn.Module):
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
