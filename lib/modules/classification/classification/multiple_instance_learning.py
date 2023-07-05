import torch
import torch.nn.functional as F
import einops

from ...layers.adn_fn import get_adn_fn
from typing import Callable
from ...layers.vit import TransformerBlockStack
from ...layers.linear_blocks import MLP, MultiHeadSelfAttention

class MILAttention(torch.nn.Module):
    """Attention module for multiple instance learning [1]. The attention is
    calculated as the softmax of the sigmoid-gated hyperbolic tangent of the
    input, i.e. 
        $A = \mathrm{softmax}(\mathrm{tanh}(A * V) * \mathrm{sigmoid}(A * U))$

    [1] https://arxiv.org/pdf/1802.04712.pdf
    """
    def __init__(self,n_dim:int,along_dim=-2):
        super().__init__()
        
        self.n_dim = n_dim
        self.along_dim = along_dim
        
        self.initialize_layers()

    def initialize_layers(self):
        self.V = torch.nn.Linear(self.n_dim,self.n_dim)
        self.U = torch.nn.Linear(self.n_dim,self.n_dim)
    
    def forward(self,X:torch.Tensor)->torch.Tensor:
        return X * F.softmax(
            torch.multiply(
                F.tanh(self.V(X)),
                F.sigmoid(self.U(X))),
            self.along_dim)

class MultipleInstanceClassifier(torch.nn.Module):
    """
    Multiple instance classifier for volumes. Extracts features from each slice
    (last dimension) using a pre-specified ``module`` and applies a linear layer
    to derive classifications from this. Three different 
    ``classification_modes`` are available to extract features from a stack of
    instances:

    1. ``mean`` - simply calculate the average prediction across instances 
        (slices)
    2. ``max`` - extract the maximum of each feature across all instances
    3. ``vocabulary`` - soft classify instances into one of 
        ``vocabulary_size`` proxy classes and calculate the average 
        proxy class composition.

    The outputs from 1., 2. and 3. are then used in a simple linear classifier.
    """
    def __init__(self,
                 module:torch.nn.Module,
                 module_out_dim:int,
                 n_classes:int,
                 classification_structure:torch.nn.Module,
                 classification_mode:str="mean",
                 classification_adn_fn:Callable=get_adn_fn(
                     1,"layer","gelu",0.1),
                 vocabulary_size:int=10,
                 n_slices:int=None,
                 use_positional_embedding:bool=True,
                 dim:int=2,
                 attention:bool=False,
                 reduce_fn:str="max"):
        """
        Args:
            module (torch.nn.Module): end-to-end module that takes a batched 
                set of 2D images and output a vector for each image.
            module_out_dim (int): output size for module.
            n_classes (int): number of output classes.
            classification_structure (torch.nn.Module): structure for the 
                classification MLP.
            classification_mode (str, optional): classification mode. Can be 
                "mean" (calculates the average prediction across all slices), 
                "max" (calculates the maximum prediction across all slices) or 
                "vocabulary" (fits a pseudo-label to the cells and uses the
                vocabulary proportion across all cases to predict the 
                final classification). Defaults to "mean".
            classification_adn_fn (Callable, optional): ADN function for the
                MLP module. Defaults to get_adn_fn( 1,"layer","gelu",0.1).
            vocabulary_size (int, optional): vocabulary size for when 
                classification_mode == "vocabulary". Defaults to 10.
            n_slices (int, optional): number of slices. Used to initialize 
                positional embedding. Defaults to None (no positional 
                embedding).
            use_positional_embedding (bool, optional): whether a positional 
                embedding should be used. Defaults to True.
            dim (int, optional): dimension along which the module is applied.
                Defaults to 2.
            attention (bool, optional): uses multi-head self-attention. 
                Defaults to False.
            reduce_fn (str, optional): function used to reduce features coming
                from feature extraction module.
        """
        
        assert dim in [0,1,2], "dim must be one of [0,1,2]"
        assert classification_mode in ["mean","max","vocabulary"],\
            'classification has to be one of ["mean","max","vocabulary"]'
        super().__init__()
        self.module = module
        self.module_out_dim = module_out_dim
        self.n_classes = n_classes
        self.classification_structure = classification_structure
        self.classification_mode = classification_mode
        self.classification_adn_fn = classification_adn_fn
        self.vocabulary_size = vocabulary_size
        self.n_slices = n_slices
        self.use_positional_embedding = use_positional_embedding
        self.dim = dim
        self.attention = attention
        self.reduce_fn = reduce_fn
        
        self.vol_to_2d = einops.layers.torch.Rearrange(
            "b c h w s -> (b s) c h w")
        self.rep_to_emb = einops.layers.torch.Rearrange(
            "(b s) v -> b s v",s=self.n_slices)

        self.init_classification_module()
        self.init_positional_embedding()

        if self.attention is True:
            if len(self.classification_structure) > 0:
                input_dim = self.classification_structure[-1]
            else:
                input_dim = self.module_out_dim
            self.attention_op = MILAttention(input_dim)

    def init_positional_embedding(self):
        """
        Initializes the positional embedding.
        """
        self.positional_embedding = None
        if self.n_slices is not None and self.use_positional_embedding is True:
            self.positional_embedding = torch.nn.Parameter(
                torch.zeros([1,self.n_slices,1]))
            torch.nn.init.trunc_normal_(
                self.positional_embedding,mean=0.0,std=0.02,a=-2.0,b=2.0)

    def extract_features(self,X:torch.Tensor)->torch.Tensor:
        if len(X.shape) > 2:
            X = X.flatten(start_dim=2)
        if self.reduce_fn == "mean":
            return X.mean(-1)
        elif self.reduce_fn == "max":
            return X.max(-1).values
        
    def v_module(self,X:torch.Tensor)->torch.Tensor:
        X = self.vol_to_2d(X)
        X = self.module(X)
        X = self.extract_features(X)
        X = self.rep_to_emb(X)
        return X

    def init_classification_module(self):
        n_classes_out = 1 if self.n_classes == 2 else self.n_classes
        self.feature_extraction = MLP(
            self.module_out_dim,self.classification_structure[-1],
            structure=self.classification_structure[:-1],
            adn_fn=self.classification_adn_fn)
        if self.classification_mode in ["mean","max"]:
            self.final_prediction = MLP(
                self.classification_structure[-1],n_classes_out,
                structure=[],adn_fn=torch.nn.Identity)
        elif self.classification_mode == "vocabulary":
            self.vocaulary_prediction = MLP(
                self.classification_structure[-1],self.vocabulary_size,
                structure=[],
                adn_fn=self.classification_adn_fn)
            self.final_prediction = MLP(
                self.vocabulary_size,n_classes_out,
                structure=[],adn_fn=torch.nn.Identity)

    def get_prediction(self,X:torch.Tensor)->torch.Tensor:
        out = self.feature_extraction(X)
        if self.attention is True:
            out = self.attention_op(out)
        if self.classification_mode == "mean":
            return self.final_prediction(out).mean(-2)
        if self.classification_mode == "max":
            return self.final_prediction(out).max(-2).values
        if self.classification_mode == "vocabulary":
            out = self.vocaulary_prediction(out)
            out = F.softmax(out)
            out = out.mean(-2)
            return self.final_prediction(out)

    def forward(self,X:torch.Tensor)->torch.Tensor:
        """Forward pass.

        Args:
            X (torch.Tensor): input tensor.

        Returns:
            torch.Tensor: output logits.
        """
        # tried to replace this with vmap but it leads to OOM errors?
        ssl_representation = self.v_module(X)
        if self.positional_embedding is not None:
            ssl_representation = ssl_representation + self.positional_embedding
        output = self.get_prediction(ssl_representation)
        return output

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
                 use_positional_embedding:bool=True,
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
            use_positional_embedding (bool, optional): whether a positional 
                embedding should be used. Defaults to True.
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
        self.use_positional_embedding = use_positional_embedding
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
        if self.n_slices is not None and self.use_positional_embedding is True:
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
        if self.use_class_token is True:
            if self.use_class_token is True:
                class_token = einops.repeat(self.class_token,
                                            '() n e -> b n e',
                                            b=batch_size)
                ssl_representation = torch.concat(
                    [class_token,ssl_representation],1)
        transformer_output = self.tbs(ssl_representation)[0]
        if self.use_class_token is True:
            transformer_output = transformer_output[:,0,:]
        else:
            transformer_output = transformer_output.mean(1)
        output = self.classification_module(transformer_output)
        return output
