import torch
import torch.nn.functional as F

from ...custom_types import *

def cos_sim(x:torch.Tensor,y:torch.Tensor)->torch.Tensor:
    """Calculates the cosine similarity between x and y.

    Args:
        x (torch.Tensor): tensor
        y (torch.Tensor): tensor, must be of same shape to x

    Returns:
        torch.Tensor: cosine similarity between x and y
    """
    x,y = x.flatten(start_dim=1),y.flatten(start_dim=1)
    x,y = x.unsqueeze(1),y.unsqueeze(0)
    n = torch.sum(x*y,axis=-1)
    d = torch.multiply(torch.norm(x,2,-1),torch.norm(y,2,-1))
    return n/d

def standardize(x:torch.Tensor,d:int=0)->torch.Tensor:
    """Standardizes x (subtracts mean and divides by std) according to 
    dimension d.

    Args:
        x (torch.Tensor): tensor
        d (int, optional): dimension along which x will be standardized. 
            Defaults to 0
        
    Returns:
        torch.Tensor: standardized x along dimension d
    """
    return torch.divide(
        x - torch.mean(x,d,keepdim=True),
        torch.std(x,d,keepdim=True))

def pearson_corr(x:torch.Tensor,y:torch.Tensor)->torch.Tensor:
    """Calculates Pearson correlation between x and y

    Args:
        x (torch.Tensor): tensor
        y (torch.Tensor): tensor

    Returns:
        torch.Tensor: Pearson correlation between x and y
    """
    x,y = x.flatten(start_dim=1),y.flatten(start_dim=1)
    x,y = standardize(x),standardize(y)
    x,y = x.unsqueeze(1),y.unsqueeze(0)
    n = torch.sum(x*y,axis=-1)
    d = torch.multiply(torch.norm(x,2,-1),torch.norm(y,2,-1))
    return n/d

def cos_dist(x:torch.Tensor,y:torch.Tensor,center)->torch.Tensor:
    """Calculates the cosine distance between x and y.

    Args:
        x (torch.Tensor): tensor
        y (torch.Tensor): tensor, must be of same shape to x

    Returns:
        torch.Tensor: cosine distance between x and y
    """
    return 1 - cos_sim(x,y,center)

def barlow_twins_loss(x:torch.Tensor,
                      y:torch.Tensor,
                      l:float=0.02)->torch.Tensor:
    """Calculates the Barlow twins loss between x and y. This loss is composed
    of two terms: the invariance term, which maximises the Pearson correlation
    with views belonging to the same image (invariance term) and minimises the
    correlation between different images (reduction term) to promote greater
    feature diversity.

    Args:
        x (torch.Tensor): tensor
        y (torch.Tensor): tensor
        l (float, optional): term that scales the reduction term. Defaults to 
            0.02.

    Returns:
        torch.Tensor: Barlow twins loss
    """
    diag_idx = torch.arange(0,x.shape)
    n = x.shape[0]
    C = pearson_corr(x,y)
    inv_term = torch.diagonal(1 - C)[diag_idx,diag_idx]
    red_term = torch.square(C)
    red_term[diag_idx,diag_idx] = 0
    loss = torch.add(
        inv_term.sum()/x.shape[0],
        red_term.sum()/(n*(n-1))*l)
    return loss

def simsiam_loss(x1:torch.Tensor,x2:torch.Tensor)->torch.Tensor:
    """Loss for the SimSiam protocol.

    Args:
        x1 (torch.Tensor): tensor
        x2 (torch.Tensor): tensor

    Returns:
        torch.Tensor: SimSiam loss
    """
    x1 = x1/torch.functional.norm(x1,2,-1).unsqueeze(1)
    x2 = x2/torch.functional.norm(x2,2,-1).unsqueeze(1)
    return -torch.sum(x1*x2,1).mean()

def byol_loss(x1:torch.Tensor,x2:torch.Tensor)->torch.Tensor:
    """Loss for the BYOL (bootstrap your own latent) protocol.

    Args:
        x1 (torch.Tensor): tensor
        x2 (torch.Tensor): tensor

    Returns:
        torch.Tensor: BYOL loss
    """
    return 2*simsiam_loss(x1,x2)+2

def unravel_index(
    indices: torch.LongTensor,
    shape: Tuple[int, ...]) -> torch.LongTensor:
    """Converts flat indices into unraveled coordinates in a target shape.

    This is a `torch` implementation of `numpy.unravel_index`.

    Args:
        indices: A tensor of indices, (*, N).
        shape: The targeted shape, (D,).

    Returns:
        unravel coordinates, (*, N, D).
    """
    # from https://github.com/pytorch/pytorch/issues/35674#issuecomment-739492875

    shape = torch.tensor(shape)
    indices = indices % shape.prod()  # prevent out-of-bounds indices

    coord = torch.zeros(indices.size() + shape.size(), dtype=int)

    for i, dim in enumerate(reversed(shape)):
        coord[..., i] = indices % dim
        indices = torch.div(indices,dim,rounding_mode="floor")

    return coord.flip(-1)

class VICRegLoss(torch.nn.Module):
    def __init__(self,
                 min_var: float=1.,
                 eps: float=1e-4,
                 lam: float=25.,
                 mu: float=25.,
                 nu: float=0.1):
        """Implementation of the VICReg loss from [1].
        
        [1] https://arxiv.org/abs/2105.04906

        Args:
            min_var (float, optional): minimum variance of the features. 
                Defaults to 1..
            eps (float, optional): epsilon term to avoid errors due to floating
                point imprecisions. Defaults to 1e-4.
            lam (float, optional): invariance term.
            mu (float, optional): variance term.
            nu (float, optional): covariance term.
        """
        super().__init__()
        self.min_var = min_var
        self.eps = eps
        self.lam = lam
        self.mu = mu
        self.nu = nu

    def off_diagonal(self,x):
        # from https://github.com/facebookresearch/vicreg/blob/a73f567660ae507b0667c68f685945ae6e2f62c3/main_vicreg.py
        n, m = x.shape
        assert n == m
        return x.flatten()[:-1].view(n - 1, n + 1)[:, 1:].flatten()

    def variance_loss(self,X:torch.Tensor)->torch.Tensor:
        """Calculates the VICReg variance loss (a Hinge loss for the variance
        which keeps it above `self.min_var`)

        Args:
            X (torch.Tensor): input tensor

        Returns:
            torch.Tensor: variance loss
        """
        reg_std = torch.sqrt(torch.var(X,0)+self.eps)
        return F.relu(self.min_var - reg_std).mean()
    
    def covariance_loss(self,X:torch.Tensor)->torch.Tensor:
        """Calculates the covariance loss for VICReg (minimises the L2 norm of
        the off diagonal elements belonging to the covariance matrix of the 
        features).

        Args:
            X (torch.Tensor): input tensor

        Returns:
            torch.Tensor: covariance loss.
        """
        X_mean = X.mean(0)
        X_centred = X - X_mean
        cov = (X_centred.T @ X_centred) / (X.shape[0]-1)
        return torch.sum(self.off_diagonal(cov).pow_(2)/X.shape[1])

    def invariance_loss(self,X1:torch.Tensor,X2:torch.Tensor)->torch.Tensor:
        """Calculates the invariance loss for VICReg (minimises the MSE 
        between the features calculated from two views of the same image).

        Args:
            X1 (torch.Tensor): input tensor from view 1
            X2 (torch.Tensor): input tensor from view 2

        Returns:
            torch.Tensor: invariance loss
        """
        return F.mse_loss(X1,X2)
    
    def vicreg_loss(self,
                    X1:torch.Tensor,
                    X2:torch.Tensor,
                    adj:float=1.0)->torch.Tensor:
        """Wrapper for the three components of the VICReg loss.

        Args:
            X1 (torch.Tensor): input tensor from view 1
            X2 (torch.Tensor): input tensor from view 2
            adj (float, optional): adjustment to the covariance loss (helpful
                for local VICReg losses. Defaults to 1.0.

        Returns:
            var_loss (torch.Tensor) variance loss
            cov_loss (torch.Tensor) covariance loss
            inv_loss (torch.Tensor) invariance loss
        """
        var_loss = torch.add(
            self.variance_loss(X1),
            self.variance_loss(X2)) / 2
        cov_loss = torch.add(
            self.covariance_loss(X1)/adj,
            self.covariance_loss(X2)/adj) / 2
        inv_loss = self.invariance_loss(X1,X2)
        return var_loss,cov_loss,inv_loss
    
    def flatten_if_necessary(self,x):
        if len(x.shape) > 2:
            return x.flatten(start_dim=2).mean(-1)
        return x
    
    def forward(self,
                X1:torch.Tensor,X2:torch.Tensor)->Tuple[torch.Tensor,
                                                        torch.Tensor,
                                                        torch.Tensor]:
        """Forward method for VICReg loss.

        Args:
            X1 (torch.Tensor): (B,C,H,W,(D)) tensor corresponding to the first
                transform.
            X2 (torch.Tensor): (B,C,H,W,(D)) tensor corresponding to the second
                transform.

        Returns:
            inv_loss (torch.Tensor) weighted invariance loss
            cov_loss (torch.Tensor) weighted covariance loss
            var_loss (torch.Tensor) weighted variance loss
        """

        flat_max_X1 = self.flatten_if_necessary(X1)
        flat_max_X2 = self.flatten_if_necessary(X2)
        var_loss,cov_loss,inv_loss = self.vicreg_loss(
            flat_max_X1,flat_max_X2)
        return self.lam*inv_loss,self.mu*var_loss,self.nu*cov_loss

class VICRegLocalLoss(VICRegLoss):
    def __init__(self,
                 min_var: float=1.,
                 eps: float=1e-4,
                 lam: float=25.,
                 mu: float=25.,
                 nu: float=0.1,
                 gamma: int=10):
        """Local VICRegL loss from [2]. This is, in essence, a version of
        VICReg which leads to better downstream solutions for segmentation 
        tasks and other tasks requiring pixel- or superpixel-level inference.
        Default values are according to the paper.
        
        [2] https://arxiv.org/pdf/2210.01571v1.pdf

        Args:
            min_var (float, optional): minimum variance of the features. 
                Defaults to 1..
            eps (float, optional): epsilon term to avoid errors due to floating
                point imprecisions. Defaults to 1e-4.
            lam (float, optional): invariance term.
            mu (float, optional): variance term.
            nu (float, optional): covariance term.
            gamma (int, optional): the local loss term is calculated only for 
                the top-gamma feature matches between input images. Defaults 
                to 10.
        """
        super().__init__()
        self.min_var = min_var
        self.eps = eps
        self.lam = lam
        self.mu = mu
        self.nu = nu
        self.gamma = gamma

        self.alpha = 0.9
        self.zeros = None
        self.sparse_coords_1 = None
        self.sparse_coords_2 = None
    
    def transform_coords(self,
                         coords:torch.Tensor,
                         box:torch.Tensor)->torch.Tensor:
        """Takes a set of coords and addapts them to a new coordinate space
        defined by box (0,0 becomes the top left corner of the bounding box).

        Args:
            coords (torch.Tensor): pixel coordinates (x,y)
            box (torch.Tensor): coordinates for a given bounding box 
                (x1,y1,x2,y2)

        Returns:
            torch.Tensor: transformed coordinates
        """
        ndim = box.shape[-1]//2
        a,b = (torch.unsqueeze(box[:,:ndim],1),
               torch.unsqueeze(box[:,ndim:],1))
        size = b - a
        return coords.unsqueeze(0) * size + a

    def local_loss(self,
                   X1:torch.Tensor,
                   X2:torch.Tensor,
                   all_dists:torch.Tensor):
        g = self.gamma
        b = X1.shape[0]
        _,idxs = torch.topk(all_dists.flatten(start_dim=1),g,1)
        idxs = [unravel_index(x,all_dists[0].shape) for x in idxs]
        indexes = torch.cat([torch.ones(g)*i for i in range(b)]).long()
        indexes_1 = torch.cat(
            [self.sparse_coords_1[idxs[i][:,0]].long() for i in range(b)])
        indexes_1 = tuple(
            [indexes,*[indexes_1[:,i] for i in range(indexes_1.shape[1])]])
        indexes_2 = torch.cat(
            [self.sparse_coords_2[idxs[i][:,0]].long() for i in range(b)])
        indexes_2 = tuple(
            [indexes,*[indexes_2[:,i] for i in range(indexes_2.shape[1])]])
        features_1 = X1.unsqueeze(-1).swapaxes(1,-1).squeeze(1)[indexes_1]
        features_2 = X2.unsqueeze(-1).swapaxes(1,-1).squeeze(1)[indexes_2]
        vrl = sum(self.vicreg_loss(features_1,features_2,g)/g)
        return vrl

    def location_local_loss(self,
                            X1:torch.Tensor,
                            X2:torch.Tensor,
                            box_X1:torch.Tensor,
                            box_X2:torch.Tensor)->torch.Tensor:
        """Given two views of the same image X1 and X2 and their bounding box
        coordinates in the original image space (box_X1 and box_X2), this loss
        function minimises the distance between nearby pixels in both images.
        It does not calculate it for *all* pixels but only for the 
        top-self.gamma pixels.

        Args:
            X1 (torch.Tensor): input tensor from view 1
            X2 (torch.Tensor): input tensor from view 2
            box_X1 (torch.Tensor): box containing X1 view
            box_X2 (torch.Tensor): box containing X2 view

        Returns:
            torch.Tensor: local loss for location
        """
        assert X1.shape[0] == X2.shape[0],"X1 and X2 need to have the same batch size"
        X1.shape[0]
        coords_X1 = self.transform_coords(self.sparse_coords_1,box_X1)
        coords_X2 = self.transform_coords(self.sparse_coords_2,box_X2)
        all_dists = torch.cdist(coords_X1,coords_X2,p=2)
        return self.local_loss(X1,X2,all_dists)

    def feature_local_loss(self,X1:torch.Tensor,X2:torch.Tensor):
        """Given two views of the same image X1 and X2, this loss
        function minimises the distance between the top-self.gamma closest
        pixels in feature space.

        Args:
            X1 (torch.Tensor): input tensor from view 1
            X2 (torch.Tensor): input tensor from view 2

        Returns:
            torch.Tensor: local loss for features
        """
        assert X1.shape[0] == X2.shape[0],"X1 and X2 need to have the same batch size"
        flat_X1 = X1.flatten(start_dim=2).swapaxes(1,2)
        flat_X2 = X2.flatten(start_dim=2).swapaxes(1,2)
        all_dists = torch.cdist(flat_X1,flat_X2,p=2)
        return self.local_loss(X1,X2,all_dists)
        
    def get_sparse_coords(self,X):
        return torch.stack(
            [x.flatten() for x in torch.meshgrid(
                *[torch.arange(0,i) for i in X.shape[2:]],indexing="ij")],
            axis=1).float().to(X.device)

    def forward(self,
                X1:torch.Tensor,X2:torch.Tensor,
                box_X1:torch.Tensor,box_X2:torch.Tensor)->Tuple[torch.Tensor,
                                                                torch.Tensor,
                                                                torch.Tensor,
                                                                torch.Tensor]:
        """Forward method for local VICReg loss.

        Args:
            X1 (torch.Tensor): (B,C,H,W,(D)) tensor corresponding to the first
                transform.
            X2 (torch.Tensor): (B,C,H,W,(D)) tensor corresponding to the second
                transform.
            box_X1 (torch.Tensor): coordinates for X1 in the original image
            box_X2 (torch.Tensor): coordinates for X2 in the original image

        Returns:
            var_loss (torch.Tensor)
            cov_loss (torch.Tensor)
            inv_loss (torch.Tensor)
            local_loss (torch.Tensor)
        """
        
        flat_max_X1 = X1.flatten(start_dim=2).mean(-1)
        flat_max_X2 = X2.flatten(start_dim=2).mean(-1)

        # these steps calculating sparse coordinates and storing them as a class
        # variable assume that image shape remains the same. if this is not the 
        # case, then these variables are redefined

        if self.sparse_coords_1 is None:
            self.sparse_coords_1 = self.get_sparse_coords(X1)
            self.shape_1 = X1.shape
        else:
            if self.shape_1 != X1.shape:
                self.sparse_coords_1 = self.get_sparse_coords(X1)
                self.shape_1 = X1.shape

        if self.sparse_coords_2 is None:
            self.sparse_coords_2 = self.get_sparse_coords(X2)
            self.shape_2 = X2.shape
        else:
            if self.shape_2 != X2.shape:
                self.sparse_coords_2 = self.get_sparse_coords(X2)
                self.shape_2 = X2.shape

        var_loss,cov_loss,inv_loss = self.vicreg_loss(
            flat_max_X1,flat_max_X2)
        
        # location and feature local losses are non-symmetric so this 
        # symmetrises them
        short_range_local_loss = torch.add(
            self.location_local_loss(
                X1,X2,box_X1,box_X2) * (1-self.alpha),
            self.location_local_loss(
                X2,X1,box_X2,box_X1) * (1-self.alpha)) / 2
        long_range_local_loss = torch.add(
            self.feature_local_loss(X1,X2) * (1-self.alpha),
            self.feature_local_loss(X2,X1) * (1-self.alpha)) / 2
        
        local_loss = short_range_local_loss + long_range_local_loss
        return (self.lam*inv_loss * self.alpha,
                self.mu*var_loss * self.alpha,
                self.nu*cov_loss * self.alpha,
                local_loss)

class ContrastiveDistanceLoss(torch.nn.Module):
    def __init__(self,dist_p=2,random_sample=False,margin=1,
                 dev="cpu",loss_type="pairwise",dist_type="euclidean"):
        super().__init__()
        self.dist_p = dist_p
        self.random_sample = random_sample
        self.margin = margin
        self.dev = dev
        self.loss_type = loss_type
        self.dist_type = dist_type
        
        self.loss_options = ["pairwise","triplet"]
        self.dist_options = ["euclidean","cosine"]
        self.torch_margin = torch.as_tensor(
            [self.margin],dtype=torch.float32,device=self.dev)

        if self.loss_type not in self.loss_options:
            raise Exception("Loss `{}` not in `{}`".format(
                self.loss_type,self.loss_options))
        
        if self.dist_type not in self.dist_options:
            raise Exception("dist_type `{}` not in `{}`".format(
                self.loss_type,self.dist_options))

    def dist(self,x:torch.Tensor,y:torch.Tensor):
        if self.dist_type == "euclidean":
            return torch.cdist(x,y,self.dist_p)
        elif self.dist_type == "cosine":
            return cos_dist(x,y)

    def pairwise_distance(self,X1,X2,is_same):
        X1 = X1.flatten(start_dim=1)
        X2 = X2.flatten(start_dim=1)
        dist = self.dist(X1,X2)
        dist = torch.add(
            is_same*dist,
            (1-is_same.float())*torch.maximum(
                torch.zeros_like(dist),
                self.torch_margin - dist))
        if self.random_sample is True:
            # randomly samples one entry for each element
            n = dist.shape[0]
            x_idx = torch.arange(0,n,1,dtype=torch.int32)
            y_idx = torch.randint(0,n,size=[n])
            dist = dist[x_idx,y_idx]
        else:
            dist = dist.sum(-1)/(dist.shape[-1]-1)
        return dist
    
    def triplet_distance(self,X1,X2,is_same):
        X1 = X1.flatten(start_dim=1)
        X2 = X2.flatten(start_dim=1)
        dist = self.dist(X1,X2)
        # retrieve negative examples with the lowest distance to 
        # each anchor
        hard_negatives = torch.where(
            is_same,
            torch.ones_like(dist)*torch.inf,
            dist).min(1).values
        # retrieve positive examples with the highest distance to
        # each anchor
        hard_positives = torch.where(
            torch.logical_not(is_same),
            -torch.ones_like(dist)*torch.inf,
            dist).max(1).values
        # calculates loss given both hard negatives and positives
        triplet_loss = torch.maximum(
            torch.zeros_like(hard_negatives),
            self.margin + hard_positives - hard_negatives)
        return triplet_loss

    def forward(self,X:torch.Tensor,y:torch.Tensor):
        if isinstance(X,list):
            X1,X2 = X
        else:
            X1,X2 = X,X
        y1,y2 = y.unsqueeze(0),y.unsqueeze(1)
        is_same = y1 == y2
        if self.loss_type == "pairwise":
            loss = self.pairwise_distance(X1,X2,is_same)
        elif self.loss_type == "triplet":
            loss = self.triplet_distance(X1,X2,is_same)
        return loss.mean()

class BarlowTwinsLoss(torch.nn.Module):
    def __init__(self,moving:bool=False,lam=0.2):
        super().__init__()
        self.moving = moving
        self.lam = lam
        
        self.count = 0.
        self.sum = None
        self.sum_of_squares = None
        self.average = None
        self.std = None

    def standardize(self,x:torch.Tensor)->torch.Tensor:
        if self.moving is False and self.sum is None:
            o = torch.divide(
                x - torch.mean(x,0,keepdim=True),
                torch.std(x,0,keepdim=True))
        else:
            o = torch.divide(x-self.average,self.std)
        return o

    def pearson_corr(self,x:torch.Tensor,y:torch.Tensor)->torch.Tensor:
        x,y = x.flatten(start_dim=1),y.flatten(start_dim=1)
        x,y = self.standardize(x),self.standardize(y)
        x,y = x.unsqueeze(1),y.unsqueeze(0)
        n = torch.sum(x*y,axis=-1)
        d = torch.multiply(torch.norm(x,2,-1),torch.norm(y,2,-1))
        return n/d

    def calculate_loss(self,x,y,update=True):
        if update is True:
            n = x.shape[0]
            f = x.shape[1]
            if self.sum is None:
                self.sum = torch.zeros([1,f],device=x.device)
                self.sum_of_squares = torch.zeros([1,f],device=x.device)
            self.sum = torch.add(
                self.sum,
                torch.sum(x+y,0,keepdim=True))
            self.sum_of_squares = torch.add(
                self.sum_of_squares,
                torch.sum(torch.square(x)+torch.square(y),0,keepdim=True))
            self.count += 2*n
        return self.barlow_twins_loss(x,y)

    def barlow_twins_loss(self,x:torch.Tensor,y:torch.Tensor)->torch.Tensor:
        diag_idx = torch.arange(0,x.shape[0])
        n = x.shape[0]
        C = self.pearson_corr(x,y)
        inv_term = torch.diagonal(1 - C)
        red_term = torch.square(C)
        red_term[diag_idx,diag_idx] = 0
        loss = torch.add(inv_term.sum()/n,red_term.sum()/n*self.lam)
        return loss

    def calculate_average_std(self):
        self.average = self.sum / self.count
        self.std = self.sum_of_squares - torch.square(self.sum)/self.count
    
    def reset(self):
        self.count = 0.
        self.sum[()] = 0
        self.sum_of_squares[()] = 0

    def forward(self,X1:torch.Tensor,X2:torch.Tensor,update:bool=True):
        loss = self.calculate_loss(X1,X2,update)
        return loss.sum()
