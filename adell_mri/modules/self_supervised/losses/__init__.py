from .barlow_twins import BarlowTwinsLoss
from .contrastive import ContrastiveDistanceLoss, KLDivergence
from .dino import DinoLoss
from .functional import (barlow_twins_loss, byol_loss, cos_dist, cos_sim,
                         pearson_corr, simsiam_loss, standardize,
                         unravel_index)
from .koleo import KoLeoLoss
from .ntxent import NTXentLoss
from .vicreg import VICRegLocalLoss, VICRegLoss
