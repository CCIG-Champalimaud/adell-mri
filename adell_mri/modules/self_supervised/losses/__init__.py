from .barlow_twins import BarlowTwinsLoss
from .contrastive import KLDivergence, ContrastiveDistanceLoss
from .functional import (
    cos_sim,
    cos_dist,
    unravel_index,
    standardize,
    pearson_corr,
    barlow_twins_loss,
    simsiam_loss,
    byol_loss,
)
from .ntxent import NTXentLoss
from .vicreg import VICRegLoss, VICRegLocalLoss
from .dino import DinoLoss
