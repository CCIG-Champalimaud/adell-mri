from adell_mri.modules.self_supervised.losses.barlow_twins import BarlowTwinsLoss
from adell_mri.modules.self_supervised.losses.contrastive import ContrastiveDistanceLoss, KLDivergence
from adell_mri.modules.self_supervised.losses.dino import DinoLoss
from adell_mri.modules.self_supervised.losses.functional import (
    barlow_twins_loss,
    byol_loss,
    cos_dist,
    cos_sim,
    pearson_corr,
    simsiam_loss,
    standardize,
    unravel_index,
)
from adell_mri.modules.self_supervised.losses.koleo import KoLeoLoss
from adell_mri.modules.self_supervised.losses.ntxent import NTXentLoss
from adell_mri.modules.self_supervised.losses.vicreg import VICRegLocalLoss, VICRegLoss
