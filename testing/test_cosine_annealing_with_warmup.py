import sys
import os
sys.path.append(os.path.join(os.path.dirname(os.path.abspath(__file__)),'..'))
from pytest import mark

import torch
from lib.modules.learning_rate import CosineAnnealingWithWarmupLR

N_warmup = 5
N_max = 10
N_start_decay = 5
lr = 0.001
min_lr = 1e-6

params = []
for mlr in [0.0,min_lr]:
    for ws in [0,N_warmup]:
        for sd in [None,N_start_decay]:
            params.append((mlr,ws,sd))

@mark.parametrize("m,warmup_steps,start_decay",params)
def test_cosine_annealing(m,warmup_steps,start_decay):
    p = torch.nn.Parameter(torch.ones([1]))
    optim = torch.optim.Adam([p],lr=lr)
    sch = CosineAnnealingWithWarmupLR(optim,
                                      T_max=N_max,
                                      eta_min=m,
                                      n_warmup_steps=warmup_steps,
                                      start_decay=start_decay)
    for _ in range(N_max):
        sch.step()
    assert sch._get_closed_form_lr()[0] == m
