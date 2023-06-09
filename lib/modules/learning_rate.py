import torch
import math
from torch.optim.lr_scheduler import _LRScheduler

def float_to_epochs(v,max_epochs):
    if isinstance(v,float) == True:
        if v >= 1.0:
            v = int(v)
        else:
            v = int(v * max_epochs)
    return v

class _enable_get_lr_call:
    def __init__(self, o):
        self.o = o

    def __enter__(self):
        self.o._get_lr_called_within_step = True
        return self

    def __exit__(self, type, value, traceback):
        self.o._get_lr_called_within_step = False

class PolynomialLRDecay(_LRScheduler):
    """Polynomial learning rate decay until step reach to max_decay_step
    
    Args:
        optimizer (Optimizer): Wrapped optimizer.
        max_decay_steps: after this step, we stop decreasing learning rate
        end_learning_rate: scheduler stoping learning rate decay, value of learning rate
            must be this value
        power: The power of the polynomial.

    From: https://github.com/cmpark0126/pytorch-polynomial-lr-decay/blob/master/torch_poly_lr_decay/torch_poly_lr_decay.py
    """
    
    def __init__(self,optimizer,max_decay_steps,end_learning_rate=0.0001,power=1.0):
        if max_decay_steps <= 1.:
            raise ValueError('max_decay_steps should be greater than 1.')
        self.max_decay_steps = max_decay_steps
        self.end_learning_rate = end_learning_rate
        self.power = power
        self.last_step = 0
        super().__init__(optimizer)
        
    def get_lr(self):
        if self.last_step > self.max_decay_steps:
            return [self.end_learning_rate for _ in self.base_lrs]

        return [(base_lr - self.end_learning_rate) * 
                ((1 - self.last_step / self.max_decay_steps) ** (self.power)) + 
                self.end_learning_rate for base_lr in self.base_lrs]
    
    def step(self, step=None):
        if step is None:
            step = self.last_step + 1
        self.last_step = step if step != 0 else 1
        pow = self.power
        if self.last_step <= self.max_decay_steps:
            decay_lrs = [(base_lr - self.end_learning_rate) * 
                         ((1 - self.last_step / self.max_decay_steps) ** (pow)) + 
                         self.end_learning_rate for base_lr in self.base_lrs]
            self._last_lr = []
            for param_group, lr in zip(self.optimizer.param_groups, decay_lrs):
                param_group['lr'] = lr
                self._last_lr.append(lr)
          
class CosineAnnealingWithWarmupLR(_LRScheduler):
    """
    """

    def __init__(self, 
                 optimizer:torch.optim.Optimizer,
                 T_max:int,
                 n_warmup_steps:int=0,
                 eta_min:int=0, 
                 last_epoch:int=-1, 
                 verbose:bool=False,
                 start_decay:int=None):
        self.T_max = T_max
        self.n_warmup_steps = n_warmup_steps
        self.eta_min = eta_min
        self.initial_lr = eta_min
        self.start_decay = start_decay
        
        if self.start_decay is None:
            self.start_decay = self.n_warmup_steps
        self.n_warmup_steps = float_to_epochs(self.n_warmup_steps,self.T_max)
        self.start_decay = float_to_epochs(self.start_decay,self.T_max)
        
        self.last_lr = None
        super(CosineAnnealingWithWarmupLR, self).__init__(
            optimizer, last_epoch, verbose)

    def get_lr(self):
        return self._get_closed_form_lr()

    def _get_closed_form_lr(self):
        le = self.last_epoch
        nws = float(self.n_warmup_steps)
        ssd = float(self.start_decay)
        if le < (nws) and nws > 0:
            lrs = [(base_lr-self.initial_lr) * ((le+1)/nws) + self.eta_min
                   for base_lr in self.base_lrs]
        elif le <= ssd:
            lrs = [base_lr for base_lr in self.base_lrs]
        else:
            r = max(nws,ssd)
            T_max = self.T_max - r
            lrs = [self.eta_min + (base_lr - self.eta_min) *
                   (1 + math.cos(math.pi * (le-r) / T_max)) / 2
                   for base_lr in self.base_lrs]
        return lrs

    def step(self,step=None):
        self._step_count += 1

        with _enable_get_lr_call(self):
            if step is None:
                self.last_epoch += 1
                values = self.get_lr()
            else:
                self.last_epoch = step
                values = self.get_lr()

        for i, data in enumerate(zip(self.optimizer.param_groups,values)):
            param_group, lr = data
            param_group['lr'] = lr
            self.print_lr(self.verbose, i, lr, step)

        self._last_lr = [group['lr'] for group in self.optimizer.param_groups]

def poly_lr_decay(optimizer:torch.optim.Optimizer,
                        step:int,
                        initial_lr:float,
                        max_decay_steps:int,
                        end_lr:float=0.0,
                        power:float=1.0):
    """Polynomial LR decay.

    Args:
        optimizer (torch.optim.Optimizer): torch optimizer.
        step (int): step.
        initial_lr (float): initial learning rate.
        max_decay_steps (int): maximum number of steps for decay.
        end_lr (float, optional): final learning rate. Defaults to 0.0.
        power (float, optional): power (scales step/max_decay_steps). Defaults 
            to 1.0.
    """
    step = min(step, max_decay_steps)
    if isinstance(initial_lr,float):
        initial_lr = [initial_lr for _ in optimizer.param_groups]
    if step <= max_decay_steps:
        for param_group,i_lr in zip(optimizer.param_groups,initial_lr):
            base_lr = max(i_lr - end_lr,0)
            new_lr = base_lr*(1-step/max_decay_steps)**power+end_lr
            param_group['lr'] = new_lr
