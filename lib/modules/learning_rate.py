import torch
from torch.optim.lr_scheduler import _LRScheduler

class PolynomialLRDecay(_LRScheduler):
    """Polynomial learning rate decay until step reach to max_decay_step
    
    Args:
        optimizer (Optimizer): Wrapped optimizer.
        max_decay_steps: after this step, we stop decreasing learning rate
        end_learning_rate: scheduler stoping learning rate decay, value of learning rate must be this value
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
        if self.last_step <= self.max_decay_steps:
            decay_lrs = [(base_lr - self.end_learning_rate) * 
                         ((1 - self.last_step / self.max_decay_steps) ** (self.power)) + 
                         self.end_learning_rate for base_lr in self.base_lrs]
            self._last_lr = []
            for param_group, lr in zip(self.optimizer.param_groups, decay_lrs):
                param_group['lr'] = lr
                self._last_lr.append(lr)
            
def polynomial_lr_decay(optimizer:torch.optim.Optimizer,
                        step:int,
                        initial_lr:float,
                        max_decay_steps:int,
                        end_lr:float=0.0,
                        power:float=1.0):
    step = min(step, max_decay_steps)
    if isinstance(initial_lr,float):
        initial_lr = [initial_lr for _ in optimizer.param_groups]
    if step <= max_decay_steps:
        for param_group,i_lr in zip(optimizer.param_groups,initial_lr):
            base_lr = max(i_lr - end_lr,0)
            new_lr = base_lr*(1-step/max_decay_steps)**power+end_lr
            param_group['lr'] = new_lr
