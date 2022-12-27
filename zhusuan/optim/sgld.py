import math
import torch
from torch.optim.optimizer import Optimizer, _use_grad_for_differentiable


class SGLD(Optimizer):
    def __init__(self, params, lr):
        defaults = dict(lr=lr, differentiable=False)
        super().__init__(params, defaults)

    @_use_grad_for_differentiable
    def step(self):
        for group in self.param_groups:
            lr = group['lr']
            for param in group['params']:
                if param.grad is not None:
                    param.add_(param.grad, alpha=0.5*lr)
                    param.add_(torch.randn_like(
                        param.data), alpha=math.sqrt(lr))
