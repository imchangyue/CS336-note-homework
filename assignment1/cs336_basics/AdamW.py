import torch
from torch.optim import Optimizer
from typing import Any

class AdamW(Optimizer):
    def __init__(self, params, lr=1e-3, betas=(0.9, 0.999), eps=1e-8, weight_decay=0.01):
        if lr <= 0.0:
            raise ValueError(f"Invalid learning rate: {lr}")
        if eps <= 0.0:
            raise ValueError(f"Invalid epsilon value: {eps}")
        if weight_decay < 0.0:
            raise ValueError(f"Invalid weight_decay value: {weight_decay}")
        if not 0.0 <= betas[0] < 1.0 or not 0.0 <= betas[1] < 1.0:
            raise ValueError(f"Invalid beta values: {betas}")

        # Initialize the parent class
        defaults = dict(lr=lr, betas=betas, eps=eps, weight_decay=weight_decay)
        super(AdamW, self).__init__(params, defaults)

    def step(self, closure=None):
        """
        Performs a single optimization step (parameter update).
        """
        # Closure is typically used for functions like L-BFGS but is not needed in AdamW here
        loss = None
        if closure is not None:
            loss = closure()

        # Iterate over all parameters
        for group in self.param_groups:
            lr = group['lr']
            beta1, beta2 = group['betas']
            eps = group['eps']
            weight_decay = group['weight_decay']

            # Update parameters for each parameter group
            for p in group['params']:
                if p.grad is None:
                    continue

                grad = p.grad.data
                state = self.state[p]

                # Initialize moment estimates on first step
                if len(state) == 0:
                    state['step'] = 0
                    state['m'] = torch.zeros_like(p.data)
                    state['v'] = torch.zeros_like(p.data)

                m, v = state['m'], state['v']
                state['step'] += 1

                # Compute moment estimates
                m.mul_(beta1).add_(grad, alpha=1 - beta1)
                v.mul_(beta2).addcmul_(grad, grad, value=1 - beta2)

                # Bias correction
                bias_correction1 = 1 - beta1**state['step']
                bias_correction2 = 1 - beta2**state['step']

                m_hat = m / bias_correction1
                v_hat = v / bias_correction2

                # Compute update and apply weight decay
                p.data.addcdiv_(-lr, m_hat, v_hat.sqrt().add_(eps))
                if weight_decay != 0:
                    p.data.add_(-lr * weight_decay, p.data)

        return loss


