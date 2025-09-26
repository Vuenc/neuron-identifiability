"""
Learning rate scheduler utilities.
"""

import torch.optim as optim
from torch.optim.lr_scheduler import LambdaLR, MultiStepLR, CosineAnnealingLR, StepLR
from ..core.registry import register


@register('scheduler', 'none')
def create_no_scheduler(optimizer, **kwargs):
    """Create no-op scheduler."""
    return None


@register('scheduler', 'step')
def create_step_scheduler(optimizer, step_size=30, gamma=0.1, **kwargs):
    """Create step scheduler."""
    return StepLR(optimizer, step_size=step_size, gamma=gamma)


@register('scheduler', 'multistep')
def create_multistep_scheduler(optimizer, milestones=[30, 60, 90], gamma=0.1, **kwargs):
    """Create multi-step scheduler."""
    return MultiStepLR(optimizer, milestones=milestones, gamma=gamma)


@register('scheduler', 'cosine')
def create_cosine_scheduler(optimizer, T_max=100, eta_min=0, **kwargs):
    """Create cosine annealing scheduler."""
    return CosineAnnealingLR(optimizer, T_max=T_max, eta_min=eta_min)


@register('scheduler', 'linear_warmup')
def create_linear_warmup_scheduler(optimizer, warmup_steps=25, **kwargs):
    """Create linear warmup scheduler."""
    def linear_warmup(step):
        if step < warmup_steps:
            return float(step + 1) / float(warmup_steps)
        else:
            return 1.0
    
    return LambdaLR(optimizer, linear_warmup)


@register('scheduler', 'resnet_scheduler')
def create_resnet_scheduler(optimizer, milestones=None, gamma=1.259, **kwargs):
    """Create ResNet-style scheduler."""
    if milestones is None:
        milestones = list(range(1, 21))
    return MultiStepLR(optimizer, milestones=milestones, gamma=gamma)


def create_scheduler(scheduler_name, optimizer, **kwargs):
    """Create scheduler by name.
    
    Args:
        scheduler_name: Name of the scheduler
        optimizer: Optimizer to schedule
        **kwargs: Scheduler-specific arguments
        
    Returns:
        Scheduler object
    """
    from ..core.registry import build_component
    # Remove 'name' from kwargs to avoid conflict with build_component parameter
    kwargs.pop('name', None)
    return build_component('scheduler', scheduler_name, optimizer=optimizer, **kwargs)
