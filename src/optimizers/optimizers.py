"""
Optimizer creation utilities.
"""

import torch.optim as optim
from ..core.registry import register


@register('optimizer', 'adam')
def create_adam_optimizer(model, lr=0.001, weight_decay=0.0, **kwargs):
    """Create Adam optimizer."""
    return optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay, **kwargs)


@register('optimizer', 'adamw')
def create_adamw_optimizer(model, lr=0.001, weight_decay=0.01, **kwargs):
    """Create AdamW optimizer."""
    return optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay, **kwargs)


@register('optimizer', 'sgd')
def create_sgd_optimizer(model, lr=0.1, momentum=0.9, weight_decay=0.0, **kwargs):
    """Create SGD optimizer."""
    return optim.SGD(model.parameters(), lr=lr, momentum=momentum, weight_decay=weight_decay, **kwargs)


@register('optimizer', 'rmsprop')
def create_rmsprop_optimizer(model, lr=0.01, weight_decay=0.0, **kwargs):
    """Create RMSprop optimizer."""
    return optim.RMSprop(model.parameters(), lr=lr, weight_decay=weight_decay, **kwargs)


def create_optimizer(optimizer_name, model, **kwargs):
    """Create optimizer by name.
    
    Args:
        optimizer_name: Name of the optimizer
        model: Model to optimize
        **kwargs: Optimizer-specific arguments
        
    Returns:
        Optimizer object
    """
    from ..core.registry import build_component
    # Remove 'name' from kwargs to avoid conflict with build_component parameter
    kwargs.pop('name', None)
    return build_component('optimizer', optimizer_name, model=model, **kwargs)
