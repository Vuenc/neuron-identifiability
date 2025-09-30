"""
Activation functions for MLP models.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


def setup_activation(activation):
    """Helper function to setup activation for MLP models."""
    if not activation:
        return None
    else:
        if activation == 'relu':
            return F.relu
        elif activation == 'gelu':
            return F.gelu
        elif activation == 'identity':
            return lambda x: x
        else:
            raise ValueError(f"Bad activation type: {activation}")
