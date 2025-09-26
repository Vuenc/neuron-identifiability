"""
Weight matching and rebasing utilities.
Refactored from lmc/rebasin/weight_matching.py
"""

import torch
import numpy as np
from scipy.optimize import linear_sum_assignment
from typing import Dict, List, Tuple


def mlp_permutation_spec(num_layers, norm=True):
    """Create permutation specification for MLP.
    
    Args:
        num_layers: Number of layers
        norm: Whether to include normalization layers
        
    Returns:
        Permutation specification
    """
    # This is a simplified version - the full implementation would be more complex
    spec = {}
    for i in range(num_layers):
        spec[f'lins.{i}.weight'] = (f'lins.{i}.weight', (0, 1))
        spec[f'lins.{i}.bias'] = (f'lins.{i}.bias', (0,))
        if norm:
            spec[f'norms.{i}.weight'] = (f'norms.{i}.weight', (0,))
            spec[f'norms.{i}.bias'] = (f'norms.{i}.bias', (0,))
    return spec


def resnet20_permutation_spec():
    """Create permutation specification for ResNet-20."""
    # This is a simplified version - the full implementation would be more complex
    spec = {}
    # Add ResNet-specific permutation specifications
    return spec


def weight_matching(perm_spec, state_dict1, state_dict2):
    """Match weights between two models using linear assignment.
    
    Args:
        perm_spec: Permutation specification
        state_dict1: First model's state dict
        state_dict2: Second model's state dict
        
    Returns:
        Dictionary of permutations
    """
    perms = {}
    
    for key, (weight_key, axes) in perm_spec.items():
        if key in state_dict1 and key in state_dict2:
            w1 = state_dict1[key]
            w2 = state_dict2[key]
            
            if len(axes) == 1:
                # 1D case (bias)
                perms[key] = torch.arange(w1.shape[0])
            elif len(axes) == 2:
                # 2D case (weight)
                # Compute similarity matrix
                sim = torch.mm(w1, w2.t())
                # Solve assignment problem
                _, perm = linear_sum_assignment(-sim.cpu().numpy())
                perms[key] = torch.tensor(perm)
    
    return perms


def apply_permutation(perm_spec, perms, state_dict):
    """Apply permutations to a state dictionary.
    
    Args:
        perm_spec: Permutation specification
        perms: Dictionary of permutations
        state_dict: State dictionary to permute
        
    Returns:
        Permuted state dictionary
    """
    new_state_dict = state_dict.copy()
    
    for key, (weight_key, axes) in perm_spec.items():
        if key in perms and key in state_dict:
            perm = perms[key]
            w = state_dict[key]
            
            if len(axes) == 1:
                # 1D case (bias)
                new_state_dict[key] = w[perm]
            elif len(axes) == 2:
                # 2D case (weight)
                if axes == (0, 1):
                    new_state_dict[key] = w[perm][:, perm]
                elif axes == (1, 0):
                    new_state_dict[key] = w[:, perm][perm]
    
    return new_state_dict


def rebasin_models(model1, model2, perm_spec):
    """Rebasin two models using weight matching.
    
    Args:
        model1: First model
        model2: Second model
        perm_spec: Permutation specification
        
    Returns:
        Rebased second model
    """
    state1 = model1.state_dict()
    state2 = model2.state_dict()
    
    perms = weight_matching(perm_spec, state1, state2)
    new_state2 = apply_permutation(perm_spec, perms, state2)
    
    model2.load_state_dict(new_state2)
    return model2
