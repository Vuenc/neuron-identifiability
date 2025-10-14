"""
Cosine similarity computation utilities for analyzing parameter updates between models.
"""

import torch
import torch.nn as nn
from typing import Dict, List, Tuple, Optional
import numpy as np


def compute_cossim_per_layer(updates1: Dict[str, torch.Tensor], 
                            updates2: Dict[str, torch.Tensor],
                            trainable: List[str]) -> Dict[str, float]:
    """
    Compute cosine similarity between parameter updates for each layer.
    
    Args:
        updates1: Parameter updates from model 1
        updates2: Parameter updates from model 2
        trainable: List of trainable parameter names
    Returns:
        Dictionary mapping parameter names to cosine similarity scores
    """
    similarities = {}
    
    for name in updates1:
        if name in updates2:
            # Only compute similarity for trainable parameters
            if name not in trainable:
                continue

            flat1 = updates1[name].flatten()
            flat2 = updates2[name].flatten()
            
            if torch.norm(flat1) > 0 and torch.norm(flat2) > 0:
                cos_sim = torch.cosine_similarity(flat1.unsqueeze(0), flat2.unsqueeze(0), dim=1).item()
                similarities[name] = cos_sim
            else:
                similarities[name] = 0.0
    
    return similarities


def compute_cossim_aggregate(updates1: Dict[str, torch.Tensor], 
                           updates2: Dict[str, torch.Tensor],
                           trainable: List[str]) -> float:
    """
    Compute cosine similarity between all parameter updates aggregated.
    
    Args:
        updates1: Parameter updates from model 1
        updates2: Parameter updates from model 2
        trainable: List of trainable parameter names
        
    Returns:
        Aggregate cosine similarity score
    """
    # Filter to only trainable parameters if model is provided
    updates1 = {name: tensor for name, tensor in updates1.items() if name in trainable}
    updates2 = {name: tensor for name, tensor in updates2.items() if name in trainable}
    
    # Concatenate all parameters
    all_params1 = torch.cat([updates1[name].flatten() for name in updates1])
    all_params2 = torch.cat([updates2[name].flatten() for name in updates2])
    
    # Compute cosine similarity
    if torch.norm(all_params1) > 0 and torch.norm(all_params2) > 0:
        cos_sim = torch.cosine_similarity(all_params1.unsqueeze(0), all_params2.unsqueeze(0), dim=1).item()
        return cos_sim
    else:
        return 0.0
