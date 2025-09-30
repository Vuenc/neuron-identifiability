"""
Cosine similarity computation utilities for analyzing parameter updates between models.
"""

import torch
import torch.nn as nn
from typing import Dict, List, Tuple, Optional
import numpy as np


def compute_parameter_updates(model: nn.Module, 
                            initial_params: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
    """
    Compute parameter updates (current - initial) for each parameter.
    
    Args:
        model: Current model state
        initial_params: Initial parameter state dict
        
    Returns:
        Dictionary mapping parameter names to their updates
    """
    updates = {}
    for name, param in model.named_parameters():
        if name in initial_params:
            updates[name] = param.detach().cpu() - initial_params[name]
    return updates


def compute_cosine_similarity_per_layer(updates1: Dict[str, torch.Tensor], 
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


def compute_cosine_similarity_aggregate(updates1: Dict[str, torch.Tensor], 
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


def compute_cosine_similarity_analysis(model1: nn.Module, 
                                     model2: nn.Module,
                                     initial_params1: Dict[str, torch.Tensor],
                                     initial_params2: Dict[str, torch.Tensor]) -> Dict[str, any]:
    """
    Compute comprehensive cosine similarity analysis between two models.
    
    Args:
        model1: Current state of model 1
        model2: Current state of model 2
        initial_params1: Initial parameters of model 1
        initial_params2: Initial parameters of model 2
        
    Returns:
        Dictionary containing cosine similarity results
    """
    # Compute parameter updates
    updates1 = compute_parameter_updates(model1, initial_params1)
    updates2 = compute_parameter_updates(model2, initial_params2)
    
    # Compute per-layer similarities
    per_layer_similarities = compute_cosine_similarity_per_layer(updates1, updates2, model1['trainable'])
    
    # Compute aggregate similarity
    aggregate_similarity = compute_cosine_similarity_aggregate(updates1, updates2, model1['trainable'])
    
    # Compute statistics
    layer_similarities = list(per_layer_similarities.values())
    mean_layer_similarity = np.mean(layer_similarities) if layer_similarities else 0.0
    std_layer_similarity = np.std(layer_similarities) if layer_similarities else 0.0
    
    return {
        'per_layer_similarities': per_layer_similarities,
        'aggregate_similarity': aggregate_similarity,
        'mean_layer_similarity': mean_layer_similarity,
        'std_layer_similarity': std_layer_similarity,
        'num_layers': len(per_layer_similarities)
    }


def save_cosine_similarity_results(results: Dict[str, any], 
                                 output_dir: str, 
                                 epoch: int, 
                                 step: int) -> str:
    """
    Save cosine similarity results to file.
    
    Args:
        results: Cosine similarity analysis results
        output_dir: Output directory
        epoch: Current epoch
        step: Current step
        
    Returns:
        Path to saved file
    """
    import os
    from pathlib import Path
    
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    filename = f"cosine_similarity_epoch_{epoch}_step_{step}.pt"
    filepath = output_path / filename
    
    torch.save({
        'epoch': epoch,
        'step': step,
        'results': results
    }, filepath)
    
    return str(filepath)

