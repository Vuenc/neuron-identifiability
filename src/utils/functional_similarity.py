"""
Functional similarity computation utilities for analyzing prediction agreement between models.
"""

import torch
import torch.nn as nn
from typing import Dict, List, Tuple, Optional
import numpy as np
from torch.utils.data import DataLoader


def compute_functional_similarity(model1: nn.Module, 
                                model2: nn.Module,
                                data_loader: DataLoader,
                                device: str = 'cuda') -> Dict[str, float]:
    """
    Compute functional similarity between two models by measuring prediction agreement.
    
    Args:
        model1: First model
        model2: Second model  
        data_loader: DataLoader for evaluation
        device: Device to run evaluation on
        
    Returns:
        Dictionary containing functional similarity metrics
    """
    model1.eval()
    model2.eval()
    
    total_samples = 0
    agreement_count = 0
    
    with torch.no_grad():
        for data, target in data_loader:
            data, target = data.to(device), target.to(device)
            
            # Get predictions from both models
            pred1 = model1(data).argmax(dim=1)
            pred2 = model2(data).argmax(dim=1)
            
            # Count agreements
            agreement = (pred1 == pred2).sum().item()
            agreement_count += agreement
            total_samples += data.size(0)
    
    functional_similarity = agreement_count / total_samples if total_samples > 0 else 0.0
    
    return {
        'funcsim': functional_similarity,
        'agreement_count': agreement_count,
        'total_samples': total_samples
    }


def compute_functional_similarity_gnn(model1: nn.Module,
                                    model2: nn.Module,
                                    data: torch.Tensor,
                                    split_idx: Dict[str, torch.Tensor],
                                    device: str = 'cuda') -> Dict[str, float]:
    """
    Compute functional similarity for GNN models using split indices.
    
    Args:
        model1: First GNN model
        model2: Second GNN model
        data: Graph data tensor
        split_idx: Dictionary containing train/val/test split indices
        device: Device to run evaluation on
        
    Returns:
        Dictionary containing functional similarity metrics for each split
    """
    model1.eval()
    model2.eval()
    
    results = {}
    
    with torch.no_grad():
        # Get predictions from both models
        pred1 = model1(data.x, data.adj_t).argmax(dim=1)
        pred2 = model2(data.x, data.adj_t).argmax(dim=1)
        
        # Compute similarity for each split
        for split_name, indices in split_idx.items():
            if len(indices) == 0:
                continue
                
            split_pred1 = pred1[indices]
            split_pred2 = pred2[indices]
            
            agreement = (split_pred1 == split_pred2).sum().item()
            total_samples = len(indices)
            functional_similarity = agreement / total_samples if total_samples > 0 else 0.0
            
            results[f'{split_name}_funcsim'] = functional_similarity
            results[f'{split_name}_agreement_count'] = agreement
            results[f'{split_name}_total_samples'] = total_samples
    
    return results


def compute_functional_similarity_aggregate(results: Dict[str, float]) -> float:
    """
    Compute aggregate functional similarity across all splits.
    
    Args:
        results: Dictionary containing functional similarity results
        
    Returns:
        Aggregate functional similarity score
    """
    # Extract functional similarity scores (excluding count and total_samples)
    similarity_scores = []
    for key, value in results.items():
        if key.endswith('_funcsim'):
            similarity_scores.append(value)
    
    return np.mean(similarity_scores) if similarity_scores else 0.0