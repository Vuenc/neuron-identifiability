"""
GNN-specific interpolation utilities.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import copy
from .interpolation import dist_sd, get_num_params, compute_barrier, compute_linearity


def reset_bn_stats(model, data):
    """Reset batch normalization statistics - from github.com/KellerJordan/REPAIR"""
    for m in model.modules():
        if type(m) == nn.BatchNorm1d:
            m.momentum = None  # use simple average
            m.reset_running_stats()
    model.train()
    with torch.no_grad():
        for _ in range(50):
            _ = model(data.x, data.adj_t)


def evaluate_gnn_model_comprehensive_batched(
        stacked_models_forward, data, split_idx, device='cuda'
) -> dict:
    """Evaluate GNN models on train/val/test splits with comprehensive metrics using batched evaluation.

    Args:
        stacked_models_forward: Function that takes (x, adj_t) and returns output tensor with one parameter batch dimension
        data: Graph data
        split_idx: Dictionary with train/val/test indices
        device: Device to run evaluation on

    Returns:
        Dictionary with train/val/test metrics:
        - 'train_loss', 'val_loss', 'test_loss' as lists of floats
        - 'train_accuracy', 'val_accuracy', 'test_accuracy' as lists of floats (range 0-100)
    """
    results = {}
    data = data.to(device)
    split_idx_device = {k: v.to(device) if v is not None else None for k, v in split_idx.items()}
    
    with torch.no_grad():
        # Get predictions for all nodes - output shape: [num_models, num_nodes, num_classes]
        out = stacked_models_forward(data.x, data.adj_t)
        
        # Evaluate on each split
        for split_name, indices in split_idx_device.items():
            if indices is None or len(indices) == 0:
                continue
            
            # Map 'valid' to 'val' for consistency with interpolation summary
            if split_name == 'valid':
                split_name = 'val'
            
            # Get predictions and targets for this split
            split_out = out[:, indices]  # [num_models, num_nodes_in_split, num_classes]
            split_targets = data.y.squeeze(1)[indices]  # [num_nodes_in_split]
            
            # Calculate loss for each model
            # loss shape: [num_models]
            loss = F.nll_loss(
                split_out.view(-1, split_out.shape[-1]),
                split_targets.repeat(out.shape[0]),
                reduction='none'
            ).reshape(out.shape[0], -1).mean(dim=1)
            
            # Calculate accuracy for each model
            pred = split_out.argmax(dim=-1)  # [num_models, num_nodes_in_split]
            correct = (pred == split_targets[None]).sum(dim=1)  # [num_models]
            accuracy = 100.0 * correct / len(split_targets)
            
            results[f'{split_name}_loss'] = loss.tolist()
            results[f'{split_name}_accuracy'] = accuracy.tolist()
    
    return results


def interpolate_gnn_models(model: torch.nn.Module, model1_state, model2_state, data, split_idx, steps=25, device='cuda', rewarm=True):
    """Perform comprehensive interpolation between two GNN models using batched evaluation.
    
    Args:
        model: GNN model template
        model1_state: First model state dict
        model2_state: Second model state dict
        data: Graph data
        split_idx: Dictionary with train/val/test indices
        steps: Number of interpolation steps
        device: Device to run evaluation on
        rewarm: Whether to reset batch norm stats for each interpolated model
        
    Returns:
        Dictionary with interpolation results
    """
    model.eval()
    
    steps += 1  # include endpoints
    
    # Create interpolation factors
    lambdas = torch.linspace(0, 1, steps).tolist()
    
    # If rewarm is needed, we need to reset BN stats for each interpolated model
    # This requires loading each state individually, so we'll create individual states first
    if rewarm:
        # Create individual interpolated states
        interpolated_states = []
        for lam in lambdas:
            # Interpolate parameters
            interpolated_state = {}
            for key in model1_state:
                if model1_state[key].dtype.is_floating_point:
                    interpolated_state[key] = (1 - lam) * model1_state[key] + lam * model2_state[key]
                else:
                    interpolated_state[key] = model1_state[key]  # Keep non-float params from model1
            
            # Load into model and reset BN stats
            model.load_state_dict(interpolated_state)
            reset_bn_stats(model, data)
            # Extract updated state (BN stats may have changed)
            interpolated_states.append(copy.deepcopy(model.state_dict()))
        
        # Now create stacked parameters from rewarmed states
        # Filter to only include float parameters/buffers (exclude num_batches_tracked, etc.)
        def stack_tensors(*tensors):
            return torch.stack(tensors, dim=0)
        
        # Filter states to only float tensors
        filtered_states = []
        for state in interpolated_states:
            filtered = {k: v for k, v in state.items() if v.dtype.is_floating_point}
            filtered_states.append(filtered)
        
        interpolated_parameters = torch.utils._pytree.tree_map(stack_tensors, *filtered_states)  # type: ignore[import]
    else:
        # No rewarm needed, create stacked parameters directly
        def interpolate_tensors(t1, t2):
            interpolation_factors = torch.linspace(0, 1, steps, device=t1.device).view(steps, *([1] * t1.ndim))
            return t1 * (1-interpolation_factors) + t2 * interpolation_factors
        
        # Filter to only include float parameters/buffers (exclude num_batches_tracked, etc.)
        filtered_state1 = {k: v for k, v in model1_state.items() if v.dtype.is_floating_point}
        filtered_state2 = {k: v for k, v in model2_state.items() if v.dtype.is_floating_point}
        
        interpolated_parameters = torch.utils._pytree.tree_map(interpolate_tensors, filtered_state1, filtered_state2)  # type: ignore[import]
    
    # Define a function (params, x, adj_t) -> outputs, where outputs has one params batch dimension
    # We need to handle the fact that GNN forward takes two arguments (x, adj_t)
    # Use strict=False to allow buffers to remain on the model (not replaced by functional_call)
    def gnn_forward(params, x, adj_t):
        return torch.func.functional_call(model, params, (x, adj_t), strict=False)
    
    # Use vmap over the params dimension (dim 0), keep x and adj_t the same
    stacked_models_forward = torch.vmap(gnn_forward, in_dims=(0, None, None))
    
    metrics_keys = ['train_accuracy', 'val_accuracy', 'train_loss', 'val_loss', 'test_accuracy', 'test_loss']
    
    # Evaluate interpolated models
    metrics = evaluate_gnn_model_comprehensive_batched(
        lambda x, adj_t: stacked_models_forward(interpolated_parameters, x, adj_t),
        data, split_idx, device
    )
    
    interpolation_factors = lambdas
    
    # Calculate distance metrics
    dist = dist_sd(model1_state, model2_state)
    num_params = get_num_params(model)
    normalized_dist = dist / np.sqrt(num_params)
    
    # Calculate barriers and linearity
    splits = ["train", "val", "test"]
    barrier_heights = {
        f"{split}_barrier_height": compute_barrier(metrics[f'{split}_accuracy']) for split in splits
    }
    linearity_values = {
        f"{split}_linearity": compute_linearity(metrics[f'{split}_accuracy']) for split in splits
    }
    
    # Store results
    results = {
        'lambdas': interpolation_factors,
        **{key: metrics[key] if key in metrics else float('nan') for key in metrics_keys},
        'distance': dist,
        'normalized_distance': normalized_dist,
        'num_params': num_params,
        **barrier_heights,
        **linearity_values
    }
    
    for (i, lam, train_acc, val_acc, test_acc,
         train_loss, val_loss, test_loss) in \
        zip(range(steps), interpolation_factors,
            metrics['train_accuracy'], metrics['val_accuracy'],
            metrics['test_accuracy'], metrics['train_loss'],
            metrics['val_loss'], metrics['test_loss']):
        print(f'Step {i+1:2d}/{steps} (λ={lam:.3f}): '
              f'Train Acc={train_acc:.2f}%, Val Acc={val_acc:.2f}%, Test Acc={test_acc:.2f}%, '
              f'Train Loss={train_loss:.4f}, Val Loss={val_loss:.4f}, Test Loss={test_loss:.4f}')
    
    return results