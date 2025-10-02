"""
GNN-specific interpolation utilities.
"""

import torch
import copy
from .interpolation import dist_sd, get_num_params, lerp_sd, compute_barrier, compute_linearity


def evaluate_gnn_model(model, data, split_idx, device='cuda'):
    """Evaluate GNN model on train/val/test splits.
    
    Args:
        model: GNN model to evaluate
        data: Graph data
        split_idx: Dictionary with train/val/test indices
        device: Device to run evaluation on
        
    Returns:
        Dictionary with train/val/test metrics
    """
    model.eval()
    model.to(device)
    data = data.to(device)
    for key in split_idx:
        split_idx[key] = split_idx[key].to(device)
    
    results = {}
    
    with torch.no_grad():
        # Get predictions for all nodes
        out = model(data.x, data.adj_t)
        
        # Evaluate on each split
        for split_name, indices in split_idx.items():
            if indices is None or len(indices) == 0:
                continue
                
            # Get predictions and targets for this split
            split_out = out[indices]
            split_targets = data.y.squeeze(1)[indices]
            
            # Calculate loss
            loss = torch.nn.functional.nll_loss(split_out, split_targets).item()
            
            # Calculate accuracy
            pred = split_out.argmax(dim=-1)
            correct = (pred == split_targets).float().sum().item()
            accuracy = correct / len(split_targets) * 100
            
            # Map 'valid' to 'val' for consistency with interpolation summary
            if split_name == 'valid':
                split_name = 'val'
            
            results[f'{split_name}_loss'] = loss
            results[f'{split_name}_accuracy'] = accuracy
    
    return results


def interpolate_gnn_models(model1, model2, data, split_idx, steps=25, device='cuda', use_wandb=False):
    """Perform comprehensive interpolation between two GNN models.
    
    Args:
        model1: First GNN model
        model2: Second GNN model
        data: Graph data
        split_idx: Dictionary with train/val/test indices
        steps: Number of interpolation steps
        device: Device to run evaluation on
        
    Returns:
        Dictionary with interpolation results
    """
    # Store original states
    state1 = copy.deepcopy(model1.state_dict())
    state2 = copy.deepcopy(model2.state_dict())
    
    # Calculate distance metrics
    dist = dist_sd(state1, state2)
    num_params = get_num_params(model1)
    normalized_dist = dist / num_params
    
    print(f'Model distance: {dist:.4f}')
    print(f'Normalized distance: {normalized_dist:.6f}')
    
    # Interpolation results
    results = {
        'lambdas': [],
        'train_loss': [],
        'train_accuracy': [],
        'val_loss': [],
        'val_accuracy': [],
        'test_loss': [],
        'test_accuracy': [],
        'distance': dist,
        'normalized_distance': normalized_dist,
        'num_params': num_params
    }
    
    lambdas = torch.linspace(0, 1, steps=steps+1)
    
    for i, lam in enumerate(lambdas):
        # Interpolate parameters
        interpolated_state = lerp_sd(lam, state1, state2)
        model1.load_state_dict(interpolated_state)
        
        # Evaluate interpolated model
        metrics = evaluate_gnn_model(model1, data, split_idx, device)
        
        # Extract metrics
        train_acc = metrics.get('train_accuracy', 0.0)
        val_acc = metrics.get('val_accuracy', 0.0)
        train_loss = metrics.get('train_loss', 0.0)
        val_loss = metrics.get('val_loss', 0.0)
        test_acc = metrics.get('test_accuracy', 0.0)
        test_loss = metrics.get('test_loss', 0.0)
        
        print(f'Step {i+1:2d}/{steps+1} (λ={lam:.3f}): Train Acc={train_acc:.2f}%, Val Acc={val_acc:.2f}%, Test Acc={test_acc:.2f}%, Train Loss={train_loss:.4f}, Val Loss={val_loss:.4f}, Test Loss={test_loss:.4f}')
        
        # Log to wandb if enabled
        if use_wandb:
            try:
                import wandb
                if wandb.run is not None:
                    wandb.log({
                        'interpolation_step': i,
                        'interpolation_lambda': lam.item(),
                        'interpolation_train_accuracy': train_acc,
                        'interpolation_val_accuracy': val_acc,
                        'interpolation_train_loss': train_loss,
                        'interpolation_val_loss': val_loss,
                        'interpolation_test_accuracy': test_acc,
                        'interpolation_test_loss': test_loss,
                    })
            except ImportError:
                pass  # wandb not available
        
        # Store results
        results['lambdas'].append(lam.item())
        results['train_loss'].append(train_loss)
        results['train_accuracy'].append(train_acc)
        results['val_loss'].append(val_loss)
        results['val_accuracy'].append(val_acc)
        results['test_loss'].append(test_loss)
        results['test_accuracy'].append(test_acc)
    
    # Restore original states
    model1.load_state_dict(state1)
    model2.load_state_dict(state2)
    
    # Calculate additional metrics
    results['barrier_height'] = compute_barrier(results['test_accuracy'])
    results['linearity'] = compute_linearity(results['test_accuracy'])
    
    # Print summary
    print(f"\nGrid interpolation summary:")
    print(f"  Best train accuracy: {max(results['train_accuracy']):.2f}%")
    print(f"  Best val accuracy: {max(results['val_accuracy']):.2f}%")
    print(f"  Best test accuracy: {max(results['test_accuracy']):.2f}%")
    print(f"  Worst train accuracy: {min(results['train_accuracy']):.2f}%")
    print(f"  Worst val accuracy: {min(results['val_accuracy']):.2f}%")
    print(f"  Worst test accuracy: {min(results['test_accuracy']):.2f}%")
    print(f"  Barrier height: {results['barrier_height']:.2f}%")
    print(f"  Linearity: {results['linearity']}")
    
    return results
