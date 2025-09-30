"""
Linear Mode Connectivity interpolation utilities.
Refactored from lmc/LMC_utils.py
"""

import torch
import numpy as np
import copy
from typing import Callable, List, Tuple, Dict, Any, Optional
from pathlib import Path


def interpolate_test(model1, model2, test_fn, steps=25, rewarm=False, data=None):
    """Perform linear interpolation test between two models.
    
    Args:
        model1: First model
        model2: Second model
        test_fn: Test function that takes a model and returns a metric
        steps: Number of interpolation steps
        rewarm: Whether to rewarm the models
        data: Data for rewarming (if applicable)
        
    Returns:
        List of test results for each interpolation step
    """
    import copy
    
    # Store original model states
    state1 = copy.deepcopy(model1.state_dict())
    state2 = copy.deepcopy(model2.state_dict())
    
    # Calculate distance and parameter count
    dist = dist_sd(state1, state2)
    num_params = get_num_params(model1)
    print(f'Distance / params: {dist / num_params:.6f}')
    
    results = []
    lambdas = torch.linspace(0, 1, steps=steps+1)
    
    for i, lam in enumerate(lambdas):
        # Interpolate parameters
        interpolated_state = lerp_sd(lam, state1, state2)
        
        # Load interpolated state into model1
        model1.load_state_dict(interpolated_state)
        
        # Test the interpolated model
        if rewarm and data is not None:
            # Rewarm the model (simplified version)
            model1.train()
            for _ in range(10):  # Short rewarming
                # This would need to be adapted based on the specific rewarming strategy
                pass
        
        result = test_fn(model1)
        results.append(result)
        
        print(f'Interpolation step {i+1}/{steps}: {result:.4f}')
    
    # Restore original model state
    model1.load_state_dict(state1)
    
    return results


def lerp_sd(lam, sd1, sd2):
    """Linear interpolation between two state dictionaries."""
    sd3 = copy.deepcopy(sd1)
    for name in sd1:
        # Only interpolate float parameters, skip integer parameters like num_batches_tracked
        if sd1[name].dtype.is_floating_point:
            sd3[name] = (1 - lam) * sd1[name] + lam * sd2[name]
        # For non-float parameters, keep the first model's values
    return sd3


def dist_sd(sd1, sd2, device=torch.device('cuda')):
    """Calculate L2 distance between two state dictionaries."""
    sqdist = torch.tensor([0.], dtype=torch.float, device=device)
    for name in sd1:
        sqdist += (sd1[name] - sd2[name]).float().square().sum()
    dist = sqdist.sqrt()
    return dist.item()


def get_num_params(model):
    """Get number of parameters in model, accounting for unused parameters."""
    num_params = sum([p.numel() for p in model.parameters()])
    if hasattr(model, 'count_unused_params'):
        num_params = num_params - model.count_unused_params()
    return num_params


def compute_barrier(results):
    """Compute the barrier height from interpolation results.
    
    Args:
        results: List of interpolation results (accuracies)
        
    Returns:
        Barrier height as percentage
    """
    if not results:
        return 0.0
    
    start_acc = results[0]
    end_acc = results[-1]
    min_acc = min(results)
    
    # Barrier is the relative drop from the higher endpoint to the minimum point
    # Positive = hill (min > max), Negative = valley/barrier (min < max)
    max_endpoint = max(start_acc, end_acc)
    if max_endpoint == 0:
        # If both endpoints have 0 accuracy, return 0 barrier
        return 0.0
    barrier = (min_acc - max_endpoint) / max_endpoint * 100
    return barrier


def compute_linearity(results, threshold=0.5):
    """Check if the interpolation is linear within threshold.
    
    Args:
        results: List of interpolation results
        threshold: Threshold for linearity check
        
    Returns:
        Boolean indicating if interpolation is linear
    """
    if len(results) < 3:
        return True
    
    # Check if the middle point deviates significantly from linear interpolation
    n = len(results)
    expected_middle = 0.5 * (results[0] + results[-1])
    actual_middle = results[n//2]
    
    deviation = abs(actual_middle - expected_middle)
    return deviation < threshold


def evaluate_model_comprehensive(model, train_loader, val_loader, test_loader, device='cuda'):
    """Evaluate model on train/val/test splits with comprehensive metrics.
    
    Args:
        model: Model to evaluate
        train_loader: Training data loader
        val_loader: Validation data loader  
        test_loader: Test data loader
        device: Device to run evaluation on
        
    Returns:
        Dictionary with train/val/test metrics
    """
    model.eval()
    results = {}
    
    for split_name, loader in [('train', train_loader), ('val', val_loader), ('test', test_loader)]:
        if loader is None:
            continue
            
        total_loss = 0.0
        correct = 0
        total = 0
        
        with torch.no_grad():
            for batch in loader:
                if isinstance(batch, (list, tuple)):
                    data, target = batch
                else:
                    data = batch.x
                    target = batch.y.squeeze()
                
                data, target = data.to(device), target.to(device)
                
                # Handle different data formats
                if hasattr(data, 'x'):
                    # Graph data
                    output = model(data.x, data.edge_index)
                else:
                    # Regular data
                    output = model(data)
                
                # Calculate loss
                if hasattr(model, 'loss_fn'):
                    loss = model.loss_fn(output, target)
                else:
                    loss = torch.nn.functional.cross_entropy(output, target)
                
                total_loss += loss.item()
                
                # Calculate accuracy
                pred = output.argmax(dim=1)
                correct += pred.eq(target).sum().item()
                total += target.size(0)
        
        results[f'{split_name}_loss'] = total_loss / len(loader)
        results[f'{split_name}_accuracy'] = 100.0 * correct / total
    
    return results


def interpolate_models(model1, model2, train_loader, val_loader, test_loader, 
                                   steps=25, device='cuda', use_wandb=False):
    """Perform comprehensive interpolation between two models.
    
    Args:
        model1: First model
        model2: Second model
        train_loader: Training data loader
        val_loader: Validation data loader
        test_loader: Test data loader
        steps: Number of interpolation steps
        device: Device to run evaluation on
        use_wandb: Whether to log to wandb
        
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
        metrics = evaluate_model_comprehensive(model1, train_loader, val_loader, test_loader, device)
        
        # Print meaningful metrics for this step
        train_acc = metrics.get('train_accuracy', 0.0)
        val_acc = metrics.get('val_accuracy', 0.0)
        train_loss = metrics.get('train_loss', 0.0)
        val_loss = metrics.get('val_loss', 0.0)
        test_acc = metrics.get('test_accuracy', 0.0)
        test_loss = metrics.get('test_loss', 0.0)
        
        print(f'Step {i+1:2d}/{steps+1} (λ={lam:.3f}): Train Acc={train_acc:.2f}%, Val Acc={val_acc:.2f}%, Train Loss={train_loss:.4f}, Val Loss={val_loss:.4f}')
        
        # Log to wandb if enabled
        if use_wandb:
            try:
                import wandb
                if wandb.run is not None:
                    wandb.log({
                        'interpolation_lambda': lam.item(),
                        'interpolation_train_loss': train_loss,
                        'interpolation_train_accuracy': train_acc,
                        'interpolation_val_loss': val_loss,
                        'interpolation_val_accuracy': val_acc,
                        'interpolation_test_loss': test_loss,
                        'interpolation_test_accuracy': test_acc,
                    })
            except ImportError:
                print("Warning: wandb not available for logging")
        
        # Store results
        results['lambdas'].append(lam.item())
        results['train_loss'].append(train_loss)
        results['train_accuracy'].append(train_acc)
        results['val_loss'].append(val_loss)
        results['val_accuracy'].append(val_acc)
        results['test_loss'].append(test_loss)
        results['test_accuracy'].append(test_acc)
    
    # Restore original state
    model1.load_state_dict(state1)
    
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


def evaluate_midpoint_models(models, train_loader, val_loader, test_loader, device='cuda', use_wandb=False):
    """Evaluate the midpoint (center) of multiple models.
    
    Args:
        models: List of models to interpolate
        train_loader: Training data loader
        val_loader: Validation data loader
        test_loader: Test data loader
        device: Device to run evaluation on
        use_wandb: Whether to log to wandb
        
    Returns:
        Dictionary with midpoint evaluation results
    """
    if len(models) < 2:
        raise ValueError("Need at least 2 models for midpoint evaluation")
    
    # Use first model as base
    base_model = models[0]
    base_state = copy.deepcopy(base_model.state_dict())
    
    # Calculate average state
    avg_state = {}
    for key in base_state.keys():
        avg_state[key] = base_state[key].clone()
    
    for model in models[1:]:
        state = model.state_dict()
        for key in avg_state.keys():
            # Only average float parameters, skip integer parameters like num_batches_tracked
            if avg_state[key].dtype.is_floating_point:
                avg_state[key] += state[key]
    
    # Normalize by number of models (only float parameters)
    for key in avg_state.keys():
        if avg_state[key].dtype.is_floating_point:
            avg_state[key] /= len(models)
    
    # Load average state into base model
    base_model.load_state_dict(avg_state)
    
    # Evaluate midpoint model
    results = evaluate_model_comprehensive(base_model, train_loader, val_loader, test_loader, device)
    
    # Add metadata
    results['num_models'] = len(models)
    results['evaluation_type'] = 'midpoint'
    
    # Log to wandb if enabled
    if use_wandb:
        try:
            import wandb
            if wandb.run is not None:
                wandb.log({
                    'midpoint_evaluation/train_loss': results.get('train_loss', 0.0),
                    'midpoint_evaluation/train_accuracy': results.get('train_accuracy', 0.0),
                    'midpoint_evaluation/val_loss': results.get('val_loss', 0.0),
                    'midpoint_evaluation/val_accuracy': results.get('val_accuracy', 0.0),
                    'midpoint_evaluation/test_loss': results.get('test_loss', 0.0),
                    'midpoint_evaluation/test_accuracy': results.get('test_accuracy', 0.0),
                    'midpoint_evaluation/num_models': len(models),
                })
        except ImportError:
            print("Warning: wandb not available for logging")
    
    # Restore original state
    base_model.load_state_dict(base_state)
    
    return results


def load_model_checkpoint(checkpoint_path, model_class, model_kwargs, device='cuda'):
    """Load a model from checkpoint.
    
    Args:
        checkpoint_path: Path to checkpoint file
        model_class: Model class to instantiate
        model_kwargs: Keyword arguments for model creation
        device: Device to load model on
        
    Returns:
        Loaded model
    """
    checkpoint = torch.load(checkpoint_path, map_location=device)
    
    # Create model
    model = model_class(**model_kwargs)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.to(device)
    
    return model


def evaluate_checkpoint_interpolation(checkpoint_paths, model_class, model_kwargs, 
                                    train_loader, val_loader, test_loader, 
                                    interpolation_type='grid', steps=25, device='cuda', use_wandb=False):
    """Evaluate interpolation between model checkpoints.
    
    Args:
        checkpoint_paths: List of checkpoint paths
        model_class: Model class to instantiate
        model_kwargs: Keyword arguments for model creation
        train_loader: Training data loader
        val_loader: Validation data loader
        test_loader: Test data loader
        interpolation_type: 'grid' for 2 models, 'midpoint' for >2 models
        steps: Number of interpolation steps (for grid)
        device: Device to run evaluation on
        use_wandb: Whether to log to wandb
        
    Returns:
        Dictionary with interpolation results
    """
    # Load models from checkpoints
    models = []
    for checkpoint_path in checkpoint_paths:
        model = load_model_checkpoint(checkpoint_path, model_class, model_kwargs, device)
        models.append(model)
    
    if interpolation_type == 'grid' and len(models) == 2:
        # Grid interpolation between 2 models
        return interpolate_models(
            models[0], models[1], train_loader, val_loader, test_loader, steps, device, use_wandb
        )
    elif len(models) >= 2:
        # Midpoint evaluation for multiple models
        return evaluate_midpoint_models(
            models, train_loader, val_loader, test_loader, device, use_wandb
        )
    else:
        raise ValueError(f"Invalid number of models: {len(models)}")


def save_interpolation_results(results, output_dir, filename='interpolation_results.pt'):
    """Save interpolation results to disk.
    
    Args:
        results: Interpolation results dictionary
        output_dir: Output directory
        filename: Filename to save results
    """
    output_path = Path(output_dir) / filename
    torch.save(results, output_path)
    print(f"Interpolation results saved to {output_path}")


def load_interpolation_results(output_dir, filename='interpolation_results.pt'):
    """Load interpolation results from disk.
    
    Args:
        output_dir: Output directory
        filename: Filename to load results from
        
    Returns:
        Loaded interpolation results
    """
    results_path = Path(output_dir) / filename
    if results_path.exists():
        return torch.load(results_path)
    else:
        raise FileNotFoundError(f"Interpolation results not found at {results_path}")
