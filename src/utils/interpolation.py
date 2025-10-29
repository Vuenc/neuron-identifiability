"""
Linear Mode Connectivity interpolation utilities.
Refactored from lmc/LMC_utils.py
"""

from typing import Dict, List
import torch
import torch.utils
import numpy as np
import copy
from pathlib import Path


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

def evaluate_model_comprehensive_batched(
        stacked_models_forward, train_loader, val_loader, test_loader, device='cuda'
) -> Dict[str, List]:
    """Evaluate model on train/val/test splits with comprehensive metrics.

    Args:
        stacked_models_forward: Function that takes an input with one data batch dimension and outputs the forward result
            of multiple models as a tensor with one parameter batch dimension and one data batch dimension
        train_loader: Training data loader
        val_loader: Validation data loader
        test_loader: Test data loader
        device: Device to run evaluation on

    Returns:
        Dictionary with train/val/test metrics:
        - 'train_loss', 'val_loss', 'test_loss' as lists of floats
        - 'train_accuracy', 'val_accuracy', 'test_accuracy' as lists of floats (range 0-100)
    """
    results = {}

    for split_name, loader in [('train', train_loader), ('val', val_loader), ('test', test_loader)]:
        if loader is None:
            continue

        total_loss = None
        correct_instances = None
        total_instances = 0

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
                    # TODO graph forward with stacked models
                    raise NotImplementedError()
                    output = model(data.x, data.edge_index)
                else:
                    # Regular data
                    output = stacked_models_forward(data)

                # In the stacked model output, dim 0 is the parameter batch dimension and dim 1 is the data batch dimension
                # Calculate loss (reduction='none' is crucial since we have a batch of model parameters and a batch of data instances)
                loss = torch.nn.functional.cross_entropy(
                    output.view(output.shape[0] * output.shape[1], *output.shape[2:]),
                    target.repeat(output.shape[0]),
                    reduction='none')
                loss = loss.reshape(*output.shape[:2]).mean(dim=1) # loss averaged over data batch, but kept separate over model params batch

                if total_loss is None:
                    total_loss = loss
                else:
                    total_loss += loss

                # Calculate accuracy
                pred = output.argmax(dim=2)
                correct = (pred == target[None]).sum(dim=1)
                if correct_instances is None:
                    correct_instances = correct
                else:
                    correct_instances += correct
                total_instances += target.size(0)

        assert total_loss is not None and correct_instances is not None

        # results[f'{split_name}_loss'] = total_loss / len(loader) # TODO this is wrong, isn't it? -> maybe it was approximately not wrong with the CE reduction='mean'
        results[f'{split_name}_loss'] = (total_loss / total_instances).tolist()
        results[f'{split_name}_accuracy'] = (100.0 * correct_instances / total_instances).tolist()

    return results


def interpolate_models(
        model: torch.nn.Module, model1_state, model2_state,
        train_loader, val_loader, test_loader,
        steps=25, device='cuda'):
    """Perform comprehensive interpolation between two models.
    
    Args:
        model1: First model
        model2: Second model
        train_loader: Training data loader
        val_loader: Validation data loader
        test_loader: Test data loader
        steps: Number of interpolation steps
        device: Device to run evaluation on
        
    Returns:
        Dictionary with interpolation results
    """
    model.eval()

    def interpolate_tensors(t1, t2):
        interpolation_factors = torch.linspace(0, 1, steps, device=t1.device).view(steps, *([1] * t1.ndim))
        return t1 * (1-interpolation_factors) + t2 * interpolation_factors
    # Define a function (params, inputs) -> outputs, where inputs have one data batch dimension and outputs has one params batch and one data batch dimension 
    stacked_models_forward = torch.vmap(lambda params, inputs: torch.func.functional_call(model, params, inputs), in_dims=(0, None))

    # Using tree_map, create a state dict where each leaf value is a stacked tensor of all interpolated versions of that parameter between the two models
    interpolated_parameters = torch.utils._pytree.tree_map(interpolate_tensors, model1_state, model2_state) # type: ignore[import]

    # TODO introduce batching over parameters (maybe for large models 20 model evaluations at once is too much)
    # TODO it might be smart to use other data loaders (larger batches for train, for example) for added efficiency

    metrics_keys = ['train_accuracy', 'val_accuracy', 'train_loss', 'val_loss', 'test_accuracy', 'test_loss']

    # Evaluate interpolated model
    metrics = evaluate_model_comprehensive_batched(
        lambda inputs: stacked_models_forward(interpolated_parameters, inputs),
        train_loader, val_loader, test_loader, device
    )
    
    interpolation_factors = torch.linspace(0, 1, steps=steps).tolist()

    # Calculate distance metrics
    dist = dist_sd(model1_state, model2_state)
    num_params = get_num_params(model)
    # TODO better divide by sqrt(num_params) = length of n-dim 1-vector?
    normalized_dist = dist / num_params

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

    # Simulate batched forward for single model
    batched_forward = lambda input: base_model.forward(input).unsqueeze(0)

    # Evaluate midpoint model and unpack the batched metrics
    results = evaluate_model_comprehensive_batched(batched_forward, train_loader, val_loader, test_loader, device)
    results = {key: metric for key, [metric] in results.items()}
    
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
