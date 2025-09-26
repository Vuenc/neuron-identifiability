"""
Utilities for analyzing saved gradients and parameters.
"""

import torch
import numpy as np
from pathlib import Path
import matplotlib.pyplot as plt
from typing import Dict, List, Optional, Tuple


def load_gradients(gradients_dir: Path) -> Dict[int, Dict[str, torch.Tensor]]:
    """Load all saved gradients from a directory.
    
    Args:
        gradients_dir: Directory containing gradient files
        
    Returns:
        Dictionary mapping epoch to gradient dictionary
    """
    gradients = {}
    
    for grad_file in sorted(gradients_dir.glob("gradients_epoch_*.pt")):
        data = torch.load(grad_file)
        epoch = data['epoch']
        gradients[epoch] = data['gradients']
    
    return gradients


def load_parameters(parameters_dir: Path) -> Dict[int, Dict[str, torch.Tensor]]:
    """Load all saved parameters from a directory.
    
    Args:
        parameters_dir: Directory containing parameter files
        
    Returns:
        Dictionary mapping epoch to parameter dictionary
    """
    parameters = {}
    
    for param_file in sorted(parameters_dir.glob("parameters_epoch_*.pt")):
        data = torch.load(param_file)
        epoch = data['epoch']
        parameters[epoch] = data['parameters']
    
    return parameters


def compute_gradient_norms(gradients: Dict[str, torch.Tensor]) -> Dict[str, float]:
    """Compute L2 norms of gradients for each parameter.
    
    Args:
        gradients: Dictionary of gradients by parameter name
        
    Returns:
        Dictionary of gradient norms by parameter name
    """
    norms = {}
    for name, grad in gradients.items():
        if grad is not None:
            norms[name] = torch.norm(grad).item()
    return norms


def compute_parameter_norms(parameters: Dict[str, torch.Tensor]) -> Dict[str, float]:
    """Compute L2 norms of parameters.
    
    Args:
        parameters: Dictionary of parameters by name
        
    Returns:
        Dictionary of parameter norms by name
    """
    norms = {}
    for name, param in parameters.items():
        if param is not None:
            norms[name] = torch.norm(param).item()
    return norms


def plot_gradient_evolution(gradients_by_epoch: Dict[int, Dict[str, torch.Tensor]], 
                          output_dir: Path,
                          param_names: Optional[List[str]] = None):
    """Plot gradient evolution over training.
    
    Args:
        gradients_by_epoch: Gradients organized by epoch
        output_dir: Directory to save plots
        param_names: Specific parameters to plot (if None, plot all)
    """
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Get all parameter names
    if param_names is None:
        param_names = list(next(iter(gradients_by_epoch.values())).keys())
    
    epochs = sorted(gradients_by_epoch.keys())
    
    for param_name in param_names:
        if param_name not in next(iter(gradients_by_epoch.values())):
            continue
            
        norms = []
        for epoch in epochs:
            grad_norms = compute_gradient_norms(gradients_by_epoch[epoch])
            if param_name in grad_norms:
                norms.append(grad_norms[param_name])
            else:
                norms.append(0.0)
        
        plt.figure(figsize=(10, 6))
        plt.plot(epochs, norms, 'b-', linewidth=2)
        plt.xlabel('Epoch')
        plt.ylabel('Gradient L2 Norm')
        plt.title(f'Gradient Evolution: {param_name}')
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.savefig(output_dir / f'gradient_evolution_{param_name.replace(".", "_")}.png')
        plt.close()


def plot_parameter_evolution(parameters_by_epoch: Dict[int, Dict[str, torch.Tensor]], 
                           output_dir: Path,
                           param_names: Optional[List[str]] = None):
    """Plot parameter evolution over training.
    
    Args:
        parameters_by_epoch: Parameters organized by epoch
        output_dir: Directory to save plots
        param_names: Specific parameters to plot (if None, plot all)
    """
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Get all parameter names
    if param_names is None:
        param_names = list(next(iter(parameters_by_epoch.values())).keys())
    
    epochs = sorted(parameters_by_epoch.keys())
    
    for param_name in param_names:
        if param_name not in next(iter(parameters_by_epoch.values())):
            continue
            
        norms = []
        for epoch in epochs:
            param_norms = compute_parameter_norms(parameters_by_epoch[epoch])
            if param_name in param_norms:
                norms.append(param_norms[param_name])
            else:
                norms.append(0.0)
        
        plt.figure(figsize=(10, 6))
        plt.plot(epochs, norms, 'r-', linewidth=2)
        plt.xlabel('Epoch')
        plt.ylabel('Parameter L2 Norm')
        plt.title(f'Parameter Evolution: {param_name}')
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.savefig(output_dir / f'parameter_evolution_{param_name.replace(".", "_")}.png')
        plt.close()


def analyze_training_dynamics(experiment_dir: Path, 
                            model_name: str = "model_1",
                            output_dir: Optional[Path] = None):
    """Analyze training dynamics for a specific model.
    
    Args:
        experiment_dir: Directory containing the experiment results
        model_name: Name of the model to analyze (e.g., "model_1")
        output_dir: Directory to save analysis results
    """
    if output_dir is None:
        output_dir = experiment_dir / "analysis" / model_name
    
    model_dir = experiment_dir / model_name
    
    # Load gradients and parameters
    gradients_by_epoch = load_gradients(model_dir)
    parameters_by_epoch = load_parameters(model_dir)
    
    print(f"Loaded gradients for {len(gradients_by_epoch)} epochs")
    print(f"Loaded parameters for {len(parameters_by_epoch)} epochs")
    
    # Create analysis directory
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Plot evolutions
    plot_gradient_evolution(gradients_by_epoch, output_dir)
    plot_parameter_evolution(parameters_by_epoch, output_dir)
    
    # Save summary statistics
    summary = {
        'num_epochs': len(gradients_by_epoch),
        'parameter_names': list(next(iter(parameters_by_epoch.values())).keys()),
        'gradient_evolution': {
            epoch: compute_gradient_norms(grads) 
            for epoch, grads in gradients_by_epoch.items()
        },
        'parameter_evolution': {
            epoch: compute_parameter_norms(params) 
            for epoch, params in parameters_by_epoch.items()
        }
    }
    
    torch.save(summary, output_dir / "analysis_summary.pt")
    print(f"Analysis saved to {output_dir}")


if __name__ == "__main__":
    import sys
    if len(sys.argv) != 2:
        print("Usage: python analysis.py <experiment_dir>")
        sys.exit(1)
    
    experiment_dir = Path(sys.argv[1])
    analyze_training_dynamics(experiment_dir)
