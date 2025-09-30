"""
Unified evaluation framework.
"""

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from typing import Dict, Any, Optional, Callable, List
import numpy as np


class Evaluator:
    """Unified evaluator for all model types."""
    
    def __init__(self, 
                 model: nn.Module,
                 device: str = 'cuda',
                 metrics: Optional[Dict[str, Callable]] = None):
        """Initialize the evaluator.
        
        Args:
            model: The model to evaluate
            device: Device to evaluate on
            metrics: Dictionary of metric functions
        """
        self.model = model
        self.device = torch.device(device)
        self.metrics = metrics or {}
        
        # Move model to device
        self.model.to(self.device)
    
    def evaluate(self, 
                 data_loader: DataLoader,
                 loss_fn: Optional[Callable] = None,
                 prefix: str = '') -> Dict[str, float]:
        """Evaluate the model on a data loader.
        
        Args:
            data_loader: Data loader to evaluate on
            loss_fn: Loss function (defaults to CrossEntropyLoss)
            prefix: Prefix for metric names
            
        Returns:
            Dictionary of evaluation metrics
        """
        if loss_fn is None:
            loss_fn = nn.CrossEntropyLoss()
        
        self.model.eval()
        total_loss = 0.0
        correct = 0
        total = 0
        metrics_results = {}
        
        with torch.no_grad():
            for data, target in data_loader:
                data, target = data.to(self.device), target.to(self.device)
                output = self.model(data)
                loss = loss_fn(output, target)
                
                total_loss += loss.item()
                pred = output.argmax(dim=1, keepdim=True)
                correct += pred.eq(target.view_as(pred)).sum().item()
                total += target.size(0)
                
                # Compute additional metrics if provided
                for metric_name, metric_fn in self.metrics.items():
                    if metric_name not in metrics_results:
                        metrics_results[metric_name] = []
                    metrics_results[metric_name].append(metric_fn(output, target).item())
        
        # Average metrics
        for metric_name in metrics_results:
            metrics_results[metric_name] = np.mean(metrics_results[metric_name])
        
        accuracy = correct / total
        avg_loss = total_loss / len(data_loader)
        
        results = {
            f'{prefix}loss': avg_loss,
            f'{prefix}accuracy': accuracy,
            **{f'{prefix}{k}': v for k, v in metrics_results.items()}
        }
        
        return results
    
    def evaluate_interpolation(self, 
                              model2: nn.Module,
                              data_loader: DataLoader,
                              steps: int = 10,
                              loss_fn: Optional[Callable] = None,
                              use_wandb: bool = False) -> List[Dict[str, float]]:
        """Evaluate linear interpolation between two models.
        
        Args:
            model2: Second model for interpolation
            data_loader: Data loader to evaluate on
            steps: Number of interpolation steps
            loss_fn: Loss function
            use_wandb: Whether to log to wandb
            
        Returns:
            List of evaluation results for each interpolation step
        """
        if loss_fn is None:
            loss_fn = nn.CrossEntropyLoss()
        
        # Store original model state
        original_state = self.model.state_dict()
        model2_state = model2.state_dict()
        
        results = []
        
        for i in range(steps + 1):
            alpha = i / steps
            
            # Interpolate parameters
            interpolated_state = {}
            for key in original_state:
                interpolated_state[key] = (1 - alpha) * original_state[key] + alpha * model2_state[key]
            
            # Load interpolated state
            self.model.load_state_dict(interpolated_state)
            
            # Evaluate
            result = self.evaluate(data_loader, loss_fn, f'step_{i}_')
            results.append(result)
            
            # Log to wandb if enabled
            if use_wandb:
                try:
                    import wandb
                    if wandb.run is not None:
                        wandb.log({
                            'evaluator_interpolation_alpha': alpha,
                            'evaluator_interpolation_loss': result.get('loss', 0.0),
                            'evaluator_interpolation_accuracy': result.get('accuracy', 0.0),
                        })
                except ImportError:
                    print("Warning: wandb not available for logging")
        
        # Restore original model state
        self.model.load_state_dict(original_state)
        
        return results
