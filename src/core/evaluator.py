import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from typing import Dict, Optional, Callable, List
import numpy as np


class Evaluator:
    
    def __init__(self, 
                 model: nn.Module,
                 device: str = 'cuda',
                 metrics: Optional[Dict[str, Callable]] = None):
        self.model = model
        self.device = torch.device(device)
        self.metrics = metrics or {}
        self.model.to(self.device)
    
    def evaluate(self, 
                 data_loader: DataLoader,
                 loss_fn: Optional[Callable] = None,
                 prefix: str = '') -> Dict[str, float]:
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
                
                for metric_name, metric_fn in self.metrics.items():
                    if metric_name not in metrics_results:
                        metrics_results[metric_name] = []
                    metrics_results[metric_name].append(metric_fn(output, target).item())
        
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
        
        if loss_fn is None:
            loss_fn = nn.CrossEntropyLoss()
        
        original_state = self.model.state_dict()
        model2_state = model2.state_dict()
        
        results = []
        
        for i in range(steps + 1):
            alpha = i / steps
            
            interpolated_state = {}
            for key in original_state:
                interpolated_state[key] = (1 - alpha) * original_state[key] + alpha * model2_state[key]
            
            self.model.load_state_dict(interpolated_state)
            
            result = self.evaluate(data_loader, loss_fn, f'step_{i}_')
            results.append(result)
            
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
        
        self.model.load_state_dict(original_state)
        
        return results
