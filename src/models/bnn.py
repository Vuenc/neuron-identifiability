"""
Bayesian Neural Network models with sparse architectures.
Refactored from bnn/train_sparse_*.py
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from ..core.registry import register


class SparseLinear(nn.Module):
    """Sparse linear layer for BNNs."""
    
    def __init__(self, in_dim, out_dim, mask_num=0, mask_constant=0, 
                 mask_type='random_subsets', do_normal_mask=True, num_fixed=6):
        super().__init__()
        
        # Create mask
        mask = self._make_mask(in_dim, out_dim, mask_type, num_fixed, mask_num)
        self.register_buffer('mask', mask, persistent=True)
        
        # Initialize weight
        self.weight = nn.Parameter(torch.empty(out_dim, in_dim))
        self.bias = nn.Parameter(torch.empty(out_dim))
        
        # Register gradient hook
        self.weight.register_hook(lambda grad: self.mask * grad)
        
        # Normal mask for initialization
        if do_normal_mask:
            normal_mask = self._normal_mask(out_dim, in_dim, mask_num)
            self.register_buffer('normal_mask', normal_mask, persistent=True)
        else:
            self.register_buffer('normal_mask', torch.ones_like(mask), persistent=True)
        
        self.mask_constant = mask_constant
        self.mask_num = mask_num
        self.reset_parameters()
    
    def _make_mask(self, in_dim, out_dim, mask_type, num_fixed, mask_num):
        """Create the sparse mask."""
        mask = torch.zeros(out_dim, in_dim)
        
        if mask_type == 'random_subsets':
            g = torch.Generator()
            g.manual_seed(abs(hash(str(mask_num))))
            
            for i in range(out_dim):
                if num_fixed < in_dim:
                    indices = torch.randperm(in_dim, generator=g)[:num_fixed]
                    mask[i, indices] = 1
                else:
                    mask[i] = 1
        
        return mask
    
    def _normal_mask(self, out_dim, in_dim, mask_num):
        """Create normal mask for initialization."""
        g = torch.Generator()
        g.manual_seed(abs(hash(str(mask_num) + 'normal')))
        return torch.randn(out_dim, in_dim, generator=g)
    
    def reset_parameters(self):
        """Initialize parameters."""
        with torch.no_grad():
            self.weight.data = self.normal_mask * self.mask
            nn.init.zeros_(self.bias)
    
    def forward(self, x):
        return F.linear(x, self.weight * self.mask, self.bias)


@register('model', 'bnn_mlp_standard')
class FullyConnected(nn.Module):
    """Standard fully connected network for BNNs."""
    
    def __init__(self, in_dim, out_dim, num_layers, size):
        super().__init__()
        
        layers = [nn.Flatten(), nn.Linear(in_dim, size), nn.LayerNorm(size), nn.ReLU(inplace=True)]
        
        for _ in range(num_layers - 1):
            layers.extend([nn.Linear(size, size), nn.LayerNorm(size), nn.ReLU(inplace=True)])
        
        layers.append(nn.Linear(size, out_dim))
        self.network = nn.Sequential(*layers)
    
    def forward(self, x):
        return self.network(x)


@register('model', 'bnn_mlp_sparse')
class SparseFullyConnected(nn.Module):
    """Sparse fully connected network for BNNs."""
    
    def __init__(self, in_dim, out_dim, num_layers, size, n=8, c=0.5):
        super().__init__()
        
        layers = [nn.Flatten(), 
                 SparseLinear(in_dim, size, mask_num=0, mask_constant=c, num_fixed=10*n), 
                 nn.LayerNorm(size), nn.ReLU(inplace=True)]
        
        for i in range(num_layers - 1):
            layers.extend([SparseLinear(size, size, mask_num=i+1, mask_constant=c, num_fixed=10*n),
                          nn.LayerNorm(size), nn.ReLU(inplace=True)])
        
        layers.append(SparseLinear(size, out_dim, mask_num=num_layers, mask_constant=c, num_fixed=10*n))
        self.network = nn.Sequential(*layers)
    
    def forward(self, x):
        return self.network(x)
    
    def count_unused_params(self):
        """Count unused parameters due to masking."""
        count = 0
        for module in self.modules():
            if hasattr(module, 'mask'):
                count += (1 - module.mask).sum().int().item()
        return count


# ResNet models for BNNs (simplified versions)
@register('model', 'bnn_resnet_standard')
class ResNetStandard(nn.Module):
    """Standard ResNet for BNNs."""
    
    def __init__(self, depth=20, w=1, num_classes=10):
        super().__init__()
        # This would be implemented based on the specific ResNet architecture
        # For now, using a placeholder
        self.network = nn.Sequential(
            nn.Conv2d(3, 16*w, 3, padding=1),
            nn.BatchNorm2d(16*w),
            nn.ReLU(),
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
            nn.Linear(16*w, num_classes)
        )
    
    def forward(self, x):
        return self.network(x)


@register('model', 'bnn_resnet_sparse')
class SparseResNet(nn.Module):
    """Sparse ResNet for BNNs."""
    
    def __init__(self, mask_params, depth=20, w=1, num_classes=10):
        super().__init__()
        # This would be implemented based on the specific sparse ResNet architecture
        # For now, using a placeholder
        self.network = nn.Sequential(
            nn.Conv2d(3, 16*w, 3, padding=1),
            nn.BatchNorm2d(16*w),
            nn.ReLU(),
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
            nn.Linear(16*w, num_classes)
        )
    
    def forward(self, x):
        return self.network(x)
    
    def count_unused_params(self):
        """Count unused parameters due to masking."""
        count = 0
        for module in self.modules():
            if hasattr(module, 'mask'):
                count += (1 - module.mask).sum().int().item()
        return count


# Convenience functions
def create_bnn_model(model_type, in_dim, out_dim, num_layers=32, size=256, 
                    n=8, c=0.5, mask_params=None, depth=20, w=1):
    """Create BNN model based on type.
    
    Args:
        model_type: Type of model ('fc', 'sfc', 'resnet', 'sparse_resnet')
        in_dim: Input dimension
        out_dim: Output dimension
        num_layers: Number of layers (for MLPs)
        size: Hidden size (for MLPs)
        n: Sparsity parameter
        c: Mask constant
        mask_params: Mask parameters for sparse models
        depth: ResNet depth
        w: ResNet width multiplier
    """
    if model_type == 'fc':
        return FullyConnected(in_dim, out_dim, num_layers, size)
    elif model_type == 'sfc':
        return SparseFullyConnected(in_dim, out_dim, num_layers, size, n, c)
    elif model_type == 'resnet':
        return ResNetStandard(depth=depth, w=w, num_classes=out_dim)
    elif model_type == 'sparse_resnet':
        if mask_params is None:
            raise ValueError("mask_params required for sparse ResNet")
        return SparseResNet(mask_params, depth=depth, w=w, num_classes=out_dim)
    else:
        raise ValueError(f"Invalid BNN model type: {model_type}")
