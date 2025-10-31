"""
Graph Neural Network models with asymmetric architectures.
Refactored from gnn/main_arxiv.py
"""

import math
import itertools
import torch
import torch.nn as nn
import torch.nn.functional as F
from ..core.registry import register
from .mlp import NoiseLinear, SparseLinear

# Global seed for mask generation (matching original)
seed = 1

# Optional imports for GNN functionality
try:
    from torch_geometric.nn import SimpleConv
except ImportError:
    print("Warning: torch_geometric not available. GNN functionality will be limited.")
    SimpleConv = None

def make_C_lst(hidden_channels, num_layers):
    import math
    g = torch.Generator()
    g.manual_seed(0)
    return [0.01 * torch.randn(hidden_channels, hidden_channels, generator=g) / 
            math.sqrt(hidden_channels) for _ in range(num_layers)]

class AsymNonlin(nn.Module):
    """Asymmetric nonlinearity."""
    
    def __init__(self, C):
        super().__init__()
        self.register_buffer("C", C)
        self.nonlin = nn.GELU()
    
    def forward(self, x):
        x = self.nonlin(x)
        x = torch.matmul(x, self.C)
        x = self.nonlin(x)
        return x


class AsymSwiGLU(nn.Module):
    """Asymmetric SwiGLU activation."""
    
    def __init__(self, C):
        super().__init__()
        self.register_buffer("C", C)
    
    def forward(self, x):
        gate = F.sigmoid(F.linear(x, self.C))
        return gate * x


class MyConv(nn.Module):
    """Custom convolution layer for GNNs."""
    
    def __init__(self, in_dim, out_dim, nonlin='gelu', C=None, lin_builder=None, mask_rng: torch.Generator|None=None):
        super().__init__()
        
        if nonlin == 'gelu':
            nonlin_module = nn.GELU()
        elif nonlin == 'asym_gelu':
            nonlin_module = AsymNonlin(C)
        elif nonlin == 'asym_swiglu':
            nonlin_module = AsymSwiGLU(C)
        else:
            raise ValueError(f"Invalid nonlinearity: {nonlin}")
        
        if SimpleConv is not None:
            self.conv = SimpleConv(aggr='mean', combine_root='sum')
        else:
            raise ImportError("torch_geometric is required for GNN functionality")
        self.mlp = nn.Sequential(
            lin_builder(in_dim, out_dim, mask_rng), 
            nn.BatchNorm1d(out_dim), 
            nonlin_module
        )
    
    def reset_parameters(self):
        """Reset parameters."""
        if hasattr(self.mlp[0], 'reset_parameters'):
            self.mlp[0].reset_parameters()
        if hasattr(self.mlp[1], 'reset_parameters'):
            self.mlp[1].reset_parameters()

    def forward(self, x, adj_t):
        x = self.conv(x, adj_t)
        x = self.mlp(x)
        return x


@register('model', 'gnn_standard')
class MyGNN(torch.nn.Module):
    """Standard GNN."""
    
    def __init__(self, in_channels, hidden_channels, out_channels, num_layers,
                 dropout, nonlin='gelu', lin_builder=None, C_lst=None, mask_seed=None):
        super().__init__()

        mask_rng = torch.Generator()
        if mask_seed is not None:
            mask_rng.manual_seed(mask_seed)
        
        self.convs = torch.nn.ModuleList()
        self.convs.append(MyConv(in_channels, hidden_channels, nonlin=nonlin, 
                                C=C_lst[0] if C_lst else None, lin_builder=lin_builder, mask_rng=mask_rng))
        
        for i in range(num_layers - 1):
            self.convs.append(
                MyConv(hidden_channels, hidden_channels, nonlin=nonlin, 
                      C=C_lst[i+1] if C_lst else None, lin_builder=lin_builder, mask_rng=mask_rng)
            )
        
        self.lin = lin_builder(hidden_channels, out_channels, mask_rng)
        self.dropout = dropout

    def reset_parameters(self):
        """Reset parameters."""
        for conv in self.convs:
            conv.reset_parameters()

    def forward(self, x, adj_t):
        for i, conv in enumerate(self.convs[:-1]):
            x = conv(x, adj_t)
            x = F.dropout(x, p=self.dropout, training=self.training)
        x = self.convs[-1](x, adj_t)
        x = self.lin(x)
        return x.log_softmax(dim=-1)
    
    def count_unused_params(self):
        """Count unused parameters due to masking."""
        count = 0
        for module in self.modules():
            if hasattr(module, 'mask'):
                count += (1 - module.mask).sum().int().item()
        return count


@register('model', 'gnn_asym_gelu')
class AsymGeluGNN(MyGNN):
    """GNN with asymmetric GELU activations."""
    
    def __init__(self, in_channels, hidden_channels, out_channels, num_layers,
                 dropout, C_lst, lin_builder=None, mask_seed=None):
        super().__init__(in_channels, hidden_channels, out_channels, num_layers,
                        dropout, nonlin='asym_gelu', lin_builder=lin_builder, C_lst=C_lst, mask_seed=mask_seed)


@register('model', 'gnn_asym_swiglu')
class AsymSwiGLUGNN(MyGNN):
    """GNN with asymmetric SwiGLU activations."""
    
    def __init__(self, in_channels, hidden_channels, out_channels, num_layers,
                 dropout, C_lst, lin_builder=None, mask_seed=None):
        super().__init__(in_channels, hidden_channels, out_channels, num_layers,
                        dropout, nonlin='asym_swiglu', lin_builder=lin_builder, C_lst=C_lst, mask_seed=mask_seed)


@register('model', 'gnn_asym_w')
class AsymWGNN(MyGNN):
    """GNN with asymmetric weights (sparse linear layers)."""
    
    def __init__(self, in_channels, hidden_channels, out_channels, num_layers,
                 dropout, C_lst, lin_builder=None, mask_seed=None):
        super().__init__(in_channels, hidden_channels, out_channels, num_layers,
                        dropout, nonlin='asym_swiglu', lin_builder=lin_builder, C_lst=C_lst, mask_seed=mask_seed)


@register('model', 'gnn_noise_asym')
class NoiseGNN(MyGNN):
    """Noise-Asymmetric GNN with noise injection for symmetry breaking."""
    
    def __init__(self, in_channels, hidden_channels, out_channels, num_layers,
                 dropout, lin_builder=None, mask_seed=None):
        super().__init__(in_channels, hidden_channels, out_channels, num_layers,
                        dropout, nonlin='gelu', lin_builder=lin_builder, C_lst=None, mask_seed=mask_seed)


# Convenience function for creating GNNs
def create_gnn(symmetry, in_channels, hidden_channels, out_channels, num_layers,
               dropout, mask_params=None, model_type=None, mask_seed=None):
    """Create GNN based on model type.
    
    Args:
        symmetry: 0=Standard, 1=W-Asym, 2=Sigma-Asym, 3=Noise-Asym
        in_channels: Input feature dimension
        hidden_channels: Hidden feature dimension
        out_channels: Output feature dimension
        num_layers: Number of layers
        dropout: Dropout rate
        model_type: Type of GNN ('gnn', 'asym_gelu_gnn', 'asym_swiglu_gnn', 'asym_w_gnn')
        mask_params: Mask parameters
    """
    C_lst = None
    if symmetry in (1, 2):
        C_lst = make_C_lst(hidden_channels, num_layers)
    if symmetry == 0:
        lin_builder = nn.Linear
        return MyGNN(in_channels, hidden_channels, out_channels, num_layers, 
                    dropout, lin_builder=lin_builder)
    elif symmetry == 2:
        if model_type == 'asym_gelu_gnn':
            lin_builder = lambda in_dim, out_dim, mask_rng: nn.Linear(in_dim, out_dim)
            return AsymGeluGNN(in_channels, hidden_channels, out_channels, num_layers,
                            dropout, C_lst, lin_builder=lin_builder, mask_seed=mask_seed)
        else: # model_type == 'asym_swiglu_gnn'
            lin_builder = lambda in_dim, out_dim, mask_rng: nn.Linear(in_dim, out_dim)
            return AsymSwiGLUGNN(in_channels, hidden_channels, out_channels, num_layers,
                                dropout, C_lst, lin_builder=lin_builder, mask_seed=mask_seed)
    # for now: mask_params only uses default values
    elif symmetry == 1:
        lin_builder = lambda in_dim, out_dim, mask_rng: SparseLinear(in_dim, out_dim, mask_rng=mask_rng, **mask_params.get('default', {}))
        return AsymWGNN(in_channels, hidden_channels, out_channels, num_layers,
                       dropout, C_lst, lin_builder=lin_builder, mask_seed=mask_seed)
    elif symmetry == 3:
        lin_builder = lambda in_dim, out_dim, mask_rng: NoiseLinear(in_dim, out_dim, mask_rng=mask_rng, **mask_params.get('default', {}))
        return NoiseGNN(in_channels, hidden_channels, out_channels, num_layers,
                       dropout, lin_builder=lin_builder, mask_seed=mask_seed)
    else:
        raise ValueError(f"Invalid GNN symmetry type: {symmetry}")
