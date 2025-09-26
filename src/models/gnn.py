"""
Graph Neural Network models with asymmetric architectures.
Refactored from gnn/main_arxiv.py
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from ..core.registry import register

# Optional imports for GNN functionality
try:
    from torch_geometric.nn import GCNConv, SimpleConv
except ImportError:
    print("Warning: torch_geometric not available. GNN functionality will be limited.")
    GCNConv = None
    SimpleConv = None


class SparseLinear(nn.Module):
    """Sparse linear layer for GNNs."""
    
    def __init__(self, in_dim, out_dim, mask_constant=0.5, num_fixed=6, 
                 do_normal_mask=True, mask_type='random_subsets'):
        super().__init__()
        
        # Create mask
        mask = self._make_mask(in_dim, out_dim, mask_type, num_fixed)
        self.register_buffer('mask', mask, persistent=True)
        
        # Initialize weight
        self.weight = nn.Parameter(torch.empty(out_dim, in_dim))
        self.bias = nn.Parameter(torch.empty(out_dim))
        
        # Register gradient hook
        self.weight.register_hook(lambda grad: self.mask * grad)
        
        # Normal mask for initialization
        if do_normal_mask:
            normal_mask = self._normal_mask(out_dim, in_dim)
            self.register_buffer('normal_mask', normal_mask, persistent=True)
        else:
            self.register_buffer('normal_mask', torch.ones_like(mask), persistent=True)
        
        self.reset_parameters()
    
    def _make_mask(self, in_dim, out_dim, mask_type, num_fixed):
        """Create the sparse mask."""
        mask = torch.zeros(out_dim, in_dim)
        
        if mask_type == 'random_subsets':
            g = torch.Generator()
            g.manual_seed(42)  # Fixed seed for reproducibility
            
            for i in range(out_dim):
                if num_fixed < in_dim:
                    indices = torch.randperm(in_dim, generator=g)[:num_fixed]
                    mask[i, indices] = 1
                else:
                    mask[i] = 1
        
        return mask
    
    def _normal_mask(self, out_dim, in_dim):
        """Create normal mask for initialization."""
        g = torch.Generator()
        g.manual_seed(42)
        return torch.randn(out_dim, in_dim, generator=g)
    
    def reset_parameters(self):
        """Initialize parameters."""
        with torch.no_grad():
            self.weight.data = self.normal_mask * self.mask
            nn.init.zeros_(self.bias)
    
    def forward(self, x):
        return F.linear(x, self.weight * self.mask, self.bias)


class AsymNonlin(nn.Module):
    """Asymmetric nonlinearity."""
    
    def __init__(self, C):
        super().__init__()
        self.register_buffer("C", C)
    
    def forward(self, x):
        gate = F.sigmoid(F.linear(x, self.C))
        return gate * x


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
    
    def __init__(self, in_dim, out_dim, nonlin='gelu', C=None, lin_builder=None):
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
            lin_builder(in_dim, out_dim), 
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
                 dropout, nonlin='gelu', lin_builder=None, C_lst=None):
        super().__init__()
        
        self.convs = torch.nn.ModuleList()
        self.convs.append(MyConv(in_channels, hidden_channels, nonlin=nonlin, 
                                C=C_lst[0] if C_lst else None, lin_builder=lin_builder))
        
        for i in range(num_layers - 1):
            self.convs.append(
                MyConv(hidden_channels, hidden_channels, nonlin=nonlin, 
                      C=C_lst[i+1] if C_lst else None, lin_builder=lin_builder)
            )
        
        self.lin = lin_builder(hidden_channels, out_channels)
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
                 dropout, C_lst, lin_builder=None):
        super().__init__(in_channels, hidden_channels, out_channels, num_layers,
                        dropout, nonlin='asym_gelu', lin_builder=lin_builder, C_lst=C_lst)


@register('model', 'gnn_asym_swiglu')
class AsymSwiGLUGNN(MyGNN):
    """GNN with asymmetric SwiGLU activations."""
    
    def __init__(self, in_channels, hidden_channels, out_channels, num_layers,
                 dropout, C_lst, lin_builder=None):
        super().__init__(in_channels, hidden_channels, out_channels, num_layers,
                        dropout, nonlin='asym_swiglu', lin_builder=lin_builder, C_lst=C_lst)


@register('model', 'gnn_asym_w')
class AsymWGNN(MyGNN):
    """GNN with asymmetric weights (sparse linear layers)."""
    
    def __init__(self, in_channels, hidden_channels, out_channels, num_layers,
                 dropout, C_lst, lin_builder=None):
        super().__init__(in_channels, hidden_channels, out_channels, num_layers,
                        dropout, nonlin='asym_swiglu', lin_builder=lin_builder, C_lst=C_lst)


# Convenience function for creating GNNs
def create_gnn(model_type, in_channels, hidden_channels, out_channels, num_layers,
               dropout, C_lst=None):
    """Create GNN based on model type.
    
    Args:
        model_type: Type of GNN ('gnn', 'asym_gelu_gnn', 'asym_swiglu_gnn', 'asym_w_gnn')
        in_channels: Input feature dimension
        hidden_channels: Hidden feature dimension
        out_channels: Output feature dimension
        num_layers: Number of layers
        dropout: Dropout rate
        C_lst: List of C matrices for asymmetric activations
    """
    if model_type == 'gnn':
        lin_builder = nn.Linear
        return MyGNN(in_channels, hidden_channels, out_channels, num_layers, 
                    dropout, lin_builder=lin_builder)
    elif model_type == 'asym_gelu_gnn':
        lin_builder = nn.Linear
        return AsymGeluGNN(in_channels, hidden_channels, out_channels, num_layers,
                          dropout, C_lst, lin_builder=lin_builder)
    elif model_type == 'asym_swiglu_gnn':
        lin_builder = nn.Linear
        return AsymSwiGLUGNN(in_channels, hidden_channels, out_channels, num_layers,
                            dropout, C_lst, lin_builder=lin_builder)
    elif model_type == 'asym_w_gnn':
        lin_builder = lambda in_dim, out_dim: SparseLinear(in_dim, out_dim, 
                                                          mask_constant=0.5, num_fixed=6, 
                                                          do_normal_mask=True, mask_type='random_subsets')
        return AsymWGNN(in_channels, hidden_channels, out_channels, num_layers,
                       dropout, C_lst, lin_builder=lin_builder)
    else:
        raise ValueError(f"Invalid GNN model type: {model_type}")
