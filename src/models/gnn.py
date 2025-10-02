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

# Global seed for mask generation (matching original)
seed = 1

# Optional imports for GNN functionality
try:
    from torch_geometric.nn import GCNConv, SimpleConv
except ImportError:
    print("Warning: torch_geometric not available. GNN functionality will be limited.")
    GCNConv = None
    SimpleConv = None


class SparseLinear(nn.Module):
    """Sparse linear layer - EXACT COPY from gnn/asym_nets.py"""
    
    def __init__(self, in_dim, out_dim, bias=True, mask_type='densest', mask_constant=1, mask_num=0, num_fixed=6, do_normal_mask=True, mask_path=None):
        super().__init__()
        assert out_dim < 2**in_dim, 'out dim cannot be much higher than in dim'
        
        if mask_path is not None:
            mask, _ = torch.load(mask_path)
        else:
            mask = make_mask(in_dim, out_dim, mask_type=mask_type, num_fixed = num_fixed, mask_num = mask_num)

        self.register_buffer('mask', mask, persistent=True)
        self.weight = nn.Parameter(torch.empty((out_dim, in_dim)))

        if mask_path is not None:
            _, n_mask = torch.load(mask_path)
            self.register_buffer('normal_mask', n_mask, persistent=True)
        else:
            if do_normal_mask:
                self.register_buffer('normal_mask', normal_mask(out_dim, in_dim, mask_num), persistent=True)
            else:
                self.register_buffer('normal_mask', torch.ones(size = (out_dim, in_dim)), persistent=True) #torch.ones -> does nothing

        hook = self.weight.register_hook(lambda grad: self.mask*grad) # zeros out gradients for masked parts

        if bias:
            self.bias = nn.Parameter(torch.empty(out_dim))
        else:
            self.register_parameter('bias', None)

        self.mask_constant = mask_constant
        self.mask_num = mask_num
        self.num_fixed = num_fixed
        self.reset_parameters()

    def forward(self, x):
        self.weight.data = (self.weight.data* self.mask + (1-self.mask)*self.mask_constant*self.normal_mask)
        return F.linear(x, (self.weight), self.bias)

    def reset_parameters(self):
        nn.init.kaiming_uniform_(self.weight, a=math.sqrt(5))
        self.weight.data = (self.weight.data* self.mask + (1-self.mask)*self.mask_constant*self.normal_mask) #set entries where mask is zero to the normal mask at that point

        if self.bias is not None:
            fan_in, _ = nn.init._calculate_fan_in_and_fan_out(self.weight)
            bound = 1 / math.sqrt(fan_in) if fan_in > 0 else 0
            nn.init.uniform_(self.bias, -bound, bound)

    def count_unused_params(self):
        return (1-self.mask.int()).sum().item()


# EXACT COPY of mask generation functions from gnn/asym_nets.py
def get_subset(num_cols, row_idx, num_sample, mask_num):
    g = torch.Generator()
    g.manual_seed(row_idx + abs(hash(str(mask_num) + str(seed))))
    indices = torch.arange(num_cols)
    return (indices[torch.randperm(num_cols, generator = g)[:num_sample]])

def normal_mask(out_dim, in_dim, mask_num):
    g = torch.Generator()
    g.manual_seed(abs(hash(str(mask_num)+ str(seed))))
    return torch.randn(size=(out_dim,in_dim), generator = g)

def make_mask(in_dim, out_dim, mask_num = 0, num_fixed = 6, mask_type='densest'):
    # out_dim x in_dim matrix
    # where each row is unique
    assert out_dim < 2**(in_dim)
    assert in_dim > 0 and out_dim > 0

    if mask_type == 'densest':
        mask = torch.ones(out_dim, in_dim)
        mask[0, :] = 1 # first row is dense
        row_idx = 1
        if out_dim == 1:
            return mask

        for nz in range(1, in_dim):
            for zeros_in_row in itertools.combinations(range(in_dim), nz):
                mask[row_idx, zeros_in_row] = 0
                row_idx += 1
                if row_idx >= out_dim:
                    return mask
    elif mask_type == 'bound_zeros':
        # other type of mask based on lower bounding sparsity to break symmetries more
        mask = torch.ones(out_dim, in_dim)
        least_zeros = 2
        row_idx = 0
        for nz in range(least_zeros, in_dim):
            for zeros_in_row in itertools.combinations(range(in_dim), nz):
                mask[row_idx, zeros_in_row] = 0
                row_idx += 1
                if row_idx >= out_dim:
                    return mask

        raise ValueError('Error in making mask, possibly because out_dim is too large for these settings')

    elif mask_type == 'random_subsets':
            # other type of mask based on lower bounding sparsity to break symmetries more
            mask = torch.ones(out_dim, in_dim)
            row_idx = 0
            least_zeros = num_fixed
            for nz in range(least_zeros, in_dim):
                while True:

                    zeros_in_row = get_subset(in_dim, row_idx, least_zeros, mask_num)
                    mask[row_idx, zeros_in_row] = 0
                    row_idx += 1
                    if row_idx >= out_dim:
                        return mask

            raise ValueError('Error in making mask, possibly because out_dim is too large for these settings')
    else:
        raise ValueError('Invalid mask type')
    
    
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
               dropout):
    """Create GNN based on model type.
    
    Args:
        model_type: Type of GNN ('gnn', 'asym_gelu_gnn', 'asym_swiglu_gnn', 'asym_w_gnn')
        in_channels: Input feature dimension
        hidden_channels: Hidden feature dimension
        out_channels: Output feature dimension
        num_layers: Number of layers
        dropout: Dropout rate
    """
    # TODO: take care of this, what is this configuring?
    C_lst = None
    if model_type in ['asym_gelu_gnn', 'asym_swiglu_gnn', 'asym_w_gnn']:
        C_lst = make_C_lst(hidden_channels, num_layers)
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
