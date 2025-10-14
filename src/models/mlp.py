"""
MLP models with asymmetric architectures.
Refactored from lmc/models/models_mlp.py
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from ..core.registry import register
from .normalization import total_output_variances_init, setup_normalization
from .activation import setup_activation


class AsymSwiGLU(nn.Module):
    def __init__(self, dim, scale=1.0, mask_num=0, fixed_C=None):
        super().__init__()
        if fixed_C is not None:
            C = fixed_C
        else:
            g = torch.Generator()
            g.manual_seed(abs(hash(str(mask_num) + str(0))))
            C = torch.randn(dim, dim, generator=g)
        self.register_buffer("C", C)
        self.mask_num = mask_num
    
    def forward(self, x):
        gate = F.sigmoid(F.linear(x, self.C))
        return gate * x


class SparseLinear(nn.Module):
    def __init__(self, in_dim, out_dim, mask_constant=0, mask_type='random_subsets', 
                 do_normal_mask=True, num_fixed=6, mask_num=0):
        super().__init__()

        mask = self._make_mask(in_dim, out_dim, mask_type, num_fixed, mask_num)
        self.register_buffer('mask', mask, persistent=True)
        
        self.weight = nn.Parameter(torch.empty(out_dim, in_dim))
        self.bias = nn.Parameter(torch.empty(out_dim))
        
        self.weight.register_hook(lambda grad: self.mask * grad)
        
        if do_normal_mask:
            normal_mask = self._normal_mask(out_dim, in_dim, mask_num)
            self.register_buffer('normal_mask', normal_mask, persistent=True)
        else:
            self.register_buffer('normal_mask', torch.ones_like(mask), persistent=True)
        
        self.mask_num = mask_num
        self.mask_constant = mask_constant
        self.reset_parameters()
    
    def _make_mask(self, in_dim, out_dim, mask_type, num_fixed, mask_num):
        if mask_type == 'random_subsets':
            # Match the original implementation
            mask = torch.ones(out_dim, in_dim)
            row_idx = 0
            least_zeros = num_fixed
            for nz in range(least_zeros, in_dim):
                while True:
                    zeros_in_row = self._get_subset(in_dim, row_idx, least_zeros, mask_num)
                    mask[row_idx, zeros_in_row] = 0
                    row_idx += 1
                    if row_idx >= out_dim:
                        return mask
            raise ValueError('Error in making mask, possibly because out_dim is too large for these settings')
        else:
            raise ValueError('Invalid mask type')
    
    def _get_subset(self, num_cols, row_idx, num_sample, mask_num):
        g = torch.Generator()
        g.manual_seed(row_idx + abs(hash(str(mask_num))))
        indices = torch.arange(num_cols)
        return (indices[torch.randperm(num_cols, generator=g)[:num_sample]])
    
    def _normal_mask(self, out_dim, in_dim, mask_num):
        g = torch.Generator()
        g.manual_seed(abs(hash(str(mask_num))))
        return torch.randn(out_dim, in_dim, generator=g)
    
    def reset_parameters(self):
        with torch.no_grad():
            # # Initialize weights randomly, then apply mask and normal_mask
            # nn.init.normal_(self.weight, mean=0.0, std=1.0 / math.sqrt(self.weight.size(1)))
            # self.weight.mul_(self.mask).add_((1 - self.mask) * self.mask_constant * self.normal_mask)
            # if self.bias is not None:
            #     nn.init.zeros_(self.bias)
            d_in = self.weight.size(1)
            nn.init.normal_(self.weight, mean=0.0, std=1.0 / math.sqrt(d_in))
            self.weight.mul_(self.mask).add_((1 - self.mask) * self.mask_constant * self.normal_mask)
            if self.bias is not None:
                fan_in, _ = nn.init._calculate_fan_in_and_fan_out(self.weight)
                bound = 1 / math.sqrt(fan_in) if fan_in > 0 else 0
                nn.init.uniform_(self.bias, -bound, bound)
    
    def forward(self, x):
        return F.linear(x, self.weight * self.mask.detach() + (1 - self.mask.detach()) * self.mask_constant * self.normal_mask, self.bias)


@register('model', 'mlp_standard')
class MLP(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim, num_layers, norm='layer', 
                 mask_params=None, elementwise_affine=True, activation='relu'):
        super().__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.output_dim = output_dim
        self.num_layers = num_layers
        self.activation = activation
        
        self.lins = nn.ModuleList()
        self.norms = nn.ModuleList()
        
        # Handle normalization
        self.norm_kind = norm
        self.norm = setup_normalization(norm, hidden_dim, elementwise_affine)
        
        # Handle activation
        self.activation_func = setup_activation(activation)
        
        if num_layers == 1:
            self.lins.append(nn.Linear(input_dim, output_dim))
        else:
            self.lins.append(nn.Linear(input_dim, hidden_dim))
            
            for i in range(num_layers - 2):
                self.lins.append(nn.Linear(hidden_dim, hidden_dim))
            
            self.lins.append(nn.Linear(hidden_dim, output_dim))
            
            # Add normalization layers
            if self.norm_kind in ('layer', 'batch'):
                for i in range(num_layers - 1):
                    self.norms.append(self.norm(hidden_dim))
            elif self.norm_kind in ('layer_linear', 'batch_linear'):
                for i in range(num_layers - 1):
                    v = total_output_variances_init(self.lins[i], use_realized_F=True)
                    self.norms.append(self.norm(v))
        
        self.norm_type = norm
    
    def forward(self, x):
        x = x.view(x.size(0), -1)  # Flatten the input
        
        for i in range(len(self.lins) - 1):
            x = self.lins[i](x)
            if self.norm_type:
                x = self.norms[i](x)
            if self.activation_func:
                x = self.activation_func(x)
        
        x = self.lins[-1](x)
        return x
    


@register('model', 'mlp_w_asym')
class WMLP(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim, num_layers, 
                 mask_params=None, norm='layer', elementwise_affine=True, activation='gelu'):
        super().__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.output_dim = output_dim
        self.num_layers = num_layers
        self.activation = activation
        
        self.lins = nn.ModuleList()
        self.norms = nn.ModuleList()
        
        # Handle normalization
        self.norm_kind = norm
        self.norm = setup_normalization(norm, hidden_dim, elementwise_affine)
        
        # Handle activation
        self.activation_func = setup_activation(activation)
        
        def get_mask_params(layer_idx, default_params=None):
            if default_params is None:
                default_params = {
                    'mask_constant': 1.0,
                    'mask_type': 'random_subsets',
                    'do_normal_mask': True,
                    'num_fixed': 64
                }
            
            layer_key = f'linear_mask_params_{layer_idx}'
            if layer_key in mask_params:
                return {**default_params, **mask_params[layer_key]}
            
            if 'default' in mask_params:
                return {**default_params, **mask_params['default']}
            
            return default_params
        
        if num_layers == 1:
            self.lins.append(SparseLinear(input_dim, output_dim, 
                                        mask_num=0, **get_mask_params(0)))
        else:
            first_layer_params = get_mask_params(0, mask_params.get('default', None))
            self.lins.append(SparseLinear(input_dim, hidden_dim, 
                                        mask_num=0, **first_layer_params))
            
            for i in range(num_layers - 2):
                hidden_layer_params = get_mask_params(i+1, mask_params.get('default', None))
                self.lins.append(SparseLinear(hidden_dim, hidden_dim, 
                                            mask_num=i+1, **hidden_layer_params))
            
            output_layer_params = get_mask_params(num_layers-1)
            self.lins.append(SparseLinear(hidden_dim, output_dim, 
                                        mask_num=num_layers-1, **output_layer_params))
            
            # Add normalization layers
            if self.norm_kind in ('layer', 'batch'):
                for i in range(num_layers - 1):
                    self.norms.append(self.norm(hidden_dim))
            elif self.norm_kind in ('layer_linear', 'batch_linear'):
                for i in range(num_layers - 1):
                    v = total_output_variances_init(self.lins[i], use_realized_F=True)
                    self.norms.append(self.norm(v))
        
        self.norm_type = norm
    
    def forward(self, x):
        x = x.view(x.size(0), -1)  # Flatten the input
        
        for i in range(len(self.lins) - 1):
            x = self.lins[i](x)
            if self.norm_type:
                x = self.norms[i](x)
            if self.activation_func:
                x = self.activation_func(x)
        
        x = self.lins[-1](x)
        return x
    
    
    def count_unused_params(self):
        count = 0
        for module in self.modules():
            if isinstance(module, SparseLinear):
                count += (1 - module.mask).sum().int().item()
        return count


@register('model', 'mlp_sigma_asym')
class SigmaMLP(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim, num_layers, 
                 norm='layer', asym_act=True):
        super().__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.output_dim = output_dim
        self.num_layers = num_layers
        
        self.lins = nn.ModuleList()
        self.activations = nn.ModuleList()
        self.norms = nn.ModuleList()
        
        if asym_act:
            for i in range(num_layers - 1):
                self.activations.append(AsymSwiGLU(hidden_dim, mask_num=i))
        else:
            for i in range(num_layers - 1):
                self.activations.append(nn.GELU())
        
        if num_layers == 1:
            self.lins.append(nn.Linear(input_dim, output_dim))
        else:
            self.lins.append(nn.Linear(input_dim, hidden_dim))
            if norm == 'layer':
                self.norms.append(nn.LayerNorm(hidden_dim))
            elif norm == 'batch':
                self.norms.append(nn.BatchNorm1d(hidden_dim))
            
            for i in range(num_layers - 2):
                self.lins.append(nn.Linear(hidden_dim, hidden_dim))
                if norm == 'layer':
                    self.norms.append(nn.LayerNorm(hidden_dim))
                elif norm == 'batch':
                    self.norms.append(nn.BatchNorm1d(hidden_dim))
            
            self.lins.append(nn.Linear(hidden_dim, output_dim))
        
        self.norm_type = norm
    
    def forward(self, x):
        x = x.view(x.size(0), -1)  # Flatten the input
        
        for i in range(len(self.lins) - 1):
            x = self.lins[i](x)
            if self.norm_type:
                x = self.norms[i](x)
            x = self.activations[i](x)
        
        x = self.lins[-1](x)
        return x


def create_mlp(symmetry, input_dim, hidden_dim, output_dim, num_layers, 
               mask_params=None, norm='layer', elementwise_affine=True, activation=None):
    if symmetry == 0:
        activation = activation or 'relu'
        return MLP(input_dim, hidden_dim, output_dim, num_layers, norm, 
                  mask_params, elementwise_affine, activation)
    elif symmetry == 1:
        if mask_params is None:
            raise ValueError("mask_params required for W-Asym MLP")
        activation = activation or 'gelu'
        return WMLP(input_dim, hidden_dim, output_dim, num_layers, mask_params, norm, 
                   elementwise_affine, activation)
    elif symmetry == 2:
        return SigmaMLP(input_dim, hidden_dim, output_dim, num_layers, norm)
    else:
        raise ValueError(f"Invalid symmetry type: {symmetry}")
