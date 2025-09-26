"""
MLP models with asymmetric architectures.
Refactored from lmc/models/models_mlp.py
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from ..core.registry import register


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
                 do_normal_mask=True, num_fixed=6, mask_num=0, fixed_mask=None):
        super().__init__()
        
        if fixed_mask is not None:
            mask = fixed_mask
        else:
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
            # Initialize weights randomly, then apply mask and normal_mask
            nn.init.normal_(self.weight, mean=0.0, std=1.0 / math.sqrt(self.weight.size(1)))
            self.weight.mul_(self.mask).add_((1 - self.mask) * self.mask_constant * self.normal_mask)
            if self.bias is not None:
                nn.init.zeros_(self.bias)
    
    def forward(self, x):
        # Apply mask and normal_mask like in the original implementation
        self.weight.data = (self.weight.data * self.mask + (1 - self.mask) * self.mask_constant * self.normal_mask)
        return F.linear(x, self.weight, self.bias)


@register('model', 'mlp_standard')
class MLP(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim, num_layers, norm='layer'):
        super().__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.output_dim = output_dim
        self.num_layers = num_layers
        
        self.lins = nn.ModuleList()
        self.norms = nn.ModuleList()
        
        if num_layers == 1:
            self.lins.append(nn.Linear(input_dim, output_dim))
        else:
            self.lins.append(nn.Linear(input_dim, hidden_dim))
            if norm == 'layer':
                self.norms.append(nn.LayerNorm(hidden_dim))
            elif norm == 'batch':
                self.norms.append(nn.BatchNorm1d(hidden_dim))
            
            for _ in range(num_layers - 2):
                self.lins.append(nn.Linear(hidden_dim, hidden_dim))
                if norm == 'layer':
                    self.norms.append(nn.LayerNorm(hidden_dim))
                elif norm == 'batch':
                    self.norms.append(nn.BatchNorm1d(hidden_dim))
            
            self.lins.append(nn.Linear(hidden_dim, output_dim))
        
        self.norm_type = norm
    
    def forward(self, x):
        for i in range(len(self.lins) - 1):
            x = self.lins[i](x)
            if self.norm_type:
                x = self.norms[i](x)
            x = F.relu(x)
        
        x = self.lins[-1](x)
        return x


@register('model', 'mlp_w_asym')
class WMLP(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim, num_layers, 
                 mask_params, norm='layer', fixed_masks=None):
        super().__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.output_dim = output_dim
        self.num_layers = num_layers
        
        self.lins = nn.ModuleList()
        self.norms = nn.ModuleList()
        
        # Handle normalization
        self.norm_kind = norm
        if not norm:
            self.norm = None
        else:
            if norm == 'layer':
                self.norm = nn.LayerNorm
            elif norm == 'batch':
                self.norm = nn.BatchNorm1d
            elif norm == 'layer_linear':
                self.norm = LayerNormLinearProxyWMLP
            elif norm == 'batch_linear':
                self.norm = BatchNormLinearProxyWMLP
            else:
                raise ValueError(f"Bad norm type: {norm}")
        
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
            fixed_mask = fixed_masks.get('lins.0') if fixed_masks else None
            self.lins.append(SparseLinear(input_dim, output_dim, 
                                        mask_num=0, fixed_mask=fixed_mask, **get_mask_params(0)))
        else:
            first_layer_params = get_mask_params(0)
            fixed_mask = fixed_masks.get('lins.0') if fixed_masks else None
            self.lins.append(SparseLinear(input_dim, hidden_dim, 
                                        mask_num=0, fixed_mask=fixed_mask, **first_layer_params))
            
            for i in range(num_layers - 2):
                hidden_layer_params = get_mask_params(i+1)
                fixed_mask = fixed_masks.get(f'lins.{i+1}') if fixed_masks else None
                self.lins.append(SparseLinear(hidden_dim, hidden_dim, 
                                            mask_num=i+1, fixed_mask=fixed_mask, **hidden_layer_params))
            
            output_layer_params = get_mask_params(num_layers-1)
            fixed_mask = fixed_masks.get(f'lins.{num_layers-1}') if fixed_masks else None
            self.lins.append(SparseLinear(hidden_dim, output_dim, 
                                        mask_num=num_layers-1, fixed_mask=fixed_mask, **output_layer_params))
            
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
            x = F.gelu(x)
        
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
                 norm='layer', asym_act=True, fixed_masks=None):
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
                fixed_C = fixed_masks.get(f'activations.{i}') if fixed_masks else None
                self.activations.append(AsymSwiGLU(hidden_dim, mask_num=i, fixed_C=fixed_C))
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


@torch.no_grad()
def total_output_variances_init(lin, use_realized_F: bool = False, eps: float = 1e-12):
    """
    Per-output variances at init under Cov[x]=I.

    - SparseLinear:
        v_i ≈ (frozen) + (trainable at init)
            frozen:
              * if use_realized_F: sum_j [ (1-mask_ij) * (mask_constant * normal_mask_ij) ]^2
              * else:              n_frozen_i * kappa,  with kappa = mask_constant^2
            trainable: n_train_i / d_in   (LeCun/fan-in init: Var = 1/d_in)
    - nn.Linear:
        v_i = 1  (LeCun/fan-in init), independent of weights.

    Returns:
        Tensor of shape (out_dim,) on lin.weight's device/dtype.
    """
    W = lin.weight
    out_dim, d_in = W.size(0), W.size(1)
    dev, dt = W.device, W.dtype

    if hasattr(lin, "mask") and hasattr(lin, "mask_constant"):
        frozen = (1.0 - lin.mask)
        n_frozen = frozen.sum(dim=1)
        n_train  = lin.mask.sum(dim=1)
        kappa = float(lin.mask_constant) ** 2

        if use_realized_F:
            F_ij = lin.mask_constant * lin.normal_mask
            v_frozen = (frozen * F_ij).pow(2).sum(dim=1)
        else:
            v_frozen = n_frozen * kappa

        sigma_w2 = 1.0 / float(d_in)
        v_train = n_train * sigma_w2
        v = v_frozen + v_train
        return v.to(device=dev, dtype=dt) + eps

    else:
        return torch.ones(out_dim, device=dev, dtype=dt) + eps


class LayerNormLinearProxyWMLP(nn.Module):
    """Linear proxy for LayerNorm in W-Asymmetric MLPs."""
    
    def __init__(self, v: torch.Tensor, eps: float = 1e-12):
        super().__init__()
        d = v.numel()
        sample_var = (1.0 - 1.0 / d) * v.mean()
        scale = (sample_var + eps).rsqrt()
        self.register_buffer("scale", scale)

    def forward(self, z):
        return (z - z.mean(dim=-1, keepdim=True)) * self.scale


class BatchNormLinearProxyWMLP(nn.Module):
    """Linear proxy for BatchNorm in W-Asymmetric MLPs."""
    
    def __init__(self, v: torch.Tensor, eps: float = 1e-12):
        super().__init__()
        inv_std = (v + eps).rsqrt().to(torch.float32)
        self.register_buffer("inv_std", inv_std)

    def forward(self, z):
        view = [1] * z.ndim
        view[-1] = z.shape[-1]
        return z * self.inv_std.view(*view)


def create_mlp(symmetry, input_dim, hidden_dim, output_dim, num_layers, 
               mask_params=None, norm='layer', fixed_masks=None):
    if symmetry == 0:
        return MLP(input_dim, hidden_dim, output_dim, num_layers, norm)
    elif symmetry == 1:
        if mask_params is None:
            raise ValueError("mask_params required for W-Asym MLP")
        return WMLP(input_dim, hidden_dim, output_dim, num_layers, mask_params, norm, fixed_masks)
    elif symmetry == 2:
        return SigmaMLP(input_dim, hidden_dim, output_dim, num_layers, norm, fixed_masks)
    else:
        raise ValueError(f"Invalid symmetry type: {symmetry}")
