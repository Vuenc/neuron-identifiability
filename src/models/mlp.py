"""
MLP models with asymmetric architectures.
Refactored from lmc/models/models_mlp.py
"""

from typing import Dict
import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from ..core.registry import register
from .normalization import total_output_variances_init, setup_normalization
from .activation import setup_activation


class AsymSwiGLU(nn.Module):
    def __init__(self, dim, mask_rng: torch.Generator, scale=1.0, mask_num=0, fixed_C=None):
        super().__init__()
        if fixed_C is not None:
            C = fixed_C
        else:
            C = torch.randn(dim, dim, generator=mask_rng)
        self.register_buffer("C", C)
        self.mask_num = mask_num
    
    def forward(self, x):
        gate = F.sigmoid(F.linear(x, self.C))
        return gate * x


class SparseLinear(nn.Module):
    def __init__(self, in_dim, out_dim, mask_rng: torch.Generator, mask_constant=0, mask_type='random_subsets', 
                 do_normal_mask=True, num_fixed=6, bias=True, mask_num=0):
        super().__init__()

        mask = self._make_mask(in_dim, out_dim, mask_type, num_fixed, mask_rng)
        self.register_buffer('mask', mask, persistent=True)
        
        self.weight = nn.Parameter(torch.empty(out_dim, in_dim))
        if bias:
            self.bias = nn.Parameter(torch.empty(out_dim))
        else:
            self.register_parameter('bias', None)
        
        self.weight.register_hook(lambda grad: self.mask * grad)
        
        if do_normal_mask:
            normal_mask = torch.randn(out_dim, in_dim, generator=mask_rng)
            self.register_buffer('normal_mask', normal_mask, persistent=True)
        else:
            self.register_buffer('normal_mask', torch.ones_like(mask), persistent=True)
        
        self.mask_num = mask_num
        self.mask_constant = mask_constant
        self.reset_parameters()
    
    def _make_mask(self, in_dim, out_dim, mask_type, num_fixed, mask_rng: torch.Generator):
        if mask_type == 'random_subsets':
            mask = torch.ones(out_dim, in_dim)
            if num_fixed >= in_dim:
                raise ValueError(f'Error in making mask: too many fixed parameters (num_fixed={num_fixed}, in_dim={in_dim})')
            for row_idx in range(out_dim):
                zeros_in_row = torch.randperm(in_dim, generator=mask_rng)[:num_fixed]
                mask[row_idx, zeros_in_row] = 0
            return mask
        else:
            raise ValueError('Invalid mask type')
    
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


class NoiseLinear(nn.Module):
    """Linear layer with fixed Gaussian noise injection for symmetry breaking."""
    def __init__(self, in_dim, out_dim, mask_rng: torch.Generator, mask_constant=1.0, **kwargs):
        super().__init__()
        
        self.weight = nn.Parameter(torch.empty(out_dim, in_dim))
        self.bias = nn.Parameter(torch.empty(out_dim))
        
        # Create fixed Gaussian noise similar to AsymSwiGLU's C initialization
        noise = torch.randn(out_dim, in_dim, generator=mask_rng)
        self.register_buffer('noise', noise, persistent=True)
        
        self.mask_constant = mask_constant
        self.reset_parameters()
    
    def reset_parameters(self):
        with torch.no_grad():
            d_in = self.weight.size(1)
            nn.init.normal_(self.weight, mean=0.0, std=1.0 / math.sqrt(d_in))
            if self.bias is not None:
                fan_in, _ = nn.init._calculate_fan_in_and_fan_out(self.weight)
                bound = 1 / math.sqrt(fan_in) if fan_in > 0 else 0
                nn.init.uniform_(self.bias, -bound, bound)
    
    def forward(self, x):
        # Add scaled fixed noise to weights during forward pass
        effective_weight = self.weight + self.mask_constant * self.noise
        return F.linear(x, effective_weight, self.bias)


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
        self.norm_type = norm
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
            if self.norm_type in ('layer', 'batch'):
                for i in range(num_layers - 1):
                    self.norms.append(self.norm(hidden_dim))
            elif self.norm_type in ('layer_linear', 'batch_linear'):
                for i in range(num_layers - 1):
                    v = total_output_variances_init(self.lins[i], use_realized_F=True)
                    self.norms.append(self.norm(v))
    
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
                 mask_params: Dict, norm='layer', elementwise_affine=True, activation='gelu', mask_seed=None):
        super().__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.output_dim = output_dim
        self.num_layers = num_layers
        self.activation = activation
        self.elementwise_affine = elementwise_affine
        
        self.lins = nn.ModuleList()
        self.norms = nn.ModuleList()
        
        # Handle normalization
        self.norm_type = norm
        self.norm = setup_normalization(norm, hidden_dim, elementwise_affine)
        
        # Handle activation
        self.activation_func = setup_activation(activation)
        
        def get_mask_params(layer_idx):
            default_params = mask_params.get('default', None)
            if default_params is None:
                print("No network-wide default params specified, falling back to global defaults")
                default_params = {
                    'mask_constant': 1.0,
                    'mask_type': 'random_subsets',
                    'do_normal_mask': True,
                    'num_fixed': 64
                }
            layer_key = f'linear_mask_params_{layer_idx}'
            if layer_key in mask_params:
                output = {**default_params, **mask_params[layer_key]}
            else:
                output = {**default_params, **mask_params['default']}
            print(f"Mask params for linear{layer_idx}: {output}")
            return output
        
        mask_rng = torch.Generator()
        if mask_seed is not None:
            mask_rng.manual_seed(mask_seed)

        if num_layers == 1:
            self.lins.append(SparseLinear(input_dim, output_dim, 
                                        mask_num=0, **get_mask_params(0), mask_rng=mask_rng))
        else:
            self.lins.append(SparseLinear(input_dim, hidden_dim, 
                                        mask_num=0, **get_mask_params(0), mask_rng=mask_rng))
            
            for i in range(num_layers - 2):
                self.lins.append(SparseLinear(hidden_dim, hidden_dim, 
                                            mask_num=i+1, **get_mask_params(i+1), mask_rng=mask_rng))
            
            self.lins.append(SparseLinear(hidden_dim, output_dim, 
                                        mask_num=num_layers-1, **get_mask_params(num_layers-1), mask_rng=mask_rng))
            
            # Add normalization layers
            if self.norm_type in ('layer', 'batch'):
                for i in range(num_layers - 1):
                    self.norms.append(self.norm(hidden_dim))
            elif self.norm_type in ('layer_linear', 'batch_linear'):
                for i in range(num_layers - 1):
                    v = total_output_variances_init(self.lins[i], use_realized_F=True)
                    self.norms.append(self.norm(v))
    
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


@register('model', 'mlp_noise_asym')
class NoiseMLP(nn.Module):
    def __init__(self, input_dim: int, hidden_dim, output_dim, num_layers,
                 mask_params: Dict, norm='layer', elementwise_affine=True, activation='gelu', mask_seed=None):
        super().__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.output_dim = output_dim
        self.num_layers = num_layers
        self.activation = activation
        self.elementwise_affine = elementwise_affine
        
        self.lins = nn.ModuleList()
        self.norms = nn.ModuleList()
        
        # Handle normalization
        self.norm_type = norm
        self.norm = setup_normalization(norm, hidden_dim, elementwise_affine)
        
        # Handle activation
        self.activation_func = setup_activation(activation)
        
        def get_mask_params(layer_idx):
            default_params = mask_params.get('default', None)
            if default_params is None:
                default_params = {'mask_constant': 1.0}
            layer_key = f'linear_mask_params_{layer_idx}'
            if layer_key in mask_params:
                output = {**default_params, **mask_params[layer_key]}
            else:
                output = default_params
            print(f"Mask params for linear{layer_idx}: {output}")
            return output

        mask_rng = torch.Generator()
        if mask_seed is not None:
            mask_rng.manual_seed(mask_seed)

        if num_layers == 1:
            first_layer_params = get_mask_params(0)
            self.lins.append(NoiseLinear(input_dim, output_dim, **first_layer_params, mask_rng=mask_rng))
        else:
            first_layer_params = get_mask_params(0)
            self.lins.append(NoiseLinear(input_dim, hidden_dim, **first_layer_params, mask_rng=mask_rng))
            
            for i in range(num_layers - 2):
                hidden_layer_params = get_mask_params(i+1)
                self.lins.append(NoiseLinear(hidden_dim, hidden_dim, **hidden_layer_params, mask_rng=mask_rng))
            
            output_layer_params = get_mask_params(num_layers-1)
            self.lins.append(NoiseLinear(hidden_dim, output_dim, **output_layer_params, mask_rng=mask_rng))
            
            # Add normalization layers
            if self.norm_type in ('layer', 'batch'):
                for i in range(num_layers - 1):
                    self.norms.append(self.norm(hidden_dim))
            elif self.norm_type in ('layer_linear', 'batch_linear'):
                for i in range(num_layers - 1):
                    v = total_output_variances_init(self.lins[i], use_realized_F=True)
                    self.norms.append(self.norm(v))
    
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


@register('model', 'mlp_sigma_asym')
class SigmaMLP(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim, num_layers,
                 norm='layer', asym_act=True, mask_seed=None):
        super().__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.output_dim = output_dim
        self.num_layers = num_layers
        
        self.lins = nn.ModuleList()
        self.activations = nn.ModuleList()
        self.norms = nn.ModuleList()
    
        mask_rng = torch.Generator()
        if mask_seed is not None:
            mask_rng.manual_seed(mask_seed)
        
        if asym_act:
            for i in range(num_layers - 1):
                self.activations.append(AsymSwiGLU(hidden_dim, mask_num=i, mask_rng=mask_rng))
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


def _initialize_from_asym_mlp(symmetry, input_dim, hidden_dim, output_dim, num_layers,
                              mask_params, norm, elementwise_affine, activation, mask_seed: int | None):
    """Initialize a symmetry type 0 (standard) MLP with weights from symmetry type 1 or 3 (W-Asym or Noise-Asym).
    
    This creates an asym model using create_mlp, runs it through forward to extract effective weights,
    then copies them into a standard model.
    
    Args:
        symmetry: 1=W-Asym, 3=Noise-Asym
        input_dim: Input dimension
        hidden_dim: Hidden dimension
        output_dim: Output dimension
        num_layers: Number of layers
        mask_params: Mask parameters for asym model
        norm: Normalization type
        elementwise_affine: Whether normalization has learnable affine parameters
        activation: Activation function
        
    Returns:
        Standard MLP with weights initialized from asym model
    """
    # Create asym model using the normal create_mlp function
    asym_model = create_mlp(symmetry=symmetry, input_dim=input_dim, hidden_dim=hidden_dim, 
                           output_dim=output_dim, num_layers=num_layers, mask_params=mask_params,
                           norm=norm, elementwise_affine=elementwise_affine, activation=activation,
                           asym_init_only=False, mask_seed=mask_seed)
    
    # Create standard model
    standard_model = create_mlp(symmetry=0, input_dim=input_dim, hidden_dim=hidden_dim,
                               output_dim=output_dim, num_layers=num_layers, mask_params=None,
                               norm=norm, elementwise_affine=elementwise_affine, activation=activation,
                               asym_init_only=False, mask_seed=mask_seed)
    
    # Extract effective weights - handle W-Asym and Noise-Asym differently
    standard_state = standard_model.state_dict()
    
    if symmetry == 1:
        # W-Asym: weight.data is already modified by reset_parameters(), need to capture what forward() actually uses
        # Hook into F.linear to capture effective weights, tracking by module name
        captured_weights = {}
        active_module = [None]
        original_linear = F.linear
        
        def linear_hook(input, weight, bias=None):
            if active_module[0] is not None:
                module_name = active_module[0]
                weight_name = module_name + '.weight' if module_name else 'weight'
                if weight_name not in captured_weights:
                    captured_weights[weight_name] = weight.detach().clone()
            return original_linear(input, weight, bias)
        
        F.linear = linear_hook
        
        # Register pre-hooks on all sparse modules
        hooks = []
        for asym_name, asym_module in asym_model.named_modules():
            if hasattr(asym_module, 'mask') and hasattr(asym_module, 'weight'):
                hook = asym_module.register_forward_pre_hook(
                    lambda module, input, name=asym_name: active_module.__setitem__(0, name),
                    with_kwargs=False
                )
                hooks.append(hook)
        
        try:
            asym_model.eval()
            dummy_input = torch.randn(1, input_dim)
            with torch.no_grad():
                _ = asym_model(dummy_input)
        finally:
            F.linear = original_linear
            for hook in hooks:
                hook.remove()
        
        # Copy captured effective weights
        for weight_name, effective_weight in captured_weights.items():
            if weight_name in standard_state:
                standard_state[weight_name].data.copy_(effective_weight)
    
    else:  # symmetry == 3 (Noise-Asym)
        # Noise-Asym: weight.data contains actual weights, just add noise directly
        for asym_name, asym_module in asym_model.named_modules():
            if hasattr(asym_module, 'noise') and hasattr(asym_module, 'mask_constant') and hasattr(asym_module, 'weight'):
                # Compute effective weight: weight + noise * mask_constant
                effective_weight = asym_module.weight.data + asym_module.mask_constant * asym_module.noise
                weight_name = asym_name + '.weight' if asym_name else 'weight'
                if weight_name in standard_state:
                    standard_state[weight_name].data.copy_(effective_weight)
    
    # Copy biases and other parameters (that aren't part of asym layers)
    asym_modules_dict = dict(asym_model.named_modules())
    for asym_name, asym_param in asym_model.named_parameters():
        # Skip weights of asym layers (we already copied effective weights)
        module_path = asym_name.rsplit('.', 1)[0]
        if module_path in asym_modules_dict:
            module = asym_modules_dict[module_path]
            if hasattr(module, 'mask') or hasattr(module, 'noise'):
                # This is an asym layer - skip raw parameters, we already copied effective weights
                if '.weight' in asym_name:
                    continue  # Already handled via captured_weights
        
        # Copy other parameters (biases, normalization params, etc.)
        if asym_name in standard_state:
            standard_state[asym_name].data.copy_(asym_param.data)
    
    # Copy buffers (normalization, etc.)
    for asym_name, asym_buffer in asym_model.named_buffers():
        if 'mask' not in asym_name and 'noise' not in asym_name and 'normal_mask' not in asym_name:
            if asym_name in standard_state:
                standard_state[asym_name].data.copy_(asym_buffer.data)
    
    standard_model.load_state_dict(standard_state)
    
    # Verify that both models implement the same function at initialization
    asym_model.eval()
    standard_model.eval()
    
    # Create a small batch of random inputs
    batch_size = 4
    test_input = torch.randn(batch_size, input_dim)
    
    with torch.no_grad():
        asym_output = asym_model(test_input)
        standard_output = standard_model(test_input)
    
    # Check if outputs match
    max_diff = (asym_output - standard_output).abs().max().item()
    mean_diff = (asym_output - standard_output).abs().mean().item()
    max_rel_diff = ((asym_output - standard_output).abs() / (asym_output.abs() + 1e-8)).max().item()
    
    sym_type_name = "W-Asym" if symmetry == 1 else "Noise-Asym"
    print(f"Initialized standard MLP from {sym_type_name} model")
    print(f"Function equivalence check:")
    print(f"  Max absolute difference: {max_diff:.2e}")
    print(f"  Mean absolute difference: {mean_diff:.2e}")
    print(f"  Max relative difference: {max_rel_diff:.2e}")
    
    # Check if they're close enough (within numerical precision)
    tolerance = 1e-5
    if max_diff < tolerance:
        print(f"  ✓ Models implement the same function (difference < {tolerance})")
    else:
        print(f"  ⚠ Warning: Models differ by more than {tolerance}")
        print(f"    This may indicate an issue with weight copying.")
    
    return standard_model


def create_mlp(symmetry, input_dim, hidden_dim, output_dim, num_layers, mask_seed=None,
               mask_params=None, norm='layer', elementwise_affine=True, activation=None, asym_init_only=False):
    """Create MLP based on symmetry type.
    
    Args:
        symmetry: 0=Standard, 1=W-Asym, 2=Sigma-Asym, 3=Noise-Asym
        input_dim: Input dimension
        hidden_dim: Hidden dimension
        output_dim: Output dimension
        num_layers: Number of layers
        mask_params: Mask parameters for asym models
        norm: Normalization type
        elementwise_affine: Whether normalization has learnable affine parameters
        activation: Activation function
        asym_init_only: If True and symmetry in [1, 3], initialize standard model with weights from asym model
    """
    # Handle asym_init_only: use asym model only for initialization, create standard model
    if asym_init_only and symmetry in [1, 3]:
        return _initialize_from_asym_mlp(symmetry, input_dim, hidden_dim, output_dim, num_layers,
                                       mask_params, norm, elementwise_affine, activation, mask_seed=mask_seed)
    
    if symmetry == 0:
        activation = activation or 'relu'
        return MLP(input_dim, hidden_dim, output_dim, num_layers, norm, 
                  mask_params, elementwise_affine, activation)
    elif symmetry == 1:
        if mask_params is None:
            raise ValueError("mask_params required for W-Asym MLP")
        activation = activation or 'gelu'
        return WMLP(input_dim, hidden_dim, output_dim, num_layers, mask_params, norm, 
                   elementwise_affine, activation, mask_seed=mask_seed)
    elif symmetry == 2:
        return SigmaMLP(input_dim, hidden_dim, output_dim, num_layers, norm, mask_seed=mask_seed)
    elif symmetry == 3:
        activation = activation or 'gelu'
        return NoiseMLP(input_dim, hidden_dim, output_dim, num_layers, mask_seed=mask_seed,
                       mask_params=mask_params, norm=norm, elementwise_affine=elementwise_affine, 
                       activation=activation)
    else:
        raise ValueError(f"Invalid symmetry type: {symmetry}")
