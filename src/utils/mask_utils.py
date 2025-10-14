"""
Utilities for managing fixed masks across multiple model instances.
"""

import torch
import os
from pathlib import Path
from typing import Dict, Any, Optional


def save_mask(mask, mask_path: str, layer_name: str):
    """Save a mask to disk."""
    Path(mask_path).mkdir(parents=True, exist_ok=True)
    torch.save(mask, os.path.join(mask_path, f"{layer_name}_mask.pt"))


def load_mask(mask_path: str, layer_name: str) -> Optional[torch.Tensor]:
    """Load a mask from disk."""
    mask_file = os.path.join(mask_path, f"{layer_name}_mask.pt")
    if os.path.exists(mask_file):
        return torch.load(mask_file)
    return None


def save_mask_params(mask_params: Dict[str, Any], mask_path: str):
    """Save mask parameters to disk."""
    Path(mask_path).mkdir(parents=True, exist_ok=True)
    torch.save(mask_params, os.path.join(mask_path, "mask_params.pt"))


def load_mask_params(mask_path: str) -> Optional[Dict[str, Any]]:
    """Load mask parameters from disk."""
    params_file = os.path.join(mask_path, "mask_params.pt")
    if os.path.exists(params_file):
        return torch.load(params_file)
    return None


def generate_fixed_masks(mask_params: Dict[str, Any], model_config: Dict[str, Any], 
                        mask_path: str, force_regenerate: bool = False):
    """Generate and save fixed masks for a model configuration."""
    
    # Check if masks already exist
    if not force_regenerate and os.path.exists(os.path.join(mask_path, "mask_params.pt")):
        print(f"Masks already exist at {mask_path}. Use force_regenerate=True to regenerate.")
        return
    
    # Generate masks for each layer
    generated_masks = {}
    
    if model_config.get('symmetry') == 1:  # W-Asymmetric
        from ..models.mlp import SparseLinear
        
        num_layers = model_config['num_layers']
        hidden_dim = model_config['hidden_dim']
        input_dim = model_config['input_dim']
        output_dim = model_config['output_dim']
        
        for layer_idx in range(num_layers):
            if layer_idx == 0:
                in_dim, out_dim = input_dim, hidden_dim
            elif layer_idx == num_layers - 1:
                in_dim, out_dim = hidden_dim, output_dim
            else:
                in_dim, out_dim = hidden_dim, hidden_dim
            
            # Get mask parameters for this layer
            layer_key = f'linear_mask_params_{layer_idx}'
            if layer_key in mask_params:
                layer_params = mask_params[layer_key]
            elif 'default' in mask_params:
                layer_params = mask_params['default']
            else:
                layer_params = {
                    'mask_constant': 1.0,
                    'mask_type': 'random_subsets',
                    'do_normal_mask': True,
                    'num_fixed': 64
                }
            
            # Create temporary layer to generate mask
            temp_layer = SparseLinear(in_dim, out_dim, mask_num=layer_idx, **layer_params)
            generated_masks[f'layer_{layer_idx}'] = temp_layer.mask.clone()
    
    elif model_config.get('symmetry') == 2:  # Sigma-Asymmetric
        from ..models.mlp import AsymSwiGLU
        
        num_layers = model_config['num_layers']
        hidden_dim = model_config['hidden_dim']
        
        for layer_idx in range(num_layers - 1):
            # Generate C matrix for AsymSwiGLU
            g = torch.Generator()
            g.manual_seed(abs(hash(str(layer_idx) + str(0))))
            C = torch.randn(hidden_dim, hidden_dim, generator=g)
            generated_masks[f'activation_{layer_idx}'] = C
    
    elif model_config.get('symmetry') == 3:  # Noise-Asymmetric
        from ..models.mlp import NoiseLinear
        
        num_layers = model_config['num_layers']
        hidden_dim = model_config['hidden_dim']
        input_dim = model_config['input_dim']
        output_dim = model_config['output_dim']
        
        for layer_idx in range(num_layers):
            if layer_idx == 0:
                in_dim, out_dim = input_dim, hidden_dim
            elif layer_idx == num_layers - 1:
                in_dim, out_dim = hidden_dim, output_dim
            else:
                in_dim, out_dim = hidden_dim, hidden_dim
            
            # Create temporary layer to generate noise
            temp_layer = NoiseLinear(in_dim, out_dim, mask_num=layer_idx)
            generated_masks[f'layer_{layer_idx}'] = temp_layer.noise.clone()
    
    # Save all masks
    for layer_name, mask in generated_masks.items():
        save_mask(mask, mask_path, layer_name)
    
    # Save mask parameters
    save_mask_params(mask_params, mask_path)
    
    print(f"Generated and saved {len(generated_masks)} masks to {mask_path}")


def apply_fixed_masks(model, mask_path: str):
    """Apply fixed masks to a model."""
    
    for name, module in model.named_modules():
        if hasattr(module, 'mask'):
            # For SparseLinear layers
            layer_name = f"layer_{module.mask_num}"
            fixed_mask = load_mask(mask_path, layer_name)
            if fixed_mask is not None:
                module.mask.data = fixed_mask
                print(f"Applied fixed mask to {name}")
        
        elif hasattr(module, 'C'):
            # For AsymSwiGLU layers
            layer_idx = getattr(module, 'mask_num', 0)
            layer_name = f"activation_{layer_idx}"
            fixed_C = load_mask(mask_path, layer_name)
            if fixed_C is not None:
                module.C.data = fixed_C
                print(f"Applied fixed C matrix to {name}")


def create_model_with_fixed_masks(model_class, mask_path: str, **kwargs):
    """Create a model and apply fixed masks."""
    model = model_class(**kwargs)
    apply_fixed_masks(model, mask_path)
    return model


def save_masks(model, output_dir):
    """Save all masks and fixed weights from a model."""
    mask_dir = output_dir / "fixed_masks"
    mask_dir.mkdir(parents=True, exist_ok=True)
    
    mask_params = {}
    for name, module in model.named_modules():
        if hasattr(module, 'mask') and module.mask is not None:
            torch.save(module.mask, mask_dir / f"{name}_mask.pt")
            mask_params[name] = {'mask_path': str(mask_dir / f"{name}_mask.pt")}
        if hasattr(module, 'C') and module.C is not None:  # For AsymSwiGLU
            torch.save(module.C, mask_dir / f"{name}_C.pt")
            mask_params[name] = {'C_path': str(mask_dir / f"{name}_C.pt")}
    
    torch.save(mask_params, mask_dir / "mask_params.pt")
    return mask_params


def load_masks(output_dir):
    """Load all masks and fixed weights for a model."""
    mask_dir = output_dir / "fixed_masks"
    mask_params_path = mask_dir / "mask_params.pt"
    if not mask_params_path.exists():
        raise FileNotFoundError(f"Mask parameters not found at {mask_params_path}")
    
    loaded_mask_params = torch.load(mask_params_path)
    
    fixed_masks = {}
    for name, params in loaded_mask_params.items():
        if 'mask_path' in params:
            fixed_masks[name] = torch.load(Path(params['mask_path']))
        if 'C_path' in params:
            fixed_masks[name] = torch.load(Path(params['C_path']))
    return fixed_masks
