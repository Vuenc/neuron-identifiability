"""
Normalization layers for MLP models.
"""

import torch
import torch.nn as nn


def setup_normalization(norm, hidden_dim, affine=True):
    """Helper function to setup normalization for MLP models."""
    if not norm:
        return None
    else:
        if norm == 'layer':
            return lambda hidden_dim: nn.LayerNorm(hidden_dim, elementwise_affine=affine)
        elif norm == 'batch':
            return lambda hidden_dim: nn.BatchNorm1d(hidden_dim, affine=affine)
        elif norm == 'layer_linear':
            return lambda v: LayerNormLinearProxyWMLP(v, affine=affine)
        elif norm == 'batch_linear':
            return lambda v: BatchNormLinearProxyWMLP(v, affine=affine)
        else:
            raise ValueError(f"Bad norm type: {norm}")


@torch.no_grad()
def total_output_variances_init(lin, use_realized_F: bool = False, eps: float = 1e-12):
    """Per-output variances at init under Cov[x]=I."""
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
    
    def __init__(self, v: torch.Tensor, eps: float = 1e-12, affine: bool = True):
        super().__init__()
        d = v.numel()
        sample_var = (1.0 - 1.0 / d) * v.mean()
        scale = (sample_var + eps).rsqrt()
        self.register_buffer("scale", scale)
        
        # Add trainable beta and gamma parameters like real LayerNorm
        if affine:
            self.weight = nn.Parameter(torch.ones(d))
            self.bias = nn.Parameter(torch.zeros(d))
        else:
            self.register_buffer("weight", torch.ones(d))
            self.register_buffer("bias", torch.zeros(d))

    def forward(self, z):
        # Normalize
        z_norm = (z - z.mean(dim=-1, keepdim=True)) * self.scale
        # Apply trainable affine transformation
        return z_norm * self.weight + self.bias


class BatchNormLinearProxyWMLP(nn.Module):
    """Linear proxy for BatchNorm in W-Asymmetric MLPs."""
    
    def __init__(self, v: torch.Tensor, eps: float = 1e-12, affine: bool = True):
        super().__init__()
        inv_std = (v + eps).rsqrt().to(torch.float32)
        self.register_buffer("inv_std", inv_std)
        
        # Add trainable beta and gamma parameters like real BatchNorm
        if affine:
            self.weight = nn.Parameter(torch.ones(v.numel()))
            self.bias = nn.Parameter(torch.zeros(v.numel()))
        else:
            self.register_buffer("weight", torch.ones(v.numel()))
            self.register_buffer("bias", torch.zeros(v.numel()))

    def forward(self, z):
        view = [1] * z.ndim
        view[-1] = z.shape[-1]
        # Normalize
        z_norm = z * self.inv_std.view(*view)
        # Apply trainable affine transformation
        return z_norm * self.weight.view(*view) + self.bias.view(*view)
