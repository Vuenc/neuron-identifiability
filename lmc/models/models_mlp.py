import math
import itertools
import copy
import torch
import torch.nn as nn
import wandb
import torch.nn.functional as F
import random
import numpy as np


class AsymSwiGLU(nn.Module):
     def __init__(self, dim, scale=1.0, mask_num=0):
         super().__init__()
         g = torch.Generator()
         g.manual_seed(abs(hash(str(mask_num)+ str(0))))
         C = torch.randn(dim, dim, generator=g)
         self.register_buffer("C", C)
     def forward(self, x):
         gate = F.sigmoid(F.linear(x, self.C))
         return gate * x


class SigmaMLP(nn.Module):
    def __init__(self, in_dim, hidden_dim, out_dim, num_layers, norm = None, asym_act=True):
        super().__init__()
        self.lins = nn.ModuleList()
        self.activations = nn.ModuleList()
        if asym_act:
            for i in range(num_layers - 1):
                self.activations.append(AsymSwiGLU(hidden_dim, mask_num=i))
        else:
            for i in range(num_layers - 1):
                self.activations.append(nn.GELU())
        if not norm:
          self.norm = None
        else:
          self.norms = nn.ModuleList()
          if norm == 'layer':
              self.norm = nn.LayerNorm
          elif norm== 'batch':
            self.norm = nn.BatchNorm1d
          else:
            raise ValueError("Bad norm type. Should be 'layer' or 'batch'")

        if num_layers == 1:
            self.lins.append(nn.Linear(in_dim, out_dim))

        else:
            if self.norm:
              for _ in range(num_layers - 1):
                self.norms.append(self.norm(hidden_dim))

            self.lins.append(nn.Linear(in_dim, hidden_dim))

            for _ in range(num_layers-2):
                self.lins.append(nn.Linear(hidden_dim, hidden_dim))
            self.lins.append(nn.Linear(hidden_dim, out_dim))
        self.flatten = nn.Flatten()


    def forward(self, x):
        x = self.flatten(x)

        for idx, lin in enumerate(self.lins[:-1]):
            x = lin(x)
            if self.norm:
              x = self.norms[idx](x)
            x = self.activations[idx](x)
        x = self.lins[-1](x)
        return x

    def count_unused_params(self):
        return 0

class WMLP(nn.Module):
    def __init__(self, in_dim, hidden_dim, out_dim, num_layers, mask_params, norm=None, act='gelu'):
        super().__init__()
        self.lins = nn.ModuleList()
        #Handle norm first
        self.norm_kind = norm
        if not norm:
          self.norm = None
        else:
          self.norms = nn.ModuleList()
          if norm == 'layer':
            self.norm = nn.LayerNorm
          elif norm == 'batch':
            self.norm = nn.BatchNorm1d
          elif norm == 'layer_linear':
            self.norm = LayerNormLinearProxyWMLP
          elif norm == 'batch_linear':
            self.norm = BatchNormLinearProxyWMLP
          else:
            raise ValueError("Bad norm type")

        #setup Lins
        if num_layers == 1:
            self.lins.append(SparseLinear(in_dim, out_dim, **mask_params[0], mask_num = 0))

        else:
            self.lins.append(SparseLinear(in_dim, hidden_dim, **mask_params[0], mask_num = 0))
            for i in range(num_layers-2):
                self.lins.append(SparseLinear(hidden_dim, hidden_dim, **mask_params[i+1], mask_num = i+1))
            self.lins.append(SparseLinear(hidden_dim, out_dim, **mask_params[num_layers - 1], mask_num = num_layers - 1))
            
            if self.norm_kind in ('layer', 'batch'):
              for i in range(num_layers - 1):
                self.norms.append(self.norm(hidden_dim))
            elif self.norm_kind in ('layer_linear', 'batch_linear'):
              for i in range(num_layers - 1):
                v = total_output_variances_init(self.lins[i], use_realized_F=True)
                self.norms.append(self.norm(v))
        if act == 'gelu':
            self.activation = nn.GELU()
        elif act == 'identity':
            self.activation = nn.Identity()
            
        self.flatten = nn.Flatten()

    def forward(self, x):
        x = self.flatten(x)

        for idx, lin in enumerate(self.lins[:-1]):
            prev=x
            x = lin(x)
    
            if self.norm:
                x = self.norms[idx](x)
            x = self.activation(x)
            
        x = self.lins[-1](x)
        return x

    def disable_sparse_linear_data_replacement(self):
        for lin in self.lins:
            lin.disable_sparse_linear_data_replacement = True

    def count_unused_params(self):
        return sum(lin.count_unused_params() for lin in self.lins if type(lin) != nn.Linear)

class MLP(nn.Module):
    
    def __init__(self, in_dim, hidden_dim, out_dim, num_layers, norm=None, act='gelu'):
        super().__init__()
        self.lins = nn.ModuleList()
        if act == 'gelu':
            self.activation = nn.GELU()
        elif act == 'identity':
            self.activation = nn.Identity()
        self.norm_kind = norm
        if not norm:
          self.norm = None
        else:
          self.norms = nn.ModuleList()
          if norm == 'layer':
              self.norm = nn.LayerNorm
          elif norm == 'batch':
              self.norm = nn.BatchNorm1d
          elif norm == 'layer_linear':
              self.norm = LayerNormLinearProxyWMLP
          elif norm == 'batch_linear':
              self.norm = BatchNormLinearProxyWMLP
          else:
              raise ValueError("Bad norm type")

        if num_layers == 1:
            self.lins.append(nn.Linear(in_dim, out_dim))

        else:
            self.lins.append(nn.Linear(in_dim, hidden_dim))
            for _ in range(num_layers-2):
                self.lins.append(nn.Linear(hidden_dim, hidden_dim))
            self.lins.append(nn.Linear(hidden_dim, out_dim))
            
            if self.norm_kind in ('layer', 'batch'):
              for i in range(num_layers - 1):
                self.norms.append(self.norm(hidden_dim))
            elif self.norm_kind in ('layer_linear', 'batch_linear'):
              for i in range(num_layers - 1):
                v = total_output_variances_init(self.lins[i], use_realized_F=True)
                self.norms.append(self.norm(v))
            
        self.flatten = nn.Flatten()
        self.reset_parameters()

    def forward(self, x):
        x = self.flatten(x)

        for idx, lin in enumerate(self.lins[:-1]):
            x = lin(x)
            if self.norm:
              x = self.norms[idx](x)
            x = self.activation(x)
        x = self.lins[-1](x)
        return x

    def count_unused_params(self):
        return 0
    
    def reset_parameters(self):
        for lin in self.lins:
            if isinstance(lin, nn.Linear):
                d_in = lin.weight.size(1)
                nn.init.normal_(lin.weight, mean=0.0, std=1.0 / math.sqrt(d_in))
                if lin.bias is not None:
                    nn.init.zeros_(lin.bias)
    
    
class SparseLinear(nn.Module):
    
    def __init__(self, in_dim, out_dim, bias=True, mask_type='densest', mask_constant = 1, mask_num = 0, num_fixed = 6, do_normal_mask = True):
        super().__init__()
        assert out_dim < 2**in_dim, 'out dim cannot be much higher than in dim'
        mask = make_mask(in_dim, out_dim, mask_type=mask_type, num_fixed = num_fixed, mask_num = mask_num)

        self.register_buffer('mask', mask, persistent=True)
        self.weight = nn.Parameter(torch.empty((out_dim, in_dim)))

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
        if self.__dict__.get("disable_sparse_linear_data_replacement", False):
            return F.linear(x, self.weight * self.mask.detach() + (1 - self.mask.detach()) * self.mask_constant * self.normal_mask, self.bias)
            #return F.linear(x, self.mask * self.weight, self.bias)
        else:
            self.weight.data = (self.weight.data* self.mask + (1-self.mask)*self.mask_constant*self.normal_mask) 
            return F.linear(x, self.weight, self.bias)

    # def reset_parameters(self):
    #     nn.init.kaiming_uniform_(self.weight, a=math.sqrt(5))
    #     self.weight.data = (self.weight.data* self.mask + (1-self.mask)*self.mask_constant*self.normal_mask) #set entries where mask is zero to the normal mask at that point

    #     if self.bias is not None:
    #         fan_in, _ = nn.init._calculate_fan_in_and_fan_out(self.weight)
    #         bound = 1 / math.sqrt(fan_in) if fan_in > 0 else 0
    #         nn.init.uniform_(self.bias, -bound, bound)
    
    @torch.no_grad()
    def reset_parameters(self):
        d_in = self.weight.size(1)
        nn.init.normal_(self.weight, mean=0.0, std=1.0 / math.sqrt(d_in))
        self.weight.mul_(self.mask).add_((1 - self.mask) * self.mask_constant * self.normal_mask)

        if self.bias is not None:
            nn.init.zeros_(self.bias)

    def count_unused_params(self):
        return (1-self.mask.int()).sum().item()
    
    @torch.no_grad()
    def total_output_variances_init(self, d_in, use_realized_F, eps=1e-12):
        """
        v_i ~ (frozen contribution) + (trainable contribution at init)
            = [ sum_j (frozen_ij * F_ij^2)  or  n_frozen_i * kappa ]
            + n_train_i / d_in
        Assumes Cov[x] = I and trainable weights ~ N(0, 1/d_in) at init.
        """
        frozen = (1.0 - self.mask)
        n_frozen = frozen.sum(dim=1)
        n_train  = self.mask.sum(dim=1)
        kappa = (self.mask_constant ** 2)

        if use_realized_F:
            F_ij = self.mask_constant * self.normal_mask
            v_frozen = (frozen * F_ij).pow(2).sum(dim=1)
        else:
            v_frozen = n_frozen * kappa

        v_train = n_train / float(d_in)
        v = v_frozen + v_train
        return v + eps


def get_subset(num_cols, row_idx, num_sample, mask_num):
    g = torch.Generator()
    g.manual_seed(row_idx + abs(hash(str(mask_num))))
    indices = torch.arange(num_cols)
    return (indices[torch.randperm(num_cols, generator = g)[:num_sample]])

def normal_mask(out_dim,in_dim, mask_num):

    g = torch.Generator()
    g.manual_seed(abs(hash(str(mask_num))))
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
        least_zeros = num_fixed
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

    def __init__(self, v: torch.Tensor, eps: float = 1e-12):
        super().__init__()
        d = v.numel()
        sample_var = (1.0 - 1.0 / d) * v.mean()
        scale = (sample_var + eps).rsqrt()
        self.register_buffer("scale", scale)

    def forward(self, z):
        return (z - z.mean(dim=-1, keepdim=True)) * self.scale


class BatchNormLinearProxyWMLP(nn.Module):

    def __init__(self, v: torch.Tensor, eps: float = 1e-12):
        super().__init__()
        inv_std = (v + eps).rsqrt().to(torch.float32)
        self.register_buffer("inv_std", inv_std)

    def forward(self, z):
        view = [1] * z.ndim
        view[-1] = z.shape[-1]
        return z * self.inv_std.view(*view)
