"""
ResNet models with asymmetric architectures.
Refactored from lmc/models/models_resnet.py
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import math
import itertools
import random
import numpy as np
from ..core.registry import register
from .mlp import NoiseLinear


class LambdaLayer(nn.Module):
    """Lambda layer for ResNet shortcuts."""
    
    def __init__(self, lambd):
        super(LambdaLayer, self).__init__()
        self.lambd = lambd

    def forward(self, x):
        return self.lambd(x)


class SparseConv2d(nn.Module):
    """Sparse 2D convolution with fixed mask."""
    
    def __init__(self, in_channels, out_channels, mask_num, mask_type='random_subsets', 
                 do_normal_mask=True, num_fixed=6, mask_constant=0, kernel_size=3, 
                 stride=1, padding=1, bias=False):
        super().__init__()
        assert 2**(in_channels * kernel_size ** 2) >= out_channels, "out dimension too big for asymmetry"

        mask = make_conv_mask(in_channels, out_channels, kernel_size, 
                             mask_type=mask_type, num_fixed=num_fixed, mask_num=mask_num)
        self.register_buffer('mask', mask, persistent=True)

        self.weight = nn.Parameter(torch.empty((out_channels, in_channels, kernel_size, kernel_size)))
        self.weight.register_hook(lambda grad: self.mask * grad)

        if do_normal_mask:
            self.register_buffer('normal_mask', conv_normal_mask(out_channels, in_channels, kernel_size, mask_num), persistent=True)
        else:
            self.register_buffer('normal_mask', torch.ones(size=(out_channels, in_channels, kernel_size, kernel_size)), persistent=True)

        if bias:
            self.bias = nn.Parameter(torch.empty(out_channels))
        else:
            self.register_parameter('bias', None)
        
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.mask_constant = mask_constant
        self.mask_num = mask_num
        self.stride = stride
        self.padding = padding
        self.reset_parameters()
    
    def reset_parameters(self):
        nn.init.kaiming_uniform_(self.weight, a=math.sqrt(5))
        self.weight.data = (self.weight.data * self.mask + (1-self.mask) * self.mask_constant * self.normal_mask)
        
        if self.bias is not None:
            fan_in, _ = nn.init._calculate_fan_in_and_fan_out(self.weight)
            bound = 1 / math.sqrt(fan_in) if fan_in > 0 else 0
            nn.init.uniform_(self.bias, -bound, bound)
    
    def forward(self, x):
        out = F.conv2d(x, (self.weight * self.mask.detach() + (1-self.mask.detach()) * self.mask_constant * self.normal_mask), stride=self.stride, padding=self.padding)
        return out
    
    def count_unused_params(self):
        """Count unused parameters due to masking."""
        return (1 - self.mask.int()).sum().item()


class NoiseConv2d(nn.Module):
    """2D convolution with fixed Gaussian noise injection for symmetry breaking."""
    
    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1, 
                 padding=1, bias=False, mask_num=0, mask_constant=1.0):
        super().__init__()
        
        self.weight = nn.Parameter(torch.empty(out_channels, in_channels, kernel_size, kernel_size))
        if bias:
            self.bias = nn.Parameter(torch.empty(out_channels))
        else:
            self.register_parameter('bias', None)
        
        # Create fixed Gaussian noise similar to AsymSwiGLU's C initialization
        g = torch.Generator()
        g.manual_seed(abs(hash(str(mask_num) + str(0))))
        noise = torch.randn(out_channels, in_channels, kernel_size, kernel_size, generator=g)
        self.register_buffer('noise', noise, persistent=True)
        
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding
        self.mask_num = mask_num
        self.mask_constant = mask_constant
        self.reset_parameters()
    
    def reset_parameters(self):
        nn.init.kaiming_uniform_(self.weight, a=math.sqrt(5))
        if self.bias is not None:
            fan_in, _ = nn.init._calculate_fan_in_and_fan_out(self.weight)
            bound = 1 / math.sqrt(fan_in) if fan_in > 0 else 0
            nn.init.uniform_(self.bias, -bound, bound)
    
    def forward(self, x):
        # Add scaled fixed noise to weights during forward pass
        effective_weight = self.weight + self.mask_constant * self.noise
        return F.conv2d(x, effective_weight, self.bias, stride=self.stride, padding=self.padding)


class SparseBasicBlock(nn.Module):
    """Sparse ResNet basic block."""
    expansion = 1

    def __init__(self, in_planes, planes, mask_num, mask_params, stride=1, option='B'):
        super(SparseBasicBlock, self).__init__()
        
        self.conv1 = SparseConv2d(in_planes, planes, mask_num, stride=stride, 
                                 **mask_params['conv'])
        self.ln1 = nn.GroupNorm(1, planes)
        
        self.conv2 = SparseConv2d(planes, planes, mask_num + 1, stride=1, 
                                 **mask_params['conv'])
        self.ln2 = nn.GroupNorm(1, planes)

        self.shortcut = nn.Sequential()
        if stride != 1 or in_planes != planes:
            if option == 'A':
                self.shortcut = LambdaLayer(lambda x:
                    F.pad(x[:, :, ::2, ::2], (0, 0, 0, 0, planes//4, planes//4), "constant", 0))
            elif option == 'B':
                self.shortcut = nn.Sequential(
                    SparseConv2d(in_planes, self.expansion * planes, mask_num + 2, 
                               kernel_size=1, stride=stride, padding=0, bias=False, 
                               **mask_params['skip']),
                    nn.GroupNorm(1, self.expansion * planes)
                )

    def forward(self, x):
        out = F.relu(self.ln1(self.conv1(x)))
        out = self.ln2(self.conv2(out))
        out += self.shortcut(x)
        out = F.relu(out)
        return out


class BasicBlock(nn.Module):
    """Standard ResNet basic block."""
    expansion = 1

    def __init__(self, in_planes, planes, stride=1, option='B'):
        super(BasicBlock, self).__init__()
        
        self.conv1 = nn.Conv2d(in_planes, planes, kernel_size=3, stride=stride, 
                              padding=1, bias=False)
        self.ln1 = nn.GroupNorm(1, planes)
        
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=1, 
                              padding=1, bias=False)
        self.ln2 = nn.GroupNorm(1, planes)

        self.shortcut = nn.Sequential()
        if stride != 1 or in_planes != planes:
            if option == 'A':
                self.shortcut = LambdaLayer(lambda x:
                    F.pad(x[:, :, ::2, ::2], (0, 0, 0, 0, planes//4, planes//4), "constant", 0))
            elif option == 'B':
                self.shortcut = nn.Sequential(
                    nn.Conv2d(in_planes, self.expansion * planes, kernel_size=1, 
                             stride=stride, bias=False),
                    nn.GroupNorm(1, self.expansion * planes)
                )

    def forward(self, x):
        out = F.relu(self.ln1(self.conv1(x)))
        out = self.ln2(self.conv2(out))
        out += self.shortcut(x)
        out = F.relu(out)
        return out


class NoiseBasicBlock(nn.Module):
    """ResNet basic block with noise injection for symmetry breaking."""
    expansion = 1

    def __init__(self, in_planes, planes, mask_num, mask_params, stride=1, option='B'):
        super(NoiseBasicBlock, self).__init__()
        
        # Get mask parameters for this block, filtering out SparseConv2d-specific params
        conv_params = mask_params.get('conv', {'mask_constant': 1.0})
        skip_params = mask_params.get('skip', {'mask_constant': 1.0})
        
        # Filter out parameters that are not relevant for NoiseConv2d
        noise_conv_params = {k: v for k, v in conv_params.items() if k in ['mask_constant']}
        noise_skip_params = {k: v for k, v in skip_params.items() if k in ['mask_constant']}
        
        self.conv1 = NoiseConv2d(in_planes, planes, kernel_size=3, stride=stride, 
                                padding=1, bias=False, mask_num=mask_num, **noise_conv_params)
        self.ln1 = nn.GroupNorm(1, planes)
        
        self.conv2 = NoiseConv2d(planes, planes, kernel_size=3, stride=1, 
                                padding=1, bias=False, mask_num=mask_num + 1, **noise_conv_params)
        self.ln2 = nn.GroupNorm(1, planes)

        self.shortcut = nn.Sequential()
        if stride != 1 or in_planes != planes:
            if option == 'A':
                self.shortcut = LambdaLayer(lambda x:
                    F.pad(x[:, :, ::2, ::2], (0, 0, 0, 0, planes//4, planes//4), "constant", 0))
            elif option == 'B':
                self.shortcut = nn.Sequential(
                    NoiseConv2d(in_planes, self.expansion * planes, kernel_size=1, 
                               stride=stride, padding=0, bias=False, mask_num=mask_num + 2, **noise_skip_params),
                    nn.GroupNorm(1, self.expansion * planes)
                )

    def forward(self, x):
        out = F.relu(self.ln1(self.conv1(x)))
        out = self.ln2(self.conv2(out))
        out += self.shortcut(x)
        out = F.relu(out)
        return out


def _weights_init(m):
    """Initialize weights for ResNet."""
    if isinstance(m, nn.Linear) or isinstance(m, nn.Conv2d):
        nn.init.kaiming_normal_(m.weight)


@register('model', 'resnet_standard')
class ResNet(nn.Module):
    """Standard ResNet."""
    
    def __init__(self, block, num_blocks, w=1, num_classes=10):
        super(ResNet, self).__init__()
        self.in_planes = 16 * w

        self.conv1 = nn.Conv2d(3, 16*w, kernel_size=3, stride=1, padding=1, bias=False)
        self.ln1 = nn.GroupNorm(1, 16*w)

        self.layer1 = self._make_layer(block, 16*w, num_blocks[0], stride=1)
        self.layer2 = self._make_layer(block, 32*w, num_blocks[1], stride=2)
        self.layer3 = self._make_layer(block, 64*w, num_blocks[2], stride=2)

        self.linear = nn.Linear(64*w, num_classes)
        self.apply(_weights_init)

    def _make_layer(self, block, planes, num_blocks, stride):
        strides = [stride] + [1] * (num_blocks - 1)
        layers = []
        for stride in strides:
            layers.append(block(self.in_planes, planes, stride))
            self.in_planes = planes * block.expansion
        return nn.Sequential(*layers)

    def forward(self, x):
        out = F.relu(self.ln1(self.conv1(x)))
        out = self.layer1(out)
        out = self.layer2(out)
        out = self.layer3(out)
        out = F.avg_pool2d(out, out.size()[3])
        out = out.view(out.size(0), -1)
        out = self.linear(out)
        return out


@register('model', 'resnet_w_asym')
class WResNet(nn.Module):
    """W-Asymmetric ResNet with sparse convolutions."""
    
    def __init__(self, block, num_blocks, mask_params, w=1, num_classes=10):
        super(WResNet, self).__init__()
        self.in_planes = 16 * w
        mask_num = 0
        
        self.conv1 = SparseConv2d(3, 16*w, mask_num, kernel_size=3, stride=1, 
                                 padding=1, bias=False, **mask_params['conv_f'])
        self.ln1 = nn.GroupNorm(1, 16*w)
        mask_num += 1

        self.layer1 = self._make_layer(block, mask_num, 16*w, num_blocks[0], 
                                     stride=1, mask_params=mask_params['conv_1'])
        mask_num += num_blocks[0] * 3

        self.layer2 = self._make_layer(block, mask_num, 32*w, num_blocks[1], 
                                     stride=2, mask_params=mask_params['conv_2'])
        mask_num += num_blocks[1] * 3

        self.layer3 = self._make_layer(block, mask_num, 64*w, num_blocks[2], 
                                     stride=2, mask_params=mask_params['conv_3'])
        mask_num += num_blocks[2] * 3

        self.linear = SparseLinear(64*w, num_classes, mask_num=mask_num, bias=False, **mask_params['linear'])
        self.apply(_weights_init)

    def _make_layer(self, block, mask_num, planes, num_blocks, stride, mask_params):
        strides = [stride] + [1] * (num_blocks - 1)
        layers = []
        for stride in strides:
            layers.append(block(self.in_planes, planes, mask_num, mask_params, stride))
            mask_num += 3
            self.in_planes = planes * block.expansion
        return nn.Sequential(*layers)

    def count_unused_params(self):
        """Count unused parameters due to masking."""
        count = 0
        for node in self.modules():
            if hasattr(node, 'mask'):
                count += (1-node.mask).sum()
        return count.item()

    def forward(self, x):
        out = F.relu(self.ln1(self.conv1(x)))
        out = self.layer1(out)
        out = self.layer2(out)
        out = self.layer3(out)
        out = F.avg_pool2d(out, out.size()[3])
        out = out.view(out.size(0), -1)
        out = self.linear(out)
        return out


@register('model', 'resnet_noise_asym')
class NoiseResNet(nn.Module):
    """Noise-Asymmetric ResNet with noise injection for symmetry breaking."""
    
    def __init__(self, block, num_blocks, mask_params, w=1, num_classes=10):
        super(NoiseResNet, self).__init__()
        self.in_planes = 16 * w
        mask_num = 0
        
        # Get mask parameters for conv_f, filtering out SparseConv2d-specific params
        conv_f_params = mask_params.get('conv_f', {'mask_constant': 1.0})
        noise_conv_f_params = {k: v for k, v in conv_f_params.items() if k in ['mask_constant']}
        self.conv1 = NoiseConv2d(3, 16*w, kernel_size=3, stride=1, 
                                padding=1, bias=False, mask_num=mask_num, **noise_conv_f_params)
        self.ln1 = nn.GroupNorm(1, 16*w)
        mask_num += 1

        self.layer1 = self._make_layer(block, mask_num, 16*w, num_blocks[0], 
                                     stride=1, mask_params=mask_params['conv_1'])
        mask_num += num_blocks[0] * 3

        self.layer2 = self._make_layer(block, mask_num, 32*w, num_blocks[1], 
                                     stride=2, mask_params=mask_params['conv_2'])
        mask_num += num_blocks[1] * 3

        self.layer3 = self._make_layer(block, mask_num, 64*w, num_blocks[2], 
                                     stride=2, mask_params=mask_params['conv_3'])
        mask_num += num_blocks[2] * 3

        # Get mask parameters for linear layer, filtering out SparseLinear-specific params
        linear_params = mask_params.get('linear', {'mask_constant': 1.0})
        noise_linear_params = {k: v for k, v in linear_params.items() if k in ['mask_constant']}
        self.linear = NoiseLinear(64*w, num_classes, mask_num=mask_num, **noise_linear_params)
        self.apply(_weights_init)

    def _make_layer(self, block, mask_num, planes, num_blocks, stride, mask_params):
        strides = [stride] + [1] * (num_blocks - 1)
        layers = []
        for stride in strides:
            layers.append(block(self.in_planes, planes, mask_num, mask_params, stride))
            mask_num += 3
            self.in_planes = planes * block.expansion
        return nn.Sequential(*layers)

    def forward(self, x):
        out = F.relu(self.ln1(self.conv1(x)))
        out = self.layer1(out)
        out = self.layer2(out)
        out = self.layer3(out)
        out = F.avg_pool2d(out, out.size()[3])
        out = out.view(out.size(0), -1)
        out = self.linear(out)
        return out


# Convenience functions for different ResNet sizes
@register('model', 'resnet20')
def resnet20(w=1, num_classes=10):
    """ResNet-20."""
    return ResNet(BasicBlock, [3, 3, 3], w=w, num_classes=num_classes)


@register('model', 'resnet32')
def resnet32(w=1, num_classes=10):
    """ResNet-32."""
    return ResNet(BasicBlock, [5, 5, 5], w=w, num_classes=num_classes)


@register('model', 'resnet56')
def resnet56(w=1, num_classes=10):
    """ResNet-56."""
    return ResNet(BasicBlock, [9, 9, 9], w=w, num_classes=num_classes)


@register('model', 'resnet110')
def resnet110(w=1, num_classes=10):
    """ResNet-110."""
    return ResNet(BasicBlock, [18, 18, 18], w=w, num_classes=num_classes)


@register('model', 'resnet_w_asym_20')
def w_resnet20(mask_params, w=1, num_classes=10):
    """W-Asymmetric ResNet-20."""
    return WResNet(SparseBasicBlock, [3, 3, 3], mask_params, w=w, num_classes=num_classes)


@register('model', 'resnet_w_asym_32')
def w_resnet32(mask_params, w=1, num_classes=10):
    """W-Asymmetric ResNet-32."""
    return WResNet(SparseBasicBlock, [5, 5, 5], mask_params, w=w, num_classes=num_classes)


@register('model', 'resnet_w_asym_56')
def w_resnet56(mask_params, w=1, num_classes=10):
    """W-Asymmetric ResNet-56."""
    return WResNet(SparseBasicBlock, [9, 9, 9], mask_params, w=w, num_classes=num_classes)


@register('model', 'resnet_w_asym_110')
def w_resnet110(mask_params, w=1, num_classes=10):
    """W-Asymmetric ResNet-110."""
    return WResNet(SparseBasicBlock, [18, 18, 18], mask_params, w=w, num_classes=num_classes)


@register('model', 'resnet_noise_asym_20')
def noise_resnet20(mask_params, w=1, num_classes=10):
    """Noise-Asymmetric ResNet-20."""
    return NoiseResNet(NoiseBasicBlock, [3, 3, 3], mask_params, w=w, num_classes=num_classes)


@register('model', 'resnet_noise_asym_32')
def noise_resnet32(mask_params, w=1, num_classes=10):
    """Noise-Asymmetric ResNet-32."""
    return NoiseResNet(NoiseBasicBlock, [5, 5, 5], mask_params, w=w, num_classes=num_classes)


@register('model', 'resnet_noise_asym_56')
def noise_resnet56(mask_params, w=1, num_classes=10):
    """Noise-Asymmetric ResNet-56."""
    return NoiseResNet(NoiseBasicBlock, [9, 9, 9], mask_params, w=w, num_classes=num_classes)


@register('model', 'resnet_noise_asym_110')
def noise_resnet110(mask_params, w=1, num_classes=10):
    """Noise-Asymmetric ResNet-110."""
    return NoiseResNet(NoiseBasicBlock, [18, 18, 18], mask_params, w=w, num_classes=num_classes)


# Convenience function for backward compatibility
def create_resnet(symmetry, depth, w=1, mask_params=None, num_classes=10):
    """Create ResNet based on symmetry type and depth.
    
    Args:
        symmetry: 0=Standard, 1=W-Asym, 3=Noise-Asym
        depth: ResNet depth (20, 32, 56, 110)
        w: Width multiplier
        mask_params: Mask parameters for W-Asym and Noise-Asym
        num_classes: Number of output classes
    """
    if symmetry == 0:
        if depth == 20:
            return resnet20(w=w, num_classes=num_classes)
        elif depth == 32:
            return resnet32(w=w, num_classes=num_classes)
        elif depth == 56:
            return resnet56(w=w, num_classes=num_classes)
        elif depth == 110:
            return resnet110(w=w, num_classes=num_classes)
        else:
            raise ValueError(f"Invalid depth: {depth}")
    elif symmetry == 1:
        if mask_params is None:
            raise ValueError("mask_params required for W-Asym ResNet")
        if depth == 20:
            return w_resnet20(mask_params, w=w, num_classes=num_classes)
        elif depth == 32:
            return w_resnet32(mask_params, w=w, num_classes=num_classes)
        elif depth == 56:
            return w_resnet56(mask_params, w=w, num_classes=num_classes)
        elif depth == 110:
            return w_resnet110(mask_params, w=w, num_classes=num_classes)
        else:
            raise ValueError(f"Invalid depth: {depth}")
    elif symmetry == 3:
        if mask_params is None:
            raise ValueError("mask_params required for Noise-Asym ResNet")
        if depth == 20:
            return noise_resnet20(mask_params, w=w, num_classes=num_classes)
        elif depth == 32:
            return noise_resnet32(mask_params, w=w, num_classes=num_classes)
        elif depth == 56:
            return noise_resnet56(mask_params, w=w, num_classes=num_classes)
        elif depth == 110:
            return noise_resnet110(mask_params, w=w, num_classes=num_classes)
        else:
            raise ValueError(f"Invalid depth: {depth}")
    else:
        raise ValueError(f"Invalid symmetry type: {symmetry}")


# Original LMC mask generation functions
def make_conv_mask(in_channels, out_channels, kernel_size, mask_num, mask_type='random_subsets', num_fixed=6):
    """Create sparse mask for convolution (original LMC implementation)."""
    if mask_type == 'densest':
        mask = torch.ones(size=(out_channels, in_channels, kernel_size, kernel_size))
        weights_per_out_channel = in_channels * kernel_size**2
        flattened_to_3d_index = lambda ind: (ind // kernel_size**2, (ind//kernel_size)%kernel_size, ind%kernel_size)
        out_channel_idx = 1
        if out_channels == 1:
            return mask

        for nz in range(1, weights_per_out_channel):
            for zeros_in_out_channel in itertools.combinations(range(weights_per_out_channel), nz):
                for zero_ind in map(flattened_to_3d_index, zeros_in_out_channel):
                    mask[out_channel_idx][zero_ind] = 0
                out_channel_idx += 1
                if out_channel_idx >= out_channels:
                    return mask

    elif mask_type == 'bound_zeros':
        mask = torch.ones(size=(out_channels, in_channels, kernel_size, kernel_size))
        weights_per_out_channel = in_channels * kernel_size**2
        flattened_to_3d_index = lambda ind: (ind // kernel_size**2, (ind//kernel_size)%kernel_size, ind%kernel_size)
        out_channel_idx = 0
        least_zeros = num_fixed
        for nz in range(least_zeros, weights_per_out_channel):
            for zeros_in_out_channel in itertools.combinations(range(weights_per_out_channel), nz):
                for zero_ind in map(flattened_to_3d_index, zeros_in_out_channel):
                    mask[out_channel_idx][zero_ind] = 0
                out_channel_idx += 1
                if out_channel_idx >= out_channels:
                    return mask

    elif mask_type == 'random_subsets':
        mask = torch.ones(size=(out_channels, in_channels, kernel_size, kernel_size))
        weights_per_out_channel = in_channels * kernel_size**2
        least_zeros = num_fixed

        flattened_to_3d_index = lambda ind: (ind // kernel_size**2, (ind//kernel_size)%kernel_size, ind%kernel_size)
        for out_channel_idx in range(out_channels):
            zeros_in_out_channel = get_subset(weights_per_out_channel, out_channel_idx, least_zeros, mask_num)
            for zero_ind in map(flattened_to_3d_index, zeros_in_out_channel):
                mask[out_channel_idx][zero_ind] = 0
        return mask
    elif mask_type == 'filter_random_subsets':
        mask = torch.ones(size=(out_channels, in_channels, kernel_size, kernel_size))
        least_zeros = num_fixed
        
        for out_channel_idx in range(out_channels):
            zeros_in_out_channel = get_subset(in_channels, out_channel_idx, least_zeros, mask_num)
            for zero_ind in zeros_in_out_channel:
                mask[out_channel_idx, zero_ind, :, :] = 0
        return mask
    elif mask_type == 'none':
        return torch.ones(size=(out_channels, in_channels, kernel_size, kernel_size))


def conv_normal_mask(out_channels, in_channels, kernel_size, mask_num):
    """Create normal mask for initialization (original LMC implementation)."""
    g = torch.Generator()
    g.manual_seed(abs(hash(str(mask_num) + str(0))))
    return torch.randn(size=(out_channels, in_channels, kernel_size, kernel_size), generator=g)


def get_subset(num_cols, row_idx, num_sample, mask_num):
    """Get random subset of indices (original LMC implementation)."""
    g = torch.Generator()
    g.manual_seed(row_idx + abs(hash(str(mask_num))))
    indices = torch.arange(num_cols)
    return (indices[torch.randperm(num_cols, generator=g)[:num_sample]])


def make_mask(in_dim, out_dim, mask_num=0, num_fixed=6, mask_type='densest'):
    """Create mask for linear layers (original LMC implementation)."""
    # out_dim x in_dim matrix
    # where each row is unique
    assert out_dim < 2**(in_dim)
    assert in_dim > 0 and out_dim > 0

    if mask_type == 'densest':
        mask = torch.ones(out_dim, in_dim)
        mask[0, :] = 1  # first row is dense
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


def normal_mask(out_dim, in_dim, mask_num):
    """Create normal mask for linear layers (original LMC implementation)."""
    g = torch.Generator()
    g.manual_seed(abs(hash(str(mask_num))))
    return torch.randn(size=(out_dim, in_dim), generator=g)


class SparseLinear(nn.Module):
    """Sparse linear layer matching original LMC implementation."""
    def __init__(self, in_dim, out_dim, bias=True, mask_type='densest', mask_constant=1, mask_num=0, num_fixed=6, do_normal_mask=True):
        super().__init__()
        assert out_dim < 2**in_dim, 'out dim cannot be much higher than in dim'
        mask = make_mask(in_dim, out_dim, mask_type=mask_type, num_fixed=num_fixed, mask_num=mask_num)

        self.register_buffer('mask', mask, persistent=True)
        self.weight = nn.Parameter(torch.empty((out_dim, in_dim)))

        if do_normal_mask:
            self.register_buffer('normal_mask', normal_mask(out_dim, in_dim, mask_num), persistent=True)
        else:
            self.register_buffer('normal_mask', torch.ones(size=(out_dim, in_dim)), persistent=True)

        self.weight.register_hook(lambda grad: self.mask*grad) # zeros out gradients for masked parts

        if bias:
            self.bias = nn.Parameter(torch.empty(out_dim))
        else:
            self.register_parameter('bias', None)

        self.mask_constant = mask_constant
        self.mask_num = mask_num
        self.num_fixed = num_fixed
        self.reset_parameters()
    
    def reset_parameters(self):
        nn.init.kaiming_uniform_(self.weight, a=math.sqrt(5))
        self.weight.data = (self.weight.data * self.mask + (1-self.mask) * self.mask_constant * self.normal_mask)

        if self.bias is not None:
            fan_in, _ = nn.init._calculate_fan_in_and_fan_out(self.weight)
            bound = 1 / math.sqrt(fan_in) if fan_in > 0 else 0
            nn.init.uniform_(self.bias, -bound, bound)

    def forward(self, x):
        return F.linear(x, (self.weight * self.mask.detach() + (1-self.mask.detach()) * self.mask_constant * self.normal_mask), self.bias)

    def count_unused_params(self):
        return (1-self.mask.int()).sum().item()
