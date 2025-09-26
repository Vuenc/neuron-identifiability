"""
ResNet models with asymmetric architectures.
Refactored from lmc/models/models_resnet.py
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import math
import itertools
from ..core.registry import register


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

        mask = self._make_conv_mask(in_channels, out_channels, kernel_size, 
                                  mask_type=mask_type, num_fixed=num_fixed, mask_num=mask_num)
        self.register_buffer('mask', mask, persistent=True)

        self.weight = nn.Parameter(torch.empty((out_channels, in_channels, kernel_size, kernel_size)))
        self.weight.register_hook(lambda grad: self.mask * grad)

        if do_normal_mask:
            normal_mask = self._conv_normal_mask(out_channels, in_channels, kernel_size, mask_num)
            self.register_buffer('normal_mask', normal_mask, persistent=True)
        else:
            self.register_buffer('normal_mask', torch.ones_like(mask), persistent=True)

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
    
    def _make_conv_mask(self, in_channels, out_channels, kernel_size, mask_num, 
                       mask_type='random_subsets', num_fixed=6):
        """Create sparse mask for convolution."""
        mask = torch.ones(out_channels, in_channels, kernel_size, kernel_size)
        
        if mask_type == 'random_subsets':
            g = torch.Generator()
            g.manual_seed(abs(hash(str(mask_num))))
            
            for i in range(out_channels):
                total_weights = in_channels * kernel_size * kernel_size
                if num_fixed < total_weights:
                    indices = torch.randperm(total_weights, generator=g)[:num_fixed]
                    flat_mask = torch.zeros(total_weights)
                    flat_mask[indices] = 1
                    mask[i] = flat_mask.view(in_channels, kernel_size, kernel_size)
        
        return mask
    
    def _conv_normal_mask(self, out_channels, in_channels, kernel_size, mask_num):
        """Create normal mask for initialization."""
        g = torch.Generator()
        g.manual_seed(abs(hash(str(mask_num) + 'conv_normal')))
        return torch.randn(out_channels, in_channels, kernel_size, kernel_size, generator=g)
    
    def reset_parameters(self):
        """Initialize parameters."""
        nn.init.kaiming_uniform_(self.weight, a=math.sqrt(5))
        self.weight.data = (self.weight.data * self.mask + 
                           (1 - self.mask) * self.mask_constant * self.normal_mask)
        
        if self.bias is not None:
            fan_in, _ = nn.init._calculate_fan_in_and_fan_out(self.weight)
            bound = 1 / math.sqrt(fan_in) if fan_in > 0 else 0
            nn.init.uniform_(self.bias, -bound, bound)
    
    def forward(self, x):
        self.weight.data = (self.weight.data * self.mask + 
                           (1 - self.mask) * self.mask_constant * self.normal_mask)
        return F.conv2d(x, self.weight, bias=self.bias, stride=self.stride, padding=self.padding)
    
    def count_unused_params(self):
        """Count unused parameters due to masking."""
        return (1 - self.mask.int()).sum().item()


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

        self.linear = nn.Linear(64*w, num_classes)
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
        for module in self.modules():
            if hasattr(module, 'count_unused_params'):
                count += module.count_unused_params()
        return count

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


# Convenience function for backward compatibility
def create_resnet(symmetry, depth, w=1, mask_params=None, num_classes=10, fixed_masks=None):
    """Create ResNet based on symmetry type and depth.
    
    Args:
        symmetry: 0=Standard, 1=W-Asym
        depth: ResNet depth (20, 32, 56, 110)
        w: Width multiplier
        mask_params: Mask parameters for W-Asym
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
        if fixed_masks is not None:
            print("Warning: fixed_masks not yet implemented for ResNet models")
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
    else:
        raise ValueError(f"Invalid symmetry type: {symmetry}")
