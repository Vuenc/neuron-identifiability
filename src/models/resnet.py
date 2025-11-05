"""
ResNet models with asymmetric architectures.
Refactored from lmc/models/models_resnet.py
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import math
import itertools
import copy
from omegaconf import OmegaConf
from ..core.registry import register
from .mlp import NoiseLinear, SparseLinear


class LambdaLayer(nn.Module):
    """Lambda layer for ResNet shortcuts."""
    
    def __init__(self, lambd):
        super(LambdaLayer, self).__init__()
        self.lambd = lambd

    def forward(self, x):
        return self.lambd(x)


class SparseConv2d(nn.Module):
    """Sparse 2D convolution with fixed mask."""
    
    def __init__(self, in_channels, out_channels, mask_num, mask_rng: torch.Generator, mask_type='random_subsets', 
                 do_normal_mask=True, num_fixed=6, mask_constant=0, kernel_size=3, 
                 stride=1, padding=1, bias=False):
        super().__init__()
        assert 2**(in_channels * kernel_size ** 2) >= out_channels, "out dimension too big for asymmetry"

        mask = make_conv_mask(in_channels, out_channels, kernel_size, 
                             mask_type=mask_type, num_fixed=num_fixed, mask_rng=mask_rng)
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
    
    def __init__(self, in_channels, out_channels, mask_rng: torch.Generator, kernel_size=3, stride=1, 
                 padding=1, bias=False, mask_num=0, mask_constant=1.0, **kwargs):
        super().__init__()
        
        self.weight = nn.Parameter(torch.empty(out_channels, in_channels, kernel_size, kernel_size))
        if bias:
            self.bias = nn.Parameter(torch.empty(out_channels))
        else:
            self.register_parameter('bias', None)
        
        # Create fixed Gaussian noise similar to AsymSwiGLU's C initialization
        noise = torch.randn(out_channels, in_channels, kernel_size, kernel_size, generator=mask_rng)
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

    def __init__(self, in_planes, planes, mask_num, mask_params, mask_rng: torch.Generator, stride=1, option='B'):
        super(NoiseBasicBlock, self).__init__()
        
        self.conv1 = NoiseConv2d(in_planes, planes, kernel_size=3, stride=stride, 
                                padding=1, bias=False, mask_num=mask_num, **mask_params['conv'], mask_rng=mask_rng)
        self.ln1 = nn.GroupNorm(1, planes)
        
        self.conv2 = NoiseConv2d(planes, planes, kernel_size=3, stride=1, 
                                padding=1, bias=False, mask_num=mask_num + 1, **mask_params['conv'], mask_rng=mask_rng)
        self.ln2 = nn.GroupNorm(1, planes)

        self.shortcut = nn.Sequential()
        if stride != 1 or in_planes != planes:
            if option == 'A':
                self.shortcut = LambdaLayer(lambda x:
                    F.pad(x[:, :, ::2, ::2], (0, 0, 0, 0, planes//4, planes//4), "constant", 0))
            elif option == 'B':
                self.shortcut = nn.Sequential(
                    NoiseConv2d(in_planes, self.expansion * planes, kernel_size=1, 
                               stride=stride, padding=0, bias=False, mask_num=mask_num + 2, **mask_params['skip'], mask_rng=mask_rng),
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
    
    def __init__(self, block, num_blocks, mask_params, w=1, num_classes=10, mask_seed=None):
        super(WResNet, self).__init__()
        self.in_planes = 16 * w
        mask_num = 0
        
        def get_mask_params(layer_key):
            default_params = mask_params.get('default', None)
            if default_params is None:
                default_params = {
                    'mask_constant': 1.0,
                    'mask_type': 'random_subsets',
                    'do_normal_mask': True,
                    'num_fixed': 64
                }
            if layer_key in ['conv_1', 'conv_2', 'conv_3']:
                if layer_key in mask_params:
                    output = {k: {**default_params, **mask_params[layer_key][k]} for k in ['conv', 'skip']}
                else:
                    output = {k: default_params for k in ['conv', 'skip']}
                print(f"Mask params for {layer_key}.conv: {output['conv']}")
                print(f"Mask params for {layer_key}.skip: {output['skip']}")
            else:
                if layer_key in mask_params:
                    output = {**default_params, **mask_params[layer_key]}
                else:
                    output = default_params
                print(f"Mask params for {layer_key}: {output}")
            return output
    
        mask_rng = torch.Generator()
        if mask_seed is not None:
            mask_rng.manual_seed(mask_seed)
        
        self.conv1 = SparseConv2d(3, 16*w, mask_num, kernel_size=3, stride=1, 
                                 padding=1, bias=False, mask_rng=mask_rng, **get_mask_params('conv_f'))
        self.ln1 = nn.GroupNorm(1, 16*w)
        mask_num += 1

        self.layer1 = self._make_layer(block, mask_num, 16*w, num_blocks[0], 
                                     stride=1, mask_params=get_mask_params('conv_1'), mask_rng=mask_rng)
        mask_num += num_blocks[0] * 3

        self.layer2 = self._make_layer(block, mask_num, 32*w, num_blocks[1], 
                                     stride=2, mask_params=get_mask_params('conv_2'), mask_rng=mask_rng)
        mask_num += num_blocks[1] * 3

        self.layer3 = self._make_layer(block, mask_num, 64*w, num_blocks[2], 
                                     stride=2, mask_params=get_mask_params('conv_3'), mask_rng=mask_rng)
        mask_num += num_blocks[2] * 3

        self.linear = SparseLinear(64*w, num_classes, mask_rng=mask_rng, mask_num=mask_num, bias=True, **get_mask_params('linear'))
        self.apply(_weights_init)

    def _make_layer(self, block, mask_num, planes, num_blocks, stride, mask_params, mask_rng: torch.Generator):
        strides = [stride] + [1] * (num_blocks - 1)
        layers = []
        for stride in strides:
            layers.append(block(self.in_planes, planes, mask_num, mask_params, stride, mask_rng=mask_rng))
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
    
    def __init__(self, block, num_blocks, mask_params, w=1, num_classes=10, mask_seed=None):
        super(NoiseResNet, self).__init__()
        self.in_planes = 16 * w
        mask_num = 0
        
        def get_mask_params(layer_key):
            
            def remove_unused_keys(params):
                # if params is None: return None
                return {k: v for k, v in params.items() if k == 'mask_constant'}
            
            default_params = remove_unused_keys(mask_params.get('default', None))
            if default_params is None:
                print("No network-wide default params specified, falling back to global defaults")
                default_params = {'mask_constant': 1.0}
            if layer_key in ['conv_1', 'conv_2', 'conv_3']:
                if layer_key in mask_params:
                    output = {k: {**default_params, **remove_unused_keys(mask_params[layer_key][k])} for k in ['conv', 'skip']}
                else:
                    output = {k: default_params for k in ['conv', 'skip']}
                print(f"Mask params for {layer_key}.conv: {output['conv']}")
                print(f"Mask params for {layer_key}.skip: {output['skip']}")
            else:
                if layer_key in mask_params:
                    output = {**default_params, **remove_unused_keys(mask_params[layer_key])}
                else:
                    output = default_params
                print(f"Mask params for {layer_key}: {output}")
            return output
    
        mask_rng = torch.Generator()
        if mask_seed:
            mask_rng.manual_seed(mask_seed)
        
        self.conv1 = NoiseConv2d(3, 16*w, kernel_size=3, stride=1, 
                                padding=1, bias=False, mask_num=mask_num, mask_rng=mask_rng, **get_mask_params('conv_f'))
        self.ln1 = nn.GroupNorm(1, 16*w)
        mask_num += 1

        self.layer1 = self._make_layer(block, mask_num, 16*w, num_blocks[0], 
                                     stride=1, mask_params=get_mask_params('conv_1'), mask_rng=mask_rng)
        mask_num += num_blocks[0] * 3

        self.layer2 = self._make_layer(block, mask_num, 32*w, num_blocks[1], 
                                     stride=2, mask_params=get_mask_params('conv_2'), mask_rng=mask_rng)
        mask_num += num_blocks[1] * 3

        self.layer3 = self._make_layer(block, mask_num, 64*w, num_blocks[2], 
                                     stride=2, mask_params=get_mask_params('conv_3'), mask_rng=mask_rng)
        mask_num += num_blocks[2] * 3

        self.linear = NoiseLinear(64*w, num_classes, mask_num=mask_num, mask_rng=mask_rng, **get_mask_params('linear'))
        self.apply(_weights_init)

    def _make_layer(self, block, mask_num, planes, num_blocks, stride, mask_params, mask_rng: torch.Generator):
        strides = [stride] + [1] * (num_blocks - 1)
        layers = []
        for stride in strides:
            layers.append(block(self.in_planes, planes, mask_num, mask_params, stride, mask_rng=mask_rng))
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
def resnet20(**kwargs):
    """ResNet-20."""
    return ResNet(BasicBlock, [3, 3, 3], **kwargs)


@register('model', 'resnet32')
def resnet32(**kwargs):
    """ResNet-32."""
    return ResNet(BasicBlock, [5, 5, 5], **kwargs)


@register('model', 'resnet56')
def resnet56(**kwargs):
    """ResNet-56."""
    return ResNet(BasicBlock, [9, 9, 9], **kwargs)


@register('model', 'resnet110')
def resnet110(**kwargs):
    """ResNet-110."""
    return ResNet(BasicBlock, [18, 18, 18], **kwargs)


@register('model', 'resnet_w_asym_20')
def w_resnet20(mask_params, **kwargs):
    """W-Asymmetric ResNet-20."""
    return WResNet(SparseBasicBlock, [3, 3, 3], mask_params, **kwargs)


@register('model', 'resnet_w_asym_32')
def w_resnet32(mask_params, **kwargs):
    """W-Asymmetric ResNet-32."""
    return WResNet(SparseBasicBlock, [5, 5, 5], mask_params, **kwargs)


@register('model', 'resnet_w_asym_56')
def w_resnet56(mask_params, **kwargs):
    """W-Asymmetric ResNet-56."""
    return WResNet(SparseBasicBlock, [9, 9, 9], mask_params, **kwargs)


@register('model', 'resnet_w_asym_110')
def w_resnet110(mask_params, **kwargs):
    """W-Asymmetric ResNet-110."""
    return WResNet(SparseBasicBlock, [18, 18, 18], mask_params, **kwargs)


@register('model', 'resnet_noise_asym_20')
def noise_resnet20(mask_params, **kwargs):
    """Noise-Asymmetric ResNet-20."""
    return NoiseResNet(NoiseBasicBlock, [3, 3, 3], mask_params, **kwargs)


@register('model', 'resnet_noise_asym_32')
def noise_resnet32(mask_params, **kwargs):
    """Noise-Asymmetric ResNet-32."""
    return NoiseResNet(NoiseBasicBlock, [5, 5, 5], mask_params, **kwargs)


@register('model', 'resnet_noise_asym_56')
def noise_resnet56(mask_params, **kwargs):
    """Noise-Asymmetric ResNet-56."""
    return NoiseResNet(NoiseBasicBlock, [9, 9, 9], mask_params, **kwargs)


@register('model', 'resnet_noise_asym_110')
def noise_resnet110(mask_params, **kwargs):
    """Noise-Asymmetric ResNet-110."""
    return NoiseResNet(NoiseBasicBlock, [18, 18, 18], mask_params, **kwargs)


def _apply_n_mul_to_mask_params(mask_params, n_mul, is_top_level=True):
    """Recursively apply n_mul multiplier to all num_fixed values in mask_params.
    
    Args:
        mask_params: Dictionary containing mask parameters (may be nested, may be OmegaConf DictConfig)
        n_mul: Multiplier to apply to num_fixed values
        is_top_level: Whether this is the top-level call (needed to convert OmegaConf only once)
        
    Returns:
        New dictionary with num_fixed values multiplied by n_mul
    """
    # Convert OmegaConf DictConfig to regular dict if needed (only at top level)
    if is_top_level:
        try:
            # Try to convert if it's an OmegaConf object
            result = OmegaConf.to_container(mask_params, resolve=True)
        except (AttributeError, TypeError, ValueError):
            # If it's already a regular dict or not OmegaConf, just deepcopy
            result = copy.deepcopy(mask_params)
    else:
        # For nested dicts, they should already be regular dicts after top-level conversion
        result = copy.deepcopy(mask_params)
    
    if isinstance(result, dict):
        for key, value in list(result.items()):  # Use list() to avoid mutation during iteration
            if key == 'num_fixed' and isinstance(value, (int, float)):
                result[key] = int(value * n_mul)
            elif isinstance(value, dict):
                result[key] = _apply_n_mul_to_mask_params(value, n_mul, is_top_level=False)
    
    return result


def _initialize_from_asym_resnet(symmetry, depth, w, mask_params, num_classes, n_mul):
    """Initialize a symmetry type 0 (standard) ResNet with weights from symmetry type 1 or 3 (W-Asym or Noise-Asym).
    
    This creates an asym model using create_resnet, runs it through forward to extract effective weights,
    then copies them into a standard model.
    
    Args:
        symmetry: 1=W-Asym, 3=Noise-Asym
        depth: ResNet depth (20, 32, 56, 110)
        w: Width multiplier
        mask_params: Mask parameters for asym model (used for initialization)
        num_classes: Number of output classes
        n_mul: Multiplier for num_fixed values
        
    Returns:
        Standard ResNet with weights initialized from asym model
    """
    # Create asym model using the normal create_resnet function
    asym_model = create_resnet(symmetry=symmetry, depth=depth, w=w, mask_params=mask_params, 
                               num_classes=num_classes, n_mul=n_mul, asym_init_only=False)
    
    # Create standard model
    standard_model = create_resnet(symmetry=0, depth=depth, w=w, mask_params=None, 
                                  num_classes=num_classes, n_mul=n_mul, asym_init_only=False)
    
    # Extract effective weights - handle W-Asym and Noise-Asym differently
    standard_state = standard_model.state_dict()
    
    if symmetry == 1:
        # W-Asym: weight.data is already modified by reset_parameters(), need to capture what forward() actually uses
        # Hook into F.conv2d and F.linear to capture effective weights, tracking by module name
        captured_weights = {}
        active_module = [None]
        original_conv2d = F.conv2d
        original_linear = F.linear
        
        def conv2d_hook(input, weight, bias=None, stride=1, padding=0, dilation=1, groups=1):
            if active_module[0] is not None:
                module_name = active_module[0]
                weight_name = module_name + '.weight' if module_name else 'weight'
                if weight_name not in captured_weights:
                    captured_weights[weight_name] = weight.detach().clone()
            return original_conv2d(input, weight, bias, stride, padding, dilation, groups)
        
        def linear_hook(input, weight, bias=None):
            if active_module[0] is not None:
                module_name = active_module[0]
                weight_name = module_name + '.weight' if module_name else 'weight'
                if weight_name not in captured_weights:
                    captured_weights[weight_name] = weight.detach().clone()
            return original_linear(input, weight, bias)
        
        F.conv2d = conv2d_hook
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
            dummy_input = torch.randn(1, 3, 32, 32)
            with torch.no_grad():
                _ = asym_model(dummy_input)
        finally:
            F.conv2d = original_conv2d
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
    
    # Create a small batch of random inputs (CIFAR format: batch_size x 3 x 32 x 32)
    batch_size = 4
    test_input = torch.randn(batch_size, 3, 32, 32)
    
    with torch.no_grad():
        asym_output = asym_model(test_input)
        standard_output = standard_model(test_input)
    
    # Check if outputs match
    max_diff = (asym_output - standard_output).abs().max().item()
    mean_diff = (asym_output - standard_output).abs().mean().item()
    max_rel_diff = ((asym_output - standard_output).abs() / (asym_output.abs() + 1e-8)).max().item()
    
    sym_type_name = "W-Asym" if symmetry == 1 else "Noise-Asym"
    print(f"Initialized standard ResNet-{depth} from {sym_type_name} model")
    print(f"Function equivalence check:")
    print(f"  Max absolute difference: {max_diff:.2e}")
    print(f"  Mean absolute difference: {mean_diff:.2e}")
    print(f"  Max relative difference: {max_rel_diff:.2e}")
    
    # Check if they're close enough (within numerical precision)
    tolerance = 1e-5
    if max_diff < tolerance:
        print(f"Models implement the same function (difference < {tolerance})")
    else:
        print(f"ERROR: Models differ by more than {tolerance}")
        print(f"    Max absolute difference: {max_diff:.2e}")
        print(f"    This indicates an issue with weight copying.")
        raise ValueError(f"Failed to achieve exact equivalence. Models differ by {max_diff:.2e} (tolerance: {tolerance})")
    
    return standard_model


# Convenience function for backward compatibility
def create_resnet(symmetry, depth, w=1, mask_params=None, num_classes=10, mask_seed=None, n_mul=1.0, asym_init_only=False):
    """Create ResNet based on symmetry type and depth.
    
    Args:
        symmetry: 0=Standard, 1=W-Asym, 3=Noise-Asym
        depth: ResNet depth (20, 32, 56, 110)
        w: Width multiplier
        mask_params: Mask parameters for W-Asym and Noise-Asym
        num_classes: Number of output classes
        n_mul: Multiplier for num_fixed values (useful for wider ResNets)
        asym_init_only: If True and symmetry in [1, 3], initialize standard model with weights from asym model
    """
    # Handle asym_init_only: use asym model only for initialization, create standard model
    if asym_init_only and symmetry in [1, 3]:
        return _initialize_from_asym_resnet(symmetry, depth, w, mask_params, num_classes, n_mul)
    
    # Apply n_mul to all num_fixed values in mask_params
    if mask_params is not None and n_mul != 1.0:
        print(f"Applying n_mul={n_mul} to mask_params")
        mask_params = _apply_n_mul_to_mask_params(mask_params, n_mul)
        print(f"After applying n_mul, sample values: {mask_params.get('conv_f', {}).get('num_fixed', 'N/A')}")
    
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
            return w_resnet20(mask_params, w=w, num_classes=num_classes, mask_seed=mask_seed)
        elif depth == 32:
            return w_resnet32(mask_params, w=w, num_classes=num_classes, mask_seed=mask_seed)
        elif depth == 56:
            return w_resnet56(mask_params, w=w, num_classes=num_classes, mask_seed=mask_seed)
        elif depth == 110:
            return w_resnet110(mask_params, w=w, num_classes=num_classes, mask_seed=mask_seed)
        else:
            raise ValueError(f"Invalid depth: {depth}")
    elif symmetry == 3:
        if mask_params is None:
            raise ValueError("mask_params required for Noise-Asym ResNet")
        if depth == 20:
            return noise_resnet20(mask_params, w=w, num_classes=num_classes, mask_seed=mask_seed)
        elif depth == 32:
            return noise_resnet32(mask_params, w=w, num_classes=num_classes, mask_seed=mask_seed)
        elif depth == 56:
            return noise_resnet56(mask_params, w=w, num_classes=num_classes, mask_seed=mask_seed)
        elif depth == 110:
            return noise_resnet110(mask_params, w=w, num_classes=num_classes, mask_seed=mask_seed)
        else:
            raise ValueError(f"Invalid depth: {depth}")
    else:
        raise ValueError(f"Invalid symmetry type: {symmetry}")


# Original LMC mask generation functions
def make_conv_mask(in_channels, out_channels, kernel_size, mask_rng: torch.Generator, mask_type='random_subsets', num_fixed=6):
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
            zeros_in_out_channel = get_subset(weights_per_out_channel, least_zeros, mask_rng)
            for zero_ind in map(flattened_to_3d_index, zeros_in_out_channel):
                mask[out_channel_idx][zero_ind] = 0
        return mask
    elif mask_type == 'filter_random_subsets':
        mask = torch.ones(size=(out_channels, in_channels, kernel_size, kernel_size))
        least_zeros = num_fixed
        
        for out_channel_idx in range(out_channels):
            zeros_in_out_channel = get_subset(in_channels, least_zeros, mask_rng)
            for zero_ind in zeros_in_out_channel:
                mask[out_channel_idx, zero_ind, :, :] = 0
        return mask
    elif mask_type == 'none':
        return torch.ones(size=(out_channels, in_channels, kernel_size, kernel_size))


def conv_normal_mask(out_channels, in_channels, kernel_size, mask_rng: torch.Generator):
    """Create normal mask for initialization (original LMC implementation)."""
    return torch.randn(size=(out_channels, in_channels, kernel_size, kernel_size), generator=mask_rng)


def get_subset(num_cols, num_sample, mask_rng: torch.Generator):
    """Get random subset of indices (original LMC implementation)."""
    return torch.randperm(num_cols, generator=mask_rng)[:num_sample]
