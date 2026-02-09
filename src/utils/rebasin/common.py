from collections import defaultdict
import enum
from typing import Callable, Dict, List, NamedTuple, Tuple
import torch
from src.utils.record_activations import ActivationRecordingPoint, HookMode, LayerName

type PermutationName = str

class ActivationCorrelationMode(str, enum.Enum):
  DotProduct = "dot_product" # the dot product as described in Git Rebasin
  PearsonCorrelation = "pearson_correlation" # the Pearson correlation between activations
  PearsonCorrelationWithZeroForConstant = "pearson_correlation_with_zero_for_constant" # the Pearson correlation between activations, but if a hidden dimension is constant, it's correlation with all other is replaced by 0 (not actually the correct correlation)

class PermutationSpec(NamedTuple):
  perm_to_axes: dict
  axes_to_perm: dict
  activation_matching_modes: Dict[str, Dict[PermutationName, List[ActivationRecordingPoint]]]
  reshape_activation_tensor: Callable[[torch.Tensor], torch.Tensor] | None=None

class MatchingResult(NamedTuple):
  optimal_permutation: torch.Tensor
  activation_similarities: torch.Tensor

def create_permutation_spec(
  axes_to_perm: dict,
  activation_matching_modes: Dict[PermutationName, Dict[PermutationName, List[ActivationRecordingPoint]]],
  reshape_activation_tensor: Callable[[torch.Tensor], torch.Tensor] | None=None
) -> PermutationSpec:
  perm_to_axes = defaultdict(list)
  for wk, axis_perms in axes_to_perm.items():
    for axis, perm in enumerate(axis_perms):
      if perm is not None:
        perm_to_axes[perm].append((wk, axis))
  return PermutationSpec(
    perm_to_axes=dict(perm_to_axes),
    axes_to_perm=axes_to_perm,
    activation_matching_modes=activation_matching_modes,
    reshape_activation_tensor=reshape_activation_tensor
  )

def mlp_permutation_spec(num_hidden_layers: int, norm = False, bias = True) -> PermutationSpec:
  """We assume that one permutation cannot appear in two axes of the same weight array."""
  assert num_hidden_layers >= 1
  
  activation_matching_modes = {
    "post_linear": {f"P_{i}": [(f"lins.{i}", HookMode.RecordOutput)] for i in range(num_hidden_layers)},
    "post_norm": {f"P_{i}": [(f"norms.{i}", HookMode.RecordOutput)] for i in range(num_hidden_layers)},
    "post_activation_function": {f"P_{i}": [(f"lins.{i+1}", HookMode.RecordInput)] for i in range(num_hidden_layers)}
  }
  if not norm:
      if bias:
          return create_permutation_spec({
              "lins.0.weight": ("P_0", None),
              **{f"lins.{i}.weight": ( f"P_{i}", f"P_{i-1}")
                 for i in range(1, num_hidden_layers)},
              **{f"lins.{i}.bias": (f"P_{i}", )
                 for i in range(num_hidden_layers)},
              f"lins.{num_hidden_layers}.weight": (None, f"P_{num_hidden_layers-1}"),
              f"lins.{num_hidden_layers}.bias": (None, ),
          },
          activation_matching_modes=activation_matching_modes)
      else:
          return create_permutation_spec({
              "lins.0.weight": ("P_0", None),
              **{f"lins.{i}.weight": ( f"P_{i}", f"P_{i-1}")
                 for i in range(1, num_hidden_layers)},
              f"lins.{num_hidden_layers}.weight": (None, f"P_{num_hidden_layers-1}"),
          },
          activation_matching_modes=activation_matching_modes)
  else:
    return create_permutation_spec({
      "lins.0.weight": ("P_0", None),
      **{f"lins.{i}.weight": ( f"P_{i}", f"P_{i-1}")
         for i in range(1, num_hidden_layers)},
      **{f"lins.{i}.bias": (f"P_{i}", )
         for i in range(num_hidden_layers)},
      **{f"norms.{i}.weight": (f"P_{i}", )
         for i in range(num_hidden_layers)},
      **{f"norms.{i}.bias": (f"P_{i}", )
         for i in range(num_hidden_layers)},
      f"lins.{num_hidden_layers}.weight": (None, f"P_{num_hidden_layers-1}"),
      f"lins.{num_hidden_layers}.bias": (None, ),
    },
    activation_matching_modes=activation_matching_modes)
      

"""
def cnn_permutation_spec() -> PermutationSpec:
  conv = lambda name, p_in, p_out: {f"{name}.weight": (p_out, p_in, None, None, )}
  dense = lambda name, p_in, p_out, bias=True: {f"{name}.weight": (p_out, p_in), f"{name}.bias": (p_out, )} if bias else  {f"{name}.weight": (p_out, p_in)}

  return permutation_spec_from_axes_to_perm({
     **conv("conv1", None, "P_bg0"),
     **conv("conv2", "P_bg0", "P_bg1"),
     **dense("fc1", "P_bg1", "P_bg2"),
     **dense("fc2", "P_bg2", None, False),
  })
"""



def resnet20_permutation_spec() -> PermutationSpec:
  conv = lambda name, p_in, p_out: {f"{name}.weight": (p_out, p_in, None, None, )}
  norm = lambda name, p: {f"{name}.weight": (p, ), f"{name}.bias": (p, )}
  dense = lambda name, p_in, p_out: {f"{name}.weight": (p_out, p_in), f"{name}.bias": (p_out, )}

  # This is for easy blocks that use a residual connection, without any change in the number of channels.
  easyblock = lambda name, p: {
      **conv(f"{name}.conv1", p, f"P_{name}_inner"),
      **norm(f"{name}.ln1", f"P_{name}_inner"),
          
      **conv(f"{name}.conv2", f"P_{name}_inner", p),
      **norm(f"{name}.ln2", p),
      
  }

  # This is for blocks that use a residual connection, but change the number of channels via a Conv.
  shortcutblock = lambda name, p_in, p_out: {
      **conv(f"{name}.conv1", p_in, f"P_{name}_inner"),
      **norm(f"{name}.ln1", f"P_{name}_inner"),
          
      **conv(f"{name}.conv2", f"P_{name}_inner", p_out),
      **norm(f"{name}.ln2", p_out),
      **conv(f"{name}.shortcut.0", p_in, p_out),
      **norm(f"{name}.shortcut.1", p_out),
  }
  NUM_LAYERS = 3
  NUM_BLOCKS_PER_LAYER = 3
  activation_matching_modes = {
    "post_activation_function": {
      # We need to record for post-activation function:
      "P_bg0": [
        ("layer1.0.conv1", HookMode.RecordInput),
        ("layer1.1.conv1", HookMode.RecordInput),
        ("layer1.2.conv1", HookMode.RecordInput),
        ("layer2.0.conv1", HookMode.RecordInput),
      ],
      # - for P_bg{i} (= "block group"/layer{i+1} outer permutation, where +1 comes from 1-indexing): the input to layer{i+1}.1.conv1 (since layer{i+1}.0.conv1 is before the shortcut + still under the same permutation as the previous)
      "P_bg1": [
        ("layer2.1.conv1", HookMode.RecordInput),
        ("layer2.2.conv1", HookMode.RecordInput),
        ("layer3.0.conv1", HookMode.RecordInput),
      ],
      "P_bg2": [
        ("layer3.1.conv1", HookMode.RecordInput),
        ("layer3.2.conv1", HookMode.RecordInput),
      ],
      # - for P_layer{i+1}.{j}_inner (i.e. permutation between conv1 and conv2 of that block): the input to layer{i+1}.{j}.conv2
      **{
        f"P_layer{i+1}.{j}_inner": [(f"layer{i+1}.{j}.conv2", HookMode.RecordInput)]
        for i in range(NUM_LAYERS)
        for j in range(NUM_BLOCKS_PER_LAYER)
      },
    }
  }

  def reshape_activation_tensor_resnet20(input_tensor: torch.Tensor):
    assert input_tensor.ndim == 4 # Expecting: (dataset size) x (num channels) x width x height
    # Convention here: average all pixels in one channels (alternatively one could treat each pixel as a separate data input)
    # return input_tensor.mean(dim=3).mean(dim=2)
    return input_tensor.moveaxis(1, 3).reshape(-1, input_tensor.shape[1])

  return create_permutation_spec({
      **conv("conv1", None, "P_bg0"),
      **norm("ln1", "P_bg0"),
            #
      **easyblock("layer1.0", "P_bg0"),
      **easyblock("layer1.1", "P_bg0",),
      **easyblock("layer1.2", "P_bg0"),
            #
      
      **shortcutblock("layer2.0", "P_bg0", "P_bg1"),
      **easyblock("layer2.1", "P_bg1",),
      **easyblock("layer2.2", "P_bg1"),
      #**easyblock("layer2.3", "P_bg2"),
      
      **shortcutblock("layer3.0", "P_bg1", "P_bg2"),
      **easyblock("layer3.1", "P_bg2",),
      **easyblock("layer3.2", "P_bg2"),
      #
      **dense("linear", "P_bg2", None),
    },
    activation_matching_modes=activation_matching_modes,
    reshape_activation_tensor=reshape_activation_tensor_resnet20,
  )

# # should be easy to generalize it to any depth
# def resnet50_permutation_spec() -> PermutationSpec:
#   conv = lambda name, p_in, p_out: {f"{name}.weight": (p_out, p_in, None, None, )}
#   norm = lambda name, p: {f"{name}.weight": (p, ), f"{name}.bias": (p, )}
#   dense = lambda name, p_in, p_out: {f"{name}.weight": (p_out, p_in), f"{name}.bias": (p_out, )}

#   # This is for easy blocks that use a residual connection, without any change in the number of channels.
#   easyblock = lambda name, p: {
#   **norm(f"{name}.bn1", p),
#   **conv(f"{name}.conv1", p, f"P_{name}_inner"),
#   **norm(f"{name}.bn2", f"P_{name}_inner"),
#   **conv(f"{name}.conv2", f"P_{name}_inner", p),
#   }

#   # This is for blocks that use a residual connection, but change the number of channels via a Conv.
#   shortcutblock = lambda name, p_in, p_out: {
#   **norm(f"{name}.bn1", p_in),
#   **conv(f"{name}.conv1", p_in, f"P_{name}_inner"),
#   **norm(f"{name}.bn2", f"P_{name}_inner"),
#   **conv(f"{name}.conv2", f"P_{name}_inner", p_out),
#   **conv(f"{name}.shortcut.0", p_in, p_out),
#   **norm(f"{name}.shortcut.1", p_out),
#   }

#   return permutation_spec_from_axes_to_perm({
#     **conv("conv1", None, "P_bg0"),
#     #
#     **shortcutblock("layer1.0", "P_bg0", "P_bg1"),
#     **easyblock("layer1.1", "P_bg1",),
#     **easyblock("layer1.2", "P_bg1"),
#     **easyblock("layer1.3", "P_bg1"),
#     **easyblock("layer1.4", "P_bg1"),
#     **easyblock("layer1.5", "P_bg1"),
#     **easyblock("layer1.6", "P_bg1"),
#     **easyblock("layer1.7", "P_bg1"),

#     #**easyblock("layer1.3", "P_bg1"),

#     **shortcutblock("layer2.0", "P_bg1", "P_bg2"),
#     **easyblock("layer2.1", "P_bg2",),
#     **easyblock("layer2.2", "P_bg2"),
#     **easyblock("layer2.3", "P_bg2"),
#     **easyblock("layer2.4", "P_bg2"),
#     **easyblock("layer2.5", "P_bg2"),
#     **easyblock("layer2.6", "P_bg2"),
#     **easyblock("layer2.7", "P_bg2"),

#     **shortcutblock("layer3.0", "P_bg2", "P_bg3"),
#     **easyblock("layer3.1", "P_bg3",),
#     **easyblock("layer3.2", "P_bg3"),
#     **easyblock("layer3.3", "P_bg3"),
#     **easyblock("layer3.4", "P_bg3"),
#     **easyblock("layer3.5", "P_bg3"),
#     **easyblock("layer3.6", "P_bg3"),
#     **easyblock("layer3.7", "P_bg3"),

#     **norm("bn1", "P_bg3"),

#     **dense("linear", "P_bg3", None),

# })



# def vgg16_permutation_spec() -> PermutationSpec:
#   layers_with_conv = [3,7,10,14,17,20,24,27,30,34,37,40]
#   layers_with_conv_b4 = [0,3,7,10,14,17,20,24,27,30,34,37]
#   layers_with_bn = [4,8,11,15,18,21,25,28,31,35,38,41]
#   dense = lambda name, p_in, p_out, bias = True: {f"{name}.weight": (p_out, p_in), f"{name}.bias": (p_out, )}
#   return permutation_spec_from_axes_to_perm({
#       # first features
#       "features.0.weight": ( "P_Conv_0",None, None, None),
#       "features.1.weight": ( "P_Conv_0", None),
#       "features.1.bias": ( "P_Conv_0", None),
#       "features.1.running_mean": ( "P_Conv_0", None),
#       "features.1.running_var": ( "P_Conv_0", None),
#       "features.1.num_batches_tracked": (),

#       **{f"features.{layers_with_conv[i]}.weight": ( f"P_Conv_{layers_with_conv[i]}", f"P_Conv_{layers_with_conv_b4[i]}", None, None, )
#         for i in range(len(layers_with_conv))},
#       **{f"features.{i}.bias": (f"P_Conv_{i}", )
#         for i in layers_with_conv + [0]},
#       # bn
#       **{f"features.{layers_with_bn[i]}.weight": ( f"P_Conv_{layers_with_conv[i]}", None)
#         for i in range(len(layers_with_bn))},
#       **{f"features.{layers_with_bn[i]}.bias": ( f"P_Conv_{layers_with_conv[i]}", None)
#         for i in range(len(layers_with_bn))},
#       **{f"features.{layers_with_bn[i]}.running_mean": ( f"P_Conv_{layers_with_conv[i]}", None)
#         for i in range(len(layers_with_bn))},
#       **{f"features.{layers_with_bn[i]}.running_var": ( f"P_Conv_{layers_with_conv[i]}", None)
#         for i in range(len(layers_with_bn))},
#       **{f"features.{layers_with_bn[i]}.num_batches_tracked": ()
#         for i in range(len(layers_with_bn))},

#       **dense("classifier", "P_Conv_40", "P_Dense_0", False),
# })

def get_permuted_param(ps: PermutationSpec, perm, k: str, params, except_axis=None):
  """Get parameter `k` from `params`, with the permutations applied."""
  w = params[k]
  if k not in ps.axes_to_perm:
    print(f"Skipping {k}")
    return w
  for axis, p in enumerate(ps.axes_to_perm[k]):
    # Skip the axis we're trying to permute.
    if axis == except_axis:
      continue

    # None indicates that there is no permutation relevant to that axis.
    if p is not None:
        w = torch.index_select(w, axis, perm[p].int())

  return w

def apply_permutation(ps: PermutationSpec, perm, params):
  """Apply a `perm` to `params`."""
  return {k: get_permuted_param(ps, perm, k, params) for k in params.keys()}

